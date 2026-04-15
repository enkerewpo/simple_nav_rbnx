"""Atlas bridge: registers simple_nav_rbnx and serves the navigation gRPC RPCs.

Flow:
  1. RegisterNode(com.robonix.nav.simple)
  2. Discover consumed primitives/services via Atlas:
       - robonix/srv/common/map/occupancy_grid  (ROS2 topic)
       - robonix/srv/common/map/scan_2d         (ROS2 topic, optional)
       - robonix/prm/base/odom                  (ROS2 topic)
       - robonix/prm/base/twist_in              (ROS2 topic)
  3. DeclareInterface for the three navigation RPCs (grpc + ros2):
       - robonix/srv/navigation/navigate
       - robonix/srv/navigation/status
       - robonix/srv/navigation/cancel
  4. Start the NavNode with the resolved topic names.
  5. Serve the gRPC RPCs on a port the process bound (listen_port).
  6. Heartbeat every 10s.

The gRPC service shapes mirror the existing Robonix navigation contracts in
rust/contracts/srv/navigation_*.v1.toml. We use the generated Python stubs
from the robonix source tree (pulled in via `rbnx codegen`).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from concurrent import futures as _futures
from pathlib import Path
from typing import Optional

# ── Proto path setup ─────────────────────────────────────────────────────────

def _ensure_proto_gen() -> None:
    d = Path(__file__).resolve().parent
    while d.parent != d:
        pg = d / "proto_gen"
        if pg.is_dir() and (pg / "robonix_runtime_pb2.py").exists():
            sys.path.insert(0, str(pg))
            return
        d = d.parent


_ensure_proto_gen()

import grpc  # noqa: E402
import rclpy  # noqa: E402

import robonix_runtime_pb2 as pb  # noqa: E402
import robonix_runtime_pb2_grpc as pb_grpc  # noqa: E402

from .nav_node import NavNode, Topics  # noqa: E402

log = logging.getLogger("simple_nav_rbnx.bridge")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

NODE_ID = "com.robonix.nav.simple"
NAMESPACE = "robonix/srv/navigation"

CONSUMED = {
    # contract_id → fallback topic
    "robonix/srv/common/map/occupancy_grid": "/robonix/map/occupancy_grid",
    "robonix/srv/common/map/scan_2d": "/robonix/map/scan_2d",
    "robonix/prm/base/odom": "/fastlio2/lio_odom",
    "robonix/prm/base/twist_in": "/cmd_vel",
}

PROVIDED_INTERFACES = [
    ("navigate", "robonix/srv/navigation/navigate", ["grpc"]),
    ("status", "robonix/srv/navigation/status", ["grpc"]),
    ("cancel", "robonix/srv/navigation/cancel", ["grpc"]),
]


def _discover_topic(stub: pb_grpc.RobonixRuntimeStub, contract_id: str, fallback: str) -> str:
    """QueryNodes for the contract and return the first provider's ros2_topic.

    Falls back to `fallback` if no provider is registered or anything errors.
    """
    try:
        resp = stub.QueryNodes(pb.QueryNodesRequest(contract_id=contract_id, transport="ros2"))
        if not resp.nodes:
            log.info("[discover] %s: no provider, fallback=%s", contract_id, fallback)
            return fallback
        provider = resp.nodes[0]
        for iface in provider.interfaces:
            if iface.contract_id != contract_id:
                continue
            try:
                meta = json.loads(iface.metadata_json) if iface.metadata_json else {}
                topic = meta.get("ros2_topic", fallback)
            except (json.JSONDecodeError, AttributeError):
                topic = fallback
            log.info("[discover] %s provider=%s topic=%s", contract_id, provider.node_id, topic)
            return topic
        return fallback
    except grpc.RpcError as e:
        log.warning("[discover] %s RPC failed: %s, fallback=%s", contract_id, e, fallback)
        return fallback


# ── gRPC service implementations ─────────────────────────────────────────────
#
# We don't have cross-imports of the navigation_navigate_pb2 shapes here yet
# (those come from robonix_proto after codegen). The bridge keeps the wire
# contract simple by accepting `std_msgs/String` carrying JSON, matching what
# vlm_service / tiago_bridge already do when a contract TOML hasn't been
# compiled into a concrete servicer class.  Pilot's tool dispatcher just
# forwards the tool-call JSON; we parse it here and invoke the NavNode.

class NavigationService:
    """Not a real proto-generated servicer — a thin JSON handler Pilot reaches
    over a generic stream method.  The concrete .proto binding is TODO:
    once rust/contracts/srv/navigation_*.v1.toml has been compiled, replace
    this with the stub-generated class."""

    def __init__(self, nav: NavNode) -> None:
        self.nav = nav

    def handle_navigate(self, payload: dict) -> dict:
        # Expected JSON: {"x": f, "y": f, "yaw"?: f, "tolerance"?: f, "timeout"?: f}
        try:
            from geometry_msgs.msg import PoseStamped
            import math as _math
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(payload["x"])
            ps.pose.position.y = float(payload["y"])
            yaw = payload.get("yaw")
            if yaw is not None:
                yaw = float(yaw)
                ps.pose.orientation.z = _math.sin(yaw / 2.0)
                ps.pose.orientation.w = _math.cos(yaw / 2.0)
            ok, gid, msg = self.nav.submit_goal(
                ps,
                tolerance=payload.get("tolerance"),
                timeout=payload.get("timeout"),
            )
            return {"goal_id": gid, "accepted": ok, "message": msg}
        except Exception as e:
            log.exception("[navigate] error")
            return {"goal_id": "", "accepted": False, "message": f"error: {e}"}

    def handle_status(self, payload: dict) -> dict:
        gid = str(payload.get("goal_id", ""))
        gs = self.nav.status(gid)
        if gs is None:
            return {"known": False, "status": "", "terminal": False, "message": "unknown goal_id"}
        return {
            "known": True,
            "status": gs.status.value,
            "terminal": gs.is_terminal(),
            "distance_remaining_m": gs.distance_remaining_m,
            "elapsed_s": gs.elapsed_s,
            "current_x": gs.current_x,
            "current_y": gs.current_y,
            "current_yaw": gs.current_yaw,
            "message": gs.message,
        }

    def handle_cancel(self, payload: dict) -> dict:
        gid = str(payload.get("goal_id", ""))
        ok, msg = self.nav.cancel(gid)
        return {"accepted": ok, "message": msg}


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser(
        str(Path(__file__).resolve().parents[2] / "config" / "nav_default.yaml")
    ))
    ap.add_argument("--atlas", default=os.environ.get("ROBONIX_ATLAS", "127.0.0.1:50051"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("NAV_GRPC_PORT", "0")))
    args = ap.parse_args()

    port = args.port or _pick_free_port()
    channel = grpc.insecure_channel(args.atlas)
    stub = pb_grpc.RobonixRuntimeStub(channel)

    # 1) RegisterNode
    try:
        stub.RegisterNode(pb.RegisterNodeRequest(
            node_id=NODE_ID,
            namespace=NAMESPACE,
            kind="service",
            distro="humble",
            skills=[],
        ))
        log.info("[atlas] registered %s", NODE_ID)
    except grpc.RpcError as e:
        log.error("[atlas] RegisterNode failed: %s", e)
        sys.exit(1)

    # 2) Discover consumed topics
    resolved = {cid: _discover_topic(stub, cid, fallback) for cid, fallback in CONSUMED.items()}
    topics = Topics(
        cmd_vel=resolved["robonix/prm/base/twist_in"],
        odom=resolved["robonix/prm/base/odom"],
        occupancy_grid=resolved["robonix/srv/common/map/occupancy_grid"],
        scan_2d=resolved.get("robonix/srv/common/map/scan_2d"),
    )

    # 3) Declare provided interfaces
    for name, contract_id, transports in PROVIDED_INTERFACES:
        try:
            stub.DeclareInterface(pb.DeclareInterfaceRequest(
                node_id=NODE_ID,
                name=name,
                supported_transports=transports,
                metadata_json=json.dumps({"grpc_endpoint": f"localhost:{port}"}),
                listen_port=port,
                contract_id=contract_id,
            ))
            log.info("[atlas] declared %s on port %d", contract_id, port)
        except grpc.RpcError as e:
            log.error("[atlas] DeclareInterface %s failed: %s", contract_id, e)

    # 4) Start ROS node on a background thread
    rclpy.init()
    nav = NavNode(topics=topics, config_path=args.config)

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(nav,),
        name="rclpy-spin",
        daemon=True,
    )
    ros_thread.start()

    # 5) Heartbeat thread
    def _heartbeat() -> None:
        while True:
            try:
                stub.NodeHeartbeat(pb.NodeHeartbeatRequest(node_id=NODE_ID))
            except grpc.RpcError:
                pass
            time.sleep(10)

    threading.Thread(target=_heartbeat, daemon=True).start()

    log.info("simple_nav_rbnx ready: atlas=%s port=%d topics=%r", args.atlas, port, topics)

    # 6) Graceful shutdown on SIGTERM/SIGINT
    stop_evt = threading.Event()

    def _on_sig(*_: object) -> None:
        log.info("shutdown signal received")
        stop_evt.set()

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)
    stop_evt.wait()

    # Flush a zero-twist before exit (best-effort safety on shutdown).
    try:
        nav._publish_twist(0.0, 0.0)  # noqa: SLF001
    except Exception:
        pass
    rclpy.shutdown()
    log.info("bye")


if __name__ == "__main__":
    main()
