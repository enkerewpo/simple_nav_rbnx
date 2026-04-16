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

# Provided interfaces. Each is served via two transports:
#   - grpc  : programmatic robot-to-robot RPC (TODO: bind generated stubs;
#             today the bridge advertises the port but doesn't run a real grpc
#             server — Pilot reaches navigation via the MCP tool path below).
#   - mcp   : VLM-callable tool surface, served by FastMCP/uvicorn from
#             `mcp_server.py`. This is the path Pilot's executor actually
#             dispatches over today (see rust/.../dispatch/mcp.rs).
PROVIDED_INTERFACES = [
    ("navigate", "robonix/srv/navigation/navigate"),
    ("status", "robonix/srv/navigation/status"),
    ("cancel", "robonix/srv/navigation/cancel"),
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


# ── Free-port helper ─────────────────────────────────────────────────────────

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

    grpc_port = args.port or _pick_free_port()
    mcp_port = int(os.environ.get("NAV_MCP_PORT", "0")) or _pick_free_port()
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

    # 3) Start ROS node on a background thread (must exist before MCP server is
    #    built, since each MCP tool closes over `nav.submit_goal/status/cancel`).
    rclpy.init()
    nav = NavNode(topics=topics, config_path=args.config)

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(nav,),
        name="rclpy-spin",
        daemon=True,
    )
    ros_thread.start()

    # 4) Build MCP app + start uvicorn in a background thread. Pilot's executor
    #    dispatches `navigate / get_navigation_status / cancel_navigation` here.
    from .mcp_server import build_mcp_app, serve as serve_mcp
    mcp_app, mcp_tools = build_mcp_app(nav)
    mcp_endpoint = f"http://127.0.0.1:{mcp_port}/mcp"

    def _run_mcp() -> None:
        try:
            serve_mcp(mcp_app, host="127.0.0.1", port=mcp_port)
        except Exception:
            log.exception("[mcp] server crashed")

    threading.Thread(target=_run_mcp, name="nav-mcp", daemon=True).start()
    log.info("[mcp] serving %d nav tools at %s", len(mcp_tools), mcp_endpoint)

    # 5) Declare provided interfaces with both transports.
    tools_by_iface = {t["iface_name"]: t for t in mcp_tools}
    for iface_name, contract_id in PROVIDED_INTERFACES:
        meta = {
            "grpc_endpoint": f"localhost:{grpc_port}",
            "endpoint": mcp_endpoint,
        }
        if iface_name in tools_by_iface:
            t = tools_by_iface[iface_name]
            meta["tools"] = [{
                "name": t["tool_name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }]
        try:
            stub.DeclareInterface(pb.DeclareInterfaceRequest(
                node_id=NODE_ID,
                name=iface_name,
                supported_transports=["grpc", "mcp"],
                metadata_json=json.dumps(meta),
                listen_port=grpc_port,
                contract_id=contract_id,
            ))
            log.info("[atlas] declared %s (grpc:%d, mcp:%s)",
                     contract_id, grpc_port, mcp_endpoint)
        except grpc.RpcError as e:
            log.error("[atlas] DeclareInterface %s failed: %s", contract_id, e)

    # 6) Heartbeat thread
    def _heartbeat() -> None:
        while True:
            try:
                stub.NodeHeartbeat(pb.NodeHeartbeatRequest(node_id=NODE_ID))
            except grpc.RpcError:
                pass
            time.sleep(10)

    threading.Thread(target=_heartbeat, daemon=True).start()

    log.info("simple_nav_rbnx ready: atlas=%s grpc=%d mcp=%d topics=%r",
             args.atlas, grpc_port, mcp_port, topics)

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
