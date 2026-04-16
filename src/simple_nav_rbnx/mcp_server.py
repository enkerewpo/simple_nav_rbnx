"""MCP HTTP server exposing the navigation contracts as VLM-callable tools.

The MCP codegen only emits classes from `.msg` files, not `.srv` (`Navigate.srv`,
`GetNavigationStatus.srv`, `CancelNavigation.srv` have rich nested fields). Until
that changes, we wrap each request as a JSON payload carried inside
`std_msgs/String.data` and document the actual fields via JSON Schema in the
tool description / Atlas metadata.

Tool surface (all I/O = std_msgs/String carrying JSON):
  - navigate(goal_x, goal_y, goal_yaw?, tolerance?, timeout_s?)
      -> {"goal_id", "accepted", "message"}
  - get_navigation_status(goal_id)
      -> {"known", "status", "terminal", "current_x", "current_y",
          "current_yaw", "distance_remaining_m", "elapsed_s", "message"}
  - cancel_navigation(goal_id)
      -> {"accepted", "message"}
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("simple_nav_rbnx.mcp")


def _ensure_mcp_types() -> None:
    """Add this package's robonix_mcp_types/ to sys.path."""
    d = Path(__file__).resolve().parent
    while d.parent != d:
        mt = d / "robonix_mcp_types"
        if mt.is_dir() and (mt / "__init__.py").exists():
            if str(mt) not in sys.path:
                sys.path.insert(0, str(mt))
            return
        d = d.parent


def _ensure_robonix_py() -> None:
    import subprocess
    try:
        out = subprocess.run(
            ["rbnx", "path", "robonix-py"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            lib = Path(out.stdout.strip())
            if lib.is_dir() and str(lib) not in sys.path:
                sys.path.insert(0, str(lib))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


_ensure_mcp_types()
_ensure_robonix_py()

import std_msgs_mcp  # noqa: E402
from robonix_py import mcp_contract  # noqa: E402


def build_mcp_app(nav) -> tuple[FastMCP, list[dict]]:
    """Build a FastMCP app bound to the given NavNode instance.

    Returns (mcp_instance, declared_tools_meta) where the meta list is what the
    bridge passes into Atlas DeclareInterface metadata_json so Pilot can show
    the tools to the VLM with proper JSON Schema.
    """
    mcp = FastMCP("simple_nav_provider")

    # ── navigate ──────────────────────────────────────────────────────────────
    NAVIGATE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "goal_x": {"type": "number", "description": "Target x in map frame (m)."},
            "goal_y": {"type": "number", "description": "Target y in map frame (m)."},
            "goal_yaw": {"type": "number", "description": "Optional target yaw (rad)."},
            "tolerance": {"type": "number", "description": "Arrival tolerance (m); 0 = default 0.3."},
            "timeout_s": {"type": "number", "description": "Abort if not done after this many seconds; 0 = 120."},
        },
        "required": ["goal_x", "goal_y"],
    }

    @mcp_contract(mcp, contract_id="robonix/srv/navigation/navigate", name="navigate")
    async def _navigate(msg: std_msgs_mcp.String) -> std_msgs_mcp.String:
        """Send the robot to (goal_x, goal_y) in the map frame. Optional yaw, tolerance, timeout_s.

        Input JSON (std_msgs/String.data): {"goal_x": float, "goal_y": float,
        "goal_yaw"?: float, "tolerance"?: float, "timeout_s"?: float}.
        Returns JSON: {"goal_id", "accepted", "message"}.
        """
        from geometry_msgs.msg import PoseStamped
        try:
            payload: dict[str, Any] = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError as e:
            return std_msgs_mcp.String(
                data=json.dumps({"goal_id": "", "accepted": False,
                                 "message": f"invalid JSON: {e}"})
            )
        try:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(payload["goal_x"])
            ps.pose.position.y = float(payload["goal_y"])
            yaw = payload.get("goal_yaw")
            if yaw is not None:
                yaw = float(yaw)
                ps.pose.orientation.z = math.sin(yaw / 2.0)
                ps.pose.orientation.w = math.cos(yaw / 2.0)
            else:
                ps.pose.orientation.w = 1.0
            ok, gid, message = nav.submit_goal(
                ps,
                tolerance=payload.get("tolerance"),
                timeout=payload.get("timeout_s"),
            )
            return std_msgs_mcp.String(
                data=json.dumps({"goal_id": gid, "accepted": bool(ok), "message": message})
            )
        except (KeyError, ValueError, TypeError) as e:
            return std_msgs_mcp.String(
                data=json.dumps({"goal_id": "", "accepted": False,
                                 "message": f"bad payload: {e}"})
            )

    # ── get_navigation_status ────────────────────────────────────────────────
    STATUS_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"goal_id": {"type": "string"}},
        "required": ["goal_id"],
    }

    @mcp_contract(mcp, contract_id="robonix/srv/navigation/status", name="get_navigation_status")
    async def _status(msg: std_msgs_mcp.String) -> std_msgs_mcp.String:
        """Poll status of a previously-submitted nav goal_id.

        Input JSON: {"goal_id": str}.
        Returns JSON with progress fields; `terminal` is True once the goal succeeded,
        was aborted, or was cancelled.
        """
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError as e:
            return std_msgs_mcp.String(
                data=json.dumps({"known": False, "message": f"invalid JSON: {e}"})
            )
        gid = str(payload.get("goal_id", ""))
        gs = nav.status(gid)
        if gs is None:
            return std_msgs_mcp.String(
                data=json.dumps({"known": False, "status": "", "terminal": False,
                                 "message": "unknown goal_id"})
            )
        return std_msgs_mcp.String(
            data=json.dumps({
                "known": True,
                "status": gs.status.value,
                "terminal": gs.is_terminal(),
                "distance_remaining_m": gs.distance_remaining_m,
                "elapsed_s": gs.elapsed_s,
                "current_x": gs.current_x,
                "current_y": gs.current_y,
                "current_yaw": gs.current_yaw,
                "message": gs.message,
            })
        )

    # ── cancel_navigation ────────────────────────────────────────────────────
    CANCEL_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"goal_id": {"type": "string"}},
        "required": ["goal_id"],
    }

    @mcp_contract(mcp, contract_id="robonix/srv/navigation/cancel", name="cancel_navigation")
    async def _cancel(msg: std_msgs_mcp.String) -> std_msgs_mcp.String:
        """Cancel an in-flight navigation goal (immediate stop + zero-twist)."""
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError as e:
            return std_msgs_mcp.String(
                data=json.dumps({"accepted": False, "message": f"invalid JSON: {e}"})
            )
        gid = str(payload.get("goal_id", ""))
        ok, message = nav.cancel(gid)
        return std_msgs_mcp.String(
            data=json.dumps({"accepted": bool(ok), "message": message})
        )

    # Tool metadata for Atlas DeclareInterface (one entry per provided contract).
    declared = [
        {
            "contract_id": "robonix/srv/navigation/navigate",
            "iface_name": "navigate",
            "tool_name": "navigate",
            "description": (
                "Send the robot to (goal_x, goal_y) in the map frame. Costmap "
                "collision-checked; safe to call with a VLM in the loop."
            ),
            "input_schema": NAVIGATE_SCHEMA,
        },
        {
            "contract_id": "robonix/srv/navigation/status",
            "iface_name": "status",
            "tool_name": "get_navigation_status",
            "description": "Poll progress for a navigation goal_id.",
            "input_schema": STATUS_SCHEMA,
        },
        {
            "contract_id": "robonix/srv/navigation/cancel",
            "iface_name": "cancel",
            "tool_name": "cancel_navigation",
            "description": "Cancel an in-flight navigation goal (zero-twist immediately).",
            "input_schema": CANCEL_SCHEMA,
        },
    ]
    return mcp, declared


def serve(mcp: FastMCP, host: str, port: int) -> None:
    """Block on uvicorn serving the MCP streamable-http app."""
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host=host, port=port, log_level="warning")
