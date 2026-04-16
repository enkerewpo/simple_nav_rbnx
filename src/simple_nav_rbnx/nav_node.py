"""ROS 2 node: wires Atlas discovery + topics into the follower.

Responsibilities:
  - Subscribe to the Atlas-resolved occupancy grid and odometry topics.
  - Subscribe (optional) to the scan_2d topic for live obstacles.
  - Publish `geometry_msgs/Twist` to the base's `twist_in` topic.
  - Maintain a `GoalState` registry indexed by goal_id.
  - Launch / cancel follower threads on navigate / cancel requests.

This module is only imported inside `atlas_bridge.py` (the process entrypoint)
so `rclpy` / `geometry_msgs` are a lazy dependency — unit tests for planner
and safety don't need ROS installed.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

from .dwa import DWAConfig
from .follower import Follower, FollowerConfig
from .goal_state import GoalState, GoalStatus
from .planner import GridInfo, inflate_costmap, plan, world_to_cell
from .pure_pursuit import PurePursuitConfig
from .safety import SafetyParams

log = logging.getLogger("simple_nav_rbnx.nav_node")


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """ROS quaternion → yaw (planar)."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


@dataclass
class Topics:
    cmd_vel: str
    odom: str
    occupancy_grid: str
    scan_2d: Optional[str] = None


class NavNode(Node):
    """Single-goal navigation node. Concurrent goals are not supported (last wins)."""

    def __init__(self, topics: Topics, config_path: Optional[str] = None) -> None:
        super().__init__("simple_nav_rbnx")
        self.topics = topics

        # Load config (with sane defaults)
        cfg: dict = {}
        if config_path:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}

        self._robot_radius = float(cfg.get("robot_radius", 0.25))
        pp_cfg = cfg.get("pure_pursuit", {}) or {}
        dwa_cfg = cfg.get("dwa", {}) or {}
        self._follower_cfg = FollowerConfig(
            hz=float(cfg.get("follower_hz", 20)),
            pure_pursuit=PurePursuitConfig(
                lookahead=float(pp_cfg.get("lookahead", 0.8)),
                lookahead_min=float(pp_cfg.get("lookahead_min", 0.25)),
                lookahead_gain=float(pp_cfg.get("lookahead_gain", 0.0)),
                cruise_speed=float(pp_cfg.get("cruise_speed", 0.35)),
                goal_slowdown_dist=float(pp_cfg.get("goal_slowdown_dist", 0.8)),
                max_linear=float(cfg.get("max_linear", 0.5)),
                max_angular=float(cfg.get("max_angular", 1.0)),
            ),
            dwa=DWAConfig(
                v_samples=int(dwa_cfg.get("v_samples", 7)),
                w_samples=int(dwa_cfg.get("w_samples", 11)),
                v_span=float(dwa_cfg.get("v_span", 0.25)),
                w_span=float(dwa_cfg.get("w_span", 1.2)),
                w_progress=float(dwa_cfg.get("w_progress", 1.0)),
                w_align=float(dwa_cfg.get("w_align", 0.5)),
                w_velocity=float(dwa_cfg.get("w_velocity", 0.1)),
            ),
            safety=SafetyParams(
                robot_radius=self._robot_radius,
                extra_margin=float(cfg.get("safety_inflate_extra", 0.1)),
                lookahead_dt=float(cfg.get("lookahead_dt", 0.4)),
                lookahead_ticks=int(cfg.get("lookahead_ticks", 5)),
                max_linear=float(cfg.get("max_linear", 0.5)),
                max_angular=float(cfg.get("max_angular", 1.0)),
            ),
        )
        self._default_tolerance = float(cfg.get("goal_tolerance", 0.3))
        self._default_timeout = float(cfg.get("timeout", 120.0))
        self._scan_safety_enabled = bool(cfg.get("scan_safety_enabled", True))

        # Shared state, guarded by `_state_lock`
        self._state_lock = threading.Lock()
        self._pose: Optional[tuple[float, float, float]] = None
        self._costmap: Optional[np.ndarray] = None
        self._grid_info: Optional[GridInfo] = None
        self._scan: Optional[tuple[list[float], float, float, tuple[float, float, float]]] = None
        self._goals: dict[str, GoalState] = {}
        self._active_goal: Optional[str] = None

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._cmd_pub = self.create_publisher(Twist, topics.cmd_vel, qos)
        self.create_subscription(Odometry, topics.odom, self._on_odom, qos)
        self.create_subscription(OccupancyGrid, topics.occupancy_grid, self._on_grid, 1)
        if topics.scan_2d and self._scan_safety_enabled:
            self.create_subscription(LaserScan, topics.scan_2d, self._on_scan, qos)

        # RViz integration: accept the "2D Nav Goal" tool's default topic,
        # publish path + markers for visualization. Remember the
        # map frame_id so published Path/Markers match the RViz fixed frame.
        self._map_frame = "map"
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_pose, 1)
        self._path_pub = self.create_publisher(Path, "/simple_nav/path", 1)
        self._goal_marker_pub = self.create_publisher(Marker, "/simple_nav/goal_marker", 1)
        self._lookahead_marker_pub = self.create_publisher(
            Marker, "/simple_nav/lookahead_marker", 1
        )

        self.get_logger().info(
            f"simple_nav_rbnx wired: cmd_vel={topics.cmd_vel} odom={topics.odom} "
            f"grid={topics.occupancy_grid} scan={topics.scan_2d or '(disabled)'} "
            "goal=/goal_pose"
        )

    # ─── subscribers ─────────────────────────────────────────────────────────
    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        yaw = _quat_to_yaw(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        with self._state_lock:
            self._pose = (p.position.x, p.position.y, yaw)

    def _on_grid(self, msg: OccupancyGrid) -> None:
        data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        info = GridInfo(
            resolution=msg.info.resolution,
            width=msg.info.width,
            height=msg.info.height,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
        )
        inflated = inflate_costmap(data, info, radius_m=self._robot_radius)
        with self._state_lock:
            self._costmap = inflated
            self._grid_info = info
            if msg.header.frame_id:
                self._map_frame = msg.header.frame_id

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        """RViz 2D Nav Goal handler: direct submit, bypassing gRPC."""
        ok, gid, reason = self.submit_goal(msg)
        if ok:
            self.get_logger().info(f"[goal_pose] accepted via RViz: {gid}")
        else:
            self.get_logger().warn(f"[goal_pose] rejected: {reason}")

    def _on_scan(self, msg: LaserScan) -> None:
        with self._state_lock:
            pose = self._pose
        if pose is None:
            return
        with self._state_lock:
            self._scan = (list(msg.ranges), float(msg.angle_min), float(msg.angle_increment), pose)

    # ─── follower callbacks ──────────────────────────────────────────────────
    def _get_pose(self) -> tuple[float, float, float]:
        with self._state_lock:
            if self._pose is None:
                return (0.0, 0.0, 0.0)
            return self._pose

    def _get_costmap(self) -> tuple[np.ndarray, GridInfo]:
        with self._state_lock:
            if self._costmap is None or self._grid_info is None:
                raise RuntimeError("costmap not yet received")
            return self._costmap, self._grid_info

    def _get_scan(self):
        with self._state_lock:
            return self._scan

    def _publish_twist(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def _publish_path_viz(self, path_world: list[tuple[float, float]]) -> None:
        p = Path()
        p.header.frame_id = self._map_frame
        p.header.stamp = self.get_clock().now().to_msg()
        for wx, wy in path_world:
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        self._path_pub.publish(p)

    def _publish_lookahead_viz(self, pt: tuple[float, float]) -> None:
        m = Marker()
        m.header.frame_id = self._map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "simple_nav"
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(pt[0])
        m.pose.position.y = float(pt[1])
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.18
        m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
        self._lookahead_marker_pub.publish(m)

    def _publish_goal_marker(self, goal: GoalState) -> None:
        m = Marker()
        m.header.frame_id = self._map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "simple_nav"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(goal.goal_x)
        m.pose.position.y = float(goal.goal_y)
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 2 * float(goal.tolerance)
        m.scale.z = 0.05
        m.color = ColorRGBA(r=0.1, g=1.0, b=0.3, a=0.5)
        self._goal_marker_pub.publish(m)

    # ─── public API called by atlas_bridge ───────────────────────────────────
    def submit_goal(
        self,
        goal_pose: PoseStamped,
        tolerance: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> tuple[bool, str, str]:
        """Start a new goal. If another goal is active, cancel it first."""
        with self._state_lock:
            if self._pose is None:
                return False, "", "no odometry yet"
            if self._costmap is None or self._grid_info is None:
                return False, "", "no costmap yet"
            start_pose = self._pose
            costmap = self._costmap.copy()
            info = self._grid_info

        goal_yaw: Optional[float] = None
        q = goal_pose.pose.orientation
        if any(abs(v) > 1e-6 for v in (q.x, q.y, q.z, q.w)):
            goal_yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

        goal = GoalState.new(
            x=goal_pose.pose.position.x,
            y=goal_pose.pose.position.y,
            yaw=goal_yaw,
            tolerance=tolerance if tolerance is not None else self._default_tolerance,
            timeout_s=timeout if timeout is not None else self._default_timeout,
        )

        start_cell = world_to_cell(info, start_pose[0], start_pose[1])
        goal_cell = world_to_cell(info, goal.goal_x, goal.goal_y)
        path = plan(costmap, info, start_cell, goal_cell)
        if not path.found:
            goal.set_status(GoalStatus.ABORTED, "no feasible path")
            with self._state_lock:
                self._goals[goal.goal_id] = goal
            return False, goal.goal_id, "no feasible path"

        # Cancel any active goal before starting a new one
        with self._state_lock:
            if self._active_goal is not None:
                prev = self._goals.get(self._active_goal)
                if prev is not None and not prev.is_terminal():
                    prev.cancel()
            self._goals[goal.goal_id] = goal
            self._active_goal = goal.goal_id

        # RViz viz: goal marker now, path + lookahead streamed by follower
        self._publish_goal_marker(goal)

        # Thread owns the follower lifecycle
        follower = Follower(
            self._follower_cfg,
            get_pose=self._get_pose,
            get_costmap=self._get_costmap,
            get_scan=self._get_scan,
            publish_twist=self._publish_twist,
            publish_path=self._publish_path_viz,
            publish_lookahead=self._publish_lookahead_viz,
        )
        t = threading.Thread(
            target=follower.run,
            args=(goal, path.world),
            name=f"follower-{goal.goal_id}",
            daemon=True,
        )
        goal.thread = t
        t.start()
        return True, goal.goal_id, "accepted"

    def cancel(self, goal_id: str) -> tuple[bool, str]:
        with self._state_lock:
            g = self._goals.get(goal_id)
        if g is None:
            return False, "unknown goal_id"
        if g.is_terminal():
            return False, f"already terminal: {g.status.value}"
        g.cancel()
        return True, "cancel signaled"

    def status(self, goal_id: str) -> Optional[GoalState]:
        with self._state_lock:
            return self._goals.get(goal_id)
