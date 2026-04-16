"""Pure Pursuit path follower with DWA-lite safety fallback.

Each 20 Hz tick:
  1. Get current pose, costmap (static + optional live scan), current speed.
  2. Pure Pursuit computes nominal (v_pp, w_pp) to track the path.
  3. safety.predict_safe checks the nominal against the combined costmap.
  4. If safe, publish it.
  5. If unsafe, DWA-lite samples around the nominal, picks the best safe
     candidate by (progress, heading-align, velocity).
  6. If every candidate is unsafe, publish zero twist and ABORT the goal —
     Pilot/user must reissue (intentional: no stuck-in-place recovery yet).

This class has zero ROS / Atlas imports so planner/safety/pure_pursuit/dwa
all stay unit-testable.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .dwa import DWAConfig, choose_safe
from .goal_state import GoalState, GoalStatus
from .planner import GridInfo
from .pure_pursuit import PurePursuitConfig, compute_command
from .safety import SafetyParams, combine_scan_into_costmap

log = logging.getLogger("simple_nav_rbnx.follower")


@dataclass
class FollowerConfig:
    hz: float = 20.0
    pure_pursuit: PurePursuitConfig = field(default_factory=PurePursuitConfig)
    dwa: DWAConfig = field(default_factory=DWAConfig)
    safety: SafetyParams = field(default_factory=SafetyParams)


class Follower:
    """Injectable-callback follower — no ROS imports in this file.

    Callbacks:
      - get_pose()   -> (x, y, yaw)
      - get_costmap()-> (costmap: np.ndarray, info: GridInfo)
      - get_scan()   -> Optional[(ranges, angle_min, angle_inc, pose)]
      - publish_twist(v, w)
      - publish_path(path_world)           — optional RViz viz hook
      - publish_lookahead(point_world)     — optional RViz viz hook
    """

    def __init__(
        self,
        cfg: FollowerConfig,
        get_pose: Callable[[], tuple[float, float, float]],
        get_costmap: Callable[[], tuple[np.ndarray, GridInfo]],
        get_scan: Callable[[], Optional[tuple[list[float], float, float, tuple[float, float, float]]]],
        publish_twist: Callable[[float, float], None],
        publish_path: Optional[Callable[[list[tuple[float, float]]], None]] = None,
        publish_lookahead: Optional[Callable[[tuple[float, float]], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.get_pose = get_pose
        self.get_costmap = get_costmap
        self.get_scan = get_scan
        self.publish_twist = publish_twist
        self.publish_path = publish_path
        self.publish_lookahead = publish_lookahead

    def run(self, goal: GoalState, path_world: list[tuple[float, float]]) -> None:
        if not path_world or len(path_world) < 2:
            goal.set_status(GoalStatus.ABORTED, "empty path")
            self.publish_twist(0.0, 0.0)
            return

        goal.set_status(GoalStatus.ACTIVE, "following")
        if self.publish_path is not None:
            try:
                self.publish_path(path_world)
            except Exception as e:
                log.warning("publish_path failed: %s", e)

        period = 1.0 / max(1.0, self.cfg.hz)
        nearest_idx = 0
        last_v = 0.0

        while True:
            t_start = time.time()

            if goal.cancel_event.is_set():
                self.publish_twist(0.0, 0.0)
                goal.set_status(GoalStatus.CANCELED, "cancel requested")
                return
            if goal.elapsed_s >= goal.timeout_s:
                self.publish_twist(0.0, 0.0)
                goal.set_status(GoalStatus.ABORTED, f"timeout after {goal.timeout_s:.1f}s")
                return

            pose = self.get_pose()
            x, y, _ = pose

            dist_to_goal = math.hypot(goal.goal_x - x, goal.goal_y - y)
            if dist_to_goal <= goal.tolerance:
                self.publish_twist(0.0, 0.0)
                goal.update_pose(x, y, pose[2], 0.0)
                goal.set_status(GoalStatus.SUCCEEDED, "goal reached")
                return

            # 1. Pure Pursuit nominal command
            v_pp, w_pp, nearest_idx, lookahead = compute_command(
                self.cfg.pure_pursuit, pose, last_v, path_world, nearest_idx
            )
            if self.publish_lookahead is not None:
                try:
                    self.publish_lookahead(lookahead)
                except Exception:
                    pass

            # 2. Combined costmap (static ⊕ live scan)
            try:
                base_cost, info = self.get_costmap()
            except RuntimeError:
                # costmap not yet available — stop and wait
                self.publish_twist(0.0, 0.0)
                time.sleep(period)
                continue

            scan = self.get_scan()
            if scan is not None:
                ranges, a_min, a_inc, scan_pose = scan
                costmap = combine_scan_into_costmap(
                    base_cost, info, scan_pose, ranges, a_min, a_inc,
                    self.cfg.safety.robot_radius,
                )
            else:
                costmap = base_cost

            # 3. Safety-gate with DWA fallback
            v_cmd, w_cmd, source = choose_safe(
                v_pp, w_pp, pose, lookahead, costmap, info, self.cfg.safety, self.cfg.dwa
            )
            if source == "blocked":
                log.warning("[follower] all DWA candidates unsafe; aborting goal %s", goal.goal_id)
                self.publish_twist(0.0, 0.0)
                goal.set_status(GoalStatus.ABORTED, "blocked: no safe local command")
                return

            self.publish_twist(v_cmd, w_cmd)
            goal.update_pose(x, y, pose[2], dist_to_goal)
            last_v = v_cmd

            dt_used = time.time() - t_start
            if dt_used < period:
                time.sleep(period - dt_used)
