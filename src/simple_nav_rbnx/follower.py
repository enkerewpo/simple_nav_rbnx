"""PID waypoint follower.

Runs on its own thread, 20 Hz by default. For each tick:
  1. Pop the current waypoint (advance when within `switch_dist`).
  2. Compute a candidate `(v, w)` twist from two 1-D PID controllers
     (linear on along-track distance, angular on heading error).
  3. Clip to max_linear / max_angular.
  4. Call safety.predict_safe(); if it returns False, ABORT with zero twist.
  5. Publish via the injected `publish_twist` callback.

The follower doesn't know about ROS or Atlas — it's a pure loop over callbacks,
so it's unit-testable with stub publishers.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .goal_state import GoalState, GoalStatus
from .planner import GridInfo
from .safety import SafetyParams, combine_scan_into_costmap, predict_safe

log = logging.getLogger("simple_nav_rbnx.follower")


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    i: float = 0.0
    prev_err: float = 0.0
    i_clamp: float = 1.0

    def reset(self) -> None:
        self.i = 0.0
        self.prev_err = 0.0

    def step(self, err: float, dt: float) -> float:
        self.i = max(-self.i_clamp, min(self.i_clamp, self.i + err * dt))
        d = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        return self.kp * err + self.ki * self.i + self.kd * d


@dataclass
class FollowerConfig:
    hz: float = 20.0
    pid_linear: tuple[float, float, float] = (0.8, 0.0, 0.1)
    pid_angular: tuple[float, float, float] = (1.5, 0.0, 0.2)
    switch_waypoint_dist: float = 0.15   # advance to next waypoint when this close
    safety: SafetyParams = None          # type: ignore  — injected by caller


def _angle_diff(a: float, b: float) -> float:
    """a - b, wrapped to [-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class Follower:
    """Drives a `GoalState` along a precomputed world-path.

    Callbacks are injected so this class has zero ROS / Atlas dependencies:
      - `get_pose()    -> (x, y, yaw)` — current robot pose in map frame
      - `get_costmap() -> (np.ndarray, GridInfo)` — live combined costmap
      - `get_scan()    -> (ranges, angle_min, angle_inc, pose) | None` — optional live scan
      - `publish_twist(v, w)` — effect a velocity command
    """

    def __init__(
        self,
        cfg: FollowerConfig,
        get_pose: Callable[[], tuple[float, float, float]],
        get_costmap: Callable[[], tuple[np.ndarray, GridInfo]],
        get_scan: Callable[[], Optional[tuple[list[float], float, float, tuple[float, float, float]]]],
        publish_twist: Callable[[float, float], None],
    ) -> None:
        self.cfg = cfg
        self.get_pose = get_pose
        self.get_costmap = get_costmap
        self.get_scan = get_scan
        self.publish_twist = publish_twist
        self.pid_lin = PID(*cfg.pid_linear)
        self.pid_ang = PID(*cfg.pid_angular)
        assert cfg.safety is not None, "SafetyParams must be provided"
        self.safety = cfg.safety

    def run(self, goal: GoalState, path_world: list[tuple[float, float]]) -> None:
        """Blocking: drive `goal` to termination. Caller runs us on a thread."""
        if not path_world:
            goal.set_status(GoalStatus.ABORTED, "empty path")
            self.publish_twist(0.0, 0.0)
            return

        goal.set_status(GoalStatus.ACTIVE, "following")
        self.pid_lin.reset()
        self.pid_ang.reset()

        wp_idx = 0
        period = 1.0 / max(1.0, self.cfg.hz)
        last_t = time.time()

        while True:
            if goal.cancel_event.is_set():
                self.publish_twist(0.0, 0.0)
                goal.set_status(GoalStatus.CANCELED, "cancel requested")
                return
            if goal.elapsed_s >= goal.timeout_s:
                self.publish_twist(0.0, 0.0)
                goal.set_status(GoalStatus.ABORTED, f"timeout after {goal.timeout_s:.1f}s")
                return

            x, y, yaw = self.get_pose()

            # Advance waypoint if close enough
            while wp_idx < len(path_world) - 1:
                wx, wy = path_world[wp_idx]
                if math.hypot(wx - x, wy - y) < self.cfg.switch_waypoint_dist:
                    wp_idx += 1
                else:
                    break

            # Goal convergence check
            gx, gy = goal.goal_x, goal.goal_y
            dist_to_goal = math.hypot(gx - x, gy - y)
            if dist_to_goal <= goal.tolerance:
                self.publish_twist(0.0, 0.0)
                goal.update_pose(x, y, yaw, 0.0)
                goal.set_status(GoalStatus.SUCCEEDED, "goal reached")
                return

            # Target is the current waypoint (or goal if at last waypoint)
            tx, ty = path_world[wp_idx]
            # Heading to target
            desired_yaw = math.atan2(ty - y, tx - x)
            yaw_err = _angle_diff(desired_yaw, yaw)

            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            # Linear speed: scale down if heading is far off (don't race forward sideways)
            align = max(0.0, math.cos(yaw_err))
            along = math.hypot(tx - x, ty - y) * align
            v_cmd = self.pid_lin.step(along, dt)
            w_cmd = self.pid_ang.step(yaw_err, dt)

            # Hard caps
            v_cmd = max(-self.safety.max_linear, min(self.safety.max_linear, v_cmd))
            w_cmd = max(-self.safety.max_angular, min(self.safety.max_angular, w_cmd))

            # Safety lookahead on combined costmap (static ⊕ live scan)
            base_cost, info = self.get_costmap()
            scan = self.get_scan()
            if scan is not None:
                ranges, a_min, a_inc, scan_pose = scan
                costmap = combine_scan_into_costmap(
                    base_cost, info, scan_pose, ranges, a_min, a_inc,
                    self.safety.robot_radius,
                )
            else:
                costmap = base_cost

            ok, reason, _poses = predict_safe(costmap, info, (x, y, yaw), (v_cmd, w_cmd), self.safety)
            if not ok:
                log.warning("[follower] unsafe command vetoed: %s; aborting goal %s", reason, goal.goal_id)
                self.publish_twist(0.0, 0.0)
                goal.set_status(GoalStatus.ABORTED, f"safety: {reason}")
                return

            self.publish_twist(v_cmd, w_cmd)
            goal.update_pose(x, y, yaw, dist_to_goal)

            # Sleep respecting rate
            sleep_left = period - (time.time() - now)
            if sleep_left > 0:
                time.sleep(sleep_left)
