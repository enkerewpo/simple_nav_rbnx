"""Pure Pursuit path tracker (Coulter 1992) for a differential/ackermann unicycle.

For each control tick:
  1. Find the point on `path` that is at lookahead distance `Ld` ahead of the
     robot along the path (measured from the point nearest to the robot).
  2. Transform that point into the robot body frame.
  3. Curvature κ = 2·y_body / Ld²   (standard PP formula).
  4. v = cruise_speed (optionally slowed near the end of the path).
  5. w = v · κ.

Advantages over per-waypoint PID:
  - No oscillation on long segments; output is a smooth curve.
  - Graceful on tight turns because Ld grows / shrinks with speed if we want.
  - Single tuning parameter (Ld) for the whole path.

We keep `cruise_speed` and `goal_slowdown_dist` as knobs; the DWA fallback
downstream handles actually-safe speeds when obstacles appear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PurePursuitConfig:
    lookahead: float = 0.8          # meters, target lookahead distance Ld
    lookahead_min: float = 0.25     # lower clamp
    lookahead_gain: float = 0.0     # if >0: Ld = lookahead + gain*v (adaptive)
    cruise_speed: float = 0.35      # m/s nominal along-path speed
    goal_slowdown_dist: float = 0.8 # start decelerating this far from the end
    max_linear: float = 0.5
    max_angular: float = 1.0


def _nearest_index(path: list[tuple[float, float]], x: float, y: float) -> int:
    best = 0
    bd2 = float("inf")
    for i, (px, py) in enumerate(path):
        d2 = (px - x) * (px - x) + (py - y) * (py - y)
        if d2 < bd2:
            bd2 = d2
            best = i
    return best


def _lookahead_point(
    path: list[tuple[float, float]],
    start_idx: int,
    x: float,
    y: float,
    ld: float,
) -> tuple[float, float, int]:
    """Walk forward from `start_idx` until cumulative path distance from (x,y)
    exceeds `ld`. Returns (px, py, segment_idx). If the path ends first, the
    final path point is returned.
    """
    # Distance from (x, y) along the path starting at start_idx.
    acc = math.hypot(path[start_idx][0] - x, path[start_idx][1] - y)
    if acc >= ld:
        return path[start_idx][0], path[start_idx][1], start_idx
    for i in range(start_idx, len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        seg = math.hypot(x2 - x1, y2 - y1)
        if acc + seg >= ld:
            # Linear interp along this segment
            remainder = ld - acc
            t = remainder / seg if seg > 1e-9 else 0.0
            return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, i
        acc += seg
    return path[-1][0], path[-1][1], len(path) - 1


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_command(
    cfg: PurePursuitConfig,
    pose: tuple[float, float, float],     # x, y, yaw
    current_v: float,                     # current linear speed (m/s)
    path: list[tuple[float, float]],
    last_nearest_idx: int = 0,
) -> tuple[float, float, int, tuple[float, float]]:
    """Return (v_cmd, w_cmd, nearest_idx, lookahead_point_world)."""
    if len(path) < 2:
        return 0.0, 0.0, 0, path[0] if path else (pose[0], pose[1])

    x, y, yaw = pose

    # Robot's nearest point on the path (cheap linear scan; path lengths
    # from A* are small in practice, a few hundred cells).
    nearest = _nearest_index(path, x, y)
    # Don't let nearest regress when the robot is moving smoothly.
    nearest = max(last_nearest_idx, nearest)

    # Adaptive lookahead if configured.
    ld = max(cfg.lookahead_min, cfg.lookahead + cfg.lookahead_gain * max(0.0, current_v))

    lx, ly, _ = _lookahead_point(path, nearest, x, y, ld)

    # Transform lookahead into body frame.
    dx = lx - x
    dy = ly - y
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    bx = cos_y * dx + sin_y * dy
    by = -sin_y * dx + cos_y * dy

    # Pure Pursuit curvature: κ = 2·by / L² where L = actual distance to LH point.
    l2 = bx * bx + by * by
    if l2 < 1e-9:
        kappa = 0.0
    else:
        kappa = 2.0 * by / l2

    # Goal-slowdown: taper cruise speed near the end of the path.
    remaining = math.hypot(path[-1][0] - x, path[-1][1] - y)
    if remaining < cfg.goal_slowdown_dist and cfg.goal_slowdown_dist > 1e-6:
        scale = max(0.0, remaining / cfg.goal_slowdown_dist)
    else:
        scale = 1.0

    # If the lookahead is behind the robot (bx < 0) the path loops back; slow
    # down and let the controller pivot in place.
    if bx < 0.1:
        v_cmd = 0.0
    else:
        v_cmd = cfg.cruise_speed * scale

    w_cmd = v_cmd * kappa
    # Small bias so we still rotate in place when v_cmd=0 but heading is off.
    if v_cmd == 0.0 and abs(by) > 0.01:
        w_cmd = math.copysign(min(cfg.max_angular, 0.6 + 2.0 * abs(by)), by)

    v_cmd = _clip(v_cmd, -cfg.max_linear, cfg.max_linear)
    w_cmd = _clip(w_cmd, -cfg.max_angular, cfg.max_angular)

    return v_cmd, w_cmd, nearest, (lx, ly)
