"""DWA-lite (Dynamic Window Approach, Fox/Burgard/Thrun 1997) local fallback.

When the Pure Pursuit command `(v_pp, w_pp)` is vetoed by
`safety.predict_safe`, we sample a grid of candidate twists around it,
score each, and return the best safe candidate. If every candidate is
unsafe, we return a zero twist — which the follower surfaces as ABORT.

Scoring (higher is better):
  + progress   : how much closer to the lookahead point the predicted pose is
  + align      : cosine of heading error to lookahead
  + velocity   : prefer higher |v|
  - unsafe     : treated as -inf (filtered out)

Deliberately not implementing the full "admissible velocity" + "dynamic
window" obstacle-clearance distance terms — those require a distance-to-
nearest-obstacle query we don't have cheaply. Safety veto via costmap
lookahead is strict enough.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .planner import GridInfo
from .safety import SafetyParams, predict_safe


@dataclass
class DWAConfig:
    v_samples: int = 7
    w_samples: int = 11
    v_span: float = 0.25       # +/- from the Pure Pursuit nominal
    w_span: float = 1.2        # +/- from the Pure Pursuit nominal
    w_progress: float = 1.0    # weight on forward progress
    w_align: float = 0.5       # weight on heading alignment
    w_velocity: float = 0.1    # weight on |v|


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def _simulate(
    x: float, y: float, yaw: float, v: float, w: float, dt: float, ticks: int
) -> tuple[float, float, float]:
    """Unicycle forward simulation; returns pose after ticks×dt."""
    for _ in range(ticks):
        if abs(w) < 1e-5:
            x = x + v * math.cos(yaw) * dt
            y = y + v * math.sin(yaw) * dt
        else:
            yaw_new = yaw + w * dt
            x = x + (v / w) * (math.sin(yaw_new) - math.sin(yaw))
            y = y - (v / w) * (math.cos(yaw_new) - math.cos(yaw))
            yaw = yaw_new
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return x, y, yaw


def choose_safe(
    pp_v: float,
    pp_w: float,
    pose: tuple[float, float, float],
    lookahead: tuple[float, float],
    costmap: np.ndarray,
    info: GridInfo,
    safety: SafetyParams,
    cfg: DWAConfig,
) -> tuple[float, float, str]:
    """Return (v, w, reason).

    If the Pure Pursuit command passes safety, return it unchanged with
    reason='pp'. Otherwise sample around it, score, and return the best safe
    candidate with reason='dwa'. If none is safe, return (0, 0, 'blocked').
    """
    # Fast path: nominal is safe.
    ok, _, _ = predict_safe(costmap, info, pose, (pp_v, pp_w), safety)
    if ok:
        return pp_v, pp_w, "pp"

    x, y, yaw = pose
    lx, ly = lookahead
    dist0 = math.hypot(lx - x, ly - y)

    # Sample grid around (pp_v, pp_w).
    vs = np.linspace(pp_v - cfg.v_span, pp_v + cfg.v_span, cfg.v_samples)
    ws = np.linspace(pp_w - cfg.w_span, pp_w + cfg.w_span, cfg.w_samples)

    best: tuple[float, float, float] | None = None  # (score, v, w)
    sim_dt = safety.lookahead_dt / max(1, safety.lookahead_ticks)
    sim_ticks = safety.lookahead_ticks

    for v_c in vs:
        v_c = max(-safety.max_linear, min(safety.max_linear, float(v_c)))
        for w_c in ws:
            w_c = max(-safety.max_angular, min(safety.max_angular, float(w_c)))
            ok, _, _ = predict_safe(costmap, info, pose, (v_c, w_c), safety)
            if not ok:
                continue
            # Forward-sim once cheaply for scoring (safety already did so but
            # didn't expose the final pose — keep it simple, re-simulate).
            nx, ny, nyaw = _simulate(x, y, yaw, v_c, w_c, sim_dt, sim_ticks)
            # Progress: reduction in distance to lookahead.
            dist1 = math.hypot(lx - nx, ly - ny)
            progress = dist0 - dist1
            # Alignment: heading error after simulation to lookahead.
            des_yaw = math.atan2(ly - nx, lx - nx) if False else math.atan2(ly - ny, lx - nx)
            align = math.cos(_angle_diff(des_yaw, nyaw))

            score = (
                cfg.w_progress * progress
                + cfg.w_align * align
                + cfg.w_velocity * abs(v_c)
            )
            if best is None or score > best[0]:
                best = (score, v_c, w_c)

    if best is None:
        return 0.0, 0.0, "blocked"
    return best[1], best[2], "dwa"
