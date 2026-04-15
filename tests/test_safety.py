"""Safety lookahead predicate tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simple_nav_rbnx.planner import GridInfo  # noqa: E402
from simple_nav_rbnx.safety import (  # noqa: E402
    SafetyParams,
    combine_scan_into_costmap,
    predict_safe,
)


def _info(w: int = 40, h: int = 40, res: float = 0.1) -> GridInfo:
    return GridInfo(resolution=res, width=w, height=h, origin_x=0.0, origin_y=0.0)


def test_clear_ahead_is_safe():
    cm = np.zeros((40, 40), dtype=np.int8)
    info = _info()
    params = SafetyParams(robot_radius=0.2, extra_margin=0.0, lookahead_dt=0.4, lookahead_ticks=4)
    ok, reason, _ = predict_safe(cm, info, (1.0, 1.0, 0.0), (0.3, 0.0), params)
    assert ok, reason


def test_wall_ahead_is_unsafe():
    cm = np.zeros((40, 40), dtype=np.int8)
    # Wall at col 15 → world x ≈ 1.5m
    cm[:, 15] = 100
    info = _info()
    # Robot at x=1.3m, moving east at 0.5 m/s. Front-edge of footprint is
    # already inside the wall's inflated region after one lookahead tick.
    params = SafetyParams(robot_radius=0.15, extra_margin=0.05, lookahead_dt=0.8, lookahead_ticks=5, max_linear=1.0)
    ok, _reason, _ = predict_safe(cm, info, (1.3, 1.0, 0.0), (0.5, 0.0), params)
    assert not ok


def test_off_map_is_unsafe():
    cm = np.zeros((10, 10), dtype=np.int8)
    info = _info(10, 10)
    params = SafetyParams(robot_radius=0.1, lookahead_dt=1.0, lookahead_ticks=5, max_linear=2.0)
    ok, reason, _ = predict_safe(cm, info, (0.5, 0.5, 0.0), (2.0, 0.0), params)
    assert not ok
    assert "bounds" in reason


def test_zero_twist_never_collides_on_clear_cell():
    cm = np.zeros((10, 10), dtype=np.int8)
    info = _info(10, 10)
    params = SafetyParams(robot_radius=0.1)
    ok, _, _ = predict_safe(cm, info, (0.5, 0.5, 0.0), (0.0, 0.0), params)
    assert ok


def test_scan_overlay_blocks_previously_clear_path():
    cm = np.zeros((40, 40), dtype=np.int8)
    info = _info()
    params = SafetyParams(robot_radius=0.15)

    # Nothing static: clear
    ok_before, _, _ = predict_safe(cm, info, (1.0, 1.0, 0.0), (0.3, 0.0), params)
    assert ok_before

    # Simulate a LaserScan with one return straight ahead at 0.4m
    ranges = [0.4]
    a_min = 0.0
    a_inc = 0.01
    overlaid = combine_scan_into_costmap(cm, info, (1.0, 1.0, 0.0), ranges, a_min, a_inc, inflate_radius_m=0.1)

    ok_after, _reason, _ = predict_safe(overlaid, info, (1.0, 1.0, 0.0), (0.3, 0.0), params)
    assert not ok_after
