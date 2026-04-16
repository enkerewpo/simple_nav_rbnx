"""Pure Pursuit controller tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simple_nav_rbnx.pure_pursuit import (  # noqa: E402
    PurePursuitConfig,
    _lookahead_point,
    _nearest_index,
    compute_command,
)


def _line(n: int = 20, y: float = 0.0) -> list[tuple[float, float]]:
    return [(i * 0.2, y) for i in range(n)]


def test_nearest_index_linear():
    path = _line()
    assert _nearest_index(path, 0.0, 0.0) == 0
    assert _nearest_index(path, 1.05, 0.0) == 5  # 0.2 × 5 = 1.0
    assert _nearest_index(path, 3.7, 0.1) == 18


def test_lookahead_point_mid_segment():
    path = _line(20)
    # From (0, 0) with Ld=0.5 along a line spaced 0.2m, the LH point is at x≈0.5
    lx, ly, _ = _lookahead_point(path, 0, 0.0, 0.0, 0.5)
    assert ly == pytest.approx(0.0)
    assert lx == pytest.approx(0.5, abs=0.01)


def test_lookahead_clamped_to_end_of_path():
    path = _line(5)   # length = 0.2 × 4 = 0.8
    lx, ly, _ = _lookahead_point(path, 0, 0.0, 0.0, 5.0)
    assert (lx, ly) == path[-1]


def test_straight_ahead_produces_zero_omega():
    cfg = PurePursuitConfig(lookahead=0.6, cruise_speed=0.3)
    path = _line(20)
    v, w, _, _ = compute_command(cfg, (0.0, 0.0, 0.0), 0.0, path)
    assert v > 0
    assert abs(w) < 1e-3


def test_offset_path_turns_toward_line():
    cfg = PurePursuitConfig(lookahead=0.8, cruise_speed=0.3)
    # Path along y=0; robot at (0, 0.4, 0) — needs to turn right (negative w)
    path = _line(20)
    v, w, _, lh = compute_command(cfg, (0.0, 0.4, 0.0), 0.0, path)
    assert v > 0
    assert w < 0  # turn right to reach y=0


def test_heading_behind_triggers_pivot_in_place():
    cfg = PurePursuitConfig(lookahead=0.8, cruise_speed=0.3)
    path = _line(20)
    # Robot facing backwards: lookahead is behind → v_cmd=0, pivot w≠0
    v, w, _, _ = compute_command(cfg, (0.0, 0.5, math.pi), 0.0, path)
    assert v == 0.0
    assert abs(w) > 0.1


def test_goal_slowdown_near_end():
    cfg = PurePursuitConfig(lookahead=0.4, cruise_speed=0.5, goal_slowdown_dist=1.0)
    path = _line(20)  # end at (3.8, 0)
    # Far from goal: near cruise speed
    v_far, _, _, _ = compute_command(cfg, (0.0, 0.0, 0.0), 0.0, path)
    # Close to goal (0.5m out): slowed to ~half
    v_near, _, _, _ = compute_command(cfg, (3.3, 0.0, 0.0), 0.0, path)
    assert v_far > v_near
    assert v_near < cfg.cruise_speed
