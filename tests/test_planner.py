"""A* planner tests on hand-rolled grids."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simple_nav_rbnx.planner import (  # noqa: E402
    GridInfo,
    cell_to_world,
    inflate_costmap,
    plan,
    world_to_cell,
)


def _info(w: int, h: int, res: float = 1.0) -> GridInfo:
    return GridInfo(resolution=res, width=w, height=h, origin_x=0.0, origin_y=0.0)


def test_straight_corridor_returns_path():
    cm = np.zeros((5, 10), dtype=np.int8)  # row, col
    info = _info(10, 5)
    r = plan(cm, info, (0, 2), (9, 2))
    assert r.found
    assert r.cells[0] == (0, 2)
    assert r.cells[-1] == (9, 2)
    assert r.distance_m == pytest.approx(9.0, rel=1e-6)


def test_wall_blocks_direct_path_and_routes_around():
    cm = np.zeros((7, 10), dtype=np.int8)
    # vertical wall at col=5 from row 0..5
    cm[0:6, 5] = 100
    info = _info(10, 7)
    r = plan(cm, info, (0, 3), (9, 3))
    assert r.found
    # Must bend south-ish around the wall
    assert any(row == 6 for _, row in r.cells)


def test_unreachable_returns_not_found():
    cm = np.zeros((5, 5), dtype=np.int8)
    cm[:, 2] = 100  # full vertical wall splits the map in half
    info = _info(5, 5)
    r = plan(cm, info, (0, 2), (4, 2))
    assert not r.found
    assert r.distance_m == math.inf


def test_start_or_goal_blocked_rejects():
    cm = np.zeros((3, 3), dtype=np.int8)
    cm[1, 1] = 100
    info = _info(3, 3)
    assert not plan(cm, info, (1, 1), (2, 2)).found
    assert not plan(cm, info, (0, 0), (1, 1)).found


def test_diagonal_corner_cutting_prevented():
    # Classic L-shape: diagonal would slip through 1-cell diagonal gap
    cm = np.zeros((3, 3), dtype=np.int8)
    cm[0, 1] = 100
    cm[1, 0] = 100
    info = _info(3, 3)
    r = plan(cm, info, (0, 0), (1, 1))
    # Can't reach (1,1) from (0,0) since both orthogonal neighbors are blocked
    assert not r.found


def test_unknown_cells_blocked_by_default():
    cm = np.full((3, 3), -1, dtype=np.int8)
    cm[0, 0] = 0
    cm[2, 2] = 0
    info = _info(3, 3)
    # All middle is unknown — blocked with default
    assert not plan(cm, info, (0, 0), (2, 2)).found
    # But when unknown is allowed, reachable
    r = plan(cm, info, (0, 0), (2, 2), unknown_is_blocked=False)
    assert r.found


def test_weighted_cells_preferred_over_expensive_ones():
    cm = np.zeros((3, 7), dtype=np.int8)
    cm[1, 1:6] = 80  # expensive direct band
    info = _info(7, 3)
    r = plan(cm, info, (0, 1), (6, 1))
    assert r.found
    # Prefer going around the weighted band
    touched_weighted = sum(1 for c, row in r.cells if 1 <= c <= 5 and row == 1)
    went_around = any(row == 0 or row == 2 for _, row in r.cells)
    assert went_around or touched_weighted == 0


def test_world_cell_roundtrip():
    info = GridInfo(resolution=0.05, width=100, height=80, origin_x=-1.0, origin_y=-2.0)
    col, row = world_to_cell(info, 0.0, 0.0)
    x, y = cell_to_world(info, col, row)
    # Rounded to cell center; should be within ±resolution/2
    assert abs(x - 0.025) < 0.026
    assert abs(y - 0.025) < 0.026


def test_inflate_costmap_dilates_lethal():
    cm = np.zeros((10, 10), dtype=np.int8)
    cm[5, 5] = 100
    info = _info(10, 10, res=0.1)
    inflated = inflate_costmap(cm, info, radius_m=0.25)
    # Cell 2 away (Manhattan) should be lethal too
    assert inflated[5, 3] == 100 or inflated[3, 5] == 100
    # Corners still free
    assert inflated[0, 0] == 0
