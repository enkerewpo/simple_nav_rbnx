"""A* shortest-path planner on a 2D costmap.

8-connected with diagonal cost √2. Costmap convention:
  0     = free
  100   = occupied / lethal (impassable)
  -1    = unknown (treated as impassable for safety)
  1..99 = traversable but weighted (cost multiplier 1 + v/100)

The grid frame is the ROS OccupancyGrid frame: origin at bottom-left of the
map in `info.origin`, cell (col=x, row=y), `info.resolution` meters per cell.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

SQRT2 = math.sqrt(2.0)


@dataclass
class GridInfo:
    """Mirrors nav_msgs/OccupancyGrid.info minus the quaternion (assumed no rotation)."""
    resolution: float        # meters per cell
    width: int               # cells along x (columns)
    height: int              # cells along y (rows)
    origin_x: float          # world x of cell (0, 0)'s corner
    origin_y: float          # world y of cell (0, 0)'s corner


@dataclass
class PathResult:
    """Result of a plan call."""
    found: bool
    cost: float              # total path cost in cells (× resolution for meters of straight-line portion)
    distance_m: float        # path length in meters (accurate for diag-aware)
    cells: list[tuple[int, int]]   # list of (col, row) including start and goal
    world: list[tuple[float, float]]  # same path in world (x, y)


def world_to_cell(info: GridInfo, x: float, y: float) -> tuple[int, int]:
    col = int(math.floor((x - info.origin_x) / info.resolution))
    row = int(math.floor((y - info.origin_y) / info.resolution))
    return col, row


def cell_to_world(info: GridInfo, col: int, row: int) -> tuple[float, float]:
    """World coordinate of the cell center."""
    x = info.origin_x + (col + 0.5) * info.resolution
    y = info.origin_y + (row + 0.5) * info.resolution
    return x, y


def in_bounds(info: GridInfo, col: int, row: int) -> bool:
    return 0 <= col < info.width and 0 <= row < info.height


def is_traversable(costmap: np.ndarray, col: int, row: int, *, unknown_is_blocked: bool = True) -> bool:
    """`costmap` shape is (height, width) as numpy; value per convention above."""
    v = int(costmap[row, col])
    if v >= 100 or v < 0:
        # -1 unknown (if unknown_is_blocked) or >=100 lethal
        return not (unknown_is_blocked and v < 0) if v < 0 else False
    return True


# 8 neighbor deltas + per-step base cost
_NEIGHBORS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
]


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Octile distance (admissible for 8-connected grid with unit/√2 costs)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def plan(
    costmap: np.ndarray,
    info: GridInfo,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    *,
    max_iters: int = 500_000,
    unknown_is_blocked: bool = True,
) -> PathResult:
    """A* on a 2D OccupancyGrid-style costmap.

    `start_cell` and `goal_cell` are (col, row). Returns an empty PathResult
    if either endpoint is out of bounds/blocked or the goal is unreachable.
    """
    if not (in_bounds(info, *start_cell) and in_bounds(info, *goal_cell)):
        return PathResult(False, math.inf, math.inf, [], [])
    if not is_traversable(costmap, *start_cell, unknown_is_blocked=unknown_is_blocked):
        return PathResult(False, math.inf, math.inf, [], [])
    if not is_traversable(costmap, *goal_cell, unknown_is_blocked=unknown_is_blocked):
        return PathResult(False, math.inf, math.inf, [], [])

    if start_cell == goal_cell:
        x, y = cell_to_world(info, *start_cell)
        return PathResult(True, 0.0, 0.0, [start_cell], [(x, y)])

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start_cell))
    came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start_cell: None}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}

    iters = 0
    while open_heap:
        iters += 1
        if iters > max_iters:
            break
        _, _, current = heapq.heappop(open_heap)

        if current == goal_cell:
            cells: list[tuple[int, int]] = []
            c: Optional[tuple[int, int]] = current
            while c is not None:
                cells.append(c)
                c = came_from[c]
            cells.reverse()
            cost = g_score[goal_cell]
            return PathResult(
                found=True,
                cost=cost,
                distance_m=cost * info.resolution,
                cells=cells,
                world=[cell_to_world(info, *cc) for cc in cells],
            )

        cc, cr = current
        for dc, dr, base in _NEIGHBORS:
            nc, nr = cc + dc, cr + dr
            if not in_bounds(info, nc, nr):
                continue
            if not is_traversable(costmap, nc, nr, unknown_is_blocked=unknown_is_blocked):
                continue
            # Prevent corner-cutting through diagonals: if moving diagonally, both
            # orthogonally adjacent cells must be traversable too. This avoids
            # squeezing the robot through a 1-cell diagonal gap between obstacles.
            if dc != 0 and dr != 0:
                if not is_traversable(costmap, cc + dc, cr, unknown_is_blocked=unknown_is_blocked):
                    continue
                if not is_traversable(costmap, cc, cr + dr, unknown_is_blocked=unknown_is_blocked):
                    continue
            v = int(costmap[nr, nc])
            weight = 1.0 + (max(v, 0) / 100.0) if 0 < v < 100 else 1.0
            tentative = g_score[current] + base * weight
            if tentative < g_score.get((nc, nr), math.inf):
                g_score[(nc, nr)] = tentative
                came_from[(nc, nr)] = current
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative + _heuristic((nc, nr), goal_cell), counter, (nc, nr)),
                )

    return PathResult(False, math.inf, math.inf, [], [])


def inflate_costmap(costmap: np.ndarray, info: GridInfo, radius_m: float) -> np.ndarray:
    """Morphological dilation of lethal cells by `radius_m` to enforce robot footprint.

    Treats value 100 as lethal, preserves -1 (unknown), keeps everything else as-is.
    Inflated cells get value 99 (just under lethal) so the planner still considers
    cells adjacent to obstacles traversable but expensive — except the newly-lethal
    ring closest to the wall, which is set to 100.
    """
    r = max(1, int(math.ceil(radius_m / info.resolution)))
    lethal = costmap == 100
    if not lethal.any():
        return costmap.copy()

    # Manhattan-ball dilation via iterated 4-neighbor max.
    out = costmap.copy()
    dist = np.full(lethal.shape, np.inf, dtype=np.float32)
    dist[lethal] = 0.0
    # One BFS-like sweep for up to r cells.
    for _ in range(r):
        rolled = np.minimum.reduce(
            [
                np.roll(dist, 1, axis=0),
                np.roll(dist, -1, axis=0),
                np.roll(dist, 1, axis=1),
                np.roll(dist, -1, axis=1),
            ]
        )
        new_dist = np.minimum(dist, rolled + 1.0)
        if np.array_equal(new_dist, dist):
            break
        dist = new_dist

    # dist < r becomes lethal; dist in [r, 2r) is soft-costed.
    lethal_ring = (dist <= r) & (~lethal)
    soft = (dist > r) & (dist <= 2 * r)
    out[lethal_ring] = 100
    out[soft] = np.maximum(out[soft], 70).astype(out.dtype)
    return out
