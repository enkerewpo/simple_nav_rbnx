"""Lookahead collision predicate.

Given the current pose, a candidate `(v, w)` twist, and the combined costmap
(static inflated + live scan footprint), simulate the pose forward by a few
ticks and check every predicted footprint cell. Returns True iff the entire
lookahead trajectory is clear.

Called before every `cmd_vel` publish in `follower.py`. A False return
overrides the command to zero twist and ABORTs the current goal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .planner import GridInfo, in_bounds, is_traversable, world_to_cell


@dataclass
class SafetyParams:
    robot_radius: float = 0.25
    extra_margin: float = 0.10     # added to robot_radius for footprint test
    lookahead_dt: float = 0.4      # seconds of forward prediction
    lookahead_ticks: int = 5       # samples along the lookahead trajectory
    max_linear: float = 0.5
    max_angular: float = 1.0


def _clip_twist(v: float, w: float, p: SafetyParams) -> tuple[float, float]:
    v = max(-p.max_linear, min(p.max_linear, v))
    w = max(-p.max_angular, min(p.max_angular, w))
    return v, w


def _forward_simulate(x: float, y: float, yaw: float, v: float, w: float, dt: float) -> tuple[float, float, float]:
    """Unicycle model, one integration step."""
    if abs(w) < 1e-5:
        nx = x + v * math.cos(yaw) * dt
        ny = y + v * math.sin(yaw) * dt
        nyaw = yaw
    else:
        nyaw = yaw + w * dt
        nx = x + (v / w) * (math.sin(nyaw) - math.sin(yaw))
        ny = y - (v / w) * (math.cos(nyaw) - math.cos(yaw))
    # Normalize yaw to [-pi, pi]
    nyaw = math.atan2(math.sin(nyaw), math.cos(nyaw))
    return nx, ny, nyaw


def _footprint_cells(info: GridInfo, x: float, y: float, radius_m: float) -> list[tuple[int, int]]:
    """All cells whose center is within `radius_m` of (x, y)."""
    r_cells = max(1, int(math.ceil(radius_m / info.resolution)))
    col0, row0 = world_to_cell(info, x, y)
    cells: list[tuple[int, int]] = []
    r2 = r_cells * r_cells
    for dr in range(-r_cells, r_cells + 1):
        for dc in range(-r_cells, r_cells + 1):
            if dc * dc + dr * dr <= r2:
                cells.append((col0 + dc, row0 + dr))
    return cells


def predict_safe(
    costmap: np.ndarray,
    info: GridInfo,
    pose: tuple[float, float, float],  # (x, y, yaw) in map frame
    twist: tuple[float, float],         # (v, w)
    params: SafetyParams,
) -> tuple[bool, str, list[tuple[float, float, float]]]:
    """Return (ok, reason, predicted_poses).

    If `ok` is False, `reason` is a short human-readable string like
    "collision at tick 3" or "left map bounds" and the follower must NOT
    publish the candidate twist.
    """
    v, w = _clip_twist(*twist, params)
    x, y, yaw = pose
    poses: list[tuple[float, float, float]] = [(x, y, yaw)]

    step_dt = params.lookahead_dt / max(1, params.lookahead_ticks)
    footprint_radius = params.robot_radius + params.extra_margin

    for tick in range(1, params.lookahead_ticks + 1):
        x, y, yaw = _forward_simulate(x, y, yaw, v, w, step_dt)
        poses.append((x, y, yaw))
        for col, row in _footprint_cells(info, x, y, footprint_radius):
            if not in_bounds(info, col, row):
                return False, f"left map bounds at tick {tick}", poses
            if not is_traversable(costmap, col, row, unknown_is_blocked=True):
                return False, f"collision at tick {tick} cell=({col},{row})", poses

    return True, "clear", poses


def combine_scan_into_costmap(
    base_cost: np.ndarray,
    info: GridInfo,
    pose: tuple[float, float, float],
    scan_ranges: list[float],
    scan_angle_min: float,
    scan_angle_increment: float,
    inflate_radius_m: float,
) -> np.ndarray:
    """Overlay a LaserScan onto the static costmap (both in map frame).

    Each scan return becomes a small inflated obstacle. Returns a new costmap;
    does not mutate the input. Ranges that are inf / NaN / 0 are ignored.
    """
    out = base_cost.copy()
    r_cells = max(1, int(math.ceil(inflate_radius_m / info.resolution)))
    px, py, pyaw = pose

    for i, r in enumerate(scan_ranges):
        if not math.isfinite(r) or r <= 0.0:
            continue
        ang = pyaw + scan_angle_min + i * scan_angle_increment
        wx = px + r * math.cos(ang)
        wy = py + r * math.sin(ang)
        col0, row0 = world_to_cell(info, wx, wy)
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dc * dc + dr * dr > r_cells * r_cells:
                    continue
                c, r2 = col0 + dc, row0 + dr
                if in_bounds(info, c, r2):
                    out[r2, c] = 100
    return out
