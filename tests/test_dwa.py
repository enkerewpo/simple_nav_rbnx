"""DWA-lite fallback selector tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simple_nav_rbnx.dwa import DWAConfig, choose_safe  # noqa: E402
from simple_nav_rbnx.planner import GridInfo  # noqa: E402
from simple_nav_rbnx.safety import SafetyParams  # noqa: E402


def _info(w: int = 40, h: int = 40, res: float = 0.1) -> GridInfo:
    return GridInfo(resolution=res, width=w, height=h, origin_x=0.0, origin_y=0.0)


def test_pp_passes_when_clear():
    cm = np.zeros((40, 40), dtype=np.int8)
    info = _info()
    safety = SafetyParams(robot_radius=0.15)
    cfg = DWAConfig()
    v, w, src = choose_safe(
        pp_v=0.3, pp_w=0.0,
        pose=(1.0, 1.0, 0.0),
        lookahead=(1.5, 1.0),
        costmap=cm, info=info, safety=safety, cfg=cfg,
    )
    assert src == "pp"
    assert v == 0.3
    assert w == 0.0


def test_dwa_finds_safe_candidate_when_pp_blocked():
    cm = np.zeros((40, 40), dtype=np.int8)
    # Wall directly in front at x=1.5 (col=15)
    cm[:, 15] = 100
    info = _info()
    safety = SafetyParams(robot_radius=0.15, lookahead_dt=0.8, lookahead_ticks=5, max_linear=1.0)
    cfg = DWAConfig(v_samples=5, w_samples=9, v_span=0.3, w_span=1.5)

    # PP wants to drive straight into the wall.
    v, w, src = choose_safe(
        pp_v=0.5, pp_w=0.0,
        pose=(1.3, 1.0, 0.0),        # close to the wall
        lookahead=(1.8, 1.0),
        costmap=cm, info=info, safety=safety, cfg=cfg,
    )
    # DWA should either find a safe alternative (typically stop / turn) or block.
    # Either way: we must NOT get the original unsafe (0.5, 0.0) through.
    assert src in {"dwa", "blocked"}
    if src == "dwa":
        # The chosen command must not be the PP nominal.
        assert not (v == 0.5 and w == 0.0)


def test_blocked_returns_zero_when_everything_unsafe():
    # Robot cornered tightly: occupied cells on all four sides within
    # footprint + lookahead range. Any non-trivial twist in any direction
    # will hit something within the lookahead horizon.
    cm = np.zeros((10, 10), dtype=np.int8)
    # Keep the 3x3 center (rows 4-5, cols 4-5) free; occupy the rest.
    cm[:, :] = 100
    cm[4:6, 4:6] = 0
    info = _info(10, 10, res=0.1)
    # Lookahead big enough that any direction pushes the footprint off-center.
    safety = SafetyParams(
        robot_radius=0.1,
        extra_margin=0.02,
        lookahead_dt=1.0,
        lookahead_ticks=5,
        max_linear=1.0,
        max_angular=2.0,
    )
    cfg = DWAConfig(v_samples=3, w_samples=3, v_span=0.5, w_span=2.0)

    # Robot centered in the tiny clear patch at world (0.5, 0.5).
    v, w, src = choose_safe(
        pp_v=0.5, pp_w=0.0,
        pose=(0.5, 0.5, 0.0),
        lookahead=(1.0, 0.5),
        costmap=cm, info=info, safety=safety, cfg=cfg,
    )
    assert src == "blocked"
    assert (v, w) == (0.0, 0.0)
