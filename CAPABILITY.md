# simple_nav_rbnx — package description

A non-Nav2 navigation provider for Robonix. Exists to (a) give Pilot a safe navigation tool without requiring the full Nav2 stack, and (b) prove that `robonix/srv/navigation/*` is genuinely backend-agnostic.

## Interfaces

### Provides
- `robonix/srv/navigation/navigate` — submit a `(x, y, yaw?)` goal, get a `goal_id`.
- `robonix/srv/navigation/status` — poll progress of a `goal_id`.
- `robonix/srv/navigation/cancel` — stop an in-flight goal.

State machine: `PENDING → ACTIVE → {SUCCEEDED | ABORTED | CANCELED}`.

### Consumes (via Atlas discovery)
- `robonix/srv/common/map/occupancy_grid` — mapping_rbnx's inflated static map.
- `robonix/srv/common/map/scan_2d` — optional live LaserScan for dynamic obstacles.
- `robonix/prm/base/odom` — current pose.
- `robonix/prm/base/twist_in` — velocity command output.

## Source layout

- `src/simple_nav_rbnx/planner.py` — A* (8-conn, diag √2, corner-cut guard, weighted traversal).
- `src/simple_nav_rbnx/safety.py` — forward-simulation footprint check + scan overlay.
- `src/simple_nav_rbnx/follower.py` — PID waypoint tracker; every cmd_vel goes through `safety.predict_safe`.
- `src/simple_nav_rbnx/goal_state.py` — per-goal state machine with thread-safe cancel.
- `src/simple_nav_rbnx/nav_node.py` — ROS 2 wiring (odom/grid/scan subs, cmd_vel pub).
- `src/simple_nav_rbnx/atlas_bridge.py` — RegisterNode + DeclareInterface + gRPC server.

## Runtime parameters

See `config/nav_default.yaml`. Key knobs:
- `robot_radius`, `safety_inflate_extra` — footprint for lookahead collision.
- `lookahead_dt`, `lookahead_ticks` — how far / how finely we predict.
- `max_linear`, `max_angular` — hard velocity caps.
- `goal_tolerance`, `timeout` — per-goal convergence / abort defaults.

## Unit tests

```bash
pip install numpy pytest pyyaml
python3 -m pytest tests/ -q
```

`tests/test_planner.py`: A* on straight corridor, wall-bypass, unreachable, corner-cut, unknown-as-blocked, weighted. `tests/test_safety.py`: clear, wall-ahead, off-map, scan-overlay-blocks.

Tests do **not** import `rclpy` or `grpc` — planner and safety modules are pure-Python for this reason.

## Launch on real robot

```bash
# On the robot, after rbnx setup + mapping_rbnx running:
rbnx start -p <this repo> -n com.robonix.nav.simple
```

It will discover topics via Atlas. If mapping_rbnx isn't publishing `occupancy_grid` yet, the first `navigate` call returns `accepted=false, message="no costmap yet"`.

## What this doesn't do (deliberately)

- Replan when path gets blocked mid-goal — we ABORT and ask Pilot to reissue.
- Rotation-in-place recovery.
- Dynamic parameters via `NavigateOptions` beyond `timeout` and `tolerance`.
- Behavior trees, lifecycle management, any Nav2 plugins.

## Relationship to Nav2

Nav2 is being integrated by another student as a separate `nav_rbnx` package. Both implement the same three RPCs. Pilot queries Atlas for a `robonix/srv/navigation/navigate` provider and picks one — doesn't care which.
