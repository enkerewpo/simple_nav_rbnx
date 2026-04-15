# simple_nav_rbnx

**A safety-first Robonix navigation provider that is NOT Nav2.**

Purpose: give Robonix a working `robonix/srv/navigation/*` implementation that a VLM can drive without risking wall/obstacle collisions. Exists primarily to validate the navigation contract as backend-agnostic (Nav2 is being implemented elsewhere in parallel).

## What it provides

| Contract | RPC shape | Behavior |
|---|---|---|
| `robonix/srv/navigation/navigate` | `PoseStamped (+ optional NavigateOptions) → {goal_id, accepted, message}` | Accept goal, compute A* path on the current costmap, start a follower thread |
| `robonix/srv/navigation/status` | `{goal_id} → {status, current_pose, distance_remaining_m, elapsed_s}` | Poll progress |
| `robonix/srv/navigation/cancel` | `{goal_id} → {accepted}` | Stop the follower, publish zero twist |

Status enum: `PENDING | ACTIVE | SUCCEEDED | ABORTED | CANCELED`.

## What it consumes

| Contract | Used for |
|---|---|
| `robonix/srv/common/map/occupancy_grid` | Static map — A* plans on the inflated costmap derived from this |
| `robonix/srv/common/map/scan_2d` | Live obstacles — obstacle layer added on top of static for safety lookahead |
| `robonix/prm/base/odom` | Current pose for waypoint tracking |
| `robonix/prm/base/twist_in` | Velocity command output |

Interfaces are discovered via Atlas, not hardcoded — exactly the same flow `mapping_rbnx` already uses.

## Safety: lookahead collision predicate

Before every `cmd_vel` publish, predict the robot's pose N ticks ahead with the candidate command and test that predicted footprint against the **combined** costmap (static ∪ current LaserScan obstacles, both inflated by robot radius). If any footprint cell is occupied, **override the command to zero twist**, mark the goal ABORTED, and stop.

Parameters (see `config/nav_default.yaml`):

```yaml
robot_radius: 0.25          # m — footprint inflation
lookahead_dt: 0.4           # s — how far ahead to predict
follower_hz: 20             # Hz
max_linear: 0.5             # m/s hard cap
max_angular: 1.0            # rad/s hard cap
goal_tolerance: 0.3         # m
goal_yaw_tolerance: 0.2     # rad (optional final yaw)
timeout: 120                # s per goal
safety_inflate_extra: 0.1   # m — extra margin on top of robot_radius
pid_linear: [0.8, 0.0, 0.1]
pid_angular: [1.5, 0.0, 0.2]
```

## Components

```
src/simple_nav_rbnx/
├── planner.py          # A* on 2D costmap (8-connected, diag √2)
├── follower.py         # PID waypoint tracker, calls safety.predict_safe
├── safety.py           # costmap + live-scan lookahead collision check
├── nav_node.py         # ROS 2 node wiring: subs odom + occupancy_grid + scan_2d, pubs cmd_vel
├── atlas_bridge.py     # Atlas register + DeclareInterface + gRPC navigate/status/cancel server
└── goal_state.py       # per-goal state machine + thread lifecycle
```

## Why not Nav2

- Nav2 costmap uses raw `PointCloud2`/`LaserScan` with its own layered plugins — binding here would collapse the "costmap is a Robonix contract" abstraction we're validating.
- Another student is writing the Nav2 bridge. Having two independent implementations of the same three RPCs is the test case for the contract itself.
- Nav2 is overkill for the scenarios we actually want to demo (single-floor indoor, low speed, no dynamic re-planning yet).

## Known limits (on purpose)

- No global re-planning mid-goal. If the path is blocked, goal ABORTs — user (or Pilot) must reissue.
- No rotation-in-place recovery behavior. If stuck, ABORT.
- No BT, no lifecycle manager. One goal, one thread.
- `NavigateOptions` fields other than `timeout_s` and `goal_tolerance_m` are currently ignored (TODO).

## Testing strategy

- `tests/test_planner.py` — A* on small hand-rolled grids (wall, U-shape, unreachable).
- `tests/test_safety.py` — lookahead predicate: predict into wall vs. predict along corridor.
- No on-robot integration tests in this repo; validation is via `robonix-exp`.

## Status

🚧 Bootstrapping. Planner + safety lib first, then the ROS node + Atlas bridge.
