"""simple_nav_rbnx — safety-first Robonix navigation provider.

Public components (see README.md for the contract surface):

  planner   — A* on a 2D costmap, 8-connected with √2 diagonals
  safety    — lookahead collision predicate over static + live scan
  follower  — PID waypoint tracker; every cmd_vel is safety-gated
  goal_state— per-goal state machine + cancellation
  nav_node  — ROS 2 wiring
  atlas_bridge — Atlas RegisterNode / DeclareInterface / gRPC RPCs

Nothing here imports rclpy or grpc at module load — the node/bridge modules
do that lazily so `planner` and `safety` stay importable for unit tests.
"""
