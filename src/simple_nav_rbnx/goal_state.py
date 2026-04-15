"""Per-goal state machine and lifecycle management."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GoalStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"
    CANCELED = "CANCELED"


@dataclass
class GoalState:
    """All state for a single navigation goal.

    Owned by the nav_node; follower thread reads `cancel_event` and writes
    status transitions, current pose estimate, and distance remaining.
    """
    goal_id: str
    goal_x: float
    goal_y: float
    goal_yaw: Optional[float]
    tolerance: float
    timeout_s: float
    created_at: float
    status: GoalStatus = GoalStatus.PENDING
    message: str = ""
    current_x: float = 0.0
    current_y: float = 0.0
    current_yaw: float = 0.0
    distance_remaining_m: float = float("inf")
    elapsed_s: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None

    @classmethod
    def new(cls, x: float, y: float, yaw: Optional[float], tolerance: float, timeout_s: float) -> "GoalState":
        return cls(
            goal_id=f"g-{uuid.uuid4().hex[:12]}",
            goal_x=x,
            goal_y=y,
            goal_yaw=yaw,
            tolerance=tolerance,
            timeout_s=timeout_s,
            created_at=time.time(),
        )

    def is_terminal(self) -> bool:
        return self.status in {GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.CANCELED}

    def cancel(self) -> bool:
        if self.is_terminal():
            return False
        self.cancel_event.set()
        return True

    def set_status(self, status: GoalStatus, message: str = "") -> None:
        self.status = status
        if message:
            self.message = message
        self.elapsed_s = time.time() - self.created_at

    def update_pose(self, x: float, y: float, yaw: float, distance_remaining: float) -> None:
        self.current_x = x
        self.current_y = y
        self.current_yaw = yaw
        self.distance_remaining_m = distance_remaining
        self.elapsed_s = time.time() - self.created_at
