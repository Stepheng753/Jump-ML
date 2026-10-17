from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from . import config as C


class Action(IntEnum):
    NONE = 0
    JUMP = 1
    DUCK = 2


class PlayerState(IntEnum):
    RUN = 0
    JUMP = 1
    DUCK = 2


class Kind(IntEnum):
    GROUND = 0   # beat it by jumping
    AIR = 1      # beat it by ducking


Rect = tuple  # (x0, y0, x1, y1), y measured upward from the ground


class Player:
    """Fixed horizontally; only its height / crouch changes."""

    def __init__(self):
        self.state = PlayerState.RUN
        self.timer = 0.0

    def apply(self, action: Action) -> Optional[str]:
        """Start a jump/duck if idle. Returns 'jump'/'duck' when one starts."""
        if self.state != PlayerState.RUN:
            return None
        if action == Action.JUMP:
            self.state, self.timer = PlayerState.JUMP, 0.0
            return "jump"
        if action == Action.DUCK:
            self.state, self.timer = PlayerState.DUCK, 0.0
            return "duck"
        return None

    def update(self):
        if self.state == PlayerState.RUN:
            return
        self.timer += C.DT
        limit = C.JUMP_TIME if self.state == PlayerState.JUMP else C.DUCK_TIME
        if self.timer >= limit:
            self.state, self.timer = PlayerState.RUN, 0.0

    @property
    def progress(self) -> float:
        """0..1 through the current jump/duck (0 when running)."""
        if self.state == PlayerState.RUN:
            return 0.0
        limit = C.JUMP_TIME if self.state == PlayerState.JUMP else C.DUCK_TIME
        return min(self.timer / limit, 1.0)

    @property
    def height_off_ground(self) -> float:
        if self.state != PlayerState.JUMP:
            return 0.0
        p = self.progress
        return C.JUMP_PEAK * 4.0 * p * (1.0 - p)   # parabola

    @property
    def body_h(self) -> float:
        return C.DUCK_H if self.state == PlayerState.DUCK else C.STAND_H

    @property
    def rect(self) -> Rect:
        y0 = self.height_off_ground
        half = C.PLAYER_W / 2
        return (C.PLAYER_X - half, y0, C.PLAYER_X + half, y0 + self.body_h)


@dataclass
class Obstacle:
    kind: Kind
    speed: float
    x: float                              # center x
    gap_prev: Optional[float] = None      # arrival-time gap to the one before
    gap_next: Optional[float] = None      # ... and to the one after
    passed: bool = False

    @property
    def rect(self) -> Rect:
        half = C.OBSTACLE_W / 2
        if self.kind == Kind.GROUND:
            y0, y1 = 0.0, C.GROUND_OBS_H
        else:
            y0, y1 = C.AIR_OBS_BOTTOM, C.AIR_OBS_BOTTOM + C.AIR_OBS_H
        return (self.x - half, y0, self.x + half, y1)


def overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
