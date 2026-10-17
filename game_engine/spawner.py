import random
from typing import Optional

from . import config as C
from .entities import Kind, Obstacle


class Spawner:
    """Plans obstacles on an *arrival timeline* so the game is always beatable.

    Each obstacle gets a random speed, and its spawn time is worked backwards
    from when it should reach the player: spawn = arrival - travel_time.
    Arrivals are at least MIN_GAP apart, so an obstacle can never catch up to
    (or overlap) the one in front of it, and the player always has time to
    finish one jump/duck before the next one arrives.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.last_arrival: Optional[float] = None
        self.last_obstacle: Optional[Obstacle] = None
        self.pending: Optional[tuple] = None   # (spawn_time, obstacle)

    def reset(self):
        self.last_arrival = None
        self.last_obstacle = None
        self.pending = None
        self._plan(now=0.0)

    def update(self, now: float) -> Optional[Obstacle]:
        """Return a newly spawned obstacle if one is due this step."""
        spawn_time, obstacle = self.pending
        if now < spawn_time:
            return None
        obstacle.x = C.SPAWN_X - obstacle.speed * (now - spawn_time)
        self._plan(now)
        return obstacle

    def _plan(self, now: float):
        speed = self.rng.uniform(C.SPEED_MIN, C.SPEED_MAX)
        kind = self.rng.choice((Kind.GROUND, Kind.AIR))
        travel = C.TRAVEL_DIST / speed

        if self.last_arrival is None:
            arrival, gap = max(C.FIRST_ARRIVAL, now + travel), None
        else:
            gap = self.rng.uniform(C.MIN_GAP, C.MAX_GAP)
            arrival = max(self.last_arrival + gap, now + travel)
            gap = arrival - self.last_arrival   # the real gap after clamping

        obstacle = Obstacle(kind=kind, speed=speed, x=C.SPAWN_X, gap_prev=gap)
        if self.last_obstacle is not None:
            self.last_obstacle.gap_next = gap
        self.last_arrival = arrival
        self.last_obstacle = obstacle
        self.pending = (arrival - travel, obstacle)
