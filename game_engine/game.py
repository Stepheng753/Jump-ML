import random
from typing import List, Optional

import numpy as np

from . import config as C
from .entities import Action, Kind, Obstacle, Player, PlayerState, overlaps
from .scoring import penalty_for, points_for
from .spawner import Spawner

OBS_SIZE = 10  # length of get_observation()


class Game:
    """Pure game logic. No graphics, no sound, no real-time clock.

    step(action) advances exactly one frame (1/60 s), so a trainer can run
    thousands of games per second while a human plays the same rules at 60 FPS.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.spawner = Spawner(self.rng)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        self.player = Player()
        self.obstacles: List[Obstacle] = []
        self.t = 0.0
        self.score = 0
        self.dodged = 0
        self.wasted = 0               # jumps/ducks that nothing needed
        self.last_penalty = 0         # the most recent penalty, and when it happened
        self.last_penalty_at = -999.0
        self.game_over = False
        self.spawner.reset()
        return self.get_observation()

    def step(self, action: int = Action.NONE):
        """Advance one frame. Returns (observation, reward, done, info).

        info['started'] is 'jump'/'duck' on the frame an action begins (the UI
        uses it for sound); info['points'] is what was earned this frame and
        info['penalty'] what was lost to a jump/duck that nothing needed.
        """
        if self.game_over:
            return self.get_observation(), 0.0, True, {"started": None, "points": 0, "penalty": 0}

        # Judge a new jump/duck before anything moves: it is about the world as it was when you pressed.
        started = self.player.apply(Action(action))
        penalty = self._wasted_action_penalty(started) if started else 0
        self.player.update()
        self.t += C.DT
        if penalty:
            self.wasted += 1
            self.last_penalty, self.last_penalty_at = penalty, self.t

        new = self.spawner.update(self.t)
        if new is not None:
            self.obstacles.append(new)

        for ob in self.obstacles:
            ob.x -= ob.speed * C.DT

        points = 0
        player_rect = self.player.rect
        for ob in self.obstacles:
            if overlaps(player_rect, ob.rect):
                self.game_over = True
            elif not ob.passed and ob.rect[2] < player_rect[0]:
                ob.passed = True
                points += points_for(ob.speed, ob.gap_prev, ob.gap_next)
                self.dodged += 1

        self.obstacles = [o for o in self.obstacles if o.x > -C.OBSTACLE_W]
        self.score += points - penalty

        reward = float(points - penalty) - (C.DEATH_PENALTY if self.game_over else 0.0)
        info = {"started": started, "points": points, "penalty": penalty}
        return self.get_observation(), reward, self.game_over, info

    def _wasted_action_penalty(self, started: str) -> int:
        """Points lost for a jump/duck, or 0 if an obstacle of the matching kind needed it.

        'Needed' means a ground obstacle (for a jump) or air obstacle (for a duck)
        will overlap the player's column at some point while the action lasts.
        Obstacles move at a constant speed, so this can be worked out on the spot;
        nothing that has not spawned yet can arrive within one action.
        """
        kind = Kind.GROUND if started == "jump" else Kind.AIR
        window = C.JUMP_TIME if started == "jump" else C.DUCK_TIME
        reach = (C.PLAYER_W + C.OBSTACLE_W) / 2
        incoming = [o for o in self.obstacles if not o.passed]
        for o in incoming:
            if o.kind == kind:
                enters = (o.x - C.PLAYER_X - reach) / o.speed    # seconds until it touches the player's column
                leaves = (o.x - C.PLAYER_X + reach) / o.speed    # ...and until it has fully gone past
                if leaves >= 0 and enters <= window:
                    return 0
        if not incoming:
            return penalty_for(None, None)
        nearest = min(incoming, key=lambda o: o.x)
        distance_n = (nearest.x - C.PLAYER_X) / C.TRAVEL_DIST
        speed_n = (nearest.speed - C.SPEED_MIN) / (C.SPEED_MAX - C.SPEED_MIN)
        return penalty_for(distance_n, speed_n)

    def get_observation(self) -> np.ndarray:
        """Numbers an AI can 'see'. All roughly in 0..1.

        [is_running, is_jumping, is_ducking, player_progress,
         then for the 2 nearest incoming obstacles:
           distance_to_player, speed, is_air]
        The state is one-hot (three separate 0/1 inputs) so the net doesn't
        read RUN < JUMP < DUCK as an order. Missing obstacles are reported as
        far away (distance 1, speed 0, type 0).
        """
        incoming = sorted((o for o in self.obstacles if not o.passed), key=lambda o: o.x)
        state = self.player.state
        obs = [
            float(state == PlayerState.RUN),
            float(state == PlayerState.JUMP),
            float(state == PlayerState.DUCK),
            self.player.progress,
        ]
        for i in range(2):
            if i < len(incoming):
                o = incoming[i]
                dist = max(0.0, (o.x - C.PLAYER_X) / C.TRAVEL_DIST)
                speed = (o.speed - C.SPEED_MIN) / (C.SPEED_MAX - C.SPEED_MIN)
                obs += [min(dist, 1.0), speed, float(o.kind)]
            else:
                obs += [1.0, 0.0, 0.0]
        return np.array(obs, dtype=np.float32)
