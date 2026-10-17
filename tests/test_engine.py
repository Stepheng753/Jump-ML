import sys

from game_engine import OBS_SIZE, Action, Game, Kind
from game_engine import config as C
from game_engine.entities import Obstacle
from game_engine.scoring import penalty_for, points_for
from ml.brains import PerfectBrain


def scripted_bot(game: Game) -> int:
    """The perfect bot, seeing only the observation numbers a net would see."""
    return PerfectBrain().act(game.get_observation())


def test_observation_shape_and_state_is_one_hot():
    g = Game(seed=4)
    obs = g.get_observation()
    assert obs.shape == (OBS_SIZE,)
    assert obs[:3].sum() == 1.0          # exactly one state flag is on
    g.step(Action.JUMP)
    assert g.get_observation()[1] == 1.0  # is_jumping
    for _ in range(40):
        g.step(Action.NONE)
    assert g.get_observation()[0] == 1.0  # back to running


def test_engine_has_no_pygame():
    import game_engine  # noqa: F401
    assert "pygame" not in sys.modules


def test_points_always_in_range():
    for speed in (C.SPEED_MIN, 350, C.SPEED_MAX):
        for gp in (None, C.MIN_GAP, 1.5, C.MAX_GAP):
            for gn in (None, C.MIN_GAP, C.MAX_GAP):
                assert C.MIN_POINTS <= points_for(speed, gp, gn) <= C.MAX_POINTS


def test_faster_and_closer_is_worth_more():
    assert points_for(C.SPEED_MAX, C.MIN_GAP, None) == C.MAX_POINTS
    assert points_for(C.SPEED_MIN, C.MAX_GAP, C.MAX_GAP) == C.MIN_POINTS
    assert points_for(450, 1.0, 1.0) > points_for(250, 1.0, 1.0)
    assert points_for(350, 0.95, 2.0) > points_for(350, 2.0, 2.0)


def test_spawned_gaps_respect_minimum():
    g = Game(seed=1)
    gaps = []
    for _ in range(60 * 120):
        g.step(Action.NONE)
        gaps += [o.gap_prev for o in g.obstacles if o.gap_prev is not None]
        if g.game_over:
            break
    # game ends on the first collision, so also check the planner directly
    g = Game(seed=2)
    seen = []
    for _ in range(200):
        seen.append(g.spawner.pending[1].gap_prev)
        g.spawner._plan(now=g.spawner.pending[0])
    assert all(x is None or x >= C.MIN_GAP - 1e-9 for x in seen)


def test_doing_nothing_eventually_dies():
    g = Game(seed=3)
    for _ in range(60 * 30):
        _, _, done, _ = g.step(Action.NONE)
        if done:
            break
    assert g.game_over


def test_perfect_timing_is_always_survivable():
    # If this fails the spawner is producing impossible patterns.
    for seed in range(30):
        g = Game(seed=seed)
        for _ in range(60 * 120):   # two minutes
            g.step(scripted_bot(g))
            if g.game_over:
                break
        assert not g.game_over, f"seed {seed} died at t={g.t:.1f}"
        assert g.dodged > 40
        assert g.dodged <= g.score <= 5 * g.dodged
        assert g.wasted == 0, f"seed {seed}: perfect timing should never waste a jump/duck"


def test_same_seed_is_reproducible():
    def run(seed):
        g = Game(seed=seed)
        for _ in range(3000):
            g.step(scripted_bot(g))
        return g.score, g.dodged

    assert run(7) == run(7)


def test_air_obstacle_cannot_be_jumped_over():
    assert C.AIR_OBS_BOTTOM + C.AIR_OBS_H > C.JUMP_PEAK   # feet never rise above its top
    assert C.AIR_OBS_BOTTOM > C.DUCK_H                    # ducking clears it
    assert C.AIR_OBS_BOTTOM < C.STAND_H                   # standing hits it


def _game_with(kind, x, speed=300.0):
    g = Game(seed=0)
    g.obstacles.append(Obstacle(kind=kind, speed=speed, x=x))
    return g


def test_penalty_range_and_direction():
    for d in (None, 0.0, 0.5, 1.0):
        for sp in (None, 0.0, 0.5, 1.0):
            assert C.MIN_PENALTY <= penalty_for(d, sp) <= C.MAX_PENALTY
    assert penalty_for(None, None) == C.MAX_PENALTY            # nothing around: no excuse
    assert penalty_for(1.0, 0.0) == C.MAX_PENALTY              # far and slow: no excuse
    assert penalty_for(0.0, 1.0) == C.MIN_PENALTY              # close and fast: smallest penalty
    assert penalty_for(0.1, 0.8) < penalty_for(0.9, 0.2)


def test_jump_with_nothing_around_costs_full_penalty():
    g = Game(seed=1)
    _, reward, done, info = g.step(Action.JUMP)
    assert (g.score, g.wasted, info["penalty"], reward, done) == (-C.MAX_PENALTY, 1, C.MAX_PENALTY, -float(C.MAX_PENALTY), False)
    assert g.last_penalty == C.MAX_PENALTY


def test_needed_jump_and_duck_are_free():
    g = _game_with(Kind.GROUND, x=520)             # reaches the player well inside a jump
    assert g.step(Action.JUMP)[3]["penalty"] == 0
    g = _game_with(Kind.AIR, x=520)
    assert g.step(Action.DUCK)[3]["penalty"] == 0


def test_wrong_kind_is_penalized():
    for kind, action in ((Kind.AIR, Action.JUMP), (Kind.GROUND, Action.DUCK)):
        g = _game_with(kind, x=520)
        assert g.step(action)[3]["penalty"] > 0


def test_obstacle_that_arrives_after_the_action_ends_does_not_count():
    g = _game_with(Kind.GROUND, x=790, speed=220.0)   # ~1.7 s away, a jump only lasts 0.6 s
    assert g.step(Action.JUMP)[3]["penalty"] > 0


def test_nearer_and_faster_obstacle_means_smaller_penalty():
    far_slow = _game_with(Kind.AIR, x=780, speed=230.0)
    near_fast = _game_with(Kind.AIR, x=700, speed=500.0)
    assert near_fast.step(Action.JUMP)[3]["penalty"] < far_slow.step(Action.JUMP)[3]["penalty"]


def test_presses_ignored_mid_action_are_not_penalized_again():
    g = Game(seed=1)
    g.step(Action.JUMP)
    for _ in range(10):
        assert g.step(Action.JUMP)[3]["penalty"] == 0
    assert g.wasted == 1


def test_reward_is_points_minus_penalty():
    g = Game(seed=2)
    total = 0.0
    for i in range(60 * 20):
        _, reward, done, info = g.step(Action.JUMP if i % 90 == 0 else Action.NONE)
        total += reward + (C.DEATH_PENALTY if done else 0.0)
        if done:
            break
    assert total == g.score
