import numpy as np

from game_interface.labels import cell_label, game_label, title_unit
from ml.brains import NetBrain, PerfectBrain
from ml.network import layer_sizes, random_genome


def net(**kw):
    sizes = layer_sizes()
    return NetBrain(random_genome(np.random.default_rng(0), sizes), sizes, **kw)


def test_the_perfect_bot_has_a_label():
    # the perfect bot has no generation, which once crashed the watcher
    assert game_label(PerfectBrain(), 0, 123) == "perfect bot | game 1 | seed 123"


def test_evolution_labels():
    b = net(generation=40, rank=2, fitness=17.34, unit="gen")
    assert game_label(b, 1, 99) == "gen 40 | individual #2 | train fit 17.3 | game 2 | seed 99"
    assert game_label(b, 4, 4002, replaying=True, n_games=3) == "gen 40 | individual #2 | fitness 17.3 | game 2/3 | seed 4002"
    assert cell_label(b) == "gen 40 | #2 | fit 17.3"
    assert title_unit([b]) == "generations"


def test_dqn_labels_say_round_and_hide_the_individual_number():
    b = net(generation=59, rank=1, fitness=122.34, unit="round", n_individuals=1)
    assert game_label(b, 0, 7) == "round 59 | eval 122.3 | game 1 | seed 7"
    assert cell_label(b) == "round 59 | eval 122.3"
    assert title_unit([b]) == "rounds"
