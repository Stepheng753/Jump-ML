import numpy as np

from game_engine import OBS_SIZE, Game
from ml.brains import NetBrain, PerfectBrain, load_brain, save_generation
from ml.network import Network, genome_size, layer_sizes, random_genome
from ml.neuroevolution.evolve import Config, crossover, mutate, next_generation, seeds_for, train
from ml.rollout import play_game


def test_default_layout_is_10_12_3_with_171_numbers():
    sizes = layer_sizes()
    assert sizes == (OBS_SIZE, 12, 3) == (10, 12, 3)
    assert genome_size(sizes) == 171


def test_network_outputs_three_scores():
    rng = np.random.default_rng(0)
    sizes = layer_sizes()
    net = Network(random_genome(rng, sizes), sizes)
    out = net.forward(Game(0).get_observation())
    assert out.shape == (3,)


def test_perfect_brain_survives_via_play_game():
    for seed in range(10):
        r = play_game(PerfectBrain(), seed, max_steps=60 * 60)
        assert not r.died and r.score > 0


def test_mutation_and_crossover():
    rng = np.random.default_rng(1)
    cfg = Config()
    a, b = np.zeros(171), np.ones(171)
    child = crossover(a, b, rng)
    assert 0 < child.sum() < 171                   # a mix of both parents
    m = mutate(a, cfg, rng)
    assert 0 < np.count_nonzero(m) < 171           # some numbers nudged, not all


def test_next_generation_keeps_size_and_elite():
    rng = np.random.default_rng(2)
    cfg = Config(pop_size=20, elite=3, n_parents=6)
    pop = [random_genome(rng, layer_sizes()) for _ in range(20)]
    fitness = np.arange(20, dtype=float)           # genome 19 is the best
    new = next_generation(pop, fitness, cfg, rng)
    assert len(new) == 20
    assert np.array_equal(new[0], pop[19]) and np.array_equal(new[2], pop[17])


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    sizes = layer_sizes()
    genomes = [random_genome(rng, sizes) for _ in range(3)]
    save_generation(tmp_path / "gen_0010.npz", genomes, [5.0, 3.0, 1.0], sizes, 10, [100, 101, 102])

    best = load_brain(str(tmp_path))                  # a folder -> latest generation, rank 1
    second = load_brain(str(tmp_path / "gen_0010.npz"), rank=2)
    obs = Game(5).get_observation()
    best.act(obs)
    assert np.allclose(best.scores, NetBrain(genomes[0], sizes).net.forward(obs))
    assert np.allclose(second.net.genome, genomes[1])
    assert (second.generation, second.rank, second.fitness, second.seeds) == (10, 2, 3.0, [100, 101, 102])


def _quiet_train(run_dir, generations, **kw):
    cfg = Config(pop_size=10, elite=2, n_parents=4, games_per_eval=1, max_steps=600)
    train(cfg, generations, str(run_dir), save_every=2, log=lambda *_: None, **kw)


def test_short_training_run_writes_files(tmp_path):
    _quiet_train(tmp_path, 3)                       # generations 0, 1, 2
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["gen_0000.npz", "gen_0002.npz", "history.csv", "state.npz"]
    assert len((tmp_path / "history.csv").read_text().splitlines()) == 4   # header + 3


def test_resuming_matches_one_long_run(tmp_path):
    _quiet_train(tmp_path / "split", 1)
    _quiet_train(tmp_path / "split", 3)             # 1 + 3 more
    _quiet_train(tmp_path / "whole", 4)
    assert (tmp_path / "split" / "history.csv").read_text() == (tmp_path / "whole" / "history.csv").read_text()


def test_seeds_differ_per_generation():
    assert set(seeds_for(0, 0, 3)).isdisjoint(seeds_for(0, 1, 3))


def test_parents_must_be_at_least_elite_and_validated():
    assert Config(elite=5, n_parents=5).n_parents == 5
    assert Config(elite=5, n_parents=10).n_parents == 10
    for bad in (dict(elite=5, n_parents=3), dict(elite=12),   # parents < elite (12 > default 10)
                dict(elite=60), dict(elite=0, n_parents=0), dict(n_parents=60), dict(elite=-1)):
        try:
            Config(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")


def test_single_parent_still_breeds():
    rng = np.random.default_rng(4)
    cfg = Config(pop_size=8, elite=1, n_parents=1)
    pop = [random_genome(rng, layer_sizes()) for _ in range(8)]
    assert len(next_generation(pop, np.arange(8.0), cfg, rng)) == 8
