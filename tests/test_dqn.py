import numpy as np
import pytest

from game_engine import OBS_SIZE
from ml.brains import load_brain, load_generation
from ml.network import genome_size, layer_sizes, random_genome
from ml.neuroevolution.evolve import Config, run_info as evolution_run_info
from ml.neuroevolution.evolve import train as evolution_train
from ml.rl.agent import DQNAgent, td_loss_and_grad
from ml.rl.replay import ReplayBuffer
from ml.rl.train import DQNConfig, epsilon, run_info, train
from ml.runs import run_method


def quiet(*_):
    pass


def tiny_cfg(**overrides):
    base = dict(hidden=6, episodes_per_round=2, eval_games=1, max_steps=150, warmup=50, batch_size=16,
                buffer_size=500, eps_decay_steps=300, target_sync=100)
    base.update(overrides)
    return DQNConfig(**base)


# --- the math --------------------------------------------------------------------

def numeric_gradient(theta, sizes, S, A, y, delta, h=1e-6):
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[i] += h
        down[i] -= h
        grad[i] = (td_loss_and_grad(up, sizes, S, A, y, delta)[0] - td_loss_and_grad(down, sizes, S, A, y, delta)[0]) / (2 * h)
    return grad


@pytest.mark.parametrize("target_shift", [0.0, 5.0], ids=["small errors", "large errors"])
def test_backprop_matches_finite_differences(target_shift):
    rng = np.random.default_rng(0)
    sizes = (OBS_SIZE, 4, 3)
    theta = random_genome(rng, sizes)
    S = rng.random((6, OBS_SIZE))
    A = rng.integers(0, 3, 6)
    y = rng.normal(0, 0.3, 6) + target_shift       # shifted targets push the errors past the Huber delta
    _, grad = td_loss_and_grad(theta, sizes, S, A, y, 1.0)
    assert np.allclose(grad, numeric_gradient(theta, sizes, S, A, y, 1.0), atol=1e-6)


def test_only_the_action_taken_gets_an_error():
    rng = np.random.default_rng(1)
    sizes = (OBS_SIZE, 4, 3)
    theta = random_genome(rng, sizes)
    _, grad = td_loss_and_grad(theta, sizes, rng.random((1, OBS_SIZE)), np.array([1]), np.array([2.0]), 1.0)
    n_in, n_h, _ = sizes
    W2 = grad[n_h * n_in + n_h:][:3 * n_h].reshape(3, n_h)
    b2 = grad[-3:]
    assert np.all(W2[0] == 0) and np.all(W2[2] == 0) and b2[0] == 0 and b2[2] == 0   # nothing, duck: untouched
    assert np.any(W2[1] != 0) and b2[1] != 0                                         # jump: gets the error


def test_target_is_reward_plus_discounted_best_next_guess_and_a_crash_has_no_future():
    agent = DQNAgent((OBS_SIZE, 2, 3), rng=np.random.default_rng(0))
    agent.target_theta[:] = 0.0
    agent.target_theta[-3:] = [1.0, 5.0, 2.0]        # the frozen network now always answers Q = [1, 5, 2]
    R = np.array([0.0, 2.0])
    D = np.array([0.0, 1.0])                         # second transition ended in a crash
    assert np.allclose(agent.targets(R, np.zeros((2, OBS_SIZE)), D, gamma=0.9), [0 + 0.9 * 5.0, 2.0])


def test_target_network_stays_frozen_until_synced():
    rng = np.random.default_rng(2)
    agent = DQNAgent((OBS_SIZE, 4, 3), rng=rng)
    before = agent.target_theta.copy()
    batch = (rng.random((8, OBS_SIZE)), rng.integers(0, 3, 8), rng.random(8), rng.random((8, OBS_SIZE)), np.zeros(8))
    for _ in range(5):
        agent.update(batch, gamma=0.9, lr=0.01)
    assert np.array_equal(agent.target_theta, before) and not np.array_equal(agent.theta, before)
    agent.sync_target()
    assert np.array_equal(agent.target_theta, agent.theta)


def test_agent_learns_which_action_pays():
    rng = np.random.default_rng(3)
    agent = DQNAgent((OBS_SIZE, 8, 3), rng=rng)
    s = rng.random(OBS_SIZE)
    buf = ReplayBuffer(200, OBS_SIZE)
    for i in range(150):
        a = i % 3
        buf.add(s, a, 3.0 if a == 1 else 0.0, s, True)     # jump pays 3, the others pay nothing; each ends the game
    for _ in range(800):
        agent.update(buf.sample(rng, 32), gamma=0.9, lr=0.01)
    q = agent.net.forward(s)
    assert int(np.argmax(q)) == 1
    assert abs(q[1] - 3.0) < 0.3 and abs(q[0]) < 0.3 and abs(q[2]) < 0.3


# --- schedule, memory, settings ---------------------------------------------------

def test_exploration_falls_in_a_straight_line_then_stays():
    cfg = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_steps=100)
    assert epsilon(cfg, 0) == 1.0
    assert epsilon(cfg, 50) == pytest.approx(0.525)
    assert epsilon(cfg, 100) == pytest.approx(0.05) and epsilon(cfg, 10_000) == pytest.approx(0.05)


def test_replay_buffer_forgets_the_oldest_and_round_trips():
    buf = ReplayBuffer(3, OBS_SIZE)
    for i in range(5):
        buf.add(np.full(OBS_SIZE, i), i % 3, float(i), np.full(OBS_SIZE, i + 1), i == 4)
    assert buf.size == 3 and sorted(buf.R.tolist()) == [2.0, 3.0, 4.0]    # transitions 0 and 1 were overwritten
    S, A, R, S2, D = buf.sample(np.random.default_rng(0), 7)
    assert S.shape == (7, OBS_SIZE) and S.dtype == np.float64 and set(R) <= {2.0, 3.0, 4.0}
    copy = ReplayBuffer.from_arrays(buf.to_arrays(), OBS_SIZE)
    assert copy.size == 3 and copy.ptr == buf.ptr and np.array_equal(copy.R, buf.R) and np.array_equal(copy.S2, buf.S2)


@pytest.mark.parametrize("bad", [dict(gamma=1.0), dict(gamma=-0.1), dict(lr=0.0), dict(hidden=0),
                                 dict(warmup=10, batch_size=64), dict(buffer_size=500, warmup=1000),
                                 dict(eps_start=0.1, eps_end=0.5)])
def test_dqn_config_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        DQNConfig(**bad)


# --- training, saving, resuming ----------------------------------------------------

def test_short_training_run_writes_files_a_watcher_can_load(tmp_path):
    train(tiny_cfg(), 3, str(tmp_path), save_every=2, log=quiet)       # rounds 0, 1, 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.npz", "gen_0000.npz", "gen_0002.npz", "history.csv", "state.npz"]
    lines = (tmp_path / "history.csv").read_text().splitlines()
    assert lines[0] == "round,eval_score,train_score,epsilon,loss,steps,round_steps" and len(lines) == 4

    brain = load_generation(str(tmp_path), 2)
    assert (brain.unit, brain.generation, brain.n_individuals, len(brain.seeds)) == ("round", 2, 1, 1)
    assert brain.act(np.zeros(OBS_SIZE, dtype=np.float32)) in (0, 1, 2)
    assert brain.net.genome.size == genome_size(layer_sizes(6))


def test_resuming_matches_one_long_run(tmp_path):
    train(tiny_cfg(), 1, str(tmp_path / "split"), log=quiet)
    train(tiny_cfg(), 2, str(tmp_path / "split"), log=quiet)           # 1 round, then 2 more
    train(tiny_cfg(), 3, str(tmp_path / "whole"), log=quiet)
    assert (tmp_path / "split" / "history.csv").read_text() == (tmp_path / "whole" / "history.csv").read_text()
    split, whole = np.load(tmp_path / "split" / "state.npz"), np.load(tmp_path / "whole" / "state.npz")
    for key in ("theta", "target_theta", "adam_m", "adam_v", "buf_S", "buf_R", "total_steps", "best_score"):
        assert np.array_equal(split[key], whole[key]), key


def test_resume_refuses_changed_settings_that_cannot_be_changed(tmp_path):
    train(tiny_cfg(), 1, str(tmp_path), log=quiet)
    with pytest.raises(ValueError, match="hidden neurons"):
        train(tiny_cfg(hidden=9), 1, str(tmp_path), log=quiet)
    with pytest.raises(ValueError, match="memory of 500"):
        train(tiny_cfg(buffer_size=900), 1, str(tmp_path), log=quiet)
    with pytest.raises(ValueError, match="seed"):
        train(tiny_cfg(), 1, str(tmp_path), seed=5, log=quiet)
    train(tiny_cfg(gamma=0.9, lr=0.0005), 1, str(tmp_path), log=quiet)   # these may change between calls


def test_run_info_reads_a_dqn_run(tmp_path):
    train(tiny_cfg(), 2, str(tmp_path), seed=4, log=quiet)
    info = run_info(str(tmp_path))
    assert (info.hidden, info.buffer_size, info.round, info.seed) == (6, 500, 2, 4)


def test_the_two_methods_refuse_each_others_runs(tmp_path):
    train(tiny_cfg(), 1, str(tmp_path / "q"), log=quiet)
    evolution_train(Config(pop_size=6, elite=1, n_parents=3, games_per_eval=1, max_steps=100), 1,
                    str(tmp_path / "e"), log=quiet)
    assert run_method(str(tmp_path / "q")) == "dqn" and run_method(str(tmp_path / "e")) == "evolution"
    assert run_method(str(tmp_path / "nothing")) is None
    with pytest.raises(ValueError, match="DQN"):
        evolution_train(Config(), 1, str(tmp_path / "q"), log=quiet)
    with pytest.raises(ValueError, match="neuroevolution"):
        train(tiny_cfg(), 1, str(tmp_path / "e"), log=quiet)
    assert evolution_run_info(str(tmp_path / "q")) is None and run_info(str(tmp_path / "e")) is None


def test_best_checkpoint_only_ever_improves_and_is_loadable(tmp_path):
    lines = []
    train(tiny_cfg(confirm_games=2), 4, str(tmp_path), log=lines.append)
    scores = [float(l.split("best: round")[1].split(",")[1].split("over")[0]) for l in lines if "new best" in l]
    assert scores and scores == sorted(set(scores))                  # each announced best beats the one before
    best = np.load(tmp_path / "best.npz")
    assert float(best["fitness"][0]) == scores[-1] and len(best["seeds"]) == 2
    brain = load_brain(str(tmp_path / "best.npz"))
    assert brain.unit == "round" and brain.fitness == scores[-1]


def test_confirmation_games_never_overlap_the_games_used_to_train_or_evaluate():
    from ml.neuroevolution.evolve import seeds_for
    from ml.rl.train import confirm_seeds, train_seed
    for seed in (0, 7):
        for rnd in (0, 3, 99):
            confirm = set(confirm_seeds(seed, rnd, 20))
            assert confirm.isdisjoint(seeds_for(seed, rnd, 20))
            assert confirm.isdisjoint({train_seed(seed, rnd, e) for e in range(20)})
