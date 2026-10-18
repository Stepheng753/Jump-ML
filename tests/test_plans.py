import os

import numpy as np
import pytest

from game_interface.plan import build_watch
from ml.brains import PerfectBrain, save_generation
from ml.network import layer_sizes, random_genome
from ml.neuroevolution.evolve import Config
from ml.plans import run_training
from ml.rl.train import DQNConfig
from ml.runs import run_method


def quiet(*_):
    pass


def evolution_plan(**overrides):
    plan = {"method": "evolution", "run": "e", "reset": False, "count": 2, "seed": 0, "save_every": 1, "verbose": False,
            "config": Config(pop_size=6, elite=1, n_parents=3, games_per_eval=1, max_steps=100)}
    plan.update(overrides)
    return plan


def dqn_plan(**overrides):
    plan = {"method": "dqn", "run": "q", "reset": False, "count": 2, "seed": 0, "save_every": 1, "verbose": False,
            "config": DQNConfig(hidden=6, episodes_per_round=1, eval_games=1, max_steps=100, warmup=50, batch_size=16,
                                buffer_size=500, eps_decay_steps=200)}
    plan.update(overrides)
    return plan


# --- training plans ----------------------------------------------------------------

def test_a_plan_trains_with_the_chosen_method_into_its_run_folder(tmp_path):
    run_training(evolution_plan(), str(tmp_path), log=quiet)
    run_training(dqn_plan(), str(tmp_path), log=quiet)
    assert run_method(str(tmp_path / "e")) == "evolution" and run_method(str(tmp_path / "q")) == "dqn"
    assert (tmp_path / "e" / "gen_0001.npz").exists() and (tmp_path / "q" / "best.npz").exists()


def test_running_a_plan_again_continues_the_run_and_reset_starts_over(tmp_path):
    run_training(dqn_plan(), str(tmp_path), log=quiet)
    run_training(dqn_plan(), str(tmp_path), log=quiet)
    assert len((tmp_path / "q" / "history.csv").read_text().splitlines()) == 1 + 4      # header + 2 + 2 rounds
    run_training(dqn_plan(reset=True), str(tmp_path), log=quiet)
    assert len((tmp_path / "q" / "history.csv").read_text().splitlines()) == 1 + 2      # wiped, then 2 rounds


def test_a_plan_cannot_continue_the_other_methods_run(tmp_path):
    run_training(evolution_plan(run="x"), str(tmp_path), log=quiet)
    with pytest.raises(ValueError, match="neuroevolution"):
        run_training(dqn_plan(run="x"), str(tmp_path), log=quiet)


@pytest.mark.parametrize("bad", ["", ".", "..", f"a{os.sep}b"])
def test_a_plan_needs_a_plain_run_name(tmp_path, bad):
    with pytest.raises(ValueError, match="plain folder name"):
        run_training(evolution_plan(run=bad), str(tmp_path), log=quiet)


def test_an_unknown_kind_of_training_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown kind"):
        run_training(evolution_plan(method="magic"), str(tmp_path), log=quiet)


# --- watch plans ---------------------------------------------------------------------

def make_run(folder, gens=(0, 5, 10), with_seeds=True):
    folder.mkdir()
    rng, sizes = np.random.default_rng(0), layer_sizes()
    for g in gens:
        save_generation(folder / f"gen_{g:04d}.npz", [random_genome(rng, sizes) for _ in range(3)], [3.0, 2.0, 1.0],
                        sizes, g, [1, 2, 3] if with_seeds else [])
    return folder


def test_a_single_plan_loads_one_brain(tmp_path):
    run = make_run(tmp_path / "r")
    job = build_watch({"kind": "single", "brain": str(run), "rank": 2, "replay": False, "seed": 5, "speed": 2.0})
    assert job["kind"] == "single" and job["brain"].generation == 10 and job["brain"].rank == 2
    assert (job["seed"], job["speed"], job["replay"]) == (5, 2.0, False)
    assert isinstance(build_watch({"kind": "single", "brain": "perfect", "rank": 1, "replay": False, "seed": None,
                                   "speed": 1.0})["brain"], PerfectBrain)


def test_a_multiple_plan_orders_cells_by_generation(tmp_path):
    run = make_run(tmp_path / "r")
    job = build_watch({"kind": "multiple", "run_dir": str(run), "generations": [10, 0, 5], "rank": 1, "rows": 1,
                       "cols": 3, "seed": None, "speed": 1.0})
    assert job["kind"] == "grid" and [b.generation for b in job["brains"]] == [0, 5, 10]
    assert (job["rows"], job["cols"], job["caption"]) == (1, 3, None)


def test_a_vs_plan_keeps_the_order_given_and_labels_each_player(tmp_path):
    run = make_run(tmp_path / "r")
    job = build_watch({"kind": "vs", "specs": ["perfect", f"{run}:5", f"{run}:0"], "rank": 1, "rows": 1, "cols": 3,
                       "seed": None, "speed": 1.0})
    assert [b.tag for b in job["brains"]] == ["perfect bot", "evo r g5 #1 fit 3.0", "evo r g0 #1 fit 3.0"]
    assert job["caption"] == "Jump-ML: VS"


@pytest.mark.parametrize("plan, message", [
    ({"kind": "vs", "specs": ["perfect"], "rows": 1, "cols": 1}, "at least two"),
    ({"kind": "vs", "specs": ["perfect", "perfect"], "rows": 2, "cols": 2}, "must match"),
    ({"kind": "multiple", "generations": [0, 0], "rows": 1, "cols": 2, "run_dir": "x", "rank": 1}, "more than once"),
    ({"kind": "teleport"}, "unknown kind"),
])
def test_bad_watch_plans_are_refused_with_a_readable_message(plan, message):
    plan = {"seed": None, "speed": 1.0, **plan}
    with pytest.raises(ValueError, match=message):
        build_watch(plan)


def test_replay_needs_a_generation_that_recorded_its_games(tmp_path):
    run = make_run(tmp_path / "r", with_seeds=False)
    with pytest.raises(ValueError, match="replay needs"):
        build_watch({"kind": "single", "brain": str(run), "rank": 1, "replay": True, "seed": None, "speed": 1.0})


def test_a_multiple_plan_shows_best_last_and_labels_it(tmp_path):
    folder = make_run(tmp_path / "d", gens=(0, 5, 10))                      # three checkpoints and a best.npz at round 5
    rng, sizes = np.random.default_rng(1), layer_sizes()
    save_generation(folder / "best.npz", [random_genome(rng, sizes)], [118.0], sizes, 5, [9, 8, 7], kind="dqn")
    plan = {"kind": "multiple", "run_dir": str(folder), "generations": ["best", 10, 0], "rank": 1, "rows": 1, "cols": 3,
            "seed": None, "speed": 1.0}
    brains = build_watch(plan)["brains"]
    assert [b.generation for b in brains] == [0, 10, 5]                      # numbers earliest first, then best (round 5)
    assert brains[-1].tag == "dqn d best r5 eval 118.0"


def test_a_multiple_plan_without_best_is_still_ordered_by_generation(tmp_path):
    folder = make_run(tmp_path / "r", gens=(0, 5, 10))
    plan = {"kind": "multiple", "run_dir": str(folder), "generations": [10, 0, 5], "rank": 1, "rows": 1, "cols": 3,
            "seed": None, "speed": 1.0}
    assert [b.generation for b in build_watch(plan)["brains"]] == [0, 5, 10]
