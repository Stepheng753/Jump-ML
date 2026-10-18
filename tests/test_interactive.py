import numpy as np
from _interactive import ask, ask_choice, ask_yes_no, train_wizard, watch_wizard

from ml.brains import save_generation
from ml.network import layer_sizes, random_genome
from ml.neuroevolution.evolve import Config, train
from ml.rl.train import DQNConfig
from ml.rl.train import train as dqn_train


def scripted(answers):
    it = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"asked more questions than the test expected; last prompt: {prompt!r}")

    return fake_input


def run_wizard(wizard, runs_dir, answers):
    lines = []
    result = wizard(str(runs_dir), input_fn=scripted(answers), out=lines.append)
    return result, lines


def make_run(runs_dir, name, gens=(0, 5, 10, 15, 20), individuals=3):
    rng, sizes = np.random.default_rng(0), layer_sizes()
    folder = runs_dir / name
    folder.mkdir(parents=True)
    for g in gens:
        save_generation(folder / f"gen_{g:04d}.npz", [random_genome(rng, sizes) for _ in range(individuals)],
                        [3.0, 2.0, 1.0][:individuals], sizes, g, [1, 2, 3])
    return folder


def make_evolution_run(runs_dir, name="abc", generations=2, hidden=None):
    kwargs = {"hidden": hidden} if hidden else {}
    train(Config(pop_size=10, elite=2, n_parents=4, games_per_eval=1, max_steps=300, **kwargs), generations,
          str(runs_dir / name), log=lambda *_: None)


def make_dqn_run(runs_dir, name="q", rounds=2, save_every=5):
    dqn_train(DQNConfig(hidden=6, episodes_per_round=1, eval_games=1, max_steps=100, warmup=50, batch_size=16,
                        buffer_size=500, eps_decay_steps=200), rounds, str(runs_dir / name), save_every=save_every,
              log=lambda *_: None)


# --- the prompt helpers --------------------------------------------------------

def test_ask_returns_default_on_enter_and_asks_again_on_bad_input():
    lines = []
    assert ask("n", 7, int, input_fn=scripted([""]), out=lines.append) == 7

    def positive(text):
        if int(text) < 1:
            raise ValueError("must be at least 1")
        return int(text)

    assert ask("n", 7, positive, input_fn=scripted(["0", "-3", "4"]), out=lines.append) == 4
    assert lines == ["  ! must be at least 1", "  ! must be at least 1"]


def test_yes_no_and_choice():
    assert ask_yes_no("ok?", False, input_fn=scripted(["y"]), out=print) is True
    assert ask_yes_no("ok?", True, input_fn=scripted([""]), out=print) is True
    assert ask_yes_no("ok?", True, input_fn=scripted(["maybe", "no"]), out=lambda _: None) is False
    assert ask_choice("pick", ["a", "b"], 1, input_fn=scripted(["9", "x", "2"]), out=lambda _: None) == 2


def test_yes_no_prompt_shows_the_default_first():
    seen = []

    def record(prompt):
        seen.append(prompt)
        return ""

    ask_yes_no("Print every individual's fitness each generation?", False, input_fn=record, out=print)
    ask_yes_no("Start?", True, input_fn=record, out=print)
    assert seen == ["Print every individual's fitness each generation? [N/y]: ", "Start? [Y/n]: "]


# --- the kind of training --------------------------------------------------------

def test_the_first_question_picks_the_kind_of_training(tmp_path):
    plan, lines = run_wizard(train_wizard, tmp_path, [""] * 12)                  # Enter = the first choice
    assert plan["method"] == "evolution" and "What kind of training?" in lines
    plan, _ = run_wizard(train_wizard, tmp_path, ["2"] + [""] * 8)
    assert plan["method"] == "dqn"
    plan, lines = run_wizard(train_wizard, tmp_path, ["3", "x", "2"] + [""] * 8)  # bad answers are asked again
    assert plan["method"] == "dqn" and sum("enter a number from 1 to 2" in l for l in lines) == 2


def test_each_kind_asks_to_confirm_with_a_summary_and_can_be_declined(tmp_path):
    plan, lines = run_wizard(train_wizard, tmp_path, [""] * 11 + ["n"])
    assert plan is None and any("Your settings:" in l for l in lines)
    assert any(l.strip().startswith("method") and "neuroevolution" in l for l in lines)
    plan, lines = run_wizard(train_wizard, tmp_path, ["2"] + [""] * 7 + ["n"])
    assert plan is None and any("reinforcement learning (DQN)" in l for l in lines)


# --- neuroevolution questions ----------------------------------------------------

def test_evolution_all_defaults(tmp_path):
    # kind, name, generations, pop, hidden, elite, parents, games, save, seed, verbose, start
    plan, _ = run_wizard(train_wizard, tmp_path, [""] * 12)
    cfg = plan.pop("config")
    assert plan == {"method": "evolution", "run": "default", "reset": False, "count": 100, "seed": 0,
                    "save_every": 5, "verbose": False}
    assert (cfg.pop_size, cfg.elite, cfg.n_parents, cfg.hidden, cfg.games_per_eval) == (50, 5, 10, 12, 3)


def test_evolution_default_parents_follows_elite_and_answers_are_used(tmp_path):
    answers = ["1", "xyz", "20", "", "8", "12", "", "2", "1", "7", "y", ""]
    plan, _ = run_wizard(train_wizard, tmp_path, answers)
    cfg = plan["config"]
    assert (plan["run"], plan["count"], plan["save_every"], plan["seed"], plan["verbose"]) == ("xyz", 20, 1, 7, True)
    assert (cfg.pop_size, cfg.hidden, cfg.elite, cfg.n_parents, cfg.games_per_eval) == (50, 8, 12, 12, 2)


def test_evolution_rejects_bad_elite_and_parents(tmp_path):
    answers = ["", "", "", "", "", "60", "8", "5", "9", "", "", "", "", ""]
    plan, lines = run_wizard(train_wizard, tmp_path, answers)
    assert (plan["config"].elite, plan["config"].n_parents) == (8, 9)
    assert any("population size (50)" in l for l in lines)
    assert any("at least the number of elites (8)" in l for l in lines)


def test_evolution_continues_an_existing_run_with_its_fixed_settings(tmp_path):
    make_evolution_run(tmp_path, hidden=7)
    # kind, name, start over? (n), generations, elite, parents, games, save, verbose, start  (no pop/hidden/seed)
    plan, lines = run_wizard(train_wizard, tmp_path, ["1", "abc", "n", "5", "", "", "", "", "", ""])
    assert (plan["run"], plan["count"], plan["seed"], plan["reset"]) == ("abc", 5, None, False)
    assert (plan["config"].pop_size, plan["config"].hidden) == (10, 7)
    assert any("trained 2 generations" in l for l in lines) and not any("DELETE" in l for l in lines)


def test_evolution_start_over_is_the_default_and_warns(tmp_path):
    make_evolution_run(tmp_path)
    # an Enter on "delete it and start over?" means yes, and then every question is asked
    plan, lines = run_wizard(train_wizard, tmp_path, ["1", "abc", ""] + [""] * 10)
    assert plan["reset"] is True and plan["config"].pop_size == 50 and plan["seed"] == 0
    assert any("DELETE runs/abc (2 generations trained)" in l for l in lines)


def test_the_delete_prompt_shows_yes_as_the_default(tmp_path):
    make_evolution_run(tmp_path, generations=1)
    prompts, answers = [], iter(["1", "abc", "n"] + [""] * 8)

    def record(prompt):
        prompts.append(prompt.strip())
        return next(answers)

    train_wizard(str(tmp_path), input_fn=record, out=lambda _: None)
    assert "Delete it and start over instead? [Y/n]:" in prompts


# --- DQN questions ---------------------------------------------------------------

def test_dqn_all_defaults(tmp_path):
    # kind, name, rounds, episodes, hidden, save, seed, advanced?, start
    plan, _ = run_wizard(train_wizard, tmp_path, ["2"] + [""] * 8)
    cfg = plan.pop("config")
    assert plan == {"method": "dqn", "run": "dqn", "reset": False, "count": 100, "seed": 0, "save_every": 5,
                    "verbose": False}
    assert (cfg.hidden, cfg.episodes_per_round, cfg.gamma, cfg.lr, cfg.batch_size, cfg.buffer_size) == (128, 10, 0.99, 0.001, 64, 100000)
    assert (cfg.train_every, cfg.target_sync, cfg.eps_decay_steps, cfg.eval_games, cfg.warmup) == (4, 4000, 50000, 3, 1000)


def test_dqn_advanced_settings_are_validated_and_used(tmp_path):
    answers = ["2", "", "", "", "", "", "", "y",
               "1", "0.9",              # gamma: 1 is rejected, then 0.9
               "0.01",                  # learning rate
               "2000",                  # batch
               "1500", "",              # memory: smaller than one batch is rejected, then the default
               "8", "500", "20000", "5",
               ""]                      # start
    plan, lines = run_wizard(train_wizard, tmp_path, answers)
    cfg = plan["config"]
    assert (cfg.gamma, cfg.lr, cfg.batch_size, cfg.buffer_size) == (0.9, 0.01, 2000, 100000)
    assert (cfg.train_every, cfg.target_sync, cfg.eps_decay_steps, cfg.eval_games) == (8, 500, 20000, 5)
    assert cfg.warmup == 2000                # a bigger batch needs a bigger warmup
    assert any("below 1" in l for l in lines) and any("at least one batch" in l for l in lines)


def test_dqn_continues_an_existing_run_with_its_fixed_settings(tmp_path):
    make_dqn_run(tmp_path)
    # kind, name, start over? (n), rounds, episodes, save, advanced?, start   (no hidden / seed questions)
    plan, lines = run_wizard(train_wizard, tmp_path, ["2", "q", "n", "5", "", "", "", ""])
    cfg = plan["config"]
    assert (plan["count"], plan["seed"], plan["reset"]) == (5, None, False)
    assert (cfg.hidden, cfg.buffer_size) == (6, 500)
    assert cfg.warmup == 500                 # the default 1000 would not fit in this run's 500-transition memory
    assert any("trained 2 rounds" in l for l in lines)


def test_dqn_start_over_is_the_default_and_warns(tmp_path):
    make_dqn_run(tmp_path)
    plan, lines = run_wizard(train_wizard, tmp_path, ["2", "q", ""] + [""] * 7)
    assert plan["reset"] is True and plan["config"].hidden == 128
    assert any("DELETE runs/q (2 rounds trained)" in l for l in lines)


def test_each_kind_refuses_the_other_kinds_run_names(tmp_path):
    make_dqn_run(tmp_path, "q")
    make_evolution_run(tmp_path, "evo")
    plan, lines = run_wizard(train_wizard, tmp_path, ["2", "evo", ""] + [""] * 7)
    assert plan["run"] == "dqn" and any("neuroevolution" in l for l in lines)
    plan, lines = run_wizard(train_wizard, tmp_path, ["1", "q", ""] + [""] * 10)
    assert plan["run"] == "default" and any("DQN" in l for l in lines)


# --- watching: the four types ------------------------------------------------------

def fake_dqn_run(runs_dir, name="d", gens=(0, 5, 10), with_best=True):
    folder = runs_dir / name
    folder.mkdir()
    rng, sizes = np.random.default_rng(0), layer_sizes()
    for g in gens:
        save_generation(folder / f"gen_{g:04d}.npz", [random_genome(rng, sizes)], [50.0], sizes, g, [1, 2, 3], kind="dqn")
    if with_best:
        save_generation(folder / "best.npz", [random_genome(rng, sizes)], [118.0], sizes, 5, [9, 8, 7], kind="dqn")
    return folder


def test_the_first_question_offers_the_four_types(tmp_path):
    make_run(tmp_path, "e")
    _, lines = run_wizard(watch_wizard, tmp_path, ["1", "3", "", "", ""])
    text = "\n".join(lines)
    for label in ("single: one player", "single VS: an evolution player on the left", "multiple: several generations",
                  "multiple VS: evolution generations on the left"):
        assert label in text


def test_single_evolution_asks_the_run_checkpoint_individual_replay_seed_speed(tmp_path):
    make_run(tmp_path, "e")
    # type single, kind evolution, run 1, checkpoint latest, individual 2, no replay, seed 123, speed 3, start
    plan, _ = run_wizard(watch_wizard, tmp_path, ["1", "1", "1", "", "2", "n", "123", "3", ""])
    assert plan == {"kind": "single", "type": "single", "brain": str(tmp_path / "e" / "gen_0020.npz"), "rank": 2,
                    "replay": False, "seed": 123, "speed": 3.0}


def test_single_dqn_offers_best_by_default(tmp_path):
    fake_dqn_run(tmp_path, "d")
    # type single, kind DQN, run 1, checkpoint (best by default), replay no, seed random, speed, start
    plan, _ = run_wizard(watch_wizard, tmp_path, ["1", "2", "1", "", "n", "", "", ""])
    assert plan["brain"] == str(tmp_path / "d" / "best.npz") and plan["rank"] == 1 and plan["seed"] is None


def test_single_perfect_bot_only_asks_seed_and_speed(tmp_path):
    plan, _ = run_wizard(watch_wizard, tmp_path, ["1", "3", "", "2", ""])
    assert plan == {"kind": "single", "type": "single", "brain": "perfect", "rank": 1, "replay": False, "seed": None, "speed": 2.0}


def test_single_says_when_there_are_no_runs_of_that_kind(tmp_path):
    make_run(tmp_path, "e")
    plan, lines = run_wizard(watch_wizard, tmp_path, ["1", "2"])        # DQN asked for, only an evolution run exists
    assert plan is None and any("there are no DQN runs" in l for l in lines)


def test_single_vs_puts_evolution_left_and_dqn_right(tmp_path):
    make_run(tmp_path, "e")
    fake_dqn_run(tmp_path, "d")
    # type single VS; left: run 1 (evolution), checkpoint latest; right: run 1 (DQN), checkpoint best; seed; speed; start
    plan, _ = run_wizard(watch_wizard, tmp_path, ["2", "1", "", "1", "", "", "", ""])
    assert plan["kind"] == "vs" and plan["type"] == "single_vs"
    assert plan["specs"] == [f"{tmp_path / 'e'}:20", f"{tmp_path / 'd'}:best"]
    assert (plan["rows"], plan["cols"]) == (1, 2)


def test_single_vs_needs_a_dqn_run_for_the_right_side(tmp_path):
    make_run(tmp_path, "e")
    plan, lines = run_wizard(watch_wizard, tmp_path, ["2", "1", "", "", ""])
    assert plan is None and any("there are no DQN runs" in l for l in lines)


def test_multiple_one_kind_side_by_side_with_chosen_generations(tmp_path):
    make_run(tmp_path, "e")
    # type multiple, kind evolution, run 1, generations 0 10 20, grid (default 1 row), rank (default), seed, speed, start
    plan, _ = run_wizard(watch_wizard, tmp_path, ["3", "1", "1", "0 10 20", "", "", "", "", ""])
    assert plan["kind"] == "multiple" and plan["type"] == "multiple"
    assert plan["generations"] == [0, 10, 20] and (plan["rows"], plan["cols"]) == (1, 3)


def test_multiple_rejects_unsaved_generations_and_asks_again(tmp_path):
    make_run(tmp_path, "e")
    plan, lines = run_wizard(watch_wizard, tmp_path, ["3", "1", "1", "0 7", "0 10", "", "", "", "", ""])
    assert plan["generations"] == [0, 10] and any("not saved: 7" in l for l in lines)


def test_multiple_vs_default_is_a_three_by_two_grid_of_evolution_and_dqn_rows(tmp_path):
    make_run(tmp_path, "e")
    fake_dqn_run(tmp_path, "d")
    # type multiple VS; left: run 1, generations 0 10 20; right: run 1, the default [0, 10, best]; seed; speed; start
    plan, _ = run_wizard(watch_wizard, tmp_path, ["4", "1", "0 10 20", "1", "", "", "", ""])
    e, d = str(tmp_path / "e"), str(tmp_path / "d")
    assert plan["kind"] == "vs" and plan["type"] == "multiple_vs"
    assert plan["specs"] == [f"{e}:0", f"{d}:0", f"{e}:10", f"{d}:10", f"{e}:20", f"{d}:best"]
    assert (plan["rows"], plan["cols"]) == (3, 2)


def test_multiple_vs_default_for_a_dqn_side_is_spread_with_best_last(tmp_path):
    make_run(tmp_path, "e")
    fake_dqn_run(tmp_path, "d")
    prompts = []
    answers = iter(["4", "1", "0 10 20", "1", "", "", "", ""])

    def record(prompt=""):
        prompts.append(prompt)
        return next(answers)

    watch_wizard(str(tmp_path), input_fn=record, out=lambda _: None)
    assert any("best = the best checkpoint" in p and "[0 10 best]" in p for p in prompts)


def test_multiple_vs_both_sides_must_list_the_same_count(tmp_path):
    make_run(tmp_path, "e")
    fake_dqn_run(tmp_path, "d")
    plan, lines = run_wizard(watch_wizard, tmp_path, ["4", "1", "0 10 20", "1", "0 10", "0 10 best", "", "", ""])
    assert plan["rows"] == 3 and any("list exactly 3 rounds" in l for l in lines)


def test_multiple_vs_needs_enough_generations_on_each_side(tmp_path):
    make_run(tmp_path, "e", gens=(0, 10))            # only two saved: the default of three can't be met
    fake_dqn_run(tmp_path, "d")
    lines = []
    plan = watch_wizard(str(tmp_path), input_fn=scripted(["4", "1", "1", "2", "", "", "", ""]), out=lines.append)
    assert plan is None and any("has only 2 saved generations to pick from" in l for l in lines)


def test_watch_can_be_declined_at_the_summary(tmp_path):
    make_run(tmp_path, "e")
    plan, lines = run_wizard(watch_wizard, tmp_path, ["1", "3", "", "", "n"])
    assert plan is None and any("Your settings:" in l for l in lines)


def test_multiple_questions_accept_latest_and_best(tmp_path):
    fake_dqn_run(tmp_path, "d")                                              # rounds 0, 5, 10 and a best.npz
    prompts = []
    answers = iter(["3", "2", "1", "0 latest best", "", "", "", "", ""])    # multiple, DQN, run 1, the list, grid, seed, speed, start

    def record(prompt=""):
        prompts.append(prompt)
        return next(answers)

    plan = watch_wizard(str(tmp_path), input_fn=record, out=lambda _: None)
    assert plan["generations"] == [0, 10, "best"] and (plan["rows"], plan["cols"]) == (1, 3)
    assert any("or latest, or best for the best checkpoint" in p and "[0 5 10 best]" in p for p in prompts)


def test_multiple_default_for_a_dqn_run_ends_with_best(tmp_path):
    fake_dqn_run(tmp_path, "d")
    plan, _ = run_wizard(watch_wizard, tmp_path, ["3", "2", "1", "", "", "", "", ""])
    assert plan["generations"] == [0, 5, 10, "best"] and (plan["rows"], plan["cols"]) == (1, 4)


def test_multiple_best_is_refused_for_an_evolution_run_and_latest_works(tmp_path):
    make_run(tmp_path, "e")
    plan, lines = run_wizard(watch_wizard, tmp_path, ["3", "1", "1", "0 best", "0 latest", "", "", "", "", ""])
    assert any("no best checkpoint for this run" in l for l in lines)
    assert plan["generations"] == [0, 20]                                    # "latest" is the highest saved generation
