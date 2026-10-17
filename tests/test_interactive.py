import os

import numpy as np
import pytest
from _interactive import ask, ask_choice, ask_yes_no, train_wizard, watch_wizard

from ml.brains import save_generation
from ml.network import layer_sizes, random_genome
from ml.neuroevolution.evolve import Config, train


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


# --- train wizard --------------------------------------------------------------

DEFAULT_TRAIN = ["--run", "default", "--generations", "100", "--pop", "50", "--hidden", "12", "--elite", "5",
                 "--parents", "10", "--games", "3", "--save-every", "5", "--seed", "0"]


def test_train_wizard_all_defaults(tmp_path):
    argv, _ = run_wizard(train_wizard, tmp_path, [""] * 11)
    assert argv == DEFAULT_TRAIN


def test_train_wizard_default_parents_follows_elite_and_answers_are_used(tmp_path):
    # name, generations, pop, hidden, elite, parents, games, save, seed, verbose, start
    argv, _ = run_wizard(train_wizard, tmp_path, ["xyz", "20", "", "8", "12", "", "2", "1", "7", "y", ""])
    assert argv == ["--run", "xyz", "--generations", "20", "--pop", "50", "--hidden", "8", "--elite", "12",
                    "--parents", "12", "--games", "2", "--save-every", "1", "--seed", "7", "--verbose"]


def test_train_wizard_rejects_bad_elite_and_parents(tmp_path):
    answers = ["", "", "", "", "60", "8", "5", "9", "", "", "", "", ""]
    argv, lines = run_wizard(train_wizard, tmp_path, answers)
    assert "--elite" in argv and argv[argv.index("--elite") + 1] == "8"
    assert argv[argv.index("--parents") + 1] == "9"
    assert any("population size (50)" in l for l in lines)
    assert any("at least the number of elites (8)" in l for l in lines)


def test_train_wizard_can_be_declined(tmp_path):
    argv, _ = run_wizard(train_wizard, tmp_path, [""] * 10 + ["n"])
    assert argv is None


def test_train_wizard_continues_an_existing_run_with_its_fixed_settings(tmp_path):
    train(Config(pop_size=10, elite=2, n_parents=4, games_per_eval=1, max_steps=300, hidden=7), 2,
          str(tmp_path / "abc"), log=lambda *_: None)
    # name, start-over? (n), generations, elite, parents, games, save, verbose, start  (no pop/hidden/seed questions)
    argv, lines = run_wizard(train_wizard, tmp_path, ["abc", "n", "5", "", "", "", "", "", ""])
    assert argv[:6] == ["--run", "abc", "--generations", "5", "--pop", "10"]
    assert argv[argv.index("--hidden") + 1] == "7"
    assert "--seed" not in argv and "--reset" not in argv
    assert any("trained 2 generations" in l for l in lines)


def test_train_wizard_start_over_asks_everything_and_resets(tmp_path):
    train(Config(pop_size=10, elite=2, n_parents=4, games_per_eval=1, max_steps=300), 1,
          str(tmp_path / "abc"), log=lambda *_: None)
    argv, _ = run_wizard(train_wizard, tmp_path, ["abc", "y"] + [""] * 10)
    assert "--reset" in argv and argv[argv.index("--pop") + 1] == "50" and "--seed" in argv


# --- watch wizard --------------------------------------------------------------

def rel(path):
    return os.path.relpath(path)


def test_watch_wizard_defaults_to_latest_generation_of_the_newest_run(tmp_path):
    make_run(tmp_path, "r1")
    # choose, mode, generation, individual, replay?, seed, speed, start
    argv, _ = run_wizard(watch_wizard, tmp_path, [""] * 8)
    assert argv == ["--brain", rel(tmp_path / "r1"), "--speed", "1.0"]


def test_watch_wizard_specific_generation_individual_and_replay(tmp_path):
    make_run(tmp_path, "r1")
    argv, _ = run_wizard(watch_wizard, tmp_path, ["", "1", "10", "2", "y", "3", ""])
    assert argv == ["--brain", rel(tmp_path / "r1" / "gen_0010.npz"), "--rank", "2", "--replay", "--speed", "3.0"]


def test_watch_wizard_rejects_unsaved_generation_and_too_high_rank(tmp_path):
    make_run(tmp_path, "r1")
    argv, lines = run_wizard(watch_wizard, tmp_path, ["", "1", "7", "10", "99", "1", "n", "123", "", ""])
    assert argv == ["--brain", rel(tmp_path / "r1" / "gen_0010.npz"), "--seed", "123", "--speed", "1.0"]
    assert any("generation 7 was not saved" in l for l in lines)
    assert any("rank must be between 1 and 3" in l for l in lines)


def test_watch_wizard_compare_several_generations(tmp_path):
    make_run(tmp_path, "r1")
    # choose, mode=2, generations (default spreads 4 over the saved ones), size 2 3 is wrong then 2 2, rank, seed, speed, start
    argv, lines = run_wizard(watch_wizard, tmp_path, ["", "2", "", "2 3", "2 2", "", "", "", ""])
    assert argv == ["--brain", rel(tmp_path / "r1"), "--multiple", "0", "5", "15", "20", "--size", "2", "2", "--speed", "1.0"]
    assert any("makes 6 cells" in l for l in lines)


def test_watch_wizard_compare_rejects_bad_generation_lists(tmp_path):
    make_run(tmp_path, "r1")
    answers = ["", "2", "5", "5 5", "5 7", "0 20", "", "", "", "", ""]
    argv, lines = run_wizard(watch_wizard, tmp_path, answers)
    assert argv[argv.index("--multiple") + 1:argv.index("--multiple") + 3] == ["0", "20"]
    assert any("at least two" in l for l in lines) and any("twice" in l for l in lines) and any("not saved: 7" in l for l in lines)


def test_watch_wizard_with_no_runs_offers_only_the_perfect_bot(tmp_path):
    argv, lines = run_wizard(watch_wizard, tmp_path, ["", "", "2", ""])
    assert argv == ["--brain", "perfect", "--speed", "2.0"]
    assert any("only the perfect bot" in l for l in lines)


def test_watch_wizard_can_pick_the_perfect_bot_when_runs_exist(tmp_path):
    make_run(tmp_path, "r1")
    argv, _ = run_wizard(watch_wizard, tmp_path, ["2", "", "", ""])
    assert argv == ["--brain", "perfect", "--speed", "1.0"]


def _existing_run(tmp_path, generations=2):
    train(Config(pop_size=10, elite=2, n_parents=4, games_per_eval=1, max_steps=300), generations,
          str(tmp_path / "abc"), log=lambda *_: None)


def test_existing_run_defaults_to_delete_and_start_over_with_a_warning(tmp_path):
    _existing_run(tmp_path)
    # name, then Enter on "delete and start over?" (the default is yes), then the 10 questions of a fresh run
    argv, lines = run_wizard(train_wizard, tmp_path, ["abc", ""] + [""] * 10)
    assert "--reset" in argv
    assert any("DELETE runs/abc (2 generations trained)" in l for l in lines)


def test_continuing_an_existing_run_has_no_delete_warning(tmp_path):
    _existing_run(tmp_path)
    argv, lines = run_wizard(train_wizard, tmp_path, ["abc", "n", "5", "", "", "", "", "", ""])
    assert "--reset" not in argv
    assert not any("DELETE" in l for l in lines)


def test_the_delete_prompt_shows_yes_as_the_default(tmp_path):
    _existing_run(tmp_path, generations=1)
    prompts, answers = [], iter(["abc", "n"] + [""] * 8)

    def record(prompt):
        prompts.append(prompt.strip())
        return next(answers)

    train_wizard(str(tmp_path), input_fn=record, out=lambda _: None)
    assert "Delete it and start over instead? [Y/n]:" in prompts
