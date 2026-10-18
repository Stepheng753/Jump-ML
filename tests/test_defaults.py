import copy
import json
import os

import numpy as np
import pytest
from _defaults import SettingsError, train_plan_from_file, watch_plan_from_file
from _interactive import train_wizard, watch_wizard

from ml.brains import save_generation
from ml.network import layer_sizes, random_genome
from ml.neuroevolution.evolve import Config
from ml.neuroevolution.evolve import train as evolution_train
from ml.rl.train import DQNConfig
from ml.rl.train import train as dqn_train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED = os.path.join(ROOT, "ml", "run", "default.json")


def quiet(*_):
    pass


def write(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def write_train(path, data):
    return write(path, {"train": data})


def write_watch(path, data):
    return write(path, {"watch": data})


def shipped(part):
    """One part ("train" or "watch") of the real ml/run/default.json."""
    with open(SHIPPED) as f:
        return json.load(f)[part]


def fake_evolution_run(runs, name, gens=(0, 50, 99)):
    rng, sizes = np.random.default_rng(0), layer_sizes()
    (runs / name).mkdir(parents=True, exist_ok=True)
    for g in gens:
        save_generation(runs / name / f"gen_{g:04d}.npz", [random_genome(rng, sizes) for _ in range(3)], [3.0, 2.0, 1.0],
                        sizes, g, [1, 2, 3])


def fake_dqn_run(runs, name, gens=(0, 50, 99)):
    rng, sizes = np.random.default_rng(0), layer_sizes()
    (runs / name).mkdir(parents=True, exist_ok=True)
    for g in gens:
        save_generation(runs / name / f"gen_{g:04d}.npz", [random_genome(rng, sizes)], [50.0], sizes, g, [1, 2, 3], kind="dqn")
    save_generation(runs / name / "best.npz", [random_genome(rng, sizes)], [118.0], sizes, 5, [9, 8, 7], kind="dqn")


def runs_for_defaults(tmp_path):
    """Fake runs for every run name the shipped watch part mentions, so it can be loaded whatever they are called."""
    runs = tmp_path / "runs"
    watch = shipped("watch")
    for section in ("single", "single_vs", "multiple", "multiple_vs"):
        fake_evolution_run(runs, watch[section]["evolution"]["run"])
        fake_dqn_run(runs, watch[section]["dqn"]["run"])
    return runs


# --- the train part of default.json -----------------------------------------------------------

@pytest.mark.parametrize("method", ["evolution", "dqn"])
def test_the_shipped_train_part_holds_each_kinds_settings(tmp_path, method):
    section = shipped("train")[method]
    plan, info, fixed = train_plan_from_file(SHIPPED, str(tmp_path / "runs"), method)   # the file as shipped, not a copy
    assert (plan["method"], plan["run"], plan["config"].hidden) == (method, section["run"], section["hidden"])
    assert plan["count"] == section["generations" if method == "evolution" else "rounds"]
    assert (info, fixed) == (None, False)


def test_the_kind_is_never_read_from_the_file(tmp_path):
    runs = str(tmp_path / "runs")
    with pytest.raises(SettingsError, match="unknown kind of training"):
        train_plan_from_file(SHIPPED, runs, "magic")
    with pytest.raises(TypeError):
        train_plan_from_file(SHIPPED, runs)                                       # the kind is not optional
    data = {**shipped("train"), "method": "evolution"}                            # an old-style file that names a kind
    with pytest.raises(SettingsError, match="train.method: not used: you choose neuroevolution or DQN when the script runs"):
        train_plan_from_file(write_train(tmp_path / "old.json", data), runs, "dqn")


def test_one_file_serves_both_kinds_when_the_kind_is_passed(tmp_path):
    path = write_train(tmp_path / "t.json", shipped("train"))
    kinds = [train_plan_from_file(path, str(tmp_path / "runs"), m)[0]["method"] for m in ("evolution", "dqn")]
    assert kinds == ["evolution", "dqn"]


def test_continuing_an_existing_run_needs_matching_fixed_settings_and_ignores_the_seed(tmp_path):
    runs = tmp_path / "runs"
    data = shipped("train")
    name = data["dqn"]["run"]
    dqn_train(DQNConfig(hidden=6, episodes_per_round=1, eval_games=1, max_steps=100, warmup=50, batch_size=16,
                        buffer_size=500, eps_decay_steps=200), 1, str(runs / name), seed=3, log=quiet)
    data["dqn"]["existing_run"] = "continue"
    with pytest.raises(SettingsError, match="started with 6 hidden neurons"):
        train_plan_from_file(write_train(tmp_path / "a.json", data), str(runs), "dqn")

    data["dqn"].update(hidden=6, memory_size=500, seed=99)
    plan, info, fixed = train_plan_from_file(write_train(tmp_path / "b.json", data), str(runs), "dqn")
    assert fixed and info.round == 1 and plan["reset"] is False and plan["seed"] is None   # the run's own seed is kept

    data["dqn"]["existing_run"] = "start_over"
    plan, info, fixed = train_plan_from_file(write_train(tmp_path / "c.json", data), str(runs), "dqn")
    assert plan["reset"] is True and plan["seed"] == 99 and not fixed


def test_evolution_continue_checks_population_and_hidden(tmp_path):
    runs = tmp_path / "runs"
    data = shipped("train")
    evolution_train(Config(pop_size=10, elite=2, n_parents=4, games_per_eval=1, max_steps=100, hidden=7), 1,
                    str(runs / data["evolution"]["run"]), log=quiet)
    data["evolution"]["existing_run"] = "continue"
    with pytest.raises(SettingsError) as e:
        train_plan_from_file(write_train(tmp_path / "a.json", data), str(runs), "evolution")
    assert "population of 10" in str(e.value) or "7 hidden neurons" in str(e.value)
    data["evolution"].update(population=10, hidden=7, elites=2, parents=4)
    plan, _, fixed = train_plan_from_file(write_train(tmp_path / "b.json", data), str(runs), "evolution")
    assert fixed and plan["count"] == data["evolution"]["generations"]


def test_a_run_name_owned_by_the_other_kind_is_refused(tmp_path):
    runs = tmp_path / "runs"
    data = shipped("train")
    evolution_train(Config(pop_size=6, elite=1, n_parents=3, games_per_eval=1, max_steps=100), 1,
                    str(runs / data["dqn"]["run"]), log=quiet)
    with pytest.raises(SettingsError, match="neuroevolution, the other kind"):
        train_plan_from_file(write_train(tmp_path / "a.json", data), str(runs), "dqn")


def test_every_problem_in_the_file_is_reported_at_once(tmp_path):
    data = shipped("train")
    data["dqn"].update(rounds=0, gamma=1.5, run="a/b", batch_size=True, learning_rate="fast", existing_run="maybe", gama=0.9)
    with pytest.raises(SettingsError) as e:
        train_plan_from_file(write_train(tmp_path / "a.json", data), str(tmp_path), "dqn")
    text = str(e.value)
    for expected in ("dqn.rounds", "dqn.gamma", "dqn.run", "dqn.batch_size: must be a number", "dqn.learning_rate: must be a number",
                     "dqn.existing_run", "dqn.gama: unknown setting"):
        assert expected in text, expected


def test_missing_broken_and_nonsense_files_give_clear_errors(tmp_path):
    with pytest.raises(SettingsError, match="was not found"):
        train_plan_from_file(str(tmp_path / "nope.json"), str(tmp_path), "dqn")
    (tmp_path / "bad.json").write_text("{ not json")
    with pytest.raises(SettingsError, match="not valid JSON"):
        train_plan_from_file(str(tmp_path / "bad.json"), str(tmp_path), "dqn")
    with pytest.raises(SettingsError, match="JSON object"):
        train_plan_from_file(write(tmp_path / "l.json", [1, 2]), str(tmp_path), "dqn")


def test_cross_setting_rules_are_checked(tmp_path):
    data = shipped("train")
    data["evolution"].update(elites=8, parents=5)
    with pytest.raises(SettingsError, match="at least elites"):
        train_plan_from_file(write_train(tmp_path / "a.json", data), str(tmp_path), "evolution")


def test_a_null_seed_means_zero_and_notes_are_ignored(tmp_path):
    data = shipped("train")
    data["dqn"]["seed"] = None
    data["dqn"]["_remark"] = "anything"
    assert train_plan_from_file(write_train(tmp_path / "a.json", data), str(tmp_path), "dqn")[0]["seed"] == 0


# --- the watch part of default.json ----------------------------------------------------

def watch_file(tmp_path, changes):
    """A file whose watch part is the shipped one with the given sections replaced."""
    watch = copy.deepcopy(shipped("watch"))
    watch.update(changes)
    return write_watch(tmp_path / "watch.json", watch)


def test_the_shipped_watch_part_builds_all_four_types_from_its_runs(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")

    for mode in ("evolution", "dqn"):
        single = watch_plan_from_file(SHIPPED, str(runs), "single", mode)
        assert single["type"] == "single" and single["brain"].startswith(str(runs / watch["single"][mode]["run"]))
        multiple = watch_plan_from_file(SHIPPED, str(runs), "multiple", mode)
        asked = watch["multiple"][mode]["generations"]
        assert multiple["generations"] == [99 if g == "latest" else g for g in asked]      # "latest" is the highest saved (99 here)
        assert multiple["run_dir"] == str(runs / watch["multiple"][mode]["run"])
    assert watch_plan_from_file(SHIPPED, str(runs), "single", "perfect")["brain"] == "perfect"

    sv = watch_plan_from_file(SHIPPED, str(runs), "single_vs")
    assert (sv["type"], sv["rows"], sv["cols"], len(sv["specs"])) == ("single_vs", 1, 2, 2)

    mv = watch_plan_from_file(SHIPPED, str(runs), "multiple_vs")
    assert (mv["type"], mv["rows"], mv["cols"], len(mv["specs"])) == ("multiple_vs", 3, 2, 6)
    evo_dir = str(runs / watch["multiple_vs"]["evolution"]["run"])
    dqn_dir = str(runs / watch["multiple_vs"]["dqn"]["run"])
    left, right = mv["specs"][0::2], mv["specs"][1::2]                       # each row is (evolution, DQN)
    assert all(sp.startswith(evo_dir + ":") for sp in left) and all(sp.startswith(dqn_dir + ":") for sp in right)
    assert [sp.rsplit(":", 1)[1] for sp in right] == [str(g) for g in watch["multiple_vs"]["dqn"]["generations"]]


def test_single_and_multiple_have_a_section_per_kind_and_no_kind_or_run_of_their_own():
    watch = shipped("watch")
    for section in ("single", "multiple"):
        keys = {k for k in watch[section] if not k.startswith("_")}
        assert {"evolution", "dqn"} <= keys and not ({"training", "run"} & keys)
        assert "run" in watch[section]["evolution"] and "run" in watch[section]["dqn"]


def test_single_and_multiple_need_the_kind_to_be_chosen_by_the_user(tmp_path):
    runs = runs_for_defaults(tmp_path)
    for kind in ("single", "multiple"):
        with pytest.raises(SettingsError, match="needs to know the kind to watch"):
            watch_plan_from_file(SHIPPED, str(runs), kind)
    with pytest.raises(SettingsError, match="needs to know the kind to watch"):
        watch_plan_from_file(SHIPPED, str(runs), "multiple", "perfect")


def test_each_kind_uses_only_its_own_sub_section(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    ghost = {"run": "ghost", "checkpoint": "latest"}
    path = watch_file(tmp_path, {"single": {**watch["single"], "evolution": ghost}})
    assert watch_plan_from_file(path, str(runs), "single", "dqn")["type"] == "single"          # the evolution part is not touched
    with pytest.raises(SettingsError, match="single.evolution.run: run 'ghost' has no saved generations"):
        watch_plan_from_file(path, str(runs), "single", "evolution")


def test_old_style_keys_say_where_they_moved(tmp_path):
    runs = runs_for_defaults(tmp_path)
    old = {"training": "dqn", "run": "dqn", "checkpoint": "best", "individual": 1, "replay": False}
    with pytest.raises(SettingsError) as e:
        watch_plan_from_file(watch_file(tmp_path, {"single": old}), str(runs), "single", "dqn")
    text = str(e.value)
    assert "single.training: not used: you choose neuroevolution or DQN when the script runs" in text
    assert "single.run: moved: put it inside the 'evolution' and 'dqn' sub-sections" in text
    assert "single.checkpoint: moved" in text


def test_a_side_must_be_the_right_kind_of_run(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    dqn_run = watch["single_vs"]["dqn"]["run"]
    path = watch_file(tmp_path, {"single_vs": {"evolution": {"run": dqn_run, "checkpoint": "latest"},
                                               "dqn": {"run": dqn_run, "checkpoint": "best"}}})
    with pytest.raises(SettingsError, match="is a DQN run, but this needs a neuroevolution run"):
        watch_plan_from_file(path, str(runs), "single_vs")
    path = watch_file(tmp_path, {"single": {**watch["single"], "evolution": {"run": dqn_run, "checkpoint": "latest"}}})
    with pytest.raises(SettingsError, match="but this needs a neuroevolution run"):
        watch_plan_from_file(path, str(runs), "single", "evolution")


def multiple_with(tmp_path, mode, generations, run=None):
    watch = shipped("watch")
    return watch_file(tmp_path, {"multiple": {**watch["multiple"], mode: {"run": run or watch["multiple"][mode]["run"],
                                                                          "generations": generations}}})


def test_multiple_accepts_latest_for_evolution_and_best_for_dqn(tmp_path):
    runs = runs_for_defaults(tmp_path)                               # saved: 0, 50, 99, and the DQN run has a best.npz
    evo = watch_plan_from_file(multiple_with(tmp_path, "evolution", [0, 50, "latest"]), str(runs), "multiple", "evolution")
    assert evo["generations"] == [0, 50, 99]                         # "latest" became the highest saved number
    dqn = watch_plan_from_file(multiple_with(tmp_path, "dqn", [0, 99, "best"]), str(runs), "multiple", "dqn")
    assert dqn["generations"] == [0, 99, "best"] and (dqn["rows"], dqn["cols"]) == (1, 3)
    both = watch_plan_from_file(multiple_with(tmp_path, "dqn", [0, "latest", "best"]), str(runs), "multiple", "dqn")
    assert both["generations"] == [0, 99, "best"]


def test_multiple_best_is_dqn_only_and_needs_a_best_checkpoint(tmp_path):
    runs = runs_for_defaults(tmp_path)
    with pytest.raises(SettingsError, match='"best" is only for DQN runs'):
        watch_plan_from_file(multiple_with(tmp_path, "evolution", [0, "best"]), str(runs), "multiple", "evolution")
    name = shipped("watch")["multiple"]["dqn"]["run"]
    (runs / name / "best.npz").unlink()
    with pytest.raises(SettingsError, match=f"run '{name}' has no best checkpoint yet"):
        watch_plan_from_file(multiple_with(tmp_path, "dqn", [0, "best"]), str(runs), "multiple", "dqn")


def test_latest_listed_next_to_its_own_number_is_a_duplicate(tmp_path):
    runs = runs_for_defaults(tmp_path)
    with pytest.raises(SettingsError, match=r'a generation is listed twice \("latest" is generation 99\)'):
        watch_plan_from_file(multiple_with(tmp_path, "evolution", [0, 99, "latest"]), str(runs), "multiple", "evolution")


def test_the_unsaved_generation_message_offers_latest_and_best(tmp_path):
    runs = runs_for_defaults(tmp_path)
    with pytest.raises(SettingsError, match=r"7 was not saved in run '.*'. Saved: 0, 50, 99, latest, best"):
        watch_plan_from_file(multiple_with(tmp_path, "dqn", [0, 7]), str(runs), "multiple", "dqn")


def test_multiple_vs_also_accepts_latest(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    path = watch_file(tmp_path, {"multiple_vs": {"evolution": {"run": watch["multiple_vs"]["evolution"]["run"], "generations": [0, 50, "latest"]},
                                                 "dqn": {"run": watch["multiple_vs"]["dqn"]["run"], "generations": [0, "latest", "best"]},
                                                 "rows": 3, "cols": 2}})
    plan = watch_plan_from_file(path, str(runs), "multiple_vs")
    assert [s.rsplit(":", 1)[1] for s in plan["specs"][0::2]] == ["0", "50", "99"]
    assert [s.rsplit(":", 1)[1] for s in plan["specs"][1::2]] == ["0", "99", "best"]


def test_single_replay_needs_a_null_seed(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    path = watch_file(tmp_path, {"seed": 4, "single": {**watch["single"], "replay": True}})
    with pytest.raises(SettingsError, match="seed must be null"):
        watch_plan_from_file(path, str(runs), "single", "dqn")


def test_multiple_vs_rows_must_pair_up(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    evo, dqn = watch["multiple_vs"]["evolution"]["run"], watch["multiple_vs"]["dqn"]["run"]

    def vs(evo_gens, dqn_gens, rows, cols):
        return watch_file(tmp_path, {"multiple_vs": {"evolution": {"run": evo, "generations": evo_gens},
                                                     "dqn": {"run": dqn, "generations": dqn_gens}, "rows": rows, "cols": cols}})

    with pytest.raises(SettingsError, match=r"must list 2 \(one per row\), got 3"):
        watch_plan_from_file(vs([0, 50], [0, 99, "best"], None, None), str(runs), "multiple_vs")
    with pytest.raises(SettingsError, match=r"must be 3 \(one row per pair"):
        watch_plan_from_file(vs([0, 50, 99], [0, 99, "best"], 2, 2), str(runs), "multiple_vs")
    with pytest.raises(SettingsError, match=r"must be 2 \(left and right\)"):
        watch_plan_from_file(vs([0, 50, 99], [0, 99, "best"], 3, 3), str(runs), "multiple_vs")


def test_best_only_exists_for_dqn_runs_and_unsaved_generations_are_named(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    evo = watch["single"]["evolution"]["run"]
    path = watch_file(tmp_path, {"single": {**watch["single"], "evolution": {"run": evo, "checkpoint": "best"}}})
    with pytest.raises(SettingsError, match="has no best checkpoint"):
        watch_plan_from_file(path, str(runs), "single", "evolution")
    path = watch_file(tmp_path, {"multiple": {**watch["multiple"], "evolution": {"run": evo, "generations": [0, 7]}}})
    with pytest.raises(SettingsError, match=f"7 was not saved in run '{evo}'. Saved: 0, 50, 99"):
        watch_plan_from_file(path, str(runs), "multiple", "evolution")


def test_missing_sections_unknown_kinds_and_bad_values_are_reported(tmp_path):
    runs = runs_for_defaults(tmp_path)
    without = write_watch(tmp_path / "w.json", {k: v for k, v in shipped("watch").items() if k != "multiple"})
    with pytest.raises(SettingsError, match="watch.multiple: this section is missing"):
        watch_plan_from_file(without, str(runs), "multiple", "dqn")
    with pytest.raises(SettingsError, match="unknown type of watching"):
        watch_plan_from_file(SHIPPED, str(runs), "vs")
    bad = watch_file(tmp_path, {"speed": 0, "tpyo": 1})
    with pytest.raises(SettingsError) as e:
        watch_plan_from_file(bad, str(runs), "single_vs")
    assert "watch.speed" in str(e.value) and "watch.tpyo: unknown setting" in str(e.value)


# --- the menu in the two wizards -----------------------------------------------------

def scripted(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def test_train_wizard_asks_the_kind_first_then_offers_the_saved_settings(tmp_path):
    path = write_train(tmp_path / "train.json", shipped("train"))
    lines = []
    plan = train_wizard(str(tmp_path / "runs"), input_fn=scripted(["2", "", ""]), out=lines.append, defaults_path=path)
    assert plan["method"] == "dqn" and plan["count"] == shipped("train")["dqn"]["rounds"]
    assert any("Use the saved DQN settings in train.json" in l for l in lines)
    assert lines.index(next(l for l in lines if "What kind of training?" in l)) < \
        next(i for i, l in enumerate(lines) if "How do you want to set this up?" in l)
    assert any(l.strip().startswith("method") and "DQN" in l for l in lines)           # the same summary as the questions give
    assert train_wizard(str(tmp_path / "runs"), input_fn=scripted(["2", "", "n"]), out=quiet, defaults_path=path) is None


def test_each_kind_uses_its_own_train_section(tmp_path):
    path = write_train(tmp_path / "train.json", shipped("train"))
    data = shipped("train")
    plan = train_wizard(str(tmp_path / "runs"), input_fn=scripted(["1", "", ""]), out=quiet, defaults_path=path)
    assert plan["method"] == "evolution" and plan["run"] == data["evolution"]["run"]
    plan = train_wizard(str(tmp_path / "runs"), input_fn=scripted(["2", "", ""]), out=quiet, defaults_path=path)
    assert plan["method"] == "dqn" and plan["run"] == data["dqn"]["run"]


def test_train_wizard_can_still_ask_the_questions_instead(tmp_path):
    path = write_train(tmp_path / "train.json", shipped("train"))
    plan = train_wizard(str(tmp_path / "runs"), input_fn=scripted(["1", "1"] + [""] * 11), out=quiet, defaults_path=path)
    assert plan["method"] == "evolution" and plan["run"] == "default"     # the questions' own default run name


def test_a_broken_train_file_is_explained_and_the_questions_follow(tmp_path):
    data = shipped("train")
    data["dqn"]["rounds"] = 0
    path = write_train(tmp_path / "train.json", data)
    lines = []
    plan = train_wizard(str(tmp_path / "runs"), input_fn=scripted(["2", ""] + [""] * 8), out=lines.append, defaults_path=path)
    text = "\n".join(lines)
    assert "dqn.rounds" in text and "Fix the file, or answer the questions now." in text
    assert plan["method"] == "dqn" and plan["count"] == 100                       # answered with the questions' defaults


def test_a_missing_settings_file_falls_back_to_the_questions(tmp_path):
    lines = []
    plan = train_wizard(str(tmp_path / "runs"), input_fn=scripted(["1", ""] + [""] * 11), out=lines.append,
                        defaults_path=str(tmp_path / "gone.json"))
    assert any("was not found" in l for l in lines) and plan["method"] == "evolution"


def test_start_over_from_the_file_warns_before_deleting(tmp_path):
    runs = tmp_path / "runs"
    data = shipped("train")
    name = data["evolution"]["run"]
    evolution_train(Config(pop_size=6, elite=1, n_parents=3, games_per_eval=1, max_steps=100), 2, str(runs / name), log=quiet)
    data["evolution"]["existing_run"] = "start_over"
    lines = []
    plan = train_wizard(str(runs), input_fn=scripted(["1", "", ""]), out=lines.append, defaults_path=write_train(tmp_path / "t.json", data))
    assert plan["reset"] is True and any(f"DELETE runs/{name} (2 generations trained)" in l for l in lines)


def test_watch_wizard_asks_the_type_then_the_kind_then_the_saved_settings(tmp_path):
    runs = runs_for_defaults(tmp_path)
    lines = []
    # type single, kind DQN, saved settings (the default), start
    plan = watch_wizard(str(runs), input_fn=scripted(["1", "2", "", ""]), out=lines.append, defaults_path=SHIPPED)
    assert plan["type"] == "single" and plan["brain"].startswith(str(runs / shipped("watch")["single"]["dqn"]["run"]))
    asked = [next(i for i, l in enumerate(lines) if text in l) for text in
             ("What do you want to watch?", "Which kind of player?", "How do you want to set this up?")]
    assert asked == sorted(asked)
    assert any("Use the saved single DQN settings in default.json" in l for l in lines)


def test_watch_wizard_uses_the_chosen_kinds_sub_section(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    evo = watch_wizard(str(runs), input_fn=scripted(["3", "1", "", ""]), out=quiet, defaults_path=SHIPPED)
    dqn = watch_wizard(str(runs), input_fn=scripted(["3", "2", "", ""]), out=quiet, defaults_path=SHIPPED)
    assert evo["run_dir"] == str(runs / watch["multiple"]["evolution"]["run"])
    assert dqn["run_dir"] == str(runs / watch["multiple"]["dqn"]["run"])
    assert evo["generations"] == watch["multiple"]["evolution"]["generations"]


def test_the_vs_types_ask_no_kind_and_use_their_saved_settings(tmp_path):
    runs = runs_for_defaults(tmp_path)
    lines = []
    plan = watch_wizard(str(runs), input_fn=scripted(["4", "", ""]), out=lines.append, defaults_path=SHIPPED)
    assert plan["type"] == "multiple_vs" and len(plan["specs"]) == 6
    assert not any("Which kind of" in l for l in lines)
    assert any("Use the saved multiple VS settings in default.json" in l for l in lines)


def test_the_perfect_bot_has_no_saved_settings_to_choose(tmp_path):
    lines = []
    plan = watch_wizard(str(tmp_path / "runs"), input_fn=scripted(["1", "3", "", "2", ""]), out=lines.append, defaults_path=SHIPPED)
    assert plan["brain"] == "perfect" and plan["speed"] == 2.0
    assert not any("How do you want to set this up?" in l for l in lines)


def test_watch_saved_settings_that_fail_fall_back_to_questions(tmp_path):
    runs = runs_for_defaults(tmp_path)
    watch = shipped("watch")
    ghost = watch_file(tmp_path, {"single": {**watch["single"], "dqn": {"run": "ghost", "checkpoint": "best"}}})
    lines = []
    # type single, kind DQN, saved settings (fail) -> questions: run, checkpoint, replay, seed, speed, start
    plan = watch_wizard(str(runs), input_fn=scripted(["1", "2", "", "", "", "", "", "", ""]), out=lines.append, defaults_path=ghost)
    assert any("ghost" in l and "Fix the file" in l for l in lines)
    assert plan["type"] == "single" and plan["brain"].endswith("best.npz")


# --- the shape of the combined file ---------------------------------------------------

def test_one_file_holds_both_parts_and_each_loader_reads_only_its_own(tmp_path):
    runs = runs_for_defaults(tmp_path)
    path = write(tmp_path / "both.json", {"train": shipped("train"), "watch": shipped("watch")})
    assert train_plan_from_file(path, str(runs), "dqn")[0]["method"] == "dqn"
    assert watch_plan_from_file(path, str(runs), "single_vs")["type"] == "single_vs"

    only_train = write(tmp_path / "only_train.json", {"train": shipped("train")})
    assert train_plan_from_file(only_train, str(runs), "dqn")[0]["method"] == "dqn"
    with pytest.raises(SettingsError, match='has no "watch" section'):
        watch_plan_from_file(only_train, str(runs), "single_vs")
    with pytest.raises(SettingsError, match='has no "train" section'):
        train_plan_from_file(write(tmp_path / "only_watch.json", {"watch": shipped("watch")}), str(runs), "dqn")


def test_messages_name_the_part_and_an_unknown_section_is_flagged(tmp_path):
    data = {"train": {**shipped("train"), "extra": 1}, "watch": shipped("watch"), "tain": {}}
    with pytest.raises(SettingsError) as e:
        train_plan_from_file(write(tmp_path / "a.json", data), str(tmp_path), "dqn")
    assert "train.extra: unknown setting" in str(e.value) and "tain: unknown section" in str(e.value)
