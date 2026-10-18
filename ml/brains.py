"""A 'brain' is anything that turns an observation into an action:

    brain.act(observation) -> 0 (nothing) | 1 (jump) | 2 (duck)

The perfect bot and a trained net use exactly the same interface and the same
10 observation numbers, so the trainer and the watcher treat them identically.
"""
import glob
import os

import numpy as np

from game_engine import Action
from game_engine import config as C

from .network import Network


class PerfectBrain:
    """Hand-written rule, no learning. Sees only the 10 observation numbers.

    It rebuilds 'seconds until the nearest obstacle reaches me' from distance
    and speed, then centers the jump/duck on that moment.
    """
    name = "perfect bot"
    scores = None   # a net has per-action scores; this one doesn't

    def reset(self):
        pass

    def act(self, obs) -> int:
        dist, speed_n, is_air = obs[4], obs[5], obs[6]
        speed = C.SPEED_MIN + speed_n * (C.SPEED_MAX - C.SPEED_MIN)
        time_to_arrival = dist * C.TRAVEL_DIST / speed
        if is_air < 0.5 and time_to_arrival <= C.JUMP_TIME / 2:
            return Action.JUMP
        if is_air >= 0.5 and time_to_arrival <= C.DUCK_TIME / 2:
            return Action.DUCK
        return Action.NONE


class NetBrain:
    """A neural network picks the action with the highest score.

    generation / rank / fitness / seeds are filled in when the brain was loaded
    from a saved generation (the watcher uses them for its label). `unit` is "gen" for
    neuroevolution and "round" for DQN, and n_individuals says how many were saved alongside.
    """

    def __init__(self, genome, sizes, name="net", generation=None, rank=None, fitness=None, seeds=None,
                 unit="gen", n_individuals=None):
        self.net = Network(genome, sizes)
        self.name = name
        self.generation, self.rank, self.fitness, self.seeds = generation, rank, fitness, seeds
        self.unit, self.n_individuals = unit, n_individuals
        self.tag = None          # short cell label set by VS mode; None uses the normal label
        self.scores = None

    def label(self, fresh):
        """Window label. `fresh` means it is playing new games, so its saved score is from other games."""
        if self.unit == "round":
            return f"round {self.generation} | eval {self.fitness:.1f}"
        word = "train fit" if fresh else "fitness"
        return f"gen {self.generation} | individual #{self.rank} | {word} {self.fitness:.1f}"

    def short_label(self):
        if self.unit == "round":
            return f"round {self.generation} | eval {self.fitness:.1f}"
        return f"gen {self.generation} | #{self.rank} | fit {self.fitness:.1f}"

    def reset(self):
        self.scores = None

    def act(self, obs) -> int:
        self.scores = self.net.forward(obs)
        return int(np.argmax(self.scores))


def save_generation(path, ranked_population, ranked_fitness, sizes, generation, seeds, kind="evolution"):
    """Save every individual of one generation, best first, with the games they played.

    kind is "evolution" or "dqn" (a DQN checkpoint is one individual, and `generation` is its round).
    """
    np.savez(path, population=np.asarray(ranked_population), fitness=np.asarray(ranked_fitness),
             sizes=np.array(sizes), generation=generation, seeds=np.array(seeds), kind=kind)


def load_brain(spec: str, rank: int = 1):
    """'perfect', a saved .npz file, or a run folder (-> its latest saved generation).

    rank 1 is that generation's best individual, rank 2 the second best, and so on.
    """
    if spec == "perfect":
        return PerfectBrain()
    path = spec
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "gen_*.npz")))
        if not files:
            raise FileNotFoundError(f"no gen_*.npz files in {path}")
        path = files[-1]
    return _load_file(path, rank)


def load_spec(spec: str, rank: int = 1):
    """One player from a VS-mode argument: 'perfect', a .npz file, or RUN:GEN (GEN a number, or 'best')."""
    if spec == "perfect":
        brain = PerfectBrain()
        brain.tag = "perfect bot"
        return brain
    if spec.endswith(".npz"):
        path = spec
    elif ":" in spec:
        run_dir, gen_text = spec.rsplit(":", 1)
        if gen_text == "best":
            path = os.path.join(run_dir, "best.npz")
        elif gen_text.isdigit():
            path = os.path.join(run_dir, f"gen_{int(gen_text):04d}.npz")
        else:
            raise ValueError(f"'{spec}': the part after the colon must be a generation or round number, or best")
    else:
        raise ValueError(f"'{spec}': use perfect, a .npz file, or RUN:GEN (like runs/default:99 or runs/dqn:best)")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    brain = _load_file(path, rank)
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if os.path.basename(path) == "best.npz":
        brain.tag = f"dqn {run} best r{brain.generation} eval {brain.fitness:.1f}"
    elif brain.unit == "round":
        brain.tag = f"dqn {run} r{brain.generation} eval {brain.fitness:.1f}"
    else:
        brain.tag = f"evo {run} g{brain.generation} #{brain.rank} fit {brain.fitness:.1f}"
    return brain


def saved_scores(run_dir: str):
    """{generation: score} for every saved generation: the best individual's fitness (evolution)
    or the test score (DQN)."""
    return {g: float(np.max(np.load(os.path.join(run_dir, f"gen_{g:04d}.npz"))["fitness"]))
            for g in saved_generations(run_dir)}


def best_checkpoint(run_dir: str):
    """(round, confirmed score) of a DQN run's best.npz, or None if there is none."""
    path = os.path.join(run_dir, "best.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return int(data["generation"]), float(data["fitness"][0])


def run_unit(run_dir: str) -> str:
    """'round' for a DQN run, 'gen' otherwise (older files have no kind and are evolution)."""
    files = sorted(glob.glob(os.path.join(run_dir, "gen_*.npz")))
    if files:
        data = np.load(files[-1])
        if "kind" in data.files and str(data["kind"]) == "dqn":
            return "round"
    return "gen"


def saved_generations(run_dir: str):
    """Generation numbers that have a gen_XXXX.npz file in run_dir, ascending."""
    return sorted(int(os.path.basename(f)[4:-4]) for f in glob.glob(os.path.join(run_dir, "gen_*.npz")))


def load_generation(run_dir: str, generation: int, rank: int = 1):
    """One specific saved generation of a run (see saved_generations for what exists)."""
    path = os.path.join(run_dir, f"gen_{generation:04d}.npz")
    if not os.path.exists(path):
        have = ", ".join(map(str, saved_generations(run_dir))) or "none"
        raise FileNotFoundError(f"generation {generation} was not saved in {run_dir}. Saved generations: {have}")
    return _load_file(path, rank)


def _load_file(path: str, rank: int):
    data = np.load(path)
    if "population" in data.files:
        population, fitness, seeds = data["population"], data["fitness"], [int(x) for x in data["seeds"]]
    else:   # older file that holds only the champion: no other individuals, no saved seeds
        population, fitness, seeds = data["genome"][None, :], np.atleast_1d(data["fitness"]), None
    if not 1 <= rank <= len(population):
        raise ValueError(f"rank must be between 1 and {len(population)}")
    gen, fit = int(data["generation"]), float(fitness[rank - 1])
    unit = "round" if "kind" in data.files and str(data["kind"]) == "dqn" else "gen"
    name = (f"round {gen} (eval {fit:.1f})" if unit == "round"
            else f"gen {gen} individual #{rank} (fitness {fit:.1f})")
    return NetBrain(population[rank - 1], tuple(int(x) for x in data["sizes"]), name=name,
                    generation=gen, rank=rank, fitness=fit, seeds=seeds, unit=unit, n_individuals=len(population))
