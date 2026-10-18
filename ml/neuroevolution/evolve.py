"""Round 1: neuroevolution.

One generation =
  1. play   every genome plays a few games (same seeds for everyone, so it's fair)
  2. rank   fitness = average score
  3. breed  keep the elite unchanged, make the rest from the top parents by
            crossover (mix two parents) + mutation (small random nudges)
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from game_engine import OBS_SIZE
from ml.brains import NetBrain, PerfectBrain, save_generation
from ml.network import DEFAULT_HIDDEN, N_ACTIONS, genome_size, layer_sizes, random_genome
from ml.rollout import DEFAULT_MAX_STEPS, mean_score


@dataclass
class Config:
    pop_size: int = 50
    elite: int = 5                 # best genomes copied unchanged into the next generation
    n_parents: int = 10            # the top N are allowed to have children
    crossover_prob: float = 0.7    # else the child is a plain copy of one parent
    mutation_rate: float = 0.15    # chance each of the numbers gets nudged
    mutation_sigma: float = 0.3    # size of a nudge
    games_per_eval: int = 3
    max_steps: int = DEFAULT_MAX_STEPS
    hidden: int = DEFAULT_HIDDEN

    def __post_init__(self):
        if not 0 <= self.elite <= self.pop_size:
            raise ValueError(f"elite must be between 0 and the population size ({self.pop_size})")
        # The elites are the top of the parent pool, so the pool can't be smaller than them.
        if self.n_parents < self.elite:
            raise ValueError(f"parents ({self.n_parents}) must be at least elites ({self.elite}): "
                             f"the elites are part of the parent pool. Raise the parents or lower the elites")
        if not 1 <= self.n_parents <= self.pop_size:
            raise ValueError(f"parents must be between 1 and the population size ({self.pop_size})")


@dataclass
class RunInfo:
    pop_size: int
    hidden: Optional[int]    # None if the saved layout is not 10 -> hidden -> 3
    generation: int          # the next generation this run will play
    seed: int


def run_info(run_dir: str) -> Optional[RunInfo]:
    """What an existing run was started with, or None if run_dir has no saved state."""
    path = os.path.join(run_dir, "state.npz")
    if not os.path.exists(path):
        return None
    state = np.load(path)
    if "population" not in state.files:      # a DQN run's state, not an evolution one
        return None
    pop = state["population"]
    # one hidden layer: genome = (inputs + 1) * h + (h + 1) * outputs, so solve for h
    hidden, leftover = divmod(pop.shape[1] - N_ACTIONS, OBS_SIZE + 1 + N_ACTIONS)
    return RunInfo(pop.shape[0], hidden if leftover == 0 and hidden >= 1 else None,
                   int(state["generation"]), int(state["seed"]))


def seeds_for(run_seed: int, generation: int, n: int):
    """New game seeds each generation (so bots can't memorize one level)."""
    base = run_seed * 1_000_000 + generation * 100
    return [base + k for k in range(n)]


def evaluate_population(population, sizes, seeds, max_steps) -> np.ndarray:
    return np.array([mean_score(NetBrain(g, sizes), seeds, max_steps) for g in population])


def crossover(a: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    """Each number comes from parent a or parent b with equal chance."""
    return np.where(rng.random(a.shape) < 0.5, a, b)


def mutate(genome: np.ndarray, cfg: Config, rng) -> np.ndarray:
    hit = rng.random(genome.shape) < cfg.mutation_rate
    return genome + hit * rng.normal(0.0, cfg.mutation_sigma, genome.shape)


def next_generation(population, fitness, cfg: Config, rng):
    order = np.argsort(-fitness)                       # best first
    new = [population[i].copy() for i in order[:cfg.elite]]
    parents = [population[i] for i in order[:cfg.n_parents]]
    while len(new) < cfg.pop_size:
        i, j = rng.choice(len(parents), size=2, replace=len(parents) < 2)
        child = crossover(parents[i], parents[j], rng) if rng.random() < cfg.crossover_prob else parents[i].copy()
        new.append(mutate(child, cfg, rng))
    return new


def train(cfg: Config, generations: int, run_dir: str, seed=None, save_every: int = 5,
          verbose: bool = False, log=print):
    """Run `generations` more generations, resuming from run_dir/state.npz if it exists.

    The population is saved after every call, so you can train 1 generation,
    look at it, then train 10 more, and it picks up exactly where it stopped.
    """
    os.makedirs(run_dir, exist_ok=True)
    sizes = layer_sizes(cfg.hidden)
    state_path = os.path.join(run_dir, "state.npz")
    history_path = os.path.join(run_dir, "history.csv")

    if os.path.exists(state_path):
        state = np.load(state_path)
        if "population" not in state.files:
            raise ValueError(f"{run_dir} was trained with DQN, not neuroevolution; use another run name (or delete it)")
        if seed is not None and seed != int(state["seed"]):
            raise ValueError(f"{run_dir} was started with seed {int(state['seed'])}; delete it or use a new run name to change it")
        seed, start = int(state["seed"]), int(state["generation"])
        population = list(state["population"])
        if len(population) != cfg.pop_size or len(population[0]) != genome_size(sizes):
            raise ValueError(f"{run_dir} was started with a different population size or hidden neurons; delete it or use a new run name")
        log(f"resuming {run_dir} at generation {start}")
    else:
        seed, start = (0 if seed is None else seed), 0
        rng = np.random.default_rng([seed, 0])
        population = [random_genome(rng, sizes) for _ in range(cfg.pop_size)]
        log(f"new run in {run_dir}")

    if not os.path.exists(history_path):
        with open(history_path, "w") as f:
            f.write("generation,best,mean\n")

    last = start + generations - 1
    baseline = mean_score(PerfectBrain(), seeds_for(seed, start, cfg.games_per_eval), cfg.max_steps)
    log(f"layout {sizes}, {len(population[0])} numbers per genome | perfect bot scores {baseline:.1f} on these games")

    for gen in range(start, last + 1):
        seeds = seeds_for(seed, gen, cfg.games_per_eval)
        fitness = evaluate_population(population, sizes, seeds, cfg.max_steps)
        order = np.argsort(-fitness)
        best, mean = float(fitness[order[0]]), float(fitness.mean())

        with open(history_path, "a") as f:
            f.write(f"{gen},{best:.3f},{mean:.3f}\n")
        log(f"gen {gen:4d} | best {best:7.1f} | mean {mean:7.1f}")
        if verbose:
            for rank, i in enumerate(order):
                role = "elite+parent" if rank < cfg.elite else "parent      " if rank < cfg.n_parents else "            "
                log(f"      #{rank + 1:<3d} {role}  fitness {fitness[i]:7.1f}")

        if gen % save_every == 0 or gen == last:
            save_generation(os.path.join(run_dir, f"gen_{gen:04d}.npz"), [population[i] for i in order],
                            fitness[order], sizes, gen, seeds)

        population = next_generation(population, fitness, cfg, np.random.default_rng([seed, gen + 1]))

    np.savez(state_path, population=np.array(population), generation=last + 1, seed=seed)
