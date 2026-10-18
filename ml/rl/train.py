"""Round 2: reinforcement learning (DQN).

One network learns *while it plays*. On every step of every game:

  1. look at the situation s, pick an action (the highest Q, or a random one with probability epsilon)
  2. the game answers with a reward r and a new situation s2
  3. store the transition (s, a, r, s2, done) in the replay buffer
  4. every few steps, train on a random batch from the buffer:
        target     = r + gamma * (best Q of s2, from a frozen copy of the network)
        prediction = Q(s, a)
        backprop on (prediction - target)

A "round" is a handful of training games followed by a test of the greedy player (epsilon = 0)
on fixed games, so the learning curve is comparable to neuroevolution's.
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from game_engine import OBS_SIZE, Game
from ml.brains import NetBrain, PerfectBrain, save_generation
from ml.network import layer_sizes
from ml.neuroevolution.evolve import seeds_for
from ml.rl.agent import DQNAgent
from ml.rl.replay import ReplayBuffer
from ml.rollout import DEFAULT_MAX_STEPS, mean_score

TRAIN_SEED_OFFSET = 500_000_000   # training games use seeds far from the evaluation games' seeds
CONFIRM_SEED_OFFSET = 900_000_000
INIT_STREAM = 0                   # random-number stream for the starting weights


@dataclass
class DQNConfig:
    hidden: int = 128
    gamma: float = 0.99             # how much a reward one step later counts, compared with one now
    lr: float = 1e-3                # learning rate for Adam
    batch_size: int = 64            # transitions per training step
    buffer_size: int = 100_000      # how many recent transitions are remembered
    train_every: int = 4            # one training step per this many game steps
    target_sync: int = 4000         # game steps between copying the network into the frozen target network
    warmup: int = 1000              # no training until the buffer holds this many transitions
    eps_start: float = 1.0          # exploration: at first every action is random...
    eps_end: float = 0.05           # ...and falls to this
    eps_decay_steps: int = 50_000   # ...over this many game steps
    episodes_per_round: int = 10    # training games per round
    eval_games: int = 3             # test games per round (greedy player, fixed seeds)
    confirm_games: int = 10         # a round that looks like a new best is re-tested on this many fresh games
    max_steps: int = DEFAULT_MAX_STEPS
    huber_delta: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be at least 0 and below 1")
        if self.lr <= 0:
            raise ValueError("the learning rate must be above 0")
        for name in ("hidden", "batch_size", "buffer_size", "train_every", "target_sync", "warmup",
                     "eps_decay_steps", "episodes_per_round", "eval_games", "confirm_games", "max_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.warmup < self.batch_size:
            raise ValueError(f"warmup ({self.warmup}) must be at least the batch size ({self.batch_size})")
        if self.buffer_size < self.warmup:
            raise ValueError(f"the buffer ({self.buffer_size}) must hold at least the warmup transitions ({self.warmup})")
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError("exploration must satisfy 0 <= final <= initial <= 1")


@dataclass
class RunInfo:
    hidden: Optional[int]
    buffer_size: int
    round: int               # the next round this run will play
    seed: int


def epsilon(cfg: DQNConfig, steps: int) -> float:
    """Chance of a random action after `steps` game steps: a straight line from eps_start down to eps_end."""
    frac = min(1.0, steps / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train_seed(run_seed: int, rnd: int, episode: int) -> int:
    return TRAIN_SEED_OFFSET + run_seed * 1_000_000 + rnd * 1000 + episode


def confirm_seeds(run_seed: int, rnd: int, n: int):
    base = CONFIRM_SEED_OFFSET + run_seed * 1_000_000 + rnd * 100
    return [base + k for k in range(n)]


def run_info(run_dir: str) -> Optional[RunInfo]:
    """What an existing DQN run was started with, or None if run_dir holds no DQN state."""
    path = os.path.join(run_dir, "state.npz")
    if not os.path.exists(path):
        return None
    state = np.load(path)
    if "theta" not in state.files:
        return None
    sizes = tuple(int(x) for x in state["sizes"])
    return RunInfo(sizes[1] if len(sizes) == 3 else None, int(state["buf_capacity"]),
                   int(state["round"]), int(state["seed"]))


def train(cfg: DQNConfig, rounds: int, run_dir: str, seed=None, save_every: int = 5, log=print):
    """Run `rounds` more rounds, resuming from run_dir/state.npz if it exists."""
    os.makedirs(run_dir, exist_ok=True)
    sizes = layer_sizes(cfg.hidden)
    state_path = os.path.join(run_dir, "state.npz")
    history_path = os.path.join(run_dir, "history.csv")

    if os.path.exists(state_path):
        state = np.load(state_path)
        if "theta" not in state.files:
            raise ValueError(f"{run_dir} was trained with neuroevolution, not DQN; use another run name (or delete it)")
        if seed is not None and seed != int(state["seed"]):
            raise ValueError(f"{run_dir} was started with seed {int(state['seed'])}; delete it or use a new run name to change it")
        if tuple(int(x) for x in state["sizes"]) != sizes:
            raise ValueError(f"{run_dir} was started with a different number of hidden neurons; delete it or use a new run name")
        if int(state["buf_capacity"]) != cfg.buffer_size:
            raise ValueError(f"{run_dir} was started with a memory of {int(state['buf_capacity']):,} transitions; delete it or use a new run name")
        seed, start, total_steps = int(state["seed"]), int(state["round"]), int(state["total_steps"])
        best_score = float(state["best_score"])
        agent = DQNAgent(sizes, theta=state["theta"])
        agent.target_theta[:] = state["target_theta"]
        agent.adam.m, agent.adam.v, agent.adam.t = state["adam_m"].copy(), state["adam_v"].copy(), int(state["adam_t"])
        buffer = ReplayBuffer.from_arrays(state, OBS_SIZE)
        log(f"resuming {run_dir} at round {start} ({total_steps:,} steps so far)")
    else:
        seed, start, total_steps, best_score = (0 if seed is None else seed), 0, 0, float("-inf")
        agent = DQNAgent(sizes, rng=np.random.default_rng([seed, INIT_STREAM]))
        buffer = ReplayBuffer(cfg.buffer_size, OBS_SIZE)
        log(f"new run in {run_dir}")

    if not os.path.exists(history_path):
        with open(history_path, "w") as f:
            f.write("round,eval_score,train_score,epsilon,loss,steps,round_steps\n")

    last = start + rounds - 1
    baseline = mean_score(PerfectBrain(), seeds_for(seed, start, cfg.eval_games), cfg.max_steps)
    log(f"layout {sizes}, {agent.theta.size} numbers | perfect bot scores {baseline:.1f} on the test games")

    for rnd in range(start, last + 1):
        rng = np.random.default_rng([seed, rnd + 1])      # this round's randomness: makes resuming exact
        train_scores, losses = [], []
        steps_at_start = total_steps

        for episode in range(cfg.episodes_per_round):
            game = Game(train_seed(seed, rnd, episode))
            obs = game.get_observation()
            for step in range(cfg.max_steps):
                action = agent.act(obs, epsilon(cfg, total_steps), rng)
                obs2, reward, done, _ = game.step(action)
                buffer.add(obs, action, reward, obs2, done)   # `done` only if it crashed, not if time ran out
                obs = obs2
                total_steps += 1

                if buffer.size >= cfg.warmup and total_steps % cfg.train_every == 0:
                    losses.append(agent.update(buffer.sample(rng, cfg.batch_size), cfg.gamma, cfg.lr, cfg.huber_delta))
                if total_steps % cfg.target_sync == 0:
                    agent.sync_target()
                if done:
                    break
            train_scores.append(game.score)

        eval_seeds = seeds_for(seed, rnd, cfg.eval_games)
        eval_score = mean_score(NetBrain(agent.theta, sizes), eval_seeds, cfg.max_steps)   # greedy: epsilon = 0
        train_score = float(np.mean(train_scores))
        loss = float(np.mean(losses)) if losses else float("nan")
        eps = epsilon(cfg, total_steps)
        round_steps = total_steps - steps_at_start

        with open(history_path, "a") as f:
            f.write(f"{rnd},{eval_score:.3f},{train_score:.3f},{eps:.3f},{loss:.4f},{total_steps},{round_steps}\n")
        shown_loss = f"{loss:7.3f}" if losses else "      -"
        log(f"round {rnd:4d} | eval {eval_score:7.1f} | train {train_score:7.1f} | eps {eps:.2f} | loss {shown_loss}"
            f" | steps this round {round_steps:,} | total {total_steps:,}")

        if rnd % save_every == 0 or rnd == last:
            save_generation(os.path.join(run_dir, f"gen_{rnd:04d}.npz"), [agent.theta.copy()], [eval_score],
                            sizes, rnd, eval_seeds, kind="dqn")

        # DQN swings from round to round, so the last round is not necessarily the best. Keep the best one.
        # A round that scores at least as well as the best so far is re-tested on more fresh games first,
        # so a lucky 3-game score can't fool us.
        if eval_score >= best_score:
            seeds = confirm_seeds(seed, rnd, cfg.confirm_games)
            confirmed = mean_score(NetBrain(agent.theta, sizes), seeds, cfg.max_steps)
            if confirmed > best_score:
                best_score = confirmed
                save_generation(os.path.join(run_dir, "best.npz"), [agent.theta.copy()], [confirmed], sizes, rnd, seeds, kind="dqn")
                log(f"      new best: round {rnd}, {confirmed:.1f} over {cfg.confirm_games} fresh games -> best.npz")

    np.savez(state_path, theta=agent.theta, target_theta=agent.target_theta, adam_m=agent.adam.m,
             adam_v=agent.adam.v, adam_t=agent.adam.t, total_steps=total_steps, round=last + 1, seed=seed,
             best_score=best_score,
             sizes=np.array(sizes), **buffer.to_arrays())
