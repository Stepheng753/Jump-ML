"""The replay buffer: a fixed-size memory of recent transitions.

A transition is one step written down: (s, a, r, s2, done)
  s     the observation before the action
  a     the action taken (0 nothing, 1 jump, 2 duck)
  r     the reward the game gave for that step
  s2    the observation after the action
  done  True only if that step ended the game in a crash

When the buffer is full the oldest transition is overwritten.
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int):
        self.capacity = capacity
        self.S = np.zeros((capacity, obs_size), np.float32)
        self.A = np.zeros(capacity, np.int64)
        self.R = np.zeros(capacity, np.float32)
        self.S2 = np.zeros((capacity, obs_size), np.float32)
        self.D = np.zeros(capacity, np.float32)
        self.ptr = 0      # where the next transition is written
        self.size = 0     # how many are stored (at most capacity)

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, batch_size: int):
        """A random batch of stored transitions (the same one can appear twice)."""
        idx = rng.integers(0, self.size, batch_size)
        return (self.S[idx].astype(np.float64), self.A[idx], self.R[idx].astype(np.float64),
                self.S2[idx].astype(np.float64), self.D[idx].astype(np.float64))

    # saving, so a run can resume with its memory intact
    def to_arrays(self) -> dict:
        n = self.size
        return {"buf_S": self.S[:n], "buf_A": self.A[:n], "buf_R": self.R[:n], "buf_S2": self.S2[:n],
                "buf_D": self.D[:n], "buf_ptr": self.ptr, "buf_capacity": self.capacity}

    @classmethod
    def from_arrays(cls, data, obs_size: int) -> "ReplayBuffer":
        buf = cls(int(data["buf_capacity"]), obs_size)
        n = len(data["buf_A"])
        buf.S[:n], buf.A[:n], buf.R[:n], buf.S2[:n], buf.D[:n] = (
            data["buf_S"], data["buf_A"], data["buf_R"], data["buf_S2"], data["buf_D"])
        buf.ptr, buf.size = int(data["buf_ptr"]), n
        return buf
