"""A tiny feed-forward neural network in plain numpy.

Same forward pass as the MNIST notebook (A = activation(W @ A_prev + B)), except:
  * hidden layers use tanh instead of sigmoid
  * the output layer is left raw (no squashing): we only take the argmax
  * there is no backprop. All the weights and biases live in ONE flat list of
    numbers, the "genome", which evolution mutates directly.
"""
import numpy as np

from game_engine import OBS_SIZE

N_ACTIONS = 3                      # 0 = nothing, 1 = jump, 2 = duck
DEFAULT_HIDDEN = 12


def layer_sizes(hidden=DEFAULT_HIDDEN):
    hidden = (hidden,) if isinstance(hidden, int) else tuple(hidden)
    return (OBS_SIZE, *hidden, N_ACTIONS)


def genome_size(sizes) -> int:
    return sum(out * inp + out for inp, out in zip(sizes[:-1], sizes[1:]))


def random_genome(rng: np.random.Generator, sizes) -> np.ndarray:
    """Weights ~ N(0,1)/sqrt(fan_in) (same scaling as the notebook), biases 0."""
    parts = []
    for inp, out in zip(sizes[:-1], sizes[1:]):
        parts.append((rng.standard_normal((out, inp)) / np.sqrt(inp)).ravel())
        parts.append(np.zeros(out))
    return np.concatenate(parts)


class Network:
    def __init__(self, genome, sizes):
        genome = np.asarray(genome, dtype=np.float64)
        if genome.shape != (genome_size(sizes),):
            raise ValueError(f"genome has {genome.size} numbers, layout {tuple(sizes)} needs {genome_size(sizes)}")
        self.sizes = tuple(sizes)
        self.genome = genome
        self.layers = []                 # [(W, b), ...] views into the genome
        pos = 0
        for inp, out in zip(sizes[:-1], sizes[1:]):
            W = genome[pos:pos + out * inp].reshape(out, inp)
            pos += out * inp
            b = genome[pos:pos + out]
            pos += out
            self.layers.append((W, b))

    def forward(self, x) -> np.ndarray:
        """observation -> one raw score per action."""
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < last:
                x = np.tanh(x)
        return x
