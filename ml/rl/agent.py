"""The DQN learner: a network that predicts future points, plus the math to train it.

Same forward pass as the neuroevolution brain (10 -> hidden (tanh) -> 3, raw outputs), but
here the 3 outputs are Q-values: predicted points still to come from this moment on, for
each action. Training uses backprop, written out by hand in numpy like the MNIST notebook.

The weights and biases live in one flat array `theta` (the same layout as a genome), so a
trained agent saves and loads like any other brain.
"""
import numpy as np

from ml.network import Network, random_genome


def unpack(theta, sizes):
    """Views of (W1, b1, W2, b2) inside the flat parameter array."""
    n_in, n_h, n_out = sizes
    o = 0
    W1 = theta[o:o + n_h * n_in].reshape(n_h, n_in); o += n_h * n_in
    b1 = theta[o:o + n_h]; o += n_h
    W2 = theta[o:o + n_out * n_h].reshape(n_out, n_h); o += n_out * n_h
    b2 = theta[o:o + n_out]
    return W1, b1, W2, b2


def q_batch(theta, sizes, S):
    """Q-values for a batch of observations: shape (batch, 3). Also returns the hidden values."""
    W1, b1, W2, b2 = unpack(theta, sizes)
    H = np.tanh(S @ W1.T + b1)
    return H @ W2.T + b2, H


def td_loss_and_grad(theta, sizes, S, A, y, delta=1.0):
    """Loss and its gradient (backprop) for a batch of transitions.

    S: observations, A: the action taken in each, y: the target for that action's Q-value.
    Only the output of the action actually taken gets an error; the other two outputs get none.
    The loss is Huber: 0.5*err^2 for small errors (like the notebook's squared error), but
    growing only linearly past `delta`, so one huge error cannot wreck the weights.
    """
    W1, b1, W2, b2 = unpack(theta, sizes)
    Q, H = q_batch(theta, sizes, S)
    batch = len(S)
    rows = np.arange(batch)
    err = Q[rows, A] - y                                    # prediction - target, one per transition
    small = np.abs(err) <= delta
    loss = float(np.mean(np.where(small, 0.5 * err ** 2, delta * (np.abs(err) - 0.5 * delta))))

    dQ = np.zeros_like(Q)
    dQ[rows, A] = np.clip(err, -delta, delta) / batch       # d loss / d output (taken action only)
    gW2 = dQ.T @ H
    gb2 = dQ.sum(axis=0)
    dZ1 = (dQ @ W2) * (1.0 - H ** 2)                        # back through tanh
    gW1 = dZ1.T @ S
    gb1 = dZ1.sum(axis=0)
    return loss, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])


class Adam:
    """Gradient descent with momentum and per-number step sizes (steadier than plain steps)."""

    def __init__(self, size, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(size), np.zeros(size), 0
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def step(self, theta, grad, lr):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + self.eps)   # in place, so views of theta stay valid


class DQNAgent:
    """Online network (acts and learns) and a frozen target network (used only to build targets)."""

    def __init__(self, sizes, theta=None, rng=None):
        if len(sizes) != 3:
            raise ValueError("DQN here uses exactly one hidden layer: (inputs, hidden, outputs)")
        self.sizes = tuple(sizes)
        self.theta = np.array(theta if theta is not None else random_genome(rng, sizes), dtype=np.float64)
        self.target_theta = self.theta.copy()
        self.net = Network(self.theta, self.sizes)          # views into theta, so it always sees updates
        self.adam = Adam(self.theta.size)

    def act(self, obs, epsilon, rng) -> int:
        """Epsilon-greedy: with probability epsilon try a random action, otherwise the highest Q."""
        explore = rng.random() < epsilon
        random_action = int(rng.integers(self.sizes[2]))    # always drawn, so the random stream is steady
        return random_action if explore else int(np.argmax(self.net.forward(obs)))

    def targets(self, R, S2, D, gamma):
        """target = r + gamma * (best Q of the next situation, from the frozen network); no future after a crash."""
        next_q, _ = q_batch(self.target_theta, self.sizes, S2)
        return R + gamma * (1.0 - D) * next_q.max(axis=1)

    def update(self, batch, gamma, lr, delta=1.0) -> float:
        """One learning step on a batch of transitions. Returns the loss."""
        S, A, R, S2, D = batch
        y = self.targets(R, S2, D, gamma)
        loss, grad = td_loss_and_grad(self.theta, self.sizes, S, A, y, delta)
        self.adam.step(self.theta, grad, lr)
        return loss

    def sync_target(self):
        self.target_theta[:] = self.theta
