"""Neural networks: gradients and XOR.

Wiki pages
  Chinese: https://dmxsf.com/wiki/foundations/neural-network-basics/
  English: https://dmxsf.com/en/wiki/foundations/neural-network-basics/
"""

import numpy as np


def loss_and_gradients(x, labels, w1, w2):
    """Return stable cross-entropy and gradients for a two-layer tanh classifier."""
    hidden = np.tanh(x @ w1)
    logits = hidden @ w2
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    rows = np.arange(len(labels))
    loss = -log_probs[rows, labels].mean()
    delta = np.exp(log_probs)
    delta[rows, labels] -= 1
    delta /= len(labels)
    hidden_delta = (delta @ w2.T) * (1 - hidden**2)
    return float(loss), (x.T @ hidden_delta, hidden.T @ delta)


def xor_fit(seed: int, hidden: int, steps: int = 4000, lr: float = 0.5):
    """Train on the four XOR rows, using a tanh hidden layer when hidden is nonzero."""
    rng = np.random.default_rng(seed)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    x = np.column_stack((x, np.ones(len(x))))
    w1 = rng.normal(size=(3, hidden))
    w2 = rng.normal(size=(hidden + 1 if hidden else 3, 2))
    for _ in range(steps):
        h = np.tanh(x @ w1) if hidden else x[:, :-1]
        design = np.column_stack((h, np.ones(len(h))))
        z = design @ w2
        exp = np.exp(z - z.max(axis=1, keepdims=True))
        error = exp / exp.sum(axis=1, keepdims=True)
        error[np.arange(len(x)), labels] -= 1
        error /= len(x)
        w1 -= lr * x.T @ ((error @ w2[:-1].T) * (1 - h**2)) if hidden else 0
        w2 -= lr * design.T @ error
    h = np.tanh(x @ w1) if hidden else x[:, :-1]
    logits = np.column_stack((h, np.ones(len(h)))) @ w2
    shifted = logits - logits.max(axis=1, keepdims=True)
    loss = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(x)), labels])
    return float(loss), int((logits.argmax(axis=1) == labels).sum())
