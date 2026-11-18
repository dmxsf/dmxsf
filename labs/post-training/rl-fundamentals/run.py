"""Lab: Policy gradients: exact checks, baselines and GAE.

Wiki pages
  Chinese: https://dmxsf.com/wiki/post-training/rl-fundamentals/
  English: https://dmxsf.com/en/wiki/post-training/rl-fundamentals/
"""

import numpy as np

from dmxsf import lab
from dmxsf.post_training.rl_basics import (
    action_probabilities,
    exact_policy_gradient,
    generalized_advantage,
)


def main() -> None:
    _args, report = lab.start(
        "Policy gradients: exact checks, baselines and GAE", wiki=["post-training/rl-fundamentals"]
    )
    report.section("1. An enumerated policy with three actions")
    logits = np.log([0.5, 0.3, 0.2])
    rewards = np.array([1.0, 0.0, 0.0])
    exact = exact_policy_gradient(logits, rewards)
    h = 1e-5
    numerical = [
        (
            action_probabilities(logits + h * axis) @ rewards
            - action_probabilities(logits - h * axis) @ rewards
        )
        / (2 * h)
        for axis in np.eye(3)
    ]
    report.check(
        "score-function gradient agrees with the finite difference of expected reward",
        np.allclose(exact, [0.25, -0.15, -0.10])
        and np.allclose(exact, numerical, atol=1e-8, rtol=0),
        str(exact),
    )
    report.check(
        "a constant action-independent baseline preserves the exact gradient",
        np.allclose(exact_policy_gradient(logits, rewards, 0.5), exact),
    )
    report.section("2. Generalized advantage estimation")
    errors = np.array([1.0, 0.0, 2.0])
    one_step = generalized_advantage(errors, 0.9, 0)
    full = generalized_advantage(errors, 0.9, 1)
    report.check(
        "lambda zero uses one TD error and lambda one accumulates the trajectory",
        np.array_equal(one_step, errors) and np.allclose(full, [2.62, 1.8, 2.0]),
        str(full),
    )
    report.finish()


if __name__ == "__main__":
    main()
