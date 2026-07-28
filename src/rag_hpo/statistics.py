from __future__ import annotations

import random
import statistics


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_inference(
    differences: list[float], *, seed: int, iterations: int
) -> dict[str, float | int]:
    """Return deterministic paired bootstrap and randomization-test results."""
    if len(differences) < 2:
        raise ValueError("paired comparison requires at least two cases")
    # This generator is intentionally reproducible for scientific resampling;
    # it does not create keys, tokens, or any security-sensitive value.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    observed = statistics.fmean(differences)
    bootstrapped = sorted(
        statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(iterations)
    )
    at_least_observed = 0
    at_most_observed = 0
    at_least_absolute = 0
    for _ in range(iterations):
        permuted = statistics.fmean(
            value if rng.getrandbits(1) else -value for value in differences
        )
        at_least_observed += permuted >= observed
        at_most_observed += permuted <= observed
        at_least_absolute += abs(permuted) >= abs(observed)
    return {
        "iterations": iterations,
        "mean_difference": observed,
        "median_difference": statistics.median(differences),
        "mean_difference_ci95_lower": _quantile(bootstrapped, 0.025),
        "mean_difference_ci95_upper": _quantile(bootstrapped, 0.975),
        "one_sided_sign_flip_p": (at_least_observed + 1) / (iterations + 1),
        "one_sided_lower_sign_flip_p": (at_most_observed + 1) / (iterations + 1),
        "two_sided_sign_flip_p": (at_least_absolute + 1) / (iterations + 1),
        "seed": seed,
    }
