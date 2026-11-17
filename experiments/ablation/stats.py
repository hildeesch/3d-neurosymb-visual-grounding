"""Paired factorial contrasts over per-expression correctness.

Every cell of the design scores the same expressions, so any effect is a
per-expression contrast ``d_i = sum_c w_c * y_c[i]`` with ``y`` in {0, 1}:

* a simple ablation is ``y_full - y_ablated`` (McNemar's pairs);
* a main effect averages that over the other factors' levels;
* an interaction is the difference-in-differences ``(y11 - y01) - (y10 - y00)``,
  averaged over any remaining factors; a three-way interaction is the
  difference of two such.

Effects are on the accuracy scale, in percentage points; sign +1 is the full
system's level, so a positive main effect is what that component adds and a
positive interaction means the two components help more together than apart.

Tests are exact sign-flip (paired permutation) tests of ``E[d] = 0``: under the
null each expression's contrast is equally likely to have either sign. For a
two-cell contrast this is exactly the two-sided exact McNemar test. Confidence
intervals are percentile bootstraps over expressions, seeded.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

__all__ = ["load_correct", "contrast", "signflip_p", "bootstrap_ci", "holm",
           "factorial_weights", "factorial_effects", "simple_contrast"]


def load_correct(dump: Path, keys: list[int]) -> np.ndarray:
    """Correctness per population key; a row absent from the dump is a miss.

    Absent rows are the ones the evaluator could not score (null parse, executor
    error); the official denominator counts them as misses, and so does this.
    """
    hits = {}
    with Path(dump).open() as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                hits[int(row["dataset_idx"])] = bool(row["correct"])
    unknown = set(hits) - set(keys)
    if unknown:
        raise ValueError(f"{dump} scores {len(unknown)} rows outside the "
                         f"population (e.g. {sorted(unknown)[:3]})")
    return np.array([hits.get(k, False) for k in keys], dtype=bool)


# --- exact sign-flip test ------------------------------------------------

def _binom_pmf(n: int) -> np.ndarray:
    k = np.arange(n + 1)
    log = (math.lgamma(n + 1) - np.array([math.lgamma(i + 1) for i in k])
           - np.array([math.lgamma(n - i + 1) for i in k]) - n * math.log(2))
    return np.exp(log)


def _integer_scale(d: np.ndarray) -> tuple[np.ndarray, int]:
    """d scaled to integers by the least common denominator of its values."""
    values = {Fraction(float(v)).limit_denominator(64) for v in np.unique(d)}
    scale = math.lcm(*(v.denominator for v in values)) if values else 1
    scaled = np.rint(d * scale).astype(np.int64)
    if not np.allclose(scaled, d * scale):
        raise ValueError("contrast values are not multiples of 1/64")
    return scaled, scale


def signflip_p(d: np.ndarray) -> float:
    """Two-sided exact p-value of sum(d) under independent sign flips."""
    scaled, _ = _integer_scale(np.asarray(d, dtype=float))
    nonzero = scaled[scaled != 0]
    if nonzero.size == 0:
        return 1.0
    magnitudes, counts = np.unique(np.abs(nonzero), return_counts=True)
    # S = sum_j m_j K_j with K_j ~ Bin(n_j, 1/2); T = 2S - sum_j m_j n_j.
    pmf = np.array([1.0])
    for m, n in zip(magnitudes.tolist(), counts.tolist()):
        spread = np.zeros(m * n + 1)
        spread[::m] = _binom_pmf(n)
        pmf = np.convolve(pmf, spread)
    total = int((magnitudes * counts).sum())
    t = 2 * np.arange(pmf.size) - total
    observed = abs(int(nonzero.sum()))
    return float(min(1.0, pmf[np.abs(t) >= observed].sum()))


def bootstrap_ci(d: np.ndarray, *, n_boot: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of mean(d), resampling expressions."""
    d = np.asarray(d, dtype=float)
    values, counts = np.unique(d, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(d.size, counts / d.size, size=n_boot)
    means = draws @ values / d.size
    alpha = (1 - level) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(low), float(high)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# --- contrasts -----------------------------------------------------------

def contrast(y: dict, weights: dict, *, n_boot: int = 10_000,
             seed: int = 0) -> dict:
    """Estimate, CI and exact test of sum_c w_c * y_c, averaged over rows."""
    d = sum(float(w) * y[c].astype(float) for c, w in weights.items() if w)
    d = np.asarray(d, dtype=float)
    low, high = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {
        "estimate_pp": 100 * float(d.mean()),
        "ci95_pp": [100 * low, 100 * high],
        "p_exact": signflip_p(d),
        "n": int(d.size),
        "n_nonzero": int((d != 0).sum()),
    }


def simple_contrast(y_arm: np.ndarray, y_ref: np.ndarray, **kw) -> dict:
    """arm - reference, with McNemar's discordant counts."""
    result = contrast({"arm": y_arm, "ref": y_ref}, {"arm": 1, "ref": -1}, **kw)
    result["wins"] = int((y_arm & ~y_ref).sum())
    result["losses"] = int((y_ref & ~y_arm).sum())
    return result


def factorial_weights(design: dict[str, tuple[str, str]], cells: list,
                      effect: tuple[str, ...]) -> dict:
    """Contrast weights of one effect in a 2^k design.

    ``design`` maps factor -> (full level, ablated level); ``cells`` are objects
    with ``.level(factor)``; ``effect`` is the subset of factors. Weight per
    cell: product of the effect factors' signs (+1 full, -1 ablated), divided
    by 2^(k - |effect|) to average over the factors not in the effect.
    """
    k = len(design)
    weights = {}
    for cell in cells:
        if any(cell.level(f) not in levels for f, levels in design.items()):
            continue
        sign = 1
        for factor in effect:
            sign *= 1 if cell.level(factor) == design[factor][0] else -1
        weights[cell] = Fraction(sign, 2 ** (k - len(effect)))
    expected = 2 ** k
    if len(weights) != expected:
        raise ValueError(f"design {design} needs {expected} cells, "
                         f"found {len(weights)}")
    return weights


def factorial_effects(design: dict[str, tuple[str, str]], y: dict,
                      **kw) -> list[dict]:
    """Every main effect and interaction of a 2^k paired design."""
    factors = list(design)
    out = []
    for order in range(1, len(factors) + 1):
        for effect in combinations(factors, order):
            weights = factorial_weights(design, list(y), effect)
            out.append({"effect": " x ".join(effect), "order": order,
                        **contrast(y, weights, **kw)})
    return out
