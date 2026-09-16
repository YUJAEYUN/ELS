"""Experimental own-distribution ES partition, NOT target-monotonic binning."""
import numpy as np
import pandas as pd


def lower_es(values, alpha):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all() or not 0 < alpha <= 1:
        raise ValueError("Invalid ES input")
    # Fractional order-statistic weight gives exactly alpha probability mass.
    mass = alpha * len(values)
    k = int(np.floor(mass))
    total = values[:k].sum()
    if k < len(values):
        total += (mass - k) * values[k]
    return float(total / mass)


def fit_bins(train, min_tail=10, slope_ratio=0.25):
    values = np.asarray(train, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Use finite training return/difference values")
    if min_tail < 1 or not 0 < slope_ratio <= 1:
        raise ValueError("Invalid bin settings")
    grid = np.arange(1, 26) / 100
    grid = grid[grid * len(values) >= min_tail]
    if len(grid) < 4:
        raise ValueError("Too few observations to estimate tail stability")
    es = np.array([lower_es(values, a) for a in grid])
    slopes = np.diff(es) / np.diff(grid)
    chosen, reason = 0.1, "fallback_alpha_0.10"
    peak = np.max(np.abs(slopes))
    if peak > 0:
        for i in range(len(slopes) - 2):
            if np.all(np.abs(slopes[i:i + 3]) <= slope_ratio * peak):
                chosen, reason = float(grid[i + 1]), "three_stable_slopes"
                break
    if chosen * len(values) < min_tail:
        chosen, reason = float(grid[0]), "minimum_tail_mass"
    # Tail threshold plus median/upper tail provide an ordered input partition.
    cuts = np.unique(np.quantile(values, [chosen, 0.5, 1 - chosen])).tolist()
    if np.ptp(values) == 0:
        cuts = []
    return {"cuts": cuts, "alpha": chosen, "reason": reason,
            "grid": grid.tolist(), "lower_es": es.tolist(), "experimental": True,
            "min_tail": min_tail, "slope_ratio": slope_ratio}


def apply_bins(values, fitted):
    values = pd.Series(values, dtype=float)
    cuts = np.asarray(fitted["cuts"], dtype=float)
    if not np.isfinite(cuts).all() or (np.diff(cuts) <= 0).any():
        raise ValueError("Invalid fitted boundaries")
    result = pd.Series(np.searchsorted(cuts, values, side="left"), index=values.index, dtype=float)
    result[~np.isfinite(values)] = np.nan
    return result
