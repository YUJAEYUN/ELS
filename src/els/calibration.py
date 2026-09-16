"""Past-only residual calibration. No time-series coverage guarantee is claimed."""
import numpy as np
import pandas as pd

from els.dataset import training_rows
from els.modeling import TAUS


def calibration_split(features, labels, origin, calibration_weeks=104, min_fit=156):
    if not isinstance(calibration_weeks, int) or calibration_weeks < 2:
        raise ValueError("Calibration window must be at least two weeks")
    mature = training_rows(features, labels, origin, min_samples=calibration_weeks + min_fit)
    calibration = mature[-calibration_weeks:]
    # Fit at the FIRST calibration origin: no label extending into its future.
    fit = training_rows(features, labels, calibration[0], min_samples=min_fit)
    if len(fit.intersection(calibration)) or labels.loc[fit, "label_end"].max() > calibration[0]:
        raise ValueError("Fit/calibration leakage")
    if labels.loc[calibration, "label_end"].max() > pd.Timestamp(origin):
        raise ValueError("Calibration outcome unavailable at prediction origin")
    return fit, calibration


def fit_calibration(y, quantiles, alpha=0.10):
    y, q = np.asarray(y, dtype=float), np.asarray(quantiles, dtype=float)
    if y.ndim != 1 or len(y) < 2 or q.shape != (len(y), len(TAUS)):
        raise ValueError("Invalid calibration shapes")
    if not np.isfinite(y).all() or not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        raise ValueError("Calibration requires finite ordered quantiles")
    if alpha != 0.10:
        raise ValueError("This quantile grid supports only the 5%-95% CQR interval")
    offsets = [float(np.quantile(y - q[:, j], tau, method="linear")) for j, tau in enumerate(TAUS)]
    scores = np.maximum(q[:, 1] - y, y - q[:, -1])
    rank = int(np.ceil((len(y) + 1) * (1 - alpha)))
    if rank > len(y):
        raise ValueError("Calibration window too small for finite CQR bound")
    expansion = max(0.0, float(np.sort(scores)[rank - 1]))
    return {"offsets": offsets, "cqr_expansion": expansion, "cqr_rank": rank,
            "alpha": alpha, "n": len(y), "taus": TAUS,
            "tail_expected_counts": {f"{t:.2f}": len(y) * min(t, 1-t) for t in TAUS},
            "method": "empirical_residual_shifts_and_separate_nonshrinking_cqr90",
            "coverage_guarantee": False, "return_lower_bound": -1.0}


def apply_calibration(quantiles, state):
    q = np.asarray(quantiles, dtype=float)
    offsets = np.asarray(state["offsets"], dtype=float)
    if (state["taus"] != TAUS or q.ndim != 2 or q.shape[1] != len(TAUS)
            or offsets.shape != (len(TAUS),) or not np.isfinite(offsets).all()
            or not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any()):
        raise ValueError("Invalid quantiles or calibration state")
    expansion = state["cqr_expansion"]
    if not np.isfinite(expansion) or expansion < 0:
        raise ValueError("Invalid CQR expansion")
    shifted = np.maximum(-1.0, q + offsets)
    ordered = np.sort(shifted, axis=1)
    # CQR modifies ONLY the interval, never relabels endpoints as calibrated quantiles.
    lower = np.maximum(-1.0, q[:, 1] - expansion)
    upper = np.maximum(lower, q[:, -1] + expansion)
    return ordered, lower, upper
