"""Metrics are descriptive: overlapping 52-week labels are NOT independent samples."""
import numpy as np

from els.modeling import TAUS


def interval_metrics(y, lower, upper, alpha=0.1):
    y, lower, upper = (np.asarray(v, dtype=float) for v in (y, lower, upper))
    if y.ndim != 1 or not len(y) or lower.shape != y.shape or upper.shape != y.shape:
        raise ValueError("Invalid interval shapes")
    if not 0 < alpha < 1 or not all(np.isfinite(v).all() for v in (y, lower, upper)) or (lower > upper).any():
        raise ValueError("Invalid intervals")
    width = upper - lower
    score = width + 2 / alpha * np.maximum(lower - y, 0) + 2 / alpha * np.maximum(y - upper, 0)
    return {"n": len(y), "coverage": float(((y >= lower) & (y <= upper)).mean()),
            "mean_width": float(width.mean()), "mean_interval_score": float(score.mean()),
            "below_rate": float((y < lower).mean()), "above_rate": float((y > upper).mean())}


def metrics(y, point, quantiles):
    y, point, quantiles = np.asarray(y), np.asarray(point), np.asarray(quantiles)
    if y.ndim != 1 or point.shape != y.shape or quantiles.shape != (len(y), len(TAUS)) or not len(y):
        raise ValueError("Invalid metric shapes")
    if not all(np.isfinite(a).all() for a in (y, point, quantiles)):
        raise ValueError("Invalid metric values")
    error = y[:, None] - quantiles
    tau = np.asarray(TAUS)
    losses = np.maximum(tau * error, (tau - 1) * error).mean(axis=0)
    return {"n": len(y), "rmse": float(np.sqrt(np.mean((y - point) ** 2))),
            "mae": float(np.mean(np.abs(y - point))), "mean_pinball": float(losses.mean()),
            "pinball": {f"{t:.2f}": float(v) for t, v in zip(TAUS, losses)},
            "coverage": {f"{t:.2f}": float(v) for t, v in zip(TAUS, (y[:, None] <= quantiles).mean(axis=0))},
            "interval90_coverage": float(((y >= quantiles[:, 1]) & (y <= quantiles[:, -1])).mean()),
            "interval90_width": float((quantiles[:, -1] - quantiles[:, 1]).mean())}
