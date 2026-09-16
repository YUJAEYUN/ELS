"""Metrics are descriptive: overlapping 52-week labels are NOT independent samples."""
import numpy as np

from els.modeling import TAUS


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
