"""Causal weekly features and exact-calendar forward labels."""
from pathlib import Path

import numpy as np
import pandas as pd

from els.preprocessing import weekly


def check_weekly(index):
    if not isinstance(index, pd.DatetimeIndex) or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Expected chronological unique DatetimeIndex")
    if len(index) < 2 or (index.dayofweek != 4).any():
        raise ValueError("Expected Friday weekly observations")
    if not (index.to_series().diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError("Weekly calendar has missing rows")


def make_features(weekly_values, quality):
    check_weekly(weekly_values.index)
    stale = quality.pivot(index="week", columns="series", values="stale")
    stale = stale.reindex(index=weekly_values.index, columns=weekly_values.columns)
    if stale.isna().any().any():
        raise ValueError("Quality records must cover all feature observations")
    if not stale.isin([True, False]).all().all():
        raise ValueError("Expected boolean freshness flags")
    # Changes depend on this and preceding level. Invalidate both sides of stale gaps.
    invalid = stale | stale.shift(1, fill_value=True)
    values = weekly_values.mask(invalid)
    features = {}
    for col in values:
        s = values[col]
        features[col] = s
        features[col + "__lag1"] = s.shift(1)
        features[col + "__lag4"] = s.shift(4)
        features[col + "__mean13"] = s.rolling(13, min_periods=13).mean()
        features[col + "__vol13"] = s.rolling(13, min_periods=13).std()
    result = pd.DataFrame(features)
    kinds = {c: "level" if c.endswith("__vol13") else "difference" for c in result}
    # Input config is explicitly restricted to return/difference columns by load_dataset.
    return result, kinds


def forward_target(levels, horizon=52):
    check_weekly(levels.index)
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer")
    valid = levels.dropna()
    if not np.isfinite(valid).all() or (valid <= 0).any():
        raise ValueError("Target index levels must be positive and finite")
    end = levels.index + pd.Timedelta(weeks=horizon)
    future = levels.reindex(end).to_numpy()
    return pd.DataFrame({"target": future / levels.to_numpy() - 1, "label_end": end}, index=levels.index)


def load_dataset(batch_dir, target="kospi200", horizon=52):
    import json
    batch_dir = Path(batch_dir)
    meta = json.loads((batch_dir / "metadata.json").read_text(encoding="utf-8"))
    specs = meta["config"]["series"]
    if any(s.get("transform", "level") not in {"return", "difference"} for s in specs):
        raise ValueError("Model input must be returns/differences")
    if target not in {s["id"] for s in specs}:
        raise ValueError("Target not in batch configuration")
    values = pd.read_csv(batch_dir / "weekly.csv", index_col="week", parse_dates=["week"])
    quality = pd.read_csv(batch_dir / "quality.csv", parse_dates=["week"])
    features, kinds = make_features(values, quality)
    raw = pd.read_csv(batch_dir / (target + "_raw.csv"))
    spec = next(s for s in specs if s["id"] == target)
    if spec.get("availability_lag_days", 0) != 0:
        raise ValueError("Target requires contemporaneous market close levels")
    levels = weekly(raw, meta["config"]["start"], meta["as_of"])
    # A price more than one week old is not accepted as an endpoint label.
    clean = levels["value"].mask(levels["age_days"] > 7)
    labels = forward_target(clean, horizon).reindex(features.index)
    return features, kinds, labels


def training_rows(features, labels, origin, min_samples=156):
    origin = pd.Timestamp(origin)
    mature = labels["label_end"].le(origin) & labels["target"].notna()
    finite = pd.Series(np.isfinite(features.to_numpy()).all(axis=1), index=features.index)
    idx = features.index[mature & finite & (features.index < origin)]
    if len(idx) < min_samples:
        raise ValueError("Insufficient mature training observations")
    # Do not stitch disjoint sequences together for ETC.
    check_weekly(idx)
    return idx


def walk_forward(features, labels, start, refit_weeks=13, min_samples=156):
    check_weekly(features.index)
    if refit_weeks < 1:
        raise ValueError("Refit interval must be positive")
    candidates = features.index[(features.index >= pd.Timestamp(start)) & labels["target"].notna()]
    if not len(candidates):
        raise ValueError("No matured evaluation targets")
    check_weekly(candidates)
    for i in range(0, len(candidates), refit_weeks):
        test = candidates[i:i + refit_weeks]
        if not np.isfinite(features.loc[test].to_numpy()).all():
            raise ValueError("Invalid or stale evaluation features")
        train = training_rows(features, labels, test[0], min_samples)
        yield train, test
