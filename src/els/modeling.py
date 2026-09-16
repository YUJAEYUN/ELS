"""Small deterministic LightGBM ensembles; fixed hyperparameters, no test tuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from els.binning import apply_bins, fit_bins
from els.selection import select_features

TAUS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
PARAMS = {"learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
          "min_data_in_leaf": 30, "lambda_l2": 5.0, "verbosity": -1,
          "num_threads": 2, "seed": 42, "deterministic": True, "force_col_wise": True}
ROUNDS = 80


def fit_preprocessor(train, kinds, variant):
    if variant not in {"all", "etc", "etc_bins"}:
        raise ValueError("Unknown model variant")
    selection = select_features(train, kinds) if variant != "all" else None
    cols = selection["selected"] if selection else list(train.columns)
    bins = {}
    if variant == "etc_bins":
        for col in cols:
            # Volatility is a level feature: retain it raw rather than reinterpret
            # level ES as a return distribution. Record this explicit exception.
            if kinds[col] != "level":
                bins[col] = fit_bins(train[col])
    return {"variant": variant, "columns": cols, "bins": bins, "selection": selection,
            "level_features_unbinned": [c for c in cols if kinds[c] == "level"]}


def apply_preprocessor(frame, state):
    result = frame[state["columns"]].copy()
    for col, fitted in state["bins"].items():
        result[col] = apply_bins(result[col], fitted)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Model features must be finite")
    return result


def fit_models(features, target, rounds=ROUNDS):
    import lightgbm as lgb
    if len(features) != len(target) or not features.index.equals(target.index):
        raise ValueError("Feature/target index mismatch")
    if not np.isfinite(target).all():
        raise ValueError("Training target must be finite")
    models = {}
    for name, tau in [("point", None)] + [(f"q{t:.2f}", t) for t in TAUS]:
        params = dict(PARAMS, objective="regression" if tau is None else "quantile")
        if tau is not None:
            params["alpha"] = tau
        dataset = lgb.Dataset(features, label=target, free_raw_data=True)
        models[name] = lgb.train(params, dataset, num_boost_round=rounds)
    return models


def predict(models, features):
    point = models["point"].predict(features, num_threads=2)
    raw = np.column_stack([models[f"q{t:.2f}"].predict(features, num_threads=2) for t in TAUS])
    return point, raw, np.sort(raw, axis=1)


def save_bundle(models, state, metadata, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.save_model(str(output / (name + ".txt")))
    (output / "manifest.json").write_text(
        json.dumps({"preprocessor": state, "metadata": metadata, "taus": TAUS}, indent=2), encoding="utf-8")


def predict_bundle(output, features):
    import lightgbm as lgb
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["taus"] != TAUS:
        raise ValueError("Unsupported model quantile grid")
    transformed = apply_preprocessor(features, manifest["preprocessor"])
    models = {name: lgb.Booster(model_file=str(output / (name + ".txt")))
              for name in ["point"] + [f"q{t:.2f}" for t in TAUS]}
    return predict(models, transformed)
