"""Purged chronological fit/calibration/test comparison on reused development dates."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from els.calibration import apply_calibration, calibration_split, fit_calibration
from els.dataset import load_dataset, walk_forward
from els.evaluation import interval_metrics, metrics
from els.modeling import (PARAMS, ROUNDS, TAUS, apply_preprocessor, fit_models,
                          fit_preprocessor, predict, predict_bundle, save_bundle)


def run_calibration_experiment(batch, protocol, output):
    batch, protocol, output = Path(batch), Path(protocol), Path(output)
    cfg = json.loads(protocol.read_text(encoding="utf-8"))
    if cfg["horizon_weeks"] != 52 or cfg["variant"] != "etc_bins" or cfg["alpha"] != 0.10:
        raise ValueError("Unsupported protocol")
    if cfg["cqr_nonnegative_expansion"] is not True or cfg["quantile_shift_method"] != "empirical_linear":
        raise ValueError("Unsupported calibration method")
    if cfg["evaluation_role"] != "development_reused_dates_not_untouched_holdout":
        raise ValueError("This experiment reuses development dates")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    x, kinds, labels = load_dataset(batch, cfg["target"])
    records, audits, intervals = [], [], []
    qcols = [f"q{t:.2f}" for t in TAUS]
    def fit_at(fit):
        state = fit_preprocessor(x.loc[fit], kinds, cfg["variant"])
        models = fit_models(apply_preprocessor(x.loc[fit], state), labels.loc[fit, "target"])
        return state, models
    for fold, (mature, test) in enumerate(walk_forward(x, labels, cfg["start"], cfg["refit_weeks"], cfg["minimum_fit_rows"])):
        fit, cal = calibration_split(x, labels, test[0], cfg["calibration_weeks"], cfg["minimum_fit_rows"])
        state, models = fit_at(fit)
        _, _, cal_q = predict(models, apply_preprocessor(x.loc[cal], state))
        correction = fit_calibration(labels.loc[cal, "target"], cal_q, cfg["alpha"])
        point, raw, split_q = predict(models, apply_preprocessor(x.loc[test], state))
        shifted, lo, hi = apply_calibration(split_q, correction)
        expanding_state, expanding_models = fit_at(mature)
        ep, er, eq = predict(expanding_models, apply_preprocessor(x.loc[test], expanding_state))
        ytrain = labels.loc[mature, "target"]
        historical = np.tile(np.quantile(ytrain, TAUS), (len(test), 1))
        variants = {"historical": (np.full(len(test), ytrain.mean()), historical),
                    "expanding_uncalibrated": (ep, eq), "split_uncalibrated": (point, split_q),
                    "split_residual_calibrated": (point, shifted)}
        for name, (p, qs) in variants.items():
            for i, date in enumerate(test):
                row = {"date": str(date.date()), "label_end": str(labels.loc[date, "label_end"].date()),
                       "fold": fold, "variant": name, "actual": float(labels.loc[date, "target"]), "point": float(p[i])}
                row.update({c: float(qs[i, j]) for j, c in enumerate(qcols)})
                records.append(row)
        for i, date in enumerate(test):
            intervals.append({"date": str(date.date()), "fold": fold, "actual": float(labels.loc[date, "target"]),
                              "lower": float(lo[i]), "upper": float(hi[i])})
        audits.append({"fold": fold, "test_first": str(test[0].date()), "test_last": str(test[-1].date()),
                       "fit_first": str(fit[0].date()), "fit_last": str(fit[-1].date()), "fit_n": len(fit),
                       "fit_max_label_end": str(labels.loc[fit, "label_end"].max().date()),
                       "cal_first": str(cal[0].date()), "cal_last": str(cal[-1].date()),
                       "cal_max_label_end": str(labels.loc[cal, "label_end"].max().date()),
                       "preprocessor": state, "correction": correction,
                       "expanding_preprocessor": expanding_state})
        print(f"Calibration fold {fold+1}: {test[0].date()}, fit={len(fit)}, cal={len(cal)}", flush=True)
    frame, interval_frame = pd.DataFrame(records), pd.DataFrame(intervals)
    summary = {}
    for name, group in frame.groupby("variant", sort=False):
        m = metrics(group.actual, group.point, group[qcols])
        m["interval"] = interval_metrics(group.actual, group["q0.05"], group["q0.95"])
        m["by_year"] = {year: metrics(g.actual, g.point, g[qcols]) for year, g in group.groupby(group.date.str[:4])}
        summary[name] = m
    cqr = interval_metrics(interval_frame.actual, interval_frame.lower, interval_frame.upper)
    cqr["by_year"] = {year: interval_metrics(g.actual, g.lower, g.upper) for year, g in interval_frame.groupby(interval_frame.date.str[:4])}
    summary["cqr90_interval_only"] = cqr
    # Archive a new independent bundle. Never overwrite the previous experiment.
    as_of = x.index[-1]
    fit, cal = calibration_split(x, labels, as_of, cfg["calibration_weeks"], cfg["minimum_fit_rows"])
    state, models = fit_at(fit)
    _, _, cq = predict(models, apply_preprocessor(x.loc[cal], state))
    correction = fit_calibration(labels.loc[cal, "target"], cq)
    metadata = {"protocol": cfg, "as_of": str(as_of.date()), "fit_first": str(fit[0].date()),
                "fit_last": str(fit[-1].date()), "fit_max_label_end": str(labels.loc[fit, "label_end"].max().date()),
                "cal_first": str(cal[0].date()), "cal_last": str(cal[-1].date()),
                "cal_max_label_end": str(labels.loc[cal, "label_end"].max().date()),
                "fit_n": len(fit), "cal_n": len(cal), "parameters": PARAMS, "rounds": ROUNDS,
                "coverage_guarantee": False, "outcome": "research_only_not_approved_for_product_risk",
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "input_hashes": {p: hashlib.sha256((batch / p).read_bytes()).hexdigest() for p in
                                 ["weekly.csv", "quality.csv", "metadata.json", cfg["target"] + "_raw.csv"]},
                "versions": {p: importlib.metadata.version(p) for p in ["lightgbm", "numpy", "pandas", "scipy"]}}
    save_bundle(models, state, metadata, output / "models")
    (output / "models/calibration.json").write_text(json.dumps(correction, indent=2), encoding="utf-8")
    p, _, base = predict(models, apply_preprocessor(x.iloc[[-1]], state))
    restored_p, _, restored_q = predict_bundle(output / "models", x.iloc[[-1]])
    restored_state = json.loads((output / "models/calibration.json").read_text())
    corrected, lower, upper = apply_calibration(restored_q, restored_state)
    np.testing.assert_allclose(p, restored_p, atol=1e-12, rtol=1e-12)
    for expected, actual in zip(apply_calibration(base, correction), (corrected, lower, upper)):
        np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=1e-12)
    forecast = {"as_of": str(as_of.date()), "forecast_end": str((as_of + pd.Timedelta(weeks=52)).date()),
                "status": metadata["outcome"], "point": float(restored_p[0]),
                "quantiles": dict(zip(qcols, corrected[0].tolist())),
                "cqr90_interval": [float(lower[0]), float(upper[0])]}
    frame.to_csv(output / "predictions.csv", index=False)
    interval_frame.to_csv(output / "cqr_intervals.csv", index=False)
    x.iloc[[-1]].to_csv(output / "latest_features.csv")
    for name, obj in [("metrics", summary), ("folds", audits), ("metadata", metadata), ("forecast", forecast)]:
        (output / (name + ".json")).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return summary
