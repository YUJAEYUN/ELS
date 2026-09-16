"""Reproducible first-index walk-forward experiment and final nine-model bundle."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from els.dataset import load_dataset, training_rows, walk_forward
from els.evaluation import metrics
from els.modeling import (PARAMS, ROUNDS, TAUS, apply_preprocessor, fit_models,
                          fit_preprocessor, predict, predict_bundle, save_bundle)


def run_experiment(batch_dir, output, target="kospi200", start="2020-01-03", refit_weeks=13):
    batch_dir, output = Path(batch_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory to preserve prior experiment artifacts")
    x, kinds, labels = load_dataset(batch_dir, target)
    variants = ["all", "etc", "etc_bins"]
    records, audits = [], []
    for fold, (train, test) in enumerate(walk_forward(x, labels, start, refit_weeks)):
        y_train = labels.loc[train, "target"]
        audit = {"fold": fold, "origin": str(test[0].date()), "test_end": str(test[-1].date()),
                 "train_first": str(train[0].date()), "train_last": str(train[-1].date()),
                 "max_label_end": str(labels.loc[train, "label_end"].max().date()),
                 "train_n": len(train), "test_n": len(test), "preprocessors": {}}
        # Both baselines see exactly the same matured labels as the learned models.
        base_q = np.tile(np.quantile(y_train, TAUS), (len(test), 1))
        outputs = {"historical": (np.full(len(test), y_train.mean()), base_q, base_q),
                   "zero": (np.zeros(len(test)), base_q, base_q)}
        for variant in variants:
            state = fit_preprocessor(x.loc[train], kinds, variant)
            audit["preprocessors"][variant] = state
            models = fit_models(apply_preprocessor(x.loc[train], state), y_train)
            outputs[variant] = predict(models, apply_preprocessor(x.loc[test], state))
        for name, (point, raw, ordered) in outputs.items():
            for i, date in enumerate(test):
                row = {"date": str(date.date()), "label_end": str(labels.loc[date, "label_end"].date()),
                       "fold": fold, "variant": name, "actual": float(labels.loc[date, "target"]),
                       "point": float(point[i]), "crossed": bool((np.diff(raw[i]) < 0).any())}
                row.update({f"q{t:.2f}": float(ordered[i, j]) for j, t in enumerate(TAUS)})
                row.update({f"raw_q{t:.2f}": float(raw[i, j]) for j, t in enumerate(TAUS)})
                records.append(row)
        audits.append(audit)
        print(f"Fold {fold + 1}: origin={audit['origin']}, train={len(train)}, test={len(test)}", flush=True)
    predictions = pd.DataFrame(records)
    summary = {}
    qcols = [f"q{t:.2f}" for t in TAUS]
    for name, frame in predictions.groupby("variant", sort=False):
        overall = metrics(frame.actual, frame.point, frame[qcols])
        overall["raw_crossing_rate"] = float(frame.crossed.mean())
        overall["by_forecast_year"] = {
            year: metrics(g.actual, g.point, g[qcols])
            for year, g in frame.groupby(frame.date.str[:4])}
        sparse = frame.iloc[::52]
        overall["nonoverlapping_first_phase"] = metrics(sparse.actual, sparse.point, sparse[qcols])
        summary[name] = overall
    as_of = x.index[-1]
    train = training_rows(x, labels, as_of)
    # The primary architecture is fixed before looking at test performance.
    state = fit_preprocessor(x.loc[train], kinds, "etc_bins")
    models = fit_models(apply_preprocessor(x.loc[train], state), labels.loc[train, "target"])
    latest_x = x.loc[[as_of]]
    expected = predict(models, apply_preprocessor(latest_x, state))
    metadata = {"target": target, "horizon_weeks": 52, "as_of": str(as_of.date()),
                "forecast_end": str((as_of + pd.Timedelta(weeks=52)).date()),
                "train_first": str(train[0].date()), "train_last": str(train[-1].date()),
                "max_label_end": str(labels.loc[train, "label_end"].max().date()), "train_n": len(train),
                "parameters": PARAMS, "rounds": ROUNDS, "refit_weeks": refit_weeks,
                "versions": {p: importlib.metadata.version(p) for p in ["lightgbm", "pandas", "numpy", "scipy"]},
                "input_hashes": {p: hashlib.sha256((batch_dir / p).read_bytes()).hexdigest()
                                 for p in ["weekly.csv", "quality.csv", "metadata.json", target + "_raw.csv"]},
                "limitations": ["Current data vintage; not point-in-time certified", "Overlapping 52-week labels",
                                "Primary etc_bins fixed a priori; no test-set tuning", "Experimental ES bins",
                                "No knock-in or early-redemption probability", "No 2008 coverage"]}
    save_bundle(models, state, metadata, output / "models")
    restored = predict_bundle(output / "models", latest_x)
    for a, b in zip(expected, restored):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    forecast = {"as_of": metadata["as_of"], "forecast_end": metadata["forecast_end"],
                "point": float(restored[0][0]),
                "quantiles": {f"{t:.2f}": float(v) for t, v in zip(TAUS, restored[2][0])},
                "raw_quantiles": {f"{t:.2f}": float(v) for t, v in zip(TAUS, restored[1][0])},
                "status": "research_only_unvalidated"}
    predictions.to_csv(output / "predictions.csv", index=False)
    latest_x.to_csv(output / "latest_features.csv")
    for name, obj in [("metrics", summary), ("folds", audits), ("metadata", metadata), ("forecast", forecast)]:
        (output / (name + ".json")).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    lines = [f"# {target.upper()} 52-week prototype", "", "Returns in decimal units; 0.01 = 1 percentage point.", "",
             "| Variant | RMSE | MAE | Mean pinball | 90% coverage | Raw crossing |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in summary.items():
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['mae']:.6f} | {m['mean_pinball']:.6f} | {m['interval90_coverage']:.3f} | {m['raw_crossing_rate']:.3f} |")
    lines += ["", f"Folds: {len(audits)}. Test observations per variant: {summary['historical']['n']}.",
              "", "No significance claim: labels overlap. Nonoverlapping diagnostics have very few observations.",
              "Current-vintage data, limited history and uncalibrated tails: not for product risk probabilities.",
              "Nine final models were saved and reloaded; predictions matched at 1e-12 tolerance."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
