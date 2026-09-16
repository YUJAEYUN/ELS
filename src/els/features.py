"""Offline training-only selection and experimental bin diagnostics."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from els.binning import apply_bins, fit_bins
from els.selection import select_features


def prepare(input_path, config_path, train_end, output, experimental_bins=False):
    input_path, output = Path(input_path), Path(output)
    frame = pd.read_csv(input_path, index_col="week", parse_dates=["week"])
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    kinds = {s["id"]: s.get("transform", "level") for s in config["series"]}
    # Slice BEFORE scoring, estimating medians, choosing cutoffs or fitting bins.
    train = frame.loc[frame.index <= pd.Timestamp(train_end)]
    report = select_features(train, kinds)
    report["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    report["requested_train_end"] = train_end
    report["bins"] = {}
    selected = frame[report["selected"]].copy()
    if experimental_bins:
        for name in report["selected"]:
            if kinds[name] == "level":
                raise ValueError("Experimental ES bins require returns/differences, not levels")
            report["bins"][name] = fit_bins(train.loc[report["common_start"]:, name])
            selected[name] = apply_bins(selected[name], report["bins"][name])
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected.csv")
    (output / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
