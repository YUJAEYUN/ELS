import numpy as np
import pandas as pd
import pytest

from els.binning import apply_bins, fit_bins, lower_es
from els.features import prepare
from els.cli import demo
from els.pipeline import run
from els.selection import binarize, cutoff, nsrps, select_features


def test_document_example_and_leftmost_ties():
    count, trace = nsrps([1, 1, 0, 1, 0, 0, 1, 0])
    assert trace[0] == {"pair": [1, 0], "sequence": [1, 2, 2, 0, 2]}
    assert count == 5
    assert nsrps([1, 2, 2, 0, 2])[1][0]["pair"] == [1, 2]


def test_termination_and_nonoverlap():
    assert nsrps([1])[0] == 0
    assert nsrps([1] * 10)[0] == 0
    assert nsrps([0, 1] * 10)[0] == 1
    assert nsrps([0, 0, 0, 1])[1][0]["sequence"] == [2, 0, 1]
    with pytest.raises(ValueError):
        nsrps([])


def test_expanding_median_is_causal():
    a = binarize([5, 2, 6, 3], "level")
    b = binarize([5, 2, 6, 3, -9999], "level")
    assert a.tolist() == [0, 0, 1, 0]
    assert a.tolist() == b.iloc[:4].tolist()
    assert binarize([-1, 0, 1], "return").tolist() == [0, 0, 1]


def test_cutoff():
    assert cutoff([10, 10, 10, 10]) == (2, "top_fraction")
    assert cutoff([100, 20, 19, 18, 17]) == (2, "elbow")
    with pytest.raises(ValueError):
        cutoff([1, 2])


def test_equal_history_and_missing_rejection():
    idx = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    data = pd.DataFrame({"a": np.arange(120), "b": np.arange(120)}, index=idx, dtype=float)
    data.iloc[:10, 1] = np.nan
    report = select_features(data, {"a": "level", "b": "level"})
    assert report["samples"] == 110
    # a already has prior history; b's first expanding-median bit is zero.
    assert report["scores"]["a"] == 0
    assert report["scores"]["b"] > 0
    data.iloc[50, 0] = np.nan
    with pytest.raises(ValueError):
        select_features(data, {"a": "level", "b": "level"})


def test_fractional_lower_es():
    assert lower_es([-4, -2, 2, 4], 0.375) == pytest.approx(-10 / 3)
    assert lower_es([-4, -2, 2, 4], 1) == 0


def test_bins_constant_and_extremes():
    fitted = fit_bins(np.zeros(200))
    assert fitted["cuts"] == []
    assert apply_bins([0, 0], fitted).tolist() == [0, 0]
    fitted = fit_bins(np.linspace(-1, 1, 200))
    assert apply_bins([-100, 100], fitted).tolist() == [0, 3]
    assert np.isnan(apply_bins([np.nan], fitted).iloc[0])


def test_future_values_cannot_change_fitted_selection_or_bins(tmp_path):
    config = demo(tmp_path / "input")
    run(config, "2025-12-31", tmp_path / "batch")
    path = tmp_path / "batch/weekly.csv"
    first = prepare(path, config, "2023-12-29", tmp_path / "first", True)
    frame = pd.read_csv(path)
    frame.loc[frame.week > "2023-12-29", frame.columns != "week"] = 999
    frame.to_csv(path, index=False)
    second = prepare(path, config, "2023-12-29", tmp_path / "second", True)
    for key in ["selected", "scores", "bins", "samples"]:
        assert first[key] == second[key]
