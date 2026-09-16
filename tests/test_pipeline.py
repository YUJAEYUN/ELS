import json

import pandas as pd
import pytest

from els.cli import demo
from els.pipeline import run
from els.preprocessing import transform, weekly
from els.sources import normalize, fetch


def frame(dates, values):
    return pd.DataFrame({"date": dates, "value": values})


def test_leading_missing_and_locf():
    result = weekly(frame(["2024-01-10"], [100]), "2024-01-01", "2024-02-02")
    assert pd.isna(result.iloc[0]["value"])
    assert result.iloc[1]["value"] == 100
    assert result.iloc[-1]["stale"]
    assert result.iloc[-1]["observed_at"] == pd.Timestamp("2024-01-10")


def test_future_and_incomplete_week_excluded():
    data = frame(["2024-01-05", "2024-01-10", "2024-01-12"], [10, 20, 999])
    result = weekly(data, "2024-01-01", "2024-01-10")
    assert list(result["value"]) == [10]


def test_availability_lag():
    result = weekly(frame(["2024-01-05"], [10]), "2024-01-01", "2024-01-12", lag_days=1)
    assert pd.isna(result.iloc[0]["value"])
    assert result.iloc[1]["value"] == 10


def test_null_does_not_erase_observation():
    result = weekly(frame(["2024-01-04", "2024-01-05"], [10, None]), "2024-01-01", "2024-01-05")
    assert result.iloc[0]["value"] == 10
    assert result.iloc[0]["observed_at"] == pd.Timestamp("2024-01-04")


def test_duplicate_dates_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        normalize(frame(["2024-01-01"] * 2, [1, 2]))


def test_no_silent_bad_values():
    with pytest.raises(ValueError):
        normalize(frame(["2024-01-01"], ["bad"]))


def test_transforms():
    values = pd.Series([100.0, 110.0, 99.0])
    assert transform(values, "return").iloc[1] == pytest.approx(0.1)
    assert transform(values, "difference").iloc[2] == -11
    with pytest.raises(ValueError):
        transform(pd.Series([0, 1]), "return")


def test_empty_source():
    result = weekly(frame([], []), "2024-01-01", "2024-01-12")
    assert result["value"].isna().all()
    assert result["stale"].all()


def test_end_to_end(tmp_path):
    config = demo(tmp_path / "input")
    result = run(config, "2025-12-31", tmp_path / "output")
    assert result.shape == (313, 5)
    meta = json.loads((tmp_path / "output/metadata.json").read_text())
    assert meta["data_kind"] == "synthetic"
    assert (tmp_path / "output/quality.csv").exists()


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FRED_API_KEY"):
        fetch({"source": "fred", "series_id": "DGS10"}, "2020-01-01", "2021-01-01", tmp_path)


def test_fred_adapter(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"count": 2, "observations": [
                {"date": "2024-01-01", "value": "."},
                {"date": "2024-01-02", "value": "4.2"}]}
    def get(self, url, **kwargs):
        assert kwargs["timeout"] == (10, 60)
        assert kwargs["params"]["series_id"] == "DGS10"
        return Response()
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr("requests.Session.get", get)
    data = fetch({"source": "fred", "series_id": "DGS10"}, "2024-01-01", "2024-01-02", tmp_path)
    assert pd.isna(data.iloc[0]["value"])
    assert data.iloc[1]["value"] == 4.2
