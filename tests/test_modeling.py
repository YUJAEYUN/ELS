import numpy as np
import pandas as pd
import pytest

from els.dataset import forward_target, make_features, training_rows, walk_forward
from els.evaluation import metrics
from els.modeling import (TAUS, apply_preprocessor, fit_models, fit_preprocessor,
                          predict, predict_bundle, save_bundle)


def sample(n=420):
    index = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    rng = np.random.default_rng(55)
    x = pd.DataFrame(rng.normal(size=(n, 3)), index=index, columns=["a", "b", "c"])
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))), index=index)
    return x, prices


def test_exact_52_week_target_and_unmatured_tail():
    index = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    price = pd.Series(100 * 1.01 ** np.arange(60), index=index)
    labels = forward_target(price)
    assert labels.target.iloc[0] == pytest.approx(1.01 ** 52 - 1)
    assert labels.label_end.iloc[0] == index[52]
    assert labels.target.iloc[-52:].isna().all()
    price.iloc[52] = np.nan
    assert np.isnan(forward_target(price).target.iloc[0])
    with pytest.raises(ValueError):
        forward_target(price.drop(index[10]))


def test_walkforward_label_maturity_and_disjoint_evaluation():
    x, prices = sample()
    labels = forward_target(prices)
    seen = []
    for train, test in walk_forward(x, labels, x.index[208], 13):
        assert labels.loc[train, "label_end"].max() <= test[0]
        assert train[-1] == test[0] - pd.Timedelta(weeks=52)
        assert len(train.intersection(test)) == 0
        seen.extend(test)
    assert len(seen) == len(set(seen))
    assert seen[-1] == x.index[-53]


def test_features_causal_and_stale_propagation():
    x, _ = sample(40)
    q = x.stack().rename("unused").reset_index()
    q.columns = ["week", "series", "unused"]
    q["stale"] = False
    before, _ = make_features(x, q)
    changed = x.copy()
    changed.iloc[30:] = 999
    after, _ = make_features(changed, q)
    pd.testing.assert_frame_equal(before.iloc[:30], after.iloc[:30])
    q.loc[(q.week == x.index[20]) & (q.series == "a"), "stale"] = True
    bad, _ = make_features(x, q)
    assert bad.a.iloc[20:22].isna().all()
    assert bad.a__mean13.iloc[20:34].isna().all()


def test_future_mutation_cannot_change_training_or_predictions():
    pytest.importorskip("lightgbm")
    x, prices = sample()
    origin = x.index[260]
    altered_x, altered_prices = x.copy(), prices.copy()
    altered_x.loc[altered_x.index > origin] = 999
    altered_prices.loc[altered_prices.index > origin] *= 10
    outputs = []
    for features, levels in [(x, prices), (altered_x, altered_prices)]:
        labels = forward_target(levels)
        train = training_rows(features, labels, origin)
        state = fit_preprocessor(features.loc[train], {c: "difference" for c in x}, "etc_bins")
        models = fit_models(apply_preprocessor(features.loc[train], state), labels.loc[train, "target"], rounds=3)
        outputs.append((state, predict(models, apply_preprocessor(features.loc[[origin]], state))))
    assert outputs[0][0] == outputs[1][0]
    for a, b in zip(outputs[0][1], outputs[1][1]):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_nine_model_roundtrip(tmp_path):
    pytest.importorskip("lightgbm")
    x, prices = sample(220)
    y = forward_target(prices).target.dropna()
    train = x.loc[y.index]
    state = fit_preprocessor(train, {c: "difference" for c in x}, "etc_bins")
    models = fit_models(apply_preprocessor(train, state), y, rounds=3)
    expected = predict(models, apply_preprocessor(x.iloc[[-1]], state))
    save_bundle(models, state, {"test": True}, tmp_path)
    assert len(list(tmp_path.glob("*.txt"))) == 9
    actual = predict_bundle(tmp_path, x.iloc[[-1]])
    for a, b in zip(expected, actual):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_rearrangement_keeps_raw_predictions():
    class Model:
        def __init__(self, value):
            self.value = value
        def predict(self, features, **kwargs):
            return np.full(len(features), self.value)
    models = {"point": Model(0)}
    models.update({f"q{t:.2f}": Model(8-i) for i,t in enumerate(TAUS)})
    _, raw, ordered = predict(models, pd.DataFrame({"x": [0]}))
    assert raw.tolist() == [list(range(8, 0, -1))]
    assert ordered.tolist() == [list(range(1, 9))]


def test_metric_pinball_direction():
    result = metrics([1], [0], np.zeros((1, 8)))
    assert result["rmse"] == 1
    assert result["pinball"]["0.01"] == 0.01
    result = metrics([-1], [0], np.zeros((1, 8)))
    assert result["pinball"]["0.01"] == 0.99
