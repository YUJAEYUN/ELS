"""Project ETC variant: overlapping pair counts, non-overlapping replacements."""
from collections import Counter
from math import ceil

import numpy as np
import pandas as pd


def binarize(values, kind):
    values = pd.Series(values, dtype=float)
    if values.empty or not np.isfinite(values).all():
        raise ValueError("Binarization requires finite nonempty values")
    if kind == "level":
        threshold = values.expanding().median()
    elif kind in {"return", "difference"}:
        threshold = 0
    else:
        raise ValueError("Unknown variable kind")
    return (values > threshold).astype(int)


def nsrps(sequence):
    """Return iteration count and trace; ties use the first pair encountered."""
    sequence = list(sequence)
    if not sequence or any(not isinstance(x, (int, np.integer)) or x < 0 for x in sequence):
        raise ValueError("Sequence must contain nonnegative integer symbols")
    next_symbol = int(max(sequence)) + 1
    trace = []
    while len(set(sequence)) > 1:
        counts = Counter(zip(sequence, sequence[1:]))
        pair = max(counts, key=counts.get)  # insertion order gives leftmost tie
        result, i = [], 0
        while i < len(sequence):
            if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                result.append(next_symbol)
                i += 2
            else:
                result.append(sequence[i])
                i += 1
        trace.append({"pair": [int(x) for x in pair], "sequence": [int(x) for x in result]})
        sequence = result
        next_symbol += 1
    return len(trace), trace


def cutoff(scores, fallback_fraction=0.3, elbow_threshold=0.1):
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or (np.diff(scores) > 0).any():
        raise ValueError("Scores must be finite and descending")
    if not 0 < fallback_fraction <= 1 or not 0 <= elbow_threshold <= 1:
        raise ValueError("Invalid cutoff settings")
    if len(scores) >= 3 and scores[0] > scores[-1]:
        x = np.linspace(0, 1, len(scores))
        y = (scores - scores[-1]) / (scores[0] - scores[-1])
        distance = 1 - x - y
        elbow = int(np.argmax(distance))
        if 0 < elbow < len(scores) - 1 and distance[elbow] >= elbow_threshold:
            return elbow + 1, "elbow"
    return max(1, ceil(len(scores) * fallback_fraction)), "top_fraction"


def select_features(train, kinds, min_samples=104):
    """Caller must supply only training-fold rows; never pass validation/test rows."""
    if not isinstance(train.index, pd.DatetimeIndex) or not train.index.is_monotonic_increasing or train.index.has_duplicates:
        raise ValueError("Expected unique chronological dates")
    if train.empty or train.columns.has_duplicates or set(train.columns) != set(kinds):
        raise ValueError("Provide a variable kind for every unique column")
    first = [train[c].first_valid_index() for c in train]
    if any(v is None for v in first):
        raise ValueError("A variable has no observations")
    common = train.loc[max(first):]
    if len(common) < min_samples or not np.isfinite(common.to_numpy(dtype=float)).all():
        raise ValueError("Insufficient common history or internal missing values")
    if len(common) > 1 and not (common.index.to_series().diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError("ETC comparison requires consecutive weekly rows")
    # Calculate expanding medians over each variable's available TRAINING history,
    # then compare equal-length sequences over the common interval.
    scores = {}
    for col in train:
        history = train[col].loc[train[col].first_valid_index():]
        symbols = binarize(history, kinds[col]).loc[common.index]
        scores[col] = nsrps(symbols)[0]
    ordered = sorted(scores, key=lambda c: (-scores[c], c))
    n, method = cutoff([scores[c] for c in ordered])
    return {"selected": ordered[:n], "scores": {c: scores[c] for c in ordered},
            "normalized_scores": {c: scores[c] / (len(common) - 1) for c in ordered},
            "method": method, "samples": len(common),
            "common_start": str(common.index[0].date()), "train_end": str(common.index[-1].date()),
            "parameters": {"fallback_fraction": 0.3, "elbow_threshold": 0.1}}
