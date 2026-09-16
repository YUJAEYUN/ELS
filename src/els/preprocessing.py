"""Friday close snapshots, available for a batch after all Friday markets close."""
import pandas as pd

from els.sources import normalize


def weekly(frame, start, as_of, lag_days=0, stale_weeks=3):
    start, as_of = pd.Timestamp(start), pd.Timestamp(as_of)
    if start > as_of or lag_days < 0 or stale_weeks < 1:
        raise ValueError("Invalid date range, lag or freshness threshold")
    frame = normalize(frame)
    # Missing observations do not erase the last known value.
    frame = frame.dropna(subset=["value"]).copy()
    frame["available_at"] = frame["date"] + pd.to_timedelta(lag_days, unit="D")
    frame = frame.loc[frame["available_at"] <= as_of]
    calendar = pd.DataFrame({"week": pd.date_range(start, as_of, freq="W-FRI")})
    if frame.empty:
        result = calendar.assign(value=float("nan"), observed_at=pd.NaT, available_at=pd.NaT)
    else:
        result = pd.merge_asof(
            calendar, frame.rename(columns={"date": "observed_at"}),
            left_on="week", right_on="available_at", direction="backward")
    # merge_asof implements LOCF without ever backfilling leading missing values.
    result["age_days"] = (result["week"] - result["observed_at"]).dt.days
    result["stale"] = result["value"].isna() | (result["age_days"] >= stale_weeks * 7)
    result["locf"] = result["value"].notna() & (result["age_days"] >= 7)
    return result.set_index("week")


def transform(values, kind):
    if kind == "level":
        return values
    if kind == "difference":
        return values.diff()
    if kind == "return":
        if (values.dropna() <= 0).any():
            raise ValueError("Return transformation requires positive levels")
        return values.pct_change(fill_method=None)
    raise ValueError("Unknown transform")
