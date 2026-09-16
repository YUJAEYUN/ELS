import json
import re
from pathlib import Path

import pandas as pd

from els.preprocessing import transform, weekly
from els.sources import fetch


def run(config_path, as_of, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    specs = config["series"]
    names = [s["id"] for s in specs]
    if not names or len(set(names)) != len(names):
        raise ValueError("Series IDs must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", n) for n in names):
        raise ValueError("Unsafe series ID")
    features, quality, raw_frames = {}, {}, {}
    for spec in specs:
        raw = fetch(spec, config["start"], as_of, config_path.parent)
        table = weekly(raw, config["start"], as_of,
                       spec.get("availability_lag_days", 0), config.get("stale_weeks", 3))
        features[spec["id"]] = transform(table["value"], spec.get("transform", "level"))
        quality[spec["id"]] = table.drop(columns="value")
        raw_frames[spec["id"]] = raw
    # Publish only after all sources succeeded. No silently successful partial batch.
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in raw_frames.items():
        raw.to_csv(output / (name + "_raw.csv"), index=False)
    panel = pd.DataFrame(features)
    panel.to_csv(output / "weekly.csv")
    pd.concat(quality, names=["series", "week"]).to_csv(output / "quality.csv")
    metadata = {
        "as_of": as_of, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": config, "rows": len(panel),
        "data_kind": config.get("data_kind", "external"),
        "warning": "Current vintage; not point-in-time backtest certified.",
        "coverage": {n: {"first": str(r["date"].min()), "last": str(r["date"].max()),
                         "valid_observations": int(r["value"].notna().sum())}
                     for n, r in raw_frames.items()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return panel
