import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from els.pipeline import run

INDICES = ["kospi200", "sp500", "eurostoxx50", "nikkei225", "hscei"]


def demo(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    rng = np.random.default_rng(42)
    specs = []
    for name in INDICES:
        values = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, len(dates))))
        values[200:220] = np.nan
        pd.DataFrame({"date": dates, "value": values}).to_csv(root / (name + ".csv"), index=False)
        specs.append({"id": name, "source": "csv", "path": name + ".csv", "transform": "return"})
    config = {"start": "2020-01-01", "data_kind": "synthetic", "stale_weeks": 3, "series": specs}
    path = root / "demo.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="ELS weekly research data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="Generate synthetic data and run offline")
    d.add_argument("--output", default="data/demo")
    b = commands.add_parser("batch")
    b.add_argument("--config", required=True)
    b.add_argument("--as-of", required=True, help="Completed Friday date (YYYY-MM-DD)")
    b.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        config = demo(Path(args.output) / "input")
        panel = run(config, "2025-12-31", Path(args.output) / "result")
        print("SYNTHETIC DEMO - not real index data")
    else:
        panel = run(args.config, args.as_of, args.output)
    print(f"Weekly panel: {panel.shape[0]} rows x {panel.shape[1]} series")


if __name__ == "__main__":
    main()
