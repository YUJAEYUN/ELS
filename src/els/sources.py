"""Source adapters. Never include authenticated URLs in errors."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def normalize(frame):
    if not {"date", "value"}.issubset(frame.columns):
        raise ValueError("Source requires date,value columns")
    result = frame[["date", "value"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result["date"].isna().any() or result["date"].duplicated().any():
        raise ValueError("Missing or duplicate observation dates")
    result["value"] = pd.to_numeric(result["value"].replace(".", np.nan), errors="raise")
    if np.isinf(result["value"].dropna()).any():
        raise ValueError("Infinite observation values")
    return result.sort_values("date").reset_index(drop=True)


def fetch(spec, start, end, base):
    if spec["source"] == "csv":
        return normalize(pd.read_csv(Path(base) / spec["path"]))
    if spec["source"] != "fred":
        raise ValueError("Unsupported source")
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY is required")
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        try:
            response = session.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"api_key": key, "series_id": spec["series_id"],
                        "file_type": "json", "observation_start": start,
                        "observation_end": end, "limit": 100000},
                timeout=(10, 60))
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError):
            raise RuntimeError("FRED request failed; check credentials/network") from None
    observations = body.get("observations", [])
    if int(body.get("count", len(observations))) > len(observations):
        raise ValueError("FRED response truncated; pagination required")
    return normalize(pd.DataFrame(observations, columns=["date", "value"]))
