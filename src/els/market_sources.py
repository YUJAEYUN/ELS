"""Experimental public chart adapters; no entitlement or historical coverage guarantee."""
import ast
import xml.etree.ElementTree as ET
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from els.sources import normalize


def parse_yahoo(body, symbol):
    chart = body.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError("Yahoo returned no chart")
    data = chart["result"][0]
    meta = data["meta"]
    if meta.get("symbol") != symbol or meta.get("instrumentType") != "INDEX":
        raise ValueError("Unexpected Yahoo instrument identity")
    dates = pd.to_datetime(data.get("timestamp", []), unit="s", utc=True)
    dates = dates.tz_convert(meta["exchangeTimezoneName"]).tz_localize(None).normalize()
    closes = data["indicators"]["quote"][0].get("close", [])
    if len(dates) == 0 or len(dates) != len(closes):
        raise ValueError("Empty or mismatched Yahoo observations")
    return normalize(pd.DataFrame({"date": dates, "value": closes}))


def parse_naver(content, symbol):
    if isinstance(content, bytes):
        content = content.decode("euc-kr")
    root = ET.fromstring(content)
    chart = root.find("chartdata")
    if chart is None or chart.get("symbol") != symbol:
        raise ValueError("Unexpected Naver instrument identity")
    rows = []
    for item in chart.findall("item"):
        fields = item.attrib["data"].split("|")
        if len(fields) != 6:
            raise ValueError("Unexpected Naver row format")
        rows.append({"date": pd.to_datetime(fields[0], format="%Y%m%d"), "value": fields[4]})
    if not rows:
        raise ValueError("Empty Naver observations")
    return normalize(pd.DataFrame(rows))


def fetch_market(spec, start, end):
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("Invalid market date range")
    symbol = spec["symbol"]
    with requests.Session() as session:
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"], respect_retry_after_header=True)
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers["User-Agent"] = "ELS-Research/0.1"
        try:
            if spec["source"] == "yahoo":
                response = session.get(
                    "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe=""),
                    params={"period1": int(start_date.tz_localize("UTC").timestamp()),
                            "period2": int((end_date + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
                            "interval": "1d"}, timeout=(10, 60))
                response.raise_for_status()
                result = parse_yahoo(response.json(), symbol)
            elif spec["source"] == "naver_history":
                if symbol != "KPI200":
                    raise ValueError("Only verified KPI200 mapping supported")
                response = session.get("https://api.finance.naver.com/siseJson.naver",
                                       params={"symbol": symbol, "requestType": 1,
                                               "startTime": start_date.strftime("%Y%m%d"),
                                               "endTime": end_date.strftime("%Y%m%d"),
                                               "timeframe": "day"}, timeout=(10, 60))
                response.raise_for_status()
                result = parse_naver_history(response.content)
            elif spec["source"] == "naver":
                if symbol != "KPI200":
                    raise ValueError("Only verified KPI200 mapping supported")
                response = session.get("https://fchart.stock.naver.com/sise.nhn",
                                       params={"symbol": symbol, "timeframe": "day", "count": 10000,
                                               "requestType": 0}, timeout=(10, 60))
                response.raise_for_status()
                result = parse_naver(response.content, symbol)
            else:
                raise ValueError("Unknown market source")
        except requests.RequestException as exc:
            raise RuntimeError(f"{spec['source']} request failed ({type(exc).__name__})") from None
    result = result.loc[result["date"].between(start_date, end_date)].reset_index(drop=True)
    if result["value"].dropna().empty or (result["value"].dropna() <= 0).any():
        raise ValueError("No usable positive index levels in requested range")
    return result


def parse_naver_history(content):
    """Naver date-range endpoint returns a Python-literal list, not strict JSON."""
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    rows = ast.literal_eval(content.strip())  # Never execute provider text.
    expected = ["날짜", "시가", "고가", "저가", "종가", "거래량", "외국인소진율"]
    if not isinstance(rows, list) or len(rows) < 2 or rows[0] != expected:
        raise ValueError("Unexpected Naver history header or empty data")
    if any(not isinstance(row, list) or len(row) != 7 for row in rows[1:]):
        raise ValueError("Unexpected Naver history row")
    frame = pd.DataFrame(rows[1:], columns=expected)
    return normalize(pd.DataFrame({"date": pd.to_datetime(frame["날짜"], format="%Y%m%d"),
                                   "value": frame["종가"]}))
