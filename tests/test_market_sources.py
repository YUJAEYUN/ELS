import pytest

from els.market_sources import parse_naver, parse_yahoo


def payload():
    return {"chart": {"result": [{"meta": {"symbol": "^N225", "instrumentType": "INDEX",
                                           "exchangeTimezoneName": "Asia/Tokyo"},
                                 "timestamp": [1704153600],
                                 "indicators": {"quote": [{"close": [30000]}]}}]}}


def test_yahoo_identity_and_value():
    frame = parse_yahoo(payload(), "^N225")
    assert str(frame.iloc[0]["date"].date()) == "2024-01-02"
    assert frame.iloc[0]["value"] == 30000
    with pytest.raises(ValueError, match="identity"):
        parse_yahoo(payload(), "^HSCE")


def test_yahoo_empty_response_is_failure():
    body = payload()
    body["chart"]["result"][0]["timestamp"] = []
    with pytest.raises(ValueError):
        parse_yahoo(body, "^N225")


def test_naver_uses_close_not_open():
    content = '<?xml version="1.0" encoding="EUC-KR"?><protocol><chartdata symbol="KPI200" name="코스피200"><item data="20240102|100|110|90|105|500" /></chartdata></protocol>'.encode("euc-kr")
    assert parse_naver(content, "KPI200").iloc[0]["value"] == 105
    with pytest.raises(ValueError):
        parse_naver(content, "KOSPI")
