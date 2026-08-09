from unittest.mock import patch, MagicMock
from finagent.sources import _emclient


def test_em_get_applies_session(monkeypatch):
    mock_session = MagicMock()
    mock_session.get.return_value = MagicMock(status_code=200)
    monkeypatch.setattr(_emclient, "EM_SESSION", mock_session)
    monkeypatch.setattr(_emclient, "_em_last_call", [0.0])
    _emclient.em_get("https://example.com", params={"a": "1"})
    mock_session.get.assert_called_once()


def test_em_get_throttles(monkeypatch):
    import time
    mock_session = MagicMock()
    mock_session.get.return_value = MagicMock(status_code=200)
    monkeypatch.setattr(_emclient, "EM_SESSION", mock_session)
    monkeypatch.setattr(_emclient, "_em_last_call", [time.time()])
    slept = []
    monkeypatch.setattr(_emclient.time, "sleep", lambda s: slept.append(s))
    _emclient.em_get("https://example.com")
    assert len(slept) == 1
    assert slept[0] >= 1.0


def test_eastmoney_datacenter_returns_rows(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"result": {"data": [{"a": 1}, {"a": 2}]}}
    monkeypatch.setattr(_emclient, "em_get", lambda *a, **kw: fake_resp)
    rows = _emclient.eastmoney_datacenter("RPT_TEST")
    assert rows == [{"a": 1}, {"a": 2}]


def test_eastmoney_datacenter_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"result": None}
    monkeypatch.setattr(_emclient, "em_get", lambda *a, **kw: fake_resp)
    assert _emclient.eastmoney_datacenter("RPT_TEST") == []
