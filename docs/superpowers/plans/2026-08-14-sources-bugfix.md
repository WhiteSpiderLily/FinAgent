# Data Sources Bugfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 verified issues in eastmoney_src.py and akshare_src.py — Sina descending order, empty DataFrame crash, bare except, non-JSON handling, NaN display, urllib divergence, dead code.

**Architecture:** Two independent groups: Group A fixes fund_flow/Sina fallback in eastmoney_src.py (4 issues), Group B fixes company_info in akshare_src.py (3 issues). Each group is one task with its own test cycle.

**Tech Stack:** Python 3.11, pytest, requests, akshare, pandas

## Global Constraints

- `.venv` is the project virtualenv — run tests via `.venv/bin/python -m pytest`
- Tests run sequentially (no parallel — memory)
- No stubs
- `em_get` wraps `EM_SESSION.get()` which raises `requests.RequestException` subtypes after retries
- `DataSourceError` is importable from `finagent.sources.registry`
- `_fund_flow_sina` Sina response is embedded JSON (not clean JSON response) — manual bracket extraction required

**Spec:** `docs/superpowers/specs/2026-08-14-sources-bugfix-design.md`

---

### Task 1: Fix fund_flow / Sina fallback (eastmoney_src.py)

**Files:**
- Modify: `finagent/sources/eastmoney_src.py` (lines 1-2 imports, lines 93-143 fund_flow + _fund_flow_sina)
- Modify: `tests/test_sources.py` (lines 295-326 existing tests, add new tests)

**Interfaces:**
- Consumes: `em_get` from `_emclient`, `requests`, `json`, `DataSourceError` from `registry`, `UA` from `_emclient`
- Produces: `_fund_flow_sina(code) -> list[dict]` (now ascending order, requests-based, raises DataSourceError on parse failure), `fund_flow(code) -> list[dict]` (narrowed except)

- [ ] **Step 1: Write failing test — Sina reverses to ascending**

Add to `tests/test_sources.py` after `test_fund_flow_sina_function`:

```python
def test_fund_flow_sina_reverses_to_ascending(monkeypatch):
    """_fund_flow_sina returns ascending order even when Sina API sends descending."""
    raw = json.dumps([
        {"opendate": "2024-10-22", "netamount": "3000000"},
        {"opendate": "2024-10-21", "netamount": "2000000"},
        {"opendate": "2024-10-20", "netamount": "1000000"},
    ])
    fake_resp = MagicMock()
    fake_resp.text = raw
    monkeypatch.setattr(eastmoney_src.requests, "get", lambda *a, **kw: fake_resp)
    result = eastmoney_src._fund_flow_sina("002415")
    assert result[0]["opendate"] == "2024-10-20"
    assert result[2]["opendate"] == "2024-10-22"
```

- [ ] **Step 2: Write failing test — Eastmoney parse error not swallowed**

```python
def test_fund_flow_eastmoney_parse_error_not_swallowed(monkeypatch):
    """Malformed kline data raises ValueError, not silent Sina fallback."""
    class FakeResponse:
        def json(self):
            return {"data": {"klines": ["2024-10-20,abc,def,ghi,jkl,mno"]}}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr(eastmoney_src, "_fund_flow_sina", lambda code: [])
    with pytest.raises(ValueError):
        eastmoney_src.fund_flow("002415")
```

- [ ] **Step 3: Write failing test — Sina non-JSON raises clear error**

```python
def test_fund_flow_sina_non_json_raises_clear_error(monkeypatch):
    """Sina returning HTML/limit-message raises DataSourceError, not ValueError."""
    fake_resp = MagicMock()
    fake_resp.text = "<html>rate limited</html>"
    monkeypatch.setattr(eastmoney_src.requests, "get", lambda *a, **kw: fake_resp)
    with pytest.raises(DataSourceError, match="资金流"):
        eastmoney_src._fund_flow_sina("002415")
```

- [ ] **Step 4: Run new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sources.py::test_fund_flow_sina_reverses_to_ascending tests/test_sources.py::test_fund_flow_eastmoney_parse_error_not_swallowed tests/test_sources.py::test_fund_flow_sina_non_json_raises_clear_error -v`
Expected: ALL FAIL — `requests` not imported, `.reverse()` not added, except too broad, DataSourceError not raised.

- [ ] **Step 5: Update imports in eastmoney_src.py**

Replace lines 1-2:

```python
import json
import urllib.request
```

With:

```python
import json
import logging

import requests

from finagent.sources.registry import DataSourceError
```

Add after imports (after line 5, the `_ticker` import):

```python
log = logging.getLogger(__name__)
```

- [ ] **Step 6: Fix `_fund_flow_sina` — use requests + reverse + error handling**

Replace the entire `_fund_flow_sina` function (lines 93-102) with:

```python
def _fund_flow_sina(stock_code: str) -> list[dict]:
    """新浪个股资金流（push2his 降级源）。返回升序（oldest-first）。"""
    pre = get_prefix(stock_code) + stock_code
    url = (f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"MoneyFlow.ssl_qsfx_zjlrqs?page=1&num=60&sort=opendate&asc=0&daima={pre}")
    r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"}, timeout=15)
    t = r.text
    try:
        arr = json.loads(t[t.index("["):t.rindex("]") + 1])
    except (json.JSONDecodeError, ValueError):
        raise DataSourceError("Sina 资金流响应解析失败（可能被限流）")
    arr.reverse()
    return arr
```

- [ ] **Step 7: Fix `fund_flow` — narrow except**

Replace the `except Exception:` line (line 133) and the body with:

```python
    except (requests.RequestException, json.JSONDecodeError, OSError) as e:
        log.warning("eastmoney fund_flow failed, falling back to Sina: %s", e)
        raw = _fund_flow_sina(code)
        return [{
            "date": x.get("opendate", ""),
            "main_net": float(x.get("netamount") or 0),
            "small_net": None,
            "mid_net": None,
            "large_net": None,
            "super_net": None,
            "source": "sina",
        } for x in raw]
```

- [ ] **Step 8: Update existing `test_fund_flow_sina_degradation`**

Find in `tests/test_sources.py` (line 295):

```python
def test_fund_flow_sina_degradation(monkeypatch):
    """push2his 失败时降级到新浪，返回主力净额，四档 None，带 source 标注。"""
    def em_fail(*a, **kw):
        raise RuntimeError("push2his RemoteDisconnected")
    fake_sina = [{"opendate": "2024-10-20", "trade": "10.5", "netamount": "-78984704.36"},
                 {"opendate": "2024-10-21", "trade": "10.8", "netamount": "50000000.0"}]
    monkeypatch.setattr(eastmoney_src, "em_get", em_fail)
    monkeypatch.setattr(eastmoney_src, "_fund_flow_sina", lambda code: fake_sina)
    result = eastmoney_src.fund_flow("002415")
    assert len(result) == 2
    assert result[0]["date"] == "2024-10-20"
    assert result[0]["main_net"] == -78984704.36
    assert result[0]["super_net"] is None
    assert result[0]["large_net"] is None
    assert result[0]["source"] == "sina"
```

Replace with:

```python
def test_fund_flow_sina_degradation(monkeypatch):
    """push2his 失败时降级到新浪，返回主力净额，四档 None，带 source 标注。"""
    def em_fail(*a, **kw):
        raise requests.ConnectionError("push2his RemoteDisconnected")
    # Sina returns descending (newest first); fund_flow passes through as-is
    fake_sina = [{"opendate": "2024-10-21", "trade": "10.8", "netamount": "50000000.0"},
                 {"opendate": "2024-10-20", "trade": "10.5", "netamount": "-78984704.36"}]
    monkeypatch.setattr(eastmoney_src, "em_get", em_fail)
    monkeypatch.setattr(eastmoney_src, "_fund_flow_sina", lambda code: fake_sina)
    result = eastmoney_src.fund_flow("002415")
    assert len(result) == 2
    assert result[0]["date"] == "2024-10-21"
    assert result[0]["main_net"] == 50000000.0
    assert result[0]["super_net"] is None
    assert result[0]["large_net"] is None
    assert result[0]["source"] == "sina"
```

- [ ] **Step 9: Update existing `test_fund_flow_sina_function`**

Find in `tests/test_sources.py` (line 312):

```python
def test_fund_flow_sina_function(monkeypatch):
    """_fund_flow_sina 从新浪 API 拉取并解析资金流。"""
    import json
    raw = json.dumps([
        {"opendate": "2024-10-20", "trade": "10.5", "netamount": "-1000000"},
        {"opendate": "2024-10-21", "trade": "10.8", "netamount": "2000000"},
    ])
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.read.return_value = raw.encode("utf-8")
    monkeypatch.setattr(eastmoney_src.urllib.request, "urlopen",
                        lambda req, timeout: fake_resp)
    result = eastmoney_src._fund_flow_sina("002415")
    assert len(result) == 2
    assert result[0]["opendate"] == "2024-10-20"
    assert result[0]["netamount"] == "-1000000"
```

Replace with:

```python
def test_fund_flow_sina_function(monkeypatch):
    """_fund_flow_sina 从新浪 API 拉取并解析资金流，返回升序。"""
    raw = json.dumps([
        {"opendate": "2024-10-21", "trade": "10.8", "netamount": "2000000"},
        {"opendate": "2024-10-20", "trade": "10.5", "netamount": "-1000000"},
    ])
    fake_resp = MagicMock()
    fake_resp.text = raw
    monkeypatch.setattr(eastmoney_src.requests, "get", lambda *a, **kw: fake_resp)
    result = eastmoney_src._fund_flow_sina("002415")
    assert len(result) == 2
    assert result[0]["opendate"] == "2024-10-20"
    assert result[0]["netamount"] == "-1000000"
```

- [ ] **Step 10: Add missing imports to test file**

Add these imports to the top of `tests/test_sources.py` (after existing imports):

```python
import json
import requests

from finagent.sources.registry import DataSourceError
```

`pytest`, `MagicMock`, `pd`, `np`, `eastmoney_src`, `akshare_src` are already imported.

- [ ] **Step 11: Run all fund_flow tests**

Run: `.venv/bin/python -m pytest tests/test_sources.py -k "fund_flow" -v`
Expected: ALL PASS

- [ ] **Step 12: Commit**

```bash
git add finagent/sources/eastmoney_src.py tests/test_sources.py
git commit -m "fix: fund_flow Sina fallback — ascending order, narrowed except, requests, clear errors"
```

---

### Task 2: Fix company_info (akshare_src.py)

**Files:**
- Modify: `finagent/sources/akshare_src.py` (lines 57-67 company_info)
- Modify: `tests/test_sources.py` (add new tests)

**Interfaces:**
- Consumes: `ak.stock_profile_cninfo`, `norm_ticker`
- Produces: `company_info(code) -> dict` (now handles empty DataFrame and NaN)

- [ ] **Step 1: Write failing test — empty DataFrame returns defaults**

Add to `tests/test_sources.py`:

```python
def test_company_info_empty_dataframe_returns_defaults(monkeypatch):
    """Empty DataFrame from cninfo returns default dict without crash."""
    import pandas as pd
    monkeypatch.setattr(akshare_src.ak, "stock_profile_cninfo",
                        lambda symbol: pd.DataFrame())
    result = akshare_src.company_info("999999")
    assert result["code"] == "999999"
    assert result["name"] == "未知"
    assert result["industry"] == "未知"
    assert result["listing"] == "N/A"
```

- [ ] **Step 2: Write failing test — NaN field returns default**

```python
def test_company_info_nan_field_returns_default(monkeypatch):
    """NaN values in DataFrame fields return defaults, not 'nan'."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame([{
        "A股简称": "测试",
        "所属行业": np.nan,
        "上市日期": np.nan,
        "主营业务": "主营业务文本",
    }])
    monkeypatch.setattr(akshare_src.ak, "stock_profile_cninfo", lambda symbol: df)
    result = akshare_src.company_info("002415")
    assert result["industry"] == "未知"
    assert result["listing"] == "N/A"
    assert result["name"] == "测试"
    assert "主营业务文本" in result["main_biz"]
```

- [ ] **Step 3: Run new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sources.py::test_company_info_empty_dataframe_returns_defaults tests/test_sources.py::test_company_info_nan_field_returns_default -v`
Expected: FAIL — IndexError on empty DataFrame, 'nan' on NaN fields.

- [ ] **Step 4: Add `_clean` helper + empty guard to company_info**

Replace the `company_info` function in `finagent/sources/akshare_src.py` (lines 57-67) with:

```python
def _clean(val, default: str) -> str:
    """Return str(val) or default if val is None, NaN, or empty."""
    s = str(val) if val is not None else ""
    return s if s and s != "nan" else default


def company_info(stock_code: str) -> dict:
    code = norm_ticker(stock_code)
    df = ak.stock_profile_cninfo(symbol=code)
    if df.empty:
        return {"code": code, "name": "未知", "industry": "未知", "listing": "N/A", "main_biz": ""}
    row = df.iloc[0].to_dict()
    return {
        "code": code,
        "name": _clean(row.get("A股简称"), "未知"),
        "industry": _clean(row.get("所属行业"), "未知"),
        "listing": _clean(row.get("上市日期"), "N/A")[:10],
        "main_biz": _clean(row.get("主营业务"), "")[:80],
    }
```

- [ ] **Step 5: Run new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sources.py::test_company_info_empty_dataframe_returns_defaults tests/test_sources.py::test_company_info_nan_field_returns_default -v`
Expected: PASS

- [ ] **Step 6: Run all company_info tests**

Run: `.venv/bin/python -m pytest tests/test_sources.py -k "company_info" -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add finagent/sources/akshare_src.py tests/test_sources.py
git commit -m "fix: company_info — empty DataFrame guard, NaN handling, remove dead code"
```

---

### Task 3: Full test suite verification

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/`
Expected: ALL PASS
