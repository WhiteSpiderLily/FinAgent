# Data Sources Bugfix Spec

## Problem

Code review found 7 issues in `finagent/sources/eastmoney_src.py` and `finagent/sources/akshare_src.py`. All verified by source tracing.

## Fixes

### Group A: eastmoney_src.py — fund_flow / Sina fallback

#### #1 Sina descending order → reverse to ascending

`_fund_flow_sina` URL uses `asc=0` (descending: newest first). The returned array is newest-first, but `tools.py:get_fund_flow` assumes ascending order — `rows[-20:]` and `rows[-5:]` take the tail, expecting most-recent entries.

When Eastmoney succeeds: klines are chronological (ascending), `[-20:]` = recent 20. Correct.
When Sina fallback fires: descending order, `[-20:]` = oldest 20. **Wrong — user sees stale data labeled as recent.**

**Fix:** reverse the Sina result before returning, matching the ascending-order contract:

```python
arr.reverse()
return arr
```

One line, guaranteed correct regardless of Sina API parameter behavior.

#### #3 bare except → narrow to transport errors only

Current code catches `except Exception` which silently masks parsing bugs (ValueError from `float()`, IndexError from kline format changes) as Sina fallback. Developer never learns the parser broke.

**Fix:** narrow to transport-level errors only:

```python
except (requests.RequestException, json.JSONDecodeError, OSError) as e:
    log.warning("eastmoney fund_flow failed, falling back to Sina: %s", e)
    ...
```

- `requests.RequestException` — network/connection errors (legitimate fallback)
- `json.JSONDecodeError` — API returned error page instead of data (legitimate fallback)
- `OSError` — low-level transport errors
- `ValueError` / `IndexError` (parsing bugs) — NOT caught, will propagate

Add `import logging` and `log = logging.getLogger(__name__)` at module level.

#### #4 Sina non-JSON response → clear error

`t.index("[")` raises `ValueError: substring not found` if Sina returns a rate-limit page or HTML with no JSON array. This propagates from the except handler, giving the user a misleading "substring not found" error.

**Fix:** wrap the JSON extraction in its own try/except:

```python
try:
    arr = json.loads(t[t.index("["):t.rindex("]") + 1])
except (json.JSONDecodeError, ValueError):
    raise DataSourceError("Sina 资金流响应解析失败（可能被限流）")
```

Requires importing `DataSourceError` from `finagent.sources.registry`.

#### #6 urllib → requests

`_fund_flow_sina` uses `urllib.request` + manual `urlopen`, diverging from the codebase's requests-based HTTP convention. Two top-level imports (`json`, `urllib.request`) exist solely for this function.

**Fix:** replace with `requests.get`:

```python
# Remove: import urllib.request
# Add: import requests

r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"}, timeout=15)
t = r.text
```

Keep `import json` — still needed for `json.loads` on Sina's embedded-JSON response.

### Group B: akshare_src.py — company_info

#### #2 Empty DataFrame → guard

`df.iloc[0]` raises `IndexError` when `stock_profile_cninfo` returns an empty DataFrame for a non-existent stock code (e.g., `999999`).

**Fix:** check `df.empty` before accessing:

```python
df = ak.stock_profile_cninfo(symbol=code)
if df.empty:
    return {"code": code, "name": "未知", "industry": "未知", "listing": "N/A", "main_biz": ""}
row = df.iloc[0].to_dict()
```

#### #5 NaN → 'nan' string + #7 double-or dead code

Current code: `str(row.get("所属行业") or "未知") or "未知"` — pandas NaN is truthy, so `NaN or "未知"` evaluates to NaN, and `str(NaN)` = `"nan"`. The second `or "未知"` is dead code (str() always returns truthy).

**Fix:** extract a `_clean` helper that handles None, NaN, and empty strings:

```python
def _clean(val, default: str) -> str:
    """Return str(val) or default if val is None, NaN, or empty."""
    s = str(val) if val is not None else ""
    return s if s and s != "nan" else default
```

Apply to all fields:

```python
"industry": _clean(row.get("所属行业"), "未知"),
"listing": _clean(row.get("上市日期"), "N/A")[:10],
"main_biz": _clean(row.get("主营业务"), "")[:80],
```

The `name` field already uses `row.get("A股简称", "未知")` which handles missing keys but not NaN — apply `_clean` there too for consistency.

## What stays unchanged

- `norm_ticker` accepting any 6-digit code — empty DataFrame guard is the safety net
- Sina `asc=0` URL parameter — `.reverse()` in code is the fix, not API parameter change
- `em_get` wrapper — no changes to the Eastmoney HTTP client
- All other functions in both files

## Existing test updates

Two existing tests break from the code changes and must be updated in the same commit:

### `test_fund_flow_sina_degradation` (line 295)

**Break:** raises `RuntimeError("push2his RemoteDisconnected")` — after narrowing except, `RuntimeError` is no longer caught. `em_get` wraps `EM_SESSION.get()` which raises `requests.RequestException` subtypes after retries.

**Update:** change `em_fail` to raise `requests.ConnectionError("push2his RemoteDisconnected")`. Also: input data should be descending (newest first) to verify the `.reverse()` fix:

```python
fake_sina = [
    {"opendate": "2024-10-21", "trade": "10.8", "netamount": "50000000.0"},
    {"opendate": "2024-10-20", "trade": "10.5", "netamount": "-78984704.36"},
]
```

Assertion: `result[0]["date"] == "2024-10-20"` — verifies reversal to ascending.

### `test_fund_flow_sina_function` (line 312)

**Break:** patches `eastmoney_src.urllib.request.urlopen` — after switching to requests, `urllib` no longer exists.

**Update:** patch `eastmoney_src.requests.get` instead. Mock returns a Response-like object with `.text` attribute:

```python
fake_resp = MagicMock()
fake_resp.text = raw
monkeypatch.setattr(eastmoney_src.requests, "get", lambda *a, **kw: fake_resp)
```

Also: input data should be descending to verify `.reverse()`. Test assertion: `result[0]["opendate"]` is the older date.

## Empty Sina array edge case

If Sina returns `[]` (valid response but no data), `_fund_flow_sina` returns `[]`. `fund_flow` returns `[]`. `tools.py:get_fund_flow` checks `if not rows: return "暂无资金流数据。"`. **No crash, graceful message.** No additional fix needed.

## Testing

### New tests

- `test_fund_flow_sina_reverses_to_ascending` — mock Sina returning descending data, verify result is ascending by date
- `test_fund_flow_eastmoney_parse_error_not_swallowed` — mock Eastmoney returning malformed kline (e.g., `"date,abc,..."` where `float("abc")` raises ValueError), verify ValueError propagates (not silently falling back to Sina)
- `test_fund_flow_sina_non_json_raises_clear_error` — mock Sina returning HTML/text, verify DataSourceError with clear message
- `test_company_info_empty_dataframe_returns_defaults` — mock empty DataFrame, verify default dict returned without crash
- `test_company_info_nan_field_returns_default` — mock DataFrame with NaN values, verify "未知" not "nan"

### Updated existing tests

- `test_fund_flow_sina_degradation` — raise `requests.ConnectionError`, descending input data, verify ascending output
- `test_fund_flow_sina_function` — patch `requests.get` instead of `urllib.request.urlopen`, descending input data, verify ascending output

### Full suite

Run full `tests/test_sources.py` + `tests/test_tui.py` — all must pass.
