# 数据源分层架构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `finagent/tools.py` 的 akshare 直调重构为三层架构（Skill / 工具 / DataSource 声明式 fallback registry），支持 8 个核心端点 + 3 个基础设施函数，最小闭环覆盖 A 股财报点评全链路。

**Architecture:** DataSource 层用声明式 `REGISTRY` dict + `fetch()` 函数实现多源 fallback；工具层 8 个 `@tool` 调 `fetch()` 并格式化输出；Skill 层在 `prompts.py` 用数据配方指导 agent 组合工具。源函数返回结构化数据，格式化逻辑留在工具层。

**Tech Stack:** Python 3.11+、langchain `@tool`、akshare、requests（腾讯/东财/新浪 HTTP）、pytest

## Global Constraints

- Python `>=3.11`（`pyproject.toml` 约束）
- 所有测试 mock 底层 HTTP / akshare，零真实网络调用（CI 兼容）
- 源函数返回结构化数据（dict / list / DataFrame），不返回格式化 str
- 格式化逻辑在工具层（`tools.py`），源函数不重复格式化
- `fetch()` 的 `registry` 作默认参数注入（DI），测试注入自定义 registry
- ticker 归一化统一走 `sources/_ticker.py` 的 `norm_ticker()` / `get_prefix()`
- 东财请求统一走 `sources/_emclient.py` 的 `em_get()` 限流
- 依赖注入：`fetch(endpoint, *, registry=REGISTRY, **kwargs)`
- 不引入 MCP、不引入 mootdx、不引入新框架（YAGNI）
- 对话使用中文；代码自我解释，不依赖注释

## 文件结构

```
finagent/
  sources/
    __init__.py          # 导出 fetch, DataSourceError（供 tools.py import）
    _ticker.py           # norm_ticker, get_prefix, SH_INDEX（从文档 §374/§410 移植）
    _emclient.py         # em_get, eastmoney_datacenter + EM_SESSION/常量（从文档 §506/§540 移植）
    akshare_src.py       # company_info, financials（从 tools.py 迁移纯逻辑）
    tencent_src.py       # valuation（从文档 §616 tencent_quote 移植）
    sina_src.py          # financials 备胎（从文档 §2193 移植）
    eastmoney_src.py     # industry_ranking, research_reports, holder_change,
                         #   dividend_history, fund_flow（从文档移植）
    registry.py          # REGISTRY dict + DataSourceError + fetch
  tools.py               # 8 @tool + 格式化函数（_format_* 保留迁移）
  prompts.py             # 追加 SKILL_RECIPES（skill 层）
  agent.py               # 不变
  config.py / report.py / tui.py / __init__.py / __main__.py  # 不变
tests/
  test_ticker.py         # 新
  test_emclient.py       # 新
  test_sources.py        # 新（akshare/tencent/sina/eastmoney 源函数）
  test_registry.py       # 新
  test_tools.py          # 改（格式化 + mock fetch）
  test_tools_helpers.py  # 改（helper 迁移后路径调整或删除）
  test_agent.py / test_prompts.py / test_report.py / test_tui.py  # 基本不变
```

---

## 阶段 1：基础设施 + 迁移现有（不改 agent 行为）

### Task 1: `sources/_ticker.py` — ticker 归一化 + 市场前缀路由

**Files:**
- Create: `finagent/sources/__init__.py`（空包占位）
- Create: `finagent/sources/_ticker.py`
- Test: `tests/test_ticker.py`

**Interfaces:**
- Produces: `norm_ticker(code: str, stock_only: bool = False) -> str`、`get_prefix(code: str) -> str`（返回小写 `sh`/`sz`/`bj`）、常量 `SH_INDEX`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ticker.py
import pytest
from finagent.sources._ticker import norm_ticker, get_prefix


def test_norm_ticker_plain():
    assert norm_ticker("688017") == "688017"


def test_norm_ticker_prefix_upper():
    assert norm_ticker("SH688017") == "688017"


def test_norm_ticker_suffix():
    assert norm_ticker("600519.SH") == "600519"


def test_norm_ticker_bj():
    assert norm_ticker("bj920982") == "920982"


def test_norm_ticker_seven_digits_rejected():
    with pytest.raises(ValueError):
        norm_ticker("6005190")


def test_norm_ticker_name_rejected():
    with pytest.raises(ValueError):
        norm_ticker("茅台")


def test_norm_ticker_conflict_prefix_suffix():
    with pytest.raises(ValueError):
        norm_ticker("SH000001.SZ")


def test_norm_ticker_stock_only_rejects_index():
    with pytest.raises(ValueError):
        norm_ticker("SH000001", stock_only=True)


def test_norm_ticker_market_mismatch():
    with pytest.raises(ValueError):
        norm_ticker("SZ600519")


def test_get_prefix_sh_stock():
    assert get_prefix("600519") == "sh"


def test_get_prefix_sz_stock():
    assert get_prefix("000001") == "sz"


def test_get_prefix_bj_new():
    assert get_prefix("920982") == "bj"


def test_get_prefix_explicit_passthrough():
    assert get_prefix("sh000001") == "sh"


def test_get_prefix_sh_etf():
    assert get_prefix("510300") == "sh"


def test_get_prefix_sh_index_whitelist():
    assert get_prefix("000300") == "sh"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ticker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finagent.sources._ticker'`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/__init__.py
```

```python
# finagent/sources/_ticker.py
import re

SH_INDEX = {"000300", "000905", "000016", "000688", "000852", "000010"}

_TICKER_RE = re.compile(
    r"^(?:(sh|sz|bj)(\d{6})|(\d{6})(?:\.(sh|sz|bj))?)$", re.IGNORECASE
)


def _natural_market(digits: str) -> str:
    if digits.startswith("92") or digits[:2] in ("43", "83", "87"):
        return "bj"
    if digits[0] in ("5", "6", "9"):
        return "sh"
    return "sz"


def norm_ticker(code: str, stock_only: bool = False) -> str:
    raw = str(code).strip()
    m = _TICKER_RE.match(raw)
    if not m:
        raise ValueError(
            f"无法把 {code!r} 解析为 6 位股票代码；"
            f"支持格式：600519 / SH600519 / 600519.SH"
        )
    digits = m.group(2) or m.group(3)
    market = (m.group(1) or m.group(4) or "").lower()
    if market:
        if digits.startswith("000"):
            if market == "bj":
                raise ValueError(f"{code!r} 市场标识与号段矛盾：000xxx 不属北交所。")
            if stock_only and market == "sh":
                raise ValueError(
                    f"{code!r} 指向沪市指数而非个股，本接口只服务个股。"
                )
        else:
            nat = _natural_market(digits)
            if market != nat:
                raise ValueError(
                    f"{code!r} 市场标识与号段矛盾：{digits} 属 {nat} 市。"
                )
    return digits


def get_prefix(code: str) -> str:
    c = code.lower()
    if c.startswith(("sh", "sz", "bj")):
        return c[:2]
    if c.startswith("92"):
        return "bj"
    if c.startswith(("5", "6", "9")):
        return "sh"
    if c.startswith(("4", "8")):
        return "bj"
    if c in SH_INDEX:
        return "sh"
    return "sz"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ticker.py -v`
Expected: PASS（16 tests）

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/__init__.py finagent/sources/_ticker.py tests/test_ticker.py
git commit -m "feat: add sources/_ticker.py with norm_ticker and get_prefix"
```

---

### Task 2: `sources/_emclient.py` — 东财统一请求入口

**Files:**
- Create: `finagent/sources/_emclient.py`
- Test: `tests/test_emclient.py`

**Interfaces:**
- Consumes: `requests`（标准库依赖，akshare 间接引入）
- Produces: `em_get(url, params, headers, timeout) -> requests.Response`、`eastmoney_datacenter(report_name, columns, filter_str, page_size, sort_columns, sort_types) -> list[dict]`、常量 `UA`、`DATACENTER_URL`、`REPORT_API`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emclient.py
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
    monkeypatch.setattr(_emclient, "em_get", lambda **kw: fake_resp)
    rows = _emclient.eastmoney_datacenter("RPT_TEST")
    assert rows == [{"a": 1}, {"a": 2}]


def test_eastmoney_datacenter_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"result": None}
    monkeypatch.setattr(_emclient, "em_get", lambda **kw: fake_resp)
    assert _emclient.eastmoney_datacenter("RPT_TEST") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emclient.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/_emclient.py
import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_API = "https://reportapi.eastmoney.com/report/list"

EM_SESSION = requests.Session()
EM_SESSION.headers.update({"User-Agent": UA})
try:
    _retry = Retry(
        total=3, connect=3, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    _adapter = HTTPAdapter(max_retries=_retry)
    EM_SESSION.mount("https://", _adapter)
    EM_SESSION.mount("http://", _adapter)
except Exception:
    pass

EM_MIN_INTERVAL = 1.0
_em_last_call = [0.0]


def em_get(url, params=None, headers=None, timeout=15, **kwargs):
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return EM_SESSION.get(url, params=params, headers=headers, timeout=timeout, **kwargs)
    finally:
        _em_last_call[0] = time.time()


def eastmoney_datacenter(report_name, columns="ALL", filter_str="",
                         page_size=50, sort_columns="", sort_types="-1"):
    params = {
        "reportName": report_name, "columns": columns,
        "filter": filter_str, "pageNumber": "1", "pageSize": str(page_size),
        "sortColumns": sort_columns, "sortTypes": sort_types,
        "source": "WEB", "client": "WEB",
    }
    r = em_get(DATACENTER_URL, params=params, timeout=15)
    d = r.json()
    if d.get("result") and d["result"].get("data"):
        return d["result"]["data"]
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emclient.py -v`
Expected: PASS（4 tests）

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/_emclient.py tests/test_emclient.py
git commit -m "feat: add sources/_emclient.py with throttled em_get"
```

---

### Task 3: `sources/akshare_src.py` — 迁移现有 akshare 逻辑

**Files:**
- Create: `finagent/sources/akshare_src.py`
- Test: `tests/test_sources.py`（新建，本任务只测 akshare 部分）

**Interfaces:**
- Consumes: `finagent.sources._ticker.norm_ticker`、`finagent.sources._ticker.get_prefix`、`akshare`、`pandas`
- Produces: `company_info(stock_code: str) -> dict`、`financials(stock_code: str, report_period: str) -> dict`

**说明：** 现有 `tools.py` 的 `get_company_info` / `get_financials` 把"取数 + 格式化"耦合在一起。本任务拆分：取数逻辑迁到 `akshare_src.py` 返回结构化 dict，格式化逻辑（`_format_current_period` / `_format_trend` 等）暂留 `tools.py`（Task 5 接线）。

源函数约定：
- 入口调 `norm_ticker()` 校验（失败抛 ValueError，由 fetch 降级）
- 返回结构化数据，不返回 str
- `_normalize_period` 等 period helper 随迁

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

from finagent.sources import akshare_src


def _fake_company_df():
    return pd.DataFrame({
        "item": ["股票简称", "行业", "总股本", "流通股", "上市时间"],
        "value": ["海康威视", "计算机设备", "91.6亿", "90.5亿", "20100528"],
    })


def test_company_info_success():
    with patch("finagent.sources.akshare_src.ak.stock_individual_basic_info_xq",
               return_value=_fake_company_df()):
        result = akshare_src.company_info("002415")
    assert result["name"] == "海康威视"
    assert result["code"] == "002415"
    assert "计算机" in result["industry"]


def test_company_info_bad_code():
    with pytest.raises(ValueError):
        akshare_src.company_info("123")


def _fake_profit_df():
    return pd.DataFrame([{
        "SECURITY_NAME_ABBR": "海康威视", "REPORT_DATE": "2024-09-30 00:00:00",
        "REPORT_TYPE": "三季报", "NOTICE_DATE": "2024-10-19",
        "OPERATE_INCOME": 100e8, "OPERATE_COST": 60e8,
        "TOTAL_OPERATE_INCOME": 100e8, "TOTAL_OPERATE_INCOME_YOY": 10.0,
        "PARENT_NETPROFIT": 20e8, "PARENT_NETPROFIT_YOY": 15.0,
        "DEDUCT_PARENT_NETPROFIT": 18e8, "DEDUCT_PARENT_NETPROFIT_YOY": 12.0,
        "SALE_EXPENSE": 5e8, "MANAGE_EXPENSE": 3e8, "RESEARCH_EXPENSE": 8e8,
    }])


def _fake_balance_df():
    return pd.DataFrame([{
        "REPORT_DATE": "2024-09-30 00:00:00",
        "TOTAL_ASSETS": 500e8, "TOTAL_LIABILITIES": 200e8,
        "MONETARYFUNDS": 100e8, "ACCOUNTS_RECE": 30e8, "INVENTORY": 40e8,
        "TOTAL_PARENT_EQUITY": 300e8,
    }])


def _fake_cashflow_df():
    return pd.DataFrame([{
        "REPORT_DATE": "2024-09-30 00:00:00",
        "NETCASH_OPERATE": 25e8,
    }])


def test_financials_success():
    with patch("finagent.sources.akshare_src.ak.stock_profit_sheet_by_report_em",
               return_value=_fake_profit_df()), \
         patch("finagent.sources.akshare_src.ak.stock_balance_sheet_by_report_em",
               return_value=_fake_balance_df()), \
         patch("finagent.sources.akshare_src.ak.stock_cash_flow_sheet_by_report_em",
               return_value=_fake_cashflow_df()):
        result = akshare_src.financials("002415", "2024Q3")
    assert result["code"] == "002415"
    assert result["name"] == "海康威视"
    assert result["target_date"] == "2024-09-30"
    assert result["profit_row"]["PARENT_NETPROFIT"] == 20e8
    assert len(result["profit_df"]) == 1


def test_financials_period_not_found():
    with patch("finagent.sources.akshare_src.ak.stock_profit_sheet_by_report_em",
               return_value=_fake_profit_df()), \
         patch("finagent.sources.akshare_src.ak.stock_balance_sheet_by_report_em",
               return_value=_fake_balance_df()), \
         patch("finagent.sources.akshare_src.ak.stock_cash_flow_sheet_by_report_em",
               return_value=_fake_cashflow_df()):
        with pytest.raises(ValueError, match="尚未披露"):
            akshare_src.financials("002415", "2024Q4")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

从现有 `tools.py` 迁移取数逻辑。`akshare_src.py` 不含 `@tool`、不含格式化。

```python
# finagent/sources/akshare_src.py
import os
import re

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("TQDM_DISABLE", "1")

import akshare as ak

from finagent.sources._ticker import norm_ticker, get_prefix

_PERIOD_MAP = {
    "一季报": "03-31", "q1": "03-31",
    "半年报": "06-30", "中报": "06-30", "q2": "06-30",
    "三季报": "09-30", "q3": "09-30",
    "年报": "12-31", "q4": "12-31",
}


def _normalize_period(period: str) -> str:
    period = period.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", period):
        return period
    if re.match(r"^\d{8}$", period):
        return f"{period[:4]}-{period[4:6]}-{period[6:8]}"
    m = re.match(r"^(\d{4})[Qq](\d)$", period)
    if m:
        md = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}[m.group(2)]
        return f"{m.group(1)}-{md}"
    for keyword, md in _PERIOD_MAP.items():
        if keyword in period:
            m = re.match(r"^(\d{4})", period)
            if m:
                return f"{m.group(1)}-{md}"
    raise ValueError(f"无法识别报告期格式: {period}")


def company_info(stock_code: str) -> dict:
    code = norm_ticker(stock_code)
    prefix = get_prefix(code).upper()
    symbol = f"{prefix}{code}"
    df = ak.stock_individual_basic_info_xq(symbol=symbol)
    rows = dict(zip(df["item"], df["value"]))
    name = rows.get("org_short_name_cn", "未知")
    industry_raw = rows.get("affiliate_industry", "")
    industry = (
        industry_raw.get("ind_name", "未知")
        if isinstance(industry_raw, dict)
        else str(industry_raw) or "未知"
    )
    listed_ms = rows.get("listed_date")
    listing = "N/A"
    if listed_ms:
        from datetime import datetime
        listing = datetime.fromtimestamp(int(listed_ms) / 1000).strftime("%Y-%m-%d")
    main_biz = str(rows.get("main_operation_business", ""))[:80]
    return {
        "code": code, "name": name, "industry": industry,
        "listing": listing, "main_biz": main_biz,
    }


def _find_row(df, date_str):
    matches = df[df["REPORT_DATE"].astype(str).str.startswith(date_str)]
    return matches.iloc[0].to_dict() if len(matches) else None


def financials(stock_code: str, report_period: str) -> dict:
    code = norm_ticker(stock_code)
    target_date = _normalize_period(report_period)
    prefix = get_prefix(code).upper()
    symbol = f"{prefix}{code}"

    profit_df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
    balance_df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
    cashflow_df = ak.stock_cash_flow_sheet_by_report_em(symbol=symbol)

    profit_row = _find_row(profit_df, target_date)
    if profit_row is None:
        raise ValueError(f"该报告期({report_period})数据尚未披露或不存在。")
    balance_row = _find_row(balance_df, target_date)
    cashflow_row = _find_row(cashflow_df, target_date)

    return {
        "code": code,
        "name": profit_row.get("SECURITY_NAME_ABBR", code),
        "report_period": report_period,
        "target_date": target_date,
        "profit_row": profit_row,
        "balance_row": balance_row or {},
        "cashflow_row": cashflow_row or {},
        "profit_df": profit_df,
        "balance_df": balance_df,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS（5 tests）

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/akshare_src.py tests/test_sources.py
git commit -m "feat: migrate akshare fetch logic to sources/akshare_src.py"
```

---

### Task 4: `sources/registry.py` — 声明式 fallback + fetch (DI)

**Files:**
- Create: `finagent/sources/registry.py`
- Modify: `finagent/sources/__init__.py`（导出 `fetch`, `DataSourceError`）
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `finagent.sources.akshare_src.company_info`、`finagent.sources.akshare_src.financials`
- Produces: `REGISTRY: dict[str, list[Callable]]`、`DataSourceError`、`fetch(endpoint, *, registry=REGISTRY, **kwargs)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import pytest
from unittest.mock import MagicMock
from finagent.sources.registry import fetch, DataSourceError


def test_fetch_first_source_success():
    fn_a = MagicMock(return_value={"ok": True})
    fn_b = MagicMock()
    reg = {"ep": [fn_a, fn_b]}
    result = fetch("ep", registry=reg, code="002415")
    assert result == {"ok": True}
    fn_b.assert_not_called()


def test_fetch_fallback_on_error():
    fn_a = MagicMock(side_effect=RuntimeError("timeout"))
    fn_b = MagicMock(return_value={"ok": True})
    reg = {"ep": [fn_a, fn_b]}
    result = fetch("ep", registry=reg, code="002415")
    assert result == {"ok": True}


def test_fetch_all_fail_raises():
    fn_a = MagicMock(side_effect=ValueError("bad"))
    fn_b = MagicMock(side_effect=RuntimeError("down"))
    reg = {"ep": [fn_a, fn_b]}
    with pytest.raises(DataSourceError, match="all 2 sources failed"):
        fetch("ep", registry=reg)


def test_fetch_unknown_endpoint():
    with pytest.raises(KeyError):
        fetch("nope", registry={"ep": []})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/registry.py
from collections.abc import Callable

from finagent.sources import akshare_src

REGISTRY: dict[str, list[Callable]] = {
    "company_info": [akshare_src.company_info],
    "financials": [akshare_src.financials],
}


class DataSourceError(Exception):
    pass


def fetch(endpoint, *, registry=REGISTRY, **kwargs):
    sources = registry[endpoint]
    last_err = None
    for fn in sources:
        try:
            return fn(**kwargs)
        except Exception as e:
            last_err = e
    raise DataSourceError(
        f"endpoint '{endpoint}' all {len(sources)} sources failed: {last_err}"
    )
```

```python
# finagent/sources/__init__.py
from finagent.sources.registry import fetch, DataSourceError

__all__ = ["fetch", "DataSourceError"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS（4 tests）

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/registry.py finagent/sources/__init__.py tests/test_registry.py
git commit -m "feat: add declarative fallback registry with DI"
```

---

### Task 5: 重构 `tools.py` — get_company_info / get_financials 调 fetch

**Files:**
- Modify: `finagent/tools.py`（重写 get_company_info / get_financials 为调 fetch，格式化函数保留）
- Modify: `tests/test_tools.py`（mock `fetch` 而非 akshare）
- Delete or modify: `tests/test_tools_helpers.py`（helper 已迁 akshare_src）

**Interfaces:**
- Consumes: `finagent.sources.fetch`、`finagent.sources.DataSourceError`
- Produces: `get_company_info`、`get_financials`（签名不变，行为不变）、`tools` list

**说明：** 这是阶段 1 收尾。重构后 agent 行为不变（工具签名/输出格式一致），只是内部走 fetch → akshare_src。

- [ ] **Step 1: Write the failing test**

改写 `tests/test_tools.py`，mock `finagent.tools.fetch` 而非 `finagent.tools.ak`：

```python
# tests/test_tools.py（关键测试节选，helper 格式化测试保留）
from unittest.mock import patch
from finagent import tools


def test_get_company_info_success():
    fake_data = {
        "code": "002415", "name": "海康威视", "industry": "计算机设备",
        "listing": "2010-05-28", "main_biz": "视频监控产品",
    }
    with patch("finagent.tools.fetch", return_value=fake_data):
        result = tools.get_company_info.invoke({"stock_code": "002415"})
    assert "海康威视" in result
    assert "计算机设备" in result


def test_get_company_info_fetch_error():
    from finagent.sources import DataSourceError
    with patch("finagent.tools.fetch", side_effect=DataSourceError("all failed")):
        result = tools.get_company_info.invoke({"stock_code": "002415"})
    assert "失败" in result or "error" in result.lower()


def test_get_financials_success():
    import pandas as pd
    fake_data = {
        "code": "002415", "name": "海康威视", "report_period": "2024Q3",
        "target_date": "2024-09-30",
        "profit_row": {
            "TOTAL_OPERATE_INCOME": 100e8, "TOTAL_OPERATE_INCOME_YOY": 10.0,
            "PARENT_NETPROFIT": 20e8, "PARENT_NETPROFIT_YOY": 15.0,
            "DEDUCT_PARENT_NETPROFIT": 18e8, "DEDUCT_PARENT_NETPROFIT_YOY": 12.0,
            "OPERATE_INCOME": 100e8, "OPERATE_COST": 60e8,
            "SALE_EXPENSE": 5e8, "MANAGE_EXPENSE": 3e8, "RESEARCH_EXPENSE": 8e8,
        },
        "balance_row": {
            "TOTAL_ASSETS": 500e8, "TOTAL_LIABILITIES": 200e8,
            "MONETARYFUNDS": 100e8, "ACCOUNTS_RECE": 30e8, "INVENTORY": 40e8,
        },
        "cashflow_row": {"NETCASH_OPERATE": 25e8},
        "profit_df": pd.DataFrame([{
            "REPORT_DATE": "2024-09-30", "OPERATE_INCOME": 100e8,
            "OPERATE_COST": 60e8, "PARENT_NETPROFIT": 20e8,
        }]),
        "balance_df": pd.DataFrame([{
            "REPORT_DATE": "2024-09-30", "TOTAL_LIABILITIES": 200e8,
            "TOTAL_ASSETS": 500e8, "TOTAL_PARENT_EQUITY": 300e8,
        }]),
    }
    with patch("finagent.tools.fetch", return_value=fake_data):
        result = tools.get_financials.invoke({"stock_code": "002415", "report_period": "2024Q3"})
    assert "海康威视" in result
    assert "利润表" in result
```

格式化 helper 测试（`_fmt_yi` / `_fmt_yoy` / `_format_current_period` 等）保留在 `test_tools.py`，不受影响。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `get_company_info` 仍直调 akshare，`fetch` 未 import

- [ ] **Step 3: Write minimal implementation**

重写 `finagent/tools.py` 的 `get_company_info` / `get_financials` 为调 `fetch`。所有 `_format_*` / `_fmt_*` / `_compute_*` helper 保留不动。删除 akshare 相关 import 和 `_patch_requests_ua` / `_validate_code` / `_market_prefix`（已由 akshare_src + _ticker 承担）。

```python
# finagent/tools.py（重写后的 @tool 部分）
from langchain_core.tools import tool

from finagent.sources import fetch, DataSourceError


@tool
def get_company_info(stock_code: str) -> str:
    """获取 A股上市公司基本信息：公司简称、所属行业、上市时间、主营业务。

    Args:
        stock_code: 6 位股票代码，如 002415
    """
    try:
        data = fetch("company_info", stock_code=stock_code)
    except (DataSourceError, ValueError) as e:
        return str(e)
    return (
        f"公司: {data['name']}({data['code']})\n"
        f"行业: {data['industry']}\n"
        f"上市时间: {data['listing']}\n"
        f"主营业务: {data['main_biz']}"
    )


@tool
def get_financials(stock_code: str, report_period: str) -> str:
    """获取 A股上市公司指定报告期的财务数据：利润表/资产负债表/现金流量表关键科目（含同比）+ 近 8 期趋势。

    Args:
        stock_code: 6 位股票代码，如 002415
        report_period: 报告期，支持 2024Q3 / 2024三季报 / 2024-09-30
    """
    try:
        data = fetch("financials", stock_code=stock_code, report_period=report_period)
    except (DataSourceError, ValueError) as e:
        return str(e)
    detail = _format_current_period(
        data["profit_row"], data["balance_row"], data["cashflow_row"]
    )
    trend = _format_trend(data["profit_df"], data["balance_df"])
    return f"=== {data['name']}({data['code']}) {data['report_period']} 财务数据 ===\n{detail}\n\n{trend}"


tools = [get_company_info, get_financials]
```

保留的 helper（不动）：`_fmt_yi` / `_fmt_yoy` / `_fmt_trend_val` / `_compute_gross_margin` / `_compute_debt_ratio` / `_compute_net_cash_ratio` / `_format_current_period` / `_format_trend`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS

同时跑全量确保无回归：
Run: `pytest -q`
Expected: PASS（迁移后测试数可能变化，应全绿）

- [ ] **Step 5: Clean up orphaned helpers**

检查 `tools.py` 是否有 `_validate_code` / `_market_prefix` / `_PERIOD_MAP` / `_normalize_period` 等已迁出的孤儿（本任务改动产生的）。删除它们。

检查 `tests/test_tools_helpers.py`：若其测试的 helper 已迁 `akshare_src.py`，迁移对应测试到 `tests/test_sources.py`，删除 `test_tools_helpers.py`。

- [ ] **Step 6: Commit**

```bash
git add finagent/tools.py tests/test_tools.py tests/test_tools_helpers.py
git commit -m "refactor: tools.py call fetch instead of akshare directly"
```

---

## 阶段 2：腾讯源（valuation）

### Task 6: `sources/tencent_src.py` — valuation

**Files:**
- Create: `finagent/sources/tencent_src.py`
- Modify: `tests/test_sources.py`（追加腾讯测试）

**Interfaces:**
- Consumes: `finagent.sources._ticker.norm_ticker`、`finagent.sources._ticker.get_prefix`、`urllib.request`
- Produces: `valuation(stock_code: str) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py 追加
from finagent.sources import tencent_src


_TENCENT_RAW = (
    'v_sz002415="name~海康威视~code~002415~'
    '~51.20~50.00~50.50~'
    '~~~~~~~~~~~~~~~' + '~' * 10 + '~'
    '~9.11~4.24~52.00~49.00~'
    '~' * 5 + '~187040~4.55~300.45~'
    '~' * 3 + '~7.22~410.88~410.88~11.51~258.01~172.01~1.20~~~~314.76";'
)


def test_valuation_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.read.return_value = _TENCENT_RAW.encode("gbk")
    monkeypatch.setattr(
        tencent_src.urllib.request, "urlopen", lambda req, timeout: mock_resp
    )
    result = tencent_src.valuation("002415")
    assert result["name"] == "海康威视"
    assert result["code"] == "002415"
    assert result["price"] == 51.20
    assert result["pe_ttm"] == 300.45
    assert result["pb"] == 11.51
    assert result["mcap_yi"] == 410.88


def test_valuation_stale_flag(monkeypatch):
    raw = _TENCENT_RAW.replace("~187040~", "~0~")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw.encode("gbk")
    monkeypatch.setattr(
        tencent_src.urllib.request, "urlopen", lambda req, timeout: mock_resp
    )
    result = tencent_src.valuation("002415")
    assert result["is_stale"] is True
```

**注意：** `_TENCENT_RAW` 的 `~` 数量需精确（腾讯返回 88 字段，索引须对齐）。实现时按文档 §1.2 的字段索引构造测试数据，确保 `vals[3]`/`vals[39]`/`vals[45]`/`vals[46]` 等对齐。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_valuation_success -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

从文档 §616 `tencent_quote` 适配：单股查询（非批量），返回单个 dict。

```python
# finagent/sources/tencent_src.py
import urllib.request

from finagent.sources._ticker import norm_ticker, get_prefix


def valuation(stock_code: str) -> dict:
    code = norm_ticker(stock_code)
    prefixed = f"{get_prefix(code)}{code}"
    url = "https://qt.gtimg.cn/q=" + prefixed
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    resp = urllib.request.urlopen(req, timeout=10)
    data = resp.read().decode("gbk")

    line = data.strip().rstrip(";")
    vals = line.split('"')[1].split("~")
    if len(vals) < 53:
        raise ValueError(f"腾讯行情数据字段不足: {len(vals)}")

    result = {
        "code": code,
        "name": vals[1],
        "price": float(vals[3]) if vals[3] else 0,
        "last_close": float(vals[4]) if vals[4] else 0,
        "open": float(vals[5]) if vals[5] else 0,
        "change_amt": float(vals[31]) if vals[31] else 0,
        "change_pct": float(vals[32]) if vals[32] else 0,
        "high": float(vals[33]) if vals[33] else 0,
        "low": float(vals[34]) if vals[34] else 0,
        "amount_wan": float(vals[37]) if vals[37] else 0,
        "turnover_pct": float(vals[38]) if vals[38] else 0,
        "pe_ttm": float(vals[39]) if vals[39] else 0,
        "amplitude_pct": float(vals[43]) if vals[43] else 0,
        "float_mcap_yi": float(vals[44]) if vals[44] else 0,
        "mcap_yi": float(vals[45]) if vals[45] else 0,
        "pb": float(vals[46]) if vals[46] else 0,
        "limit_up": float(vals[47]) if vals[47] else 0,
        "limit_down": float(vals[48]) if vals[48] else 0,
        "vol_ratio": float(vals[49]) if vals[49] else 0,
        "pe_static": float(vals[52]) if vals[52] else 0,
    }
    result["is_stale"] = (
        result["amount_wan"] == 0
        and result["price"] == result["last_close"]
        and result["price"] > 0
    )
    if result["is_stale"]:
        result["stale_reason"] = "成交量为 0（停牌 / 未开盘 / 废码）"
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS（含 akshare + tencent）

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/tencent_src.py tests/test_sources.py
git commit -m "feat: add tencent valuation source"
```

---

### Task 7: 工具层加 `get_valuation` + 注册

**Files:**
- Modify: `finagent/sources/registry.py`（REGISTRY 加 valuation）
- Modify: `finagent/tools.py`（加 get_valuation @tool + 格式化）
- Modify: `tests/test_tools.py`（加 get_valuation 测试）

**Interfaces:**
- Consumes: `finagent.sources.tencent_src.valuation`、`finagent.sources.fetch`
- Produces: `get_valuation` @tool、REGISTRY `"valuation"` 端点

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools.py 追加
def test_get_valuation_success():
    fake = {
        "code": "002415", "name": "海康威视", "price": 32.50,
        "pe_ttm": 25.0, "pb": 5.5, "mcap_yi": 3000.0,
        "float_mcap_yi": 2900.0, "turnover_pct": 1.2,
        "change_pct": 3.5, "limit_up": 35.0, "limit_down": 29.0,
        "is_stale": False,
    }
    with patch("finagent.tools.fetch", return_value=fake):
        result = tools.get_valuation.invoke({"stock_code": "002415"})
    assert "海康威视" in result
    assert "PE" in result or "市盈" in result
    assert "3000" in result


def test_get_valuation_stale_warning():
    fake = {
        "code": "002415", "name": "海康威视", "price": 0,
        "pe_ttm": 0, "pb": 0, "mcap_yi": 0, "float_mcap_yi": 0,
        "turnover_pct": 0, "change_pct": 0, "limit_up": 0, "limit_down": 0,
        "is_stale": True, "stale_reason": "停牌",
    }
    with patch("finagent.tools.fetch", return_value=fake):
        result = tools.get_valuation.invoke({"stock_code": "002415"})
    assert "停牌" in result or "stale" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py::test_get_valuation_success -v`
Expected: FAIL — `AttributeError: module 'finagent.tools' has no attribute 'get_valuation'`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/registry.py 追加 import 和端点
from finagent.sources import tencent_src

REGISTRY: dict[str, list[Callable]] = {
    "company_info": [akshare_src.company_info],
    "financials": [akshare_src.financials],
    "valuation": [tencent_src.valuation],
}
```

```python
# finagent/tools.py 追加
@tool
def get_valuation(stock_code: str) -> str:
    """获取 A股实时估值：PE(TTM)/PB/总市值/流通市值/换手率/涨跌停/涨跌幅。

    Args:
        stock_code: 6 位股票代码，如 002415
    """
    try:
        d = fetch("valuation", stock_code=stock_code)
    except (DataSourceError, ValueError) as e:
        return str(e)
    lines = [f"{d['name']}({d['code']}) 实时估值:"]
    lines.append(f"  现价: {d['price']:.2f} ({d['change_pct']:+.2f}%)")
    lines.append(f"  PE(TTM): {d['pe_ttm']:.1f} | PB: {d['pb']:.2f}")
    lines.append(f"  总市值: {d['mcap_yi']:.1f}亿 | 流通: {d['float_mcap_yi']:.1f}亿")
    lines.append(f"  换手率: {d['turnover_pct']:.2f}%")
    lines.append(f"  涨停: {d['limit_up']:.2f} | 跌停: {d['limit_down']:.2f}")
    if d.get("is_stale"):
        lines.append(f"  ⚠️ 疑似停牌/废码: {d.get('stale_reason', '成交量为0')}")
    return "\n".join(lines)


# tools list 追加
tools = [get_company_info, get_financials, get_valuation]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/registry.py finagent/tools.py tests/test_tools.py
git commit -m "feat: add get_valuation tool with tencent source"
```

---

## 阶段 3：东财源（5 端点）

### Task 8: `sources/eastmoney_src.py` — industry_ranking

**Files:**
- Create: `finagent/sources/eastmoney_src.py`
- Modify: `tests/test_sources.py`

**Interfaces:**
- Consumes: `finagent.sources._emclient.em_get`、`finagent.sources._emclient.UA`
- Produces: `industry_ranking(top_n: int = 20) -> dict`（注意：无 stock_code 参数，返回全行业排名）

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py 追加
from finagent.sources import eastmoney_src


def test_industry_ranking_success(monkeypatch):
    fake_json = {
        "data": {"diff": [
            {"f14": "电力设备", "f3": 3.5, "f12": "BK1", "f104": 200, "f105": 50,
             "f140": "宁德时代", "f136": 5.0},
            {"f14": "白酒", "f3": -1.2, "f12": "BK2", "f104": 10, "f105": 90,
             "f140": "贵州茅台", "f136": 0.5},
        ]}
    }
    fake_resp = MagicMock()
    fake_resp.json.return_value = fake_json
    monkeypatch.setattr(eastmoney_src, "em_get", lambda **kw: fake_resp)
    result = eastmoney_src.industry_ranking(top_n=5)
    assert result["total"] == 2
    assert result["top"][0]["name"] == "电力设备"
    assert result["bottom"][-1]["name"] == "白酒"


def test_industry_ranking_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": {"diff": []}}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda **kw: fake_resp)
    result = eastmoney_src.industry_ranking()
    assert result["total"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_industry_ranking_success -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/eastmoney_src.py
from finagent.sources._emclient import em_get, UA


def industry_ranking(top_n: int = 20) -> dict:
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "100", "po": "1", "np": "1",
        "fltt": "2", "invt": "2", "fid": "f3",
        "fs": "m:90+t:2",
        "fields": "f2,f3,f4,f12,f13,f14,f104,f105,f128,f136,f140,f141,f207",
    }
    r = em_get(url, params=params, headers={"User-Agent": UA}, timeout=15)
    items = r.json().get("data", {}).get("diff", [])
    if not items:
        return {"top": [], "bottom": [], "total": 0}
    rows = []
    for i, item in enumerate(items):
        rows.append({
            "rank": i + 1,
            "name": item.get("f14", ""),
            "change_pct": item.get("f3", 0),
            "code": item.get("f12", ""),
            "up_count": item.get("f104", 0),
            "down_count": item.get("f105", 0),
            "leader": item.get("f140", ""),
            "leader_change": item.get("f136", 0),
        })
    return {"top": rows[:top_n], "bottom": rows[-top_n:], "total": len(rows)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/eastmoney_src.py tests/test_sources.py
git commit -m "feat: add eastmoney industry_ranking source"
```

---

### Task 9: `sources/eastmoney_src.py` 追加 research_reports

**Files:**
- Modify: `finagent/sources/eastmoney_src.py`
- Modify: `tests/test_sources.py`

**Interfaces:**
- Consumes: `finagent.sources._emclient.em_get`、`finagent.sources._emclient.REPORT_API`、`finagent.sources._ticker.norm_ticker`
- Produces: `research_reports(stock_code: str, max_pages: int = 3) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py 追加
def test_research_reports_success(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": [{"title": "买入", "orgSName": "中信"}], "TotalPage": 1}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda **kw: fake_resp)
    result = eastmoney_src.research_reports("002415", max_pages=1)
    assert len(result) == 1
    assert result[0]["title"] == "买入"


def test_research_reports_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": [], "TotalPage": 1}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda **kw: fake_resp)
    assert eastmoney_src.research_reports("002415", max_pages=1) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_research_reports_success -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'research_reports'`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/eastmoney_src.py 追加
from finagent.sources._emclient import em_get, UA, REPORT_API
from finagent.sources._ticker import norm_ticker


def research_reports(stock_code: str, max_pages: int = 3) -> list[dict]:
    code = norm_ticker(stock_code, stock_only=True)
    all_records = []
    for page in range(1, max_pages + 1):
        params = {
            "industryCode": "*", "pageSize": "100", "industry": "*",
            "rating": "*", "ratingChange": "*",
            "beginTime": "2000-01-01", "endTime": "2030-01-01",
            "pageNo": str(page), "fields": "", "qType": "0",
            "orgCode": "", "code": code, "rcode": "",
            "p": str(page), "pageNum": str(page), "pageNumber": str(page),
        }
        r = em_get(REPORT_API, params=params,
                   headers={"Referer": "https://data.eastmoney.com/"}, timeout=30)
        d = r.json()
        rows = d.get("data") or []
        if not rows:
            break
        all_records.extend(rows)
        if page >= (d.get("TotalPage", 1) or 1):
            break
    return all_records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/eastmoney_src.py tests/test_sources.py
git commit -m "feat: add eastmoney research_reports source"
```

---

### Task 10: `sources/eastmoney_src.py` 追加 holder_change

**Files:**
- Modify: `finagent/sources/eastmoney_src.py`
- Modify: `tests/test_sources.py`

**Interfaces:**
- Consumes: `finagent.sources._emclient.eastmoney_datacenter`、`finagent.sources._ticker.norm_ticker`
- Produces: `holder_change(stock_code: str, page_size: int = 10) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py 追加
def test_holder_change_success(monkeypatch):
    fake_data = [{
        "END_DATE": "2024-09-30 00:00:00", "HOLDER_NUM": 50000,
        "HOLDER_NUM_CHANGE": -2000, "HOLDER_NUM_RATIO": -3.8,
        "AVG_FREE_SHARES": 1500,
    }]
    monkeypatch.setattr(eastmoney_src, "eastmoney_datacenter", lambda **kw: fake_data)
    result = eastmoney_src.holder_change("002415", page_size=5)
    assert result[0]["date"] == "2024-09-30"
    assert result[0]["holder_num"] == 50000
    assert result[0]["change_ratio"] == -3.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_holder_change_success -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/eastmoney_src.py 追加
from finagent.sources._emclient import eastmoney_datacenter


def holder_change(stock_code: str, page_size: int = 10) -> list[dict]:
    code = norm_ticker(stock_code)
    data = eastmoney_datacenter(
        "RPT_HOLDERNUMLATEST",
        filter_str=f'(SECURITY_CODE="{code}")',
        page_size=page_size,
        sort_columns="END_DATE", sort_types="-1",
    )
    return [{
        "date": str(row.get("END_DATE", ""))[:10],
        "holder_num": row.get("HOLDER_NUM", 0),
        "change_num": row.get("HOLDER_NUM_CHANGE", 0),
        "change_ratio": row.get("HOLDER_NUM_RATIO", 0),
        "avg_shares": row.get("AVG_FREE_SHARES", 0),
    } for row in data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/eastmoney_src.py tests/test_sources.py
git commit -m "feat: add eastmoney holder_change source"
```

---

### Task 11: `sources/eastmoney_src.py` 追加 dividend_history

**Files:**
- Modify: `finagent/sources/eastmoney_src.py`
- Modify: `tests/test_sources.py`

**Interfaces:**
- Produces: `dividend_history(stock_code: str, page_size: int = 20) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py 追加
def test_dividend_history_success(monkeypatch):
    fake_data = [{
        "EX_DIVIDEND_DATE": "2024-06-20 00:00:00", "PRETAX_BONUS_RMB": 0.5,
        "TRANSFER_RATIO": 0, "BONUS_RATIO": 0, "ASSIGN_PROGRESS": "实施",
    }]
    monkeypatch.setattr(eastmoney_src, "eastmoney_datacenter", lambda **kw: fake_data)
    result = eastmoney_src.dividend_history("002415")
    assert result[0]["date"] == "2024-06-20"
    assert result[0]["bonus_rmb"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_dividend_history_success -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/eastmoney_src.py 追加
def dividend_history(stock_code: str, page_size: int = 20) -> list[dict]:
    code = norm_ticker(stock_code)
    data = eastmoney_datacenter(
        "RPT_SHAREBONUS_DET",
        filter_str=f'(SECURITY_CODE="{code}")',
        page_size=page_size,
        sort_columns="EX_DIVIDEND_DATE", sort_types="-1",
    )
    return [{
        "date": str(row.get("EX_DIVIDEND_DATE", ""))[:10],
        "bonus_rmb": row.get("PRETAX_BONUS_RMB", 0),
        "transfer_ratio": row.get("TRANSFER_RATIO", 0),
        "bonus_ratio": row.get("BONUS_RATIO", 0),
        "plan": row.get("ASSIGN_PROGRESS", ""),
    } for row in data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/eastmoney_src.py tests/test_sources.py
git commit -m "feat: add eastmoney dividend_history source"
```

---

### Task 12: `sources/eastmoney_src.py` 追加 fund_flow

**Files:**
- Modify: `finagent/sources/eastmoney_src.py`
- Modify: `tests/test_sources.py`

**Interfaces:**
- Produces: `fund_flow(stock_code: str) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py 追加
def test_fund_flow_success(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "data": {"klines": [
            "2024-10-20,1000000,200000,300000,400000,100000",
            "2024-10-21,-500000,100000,-200000,-300000,-100000",
        ]}
    }
    monkeypatch.setattr(eastmoney_src, "em_get", lambda **kw: fake_resp)
    result = eastmoney_src.fund_flow("002415")
    assert len(result) == 2
    assert result[0]["date"] == "2024-10-20"
    assert result[0]["main_net"] == 1000000
    assert result[1]["main_net"] == -500000


def test_fund_flow_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": {"klines": []}}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda **kw: fake_resp)
    assert eastmoney_src.fund_flow("002415") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_fund_flow_success -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/eastmoney_src.py 追加
from finagent.sources._ticker import get_prefix


def fund_flow(stock_code: str) -> list[dict]:
    code = norm_ticker(stock_code)
    market_code = 1 if get_prefix(code) == "sh" else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": f"{market_code}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "lmt": "120",
    }
    r = em_get(url, params=params,
               headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/",
                        "Origin": "https://quote.eastmoney.com"}, timeout=15)
    klines = r.json().get("data", {}).get("klines", [])
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 7:
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]) if parts[1] != "-" else 0,
                "small_net": float(parts[2]) if parts[2] != "-" else 0,
                "mid_net": float(parts[3]) if parts[3] != "-" else 0,
                "large_net": float(parts[4]) if parts[4] != "-" else 0,
                "super_net": float(parts[5]) if parts[5] != "-" else 0,
            })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/eastmoney_src.py tests/test_sources.py
git commit -m "feat: add eastmoney fund_flow source"
```

---

### Task 13: 工具层加 5 个东财工具 + 注册

**Files:**
- Modify: `finagent/sources/registry.py`（加 5 端点）
- Modify: `finagent/tools.py`（加 5 @tool + 格式化）
- Modify: `tests/test_tools.py`（加 5 工具测试）

**Interfaces:**
- Consumes: `eastmoney_src` 的 5 函数、`fetch`
- Produces: `get_industry_ranking` / `get_research_reports` / `get_holder_change` / `get_dividend_history` / `get_fund_flow`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools.py 追加（5 个工具各 1 个关键测试）
def test_get_industry_ranking():
    fake = {"total": 2, "top": [{"rank": 1, "name": "电力设备", "change_pct": 3.5,
            "up_count": 200, "down_count": 50, "leader": "宁德时代"}],
            "bottom": [{"rank": 2, "name": "白酒", "change_pct": -1.2}]}
    with patch("finagent.tools.fetch", return_value=fake):
        result = tools.get_industry_ranking.invoke({"top_n": 5})
    assert "电力设备" in result


def test_get_research_reports():
    fake = [{"title": "业绩超预期", "publishDate": "2024-10-20", "orgSName": "中信",
             "emRatingName": "买入", "predictThisYearEps": 1.5}]
    with patch("finagent.tools.fetch", return_value=fake):
        result = tools.get_research_reports.invoke({"stock_code": "002415"})
    assert "业绩超预期" in result


def test_get_holder_change():
    fake = [{"date": "2024-09-30", "holder_num": 50000, "change_ratio": -3.8, "avg_shares": 1500}]
    with patch("finagent.tools.fetch", return_value=fake):
        result = tools.get_holder_change.invoke({"stock_code": "002415"})
    assert "50000" in result
    assert "-3.8" in result


def test_get_dividend_history():
    fake = [{"date": "2024-06-20", "bonus_rmb": 0.5, "transfer_ratio": 0,
             "bonus_ratio": 0, "plan": "实施"}]
    with patch("finagent.tools.fetch", return_value=fake):
        result = tools.get_dividend_history.invoke({"stock_code": "002415"})
    assert "0.5" in result


def test_get_fund_flow():
    fake = [{"date": "2024-10-20", "main_net": 1e8, "super_net": 5e7},
            {"date": "2024-10-21", "main_net": -5e7, "super_net": -2e7}]
    with patch("finagent.tools.fetch", return_value=fake):
        result = tools.get_fund_flow.invoke({"stock_code": "002415"})
    assert "主力" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v -k "industry_ranking or research_reports or holder_change or dividend_history or fund_flow"`
Expected: FAIL — 5 个 `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/registry.py 追加
from finagent.sources import eastmoney_src

REGISTRY: dict[str, list[Callable]] = {
    "company_info": [akshare_src.company_info],
    "financials": [akshare_src.financials],
    "valuation": [tencent_src.valuation],
    "industry_ranking": [eastmoney_src.industry_ranking],
    "research_reports": [eastmoney_src.research_reports],
    "holder_change": [eastmoney_src.holder_change],
    "dividend_history": [eastmoney_src.dividend_history],
    "fund_flow": [eastmoney_src.fund_flow],
}
```

```python
# finagent/tools.py 追加 5 个 @tool
@tool
def get_industry_ranking(top_n: int = 20) -> str:
    """获取全行业涨跌幅排名（东财行业板块），定位公司所在行业的热度。

    Args:
        top_n: 返回前 N 名（默认 20）
    """
    try:
        d = fetch("industry_ranking", top_n=top_n)
    except (DataSourceError, ValueError) as e:
        return str(e)
    lines = [f"行业涨跌幅排名（共 {d['total']} 个行业）:"]
    lines.append("涨幅前列:")
    for r in d["top"][:10]:
        lines.append(f"  {r['rank']}. {r['name']} {r['change_pct']:+.2f}% 涨{r['up_count']}跌{r['down_count']} 领涨:{r['leader']}")
    return "\n".join(lines)


@tool
def get_research_reports(stock_code: str, max_pages: int = 3) -> str:
    """获取近期卖方研报标题与摘要。

    Args:
        stock_code: 6 位股票代码
        max_pages: 最多拉取页数（默认 3）
    """
    try:
        reports = fetch("research_reports", stock_code=stock_code, max_pages=max_pages)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not reports:
        return "暂无研报覆盖。"
    lines = [f"共 {len(reports)} 篇研报:"]
    for r in reports[:10]:
        date = (r.get("publishDate") or "")[:10]
        org = r.get("orgSName", "")
        title = (r.get("title") or "")[:60]
        rating = r.get("emRatingName", "")
        lines.append(f"  {date} | {org} | {rating} | {title}")
    return "\n".join(lines)


@tool
def get_holder_change(stock_code: str, page_size: int = 10) -> str:
    """获取股东户数变化趋势，判断筹码集中/分散。

    Args:
        stock_code: 6 位股票代码
        page_size: 取最近 N 期（默认 10）
    """
    try:
        rows = fetch("holder_change", stock_code=stock_code, page_size=page_size)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not rows:
        return "暂无股东户数数据。"
    lines = ["报告期      股东数    环比变化  户均持股"]
    for r in rows[:8]:
        lines.append(f"  {r['date']}  {r['holder_num']:>8}  {r['change_ratio']:>+6.1f}%  {r['avg_shares']:>8.0f}")
    return "\n".join(lines)


@tool
def get_dividend_history(stock_code: str, page_size: int = 20) -> str:
    """获取分红送转历史。

    Args:
        stock_code: 6 位股票代码
        page_size: 取最近 N 期（默认 20）
    """
    try:
        rows = fetch("dividend_history", stock_code=stock_code, page_size=page_size)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not rows:
        return "暂无分红记录。"
    lines = ["日期        每股派息  转增  送股  进度"]
    for r in rows[:10]:
        lines.append(f"  {r['date']}  {r['bonus_rmb']:>6.3f}  {r['transfer_ratio']:>4}  {r['bonus_ratio']:>4}  {r['plan']}")
    return "\n".join(lines)


@tool
def get_fund_flow(stock_code: str) -> str:
    """获取近 120 日主力资金流向趋势（超大/大/中/小单日级）。

    Args:
        stock_code: 6 位股票代码
    """
    try:
        rows = fetch("fund_flow", stock_code=stock_code)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not rows:
        return "暂无资金流数据。"
    recent = rows[-20:]
    total_main = sum(r["main_net"] for r in recent)
    total_super = sum(r["super_net"] for r in recent)
    lines = [f"近 {len(rows)} 日资金流（近 20 日汇总）:"]
    lines.append(f"  主力净流入: {total_main / 1e8:+.2f}亿")
    lines.append(f"  超大单净额: {total_super / 1e8:+.2f}亿")
    lines.append("近 5 日:")
    for r in rows[-5:]:
        lines.append(f"  {r['date']}  主力 {r['main_net'] / 1e4:+.0f}万  超大 {r['super_net'] / 1e4:+.0f}万")
    return "\n".join(lines)


# tools list 最终版
tools = [
    get_company_info, get_financials, get_valuation,
    get_industry_ranking, get_research_reports,
    get_holder_change, get_dividend_history, get_fund_flow,
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS（8 工具齐全）

Run: `pytest -q`
Expected: PASS（全量绿）

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/registry.py finagent/tools.py tests/test_tools.py
git commit -m "feat: add 5 eastmoney tools (industry/reports/holder/dividend/fundflow)"
```

---

## 阶段 4：新浪备胎 + Skill 配方

### Task 14: `sources/sina_src.py` — financials 备胎 + 注册更新

**Files:**
- Create: `finagent/sources/sina_src.py`
- Modify: `finagent/sources/registry.py`（financials 加 sina 备胎）
- Modify: `tests/test_sources.py`、`tests/test_registry.py`

**Interfaces:**
- Produces: `financials(stock_code: str, report_period: str) -> dict`（与 akshare_src.financialals 签名一致，返回结构兼容）

**说明：** sina 返回的财报结构与 akshare 不同（字段名/格式各异）。为兼容 registry fallback，sina_src.financials 需将新浪数据适配为与 akshare_src.financials 相同的 dict 结构。MVP 阶段若适配复杂，可返回部分字段（profit_row 至少含 PARENT_NETPROFIT / OPERATE_INCOME），缺字段由工具层 `_format_current_period` 的 None 兜底处理。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py 追加
from finagent.sources import sina_src


def test_sina_financials_success(monkeypatch):
    fake_json = {
        "result": {"data": {"report_list": {
            "20240930": {"data": [
                {"item_title": "净利润", "item_value": "20000000000", "item_tongbi": "15.0"},
                {"item_title": "营业收入", "item_value": "100000000000", "item_tongbi": "10.0"},
            ]},
        }}}
    }
    fake_resp = MagicMock()
    fake_resp.json.return_value = fake_json
    monkeypatch.setattr(sina_src.requests, "get", lambda **kw: fake_resp)
    result = sina_src.financials("002415", "2024Q3")
    assert result["code"] == "002415"
    assert result["target_date"] == "2024-09-30"
    assert "profit_row" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py::test_sina_financials_success -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/sources/sina_src.py
import requests

from finagent.sources._ticker import norm_ticker, get_prefix
from finagent.sources.akshare_src import _normalize_period

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

_SINA_TO_AKSHARE = {
    "净利润": "PARENT_NETPROFIT",
    "营业收入": "TOTAL_OPERATE_INCOME",
    "营业成本": "OPERATE_COST",
}


def financials(stock_code: str, report_period: str) -> dict:
    code = norm_ticker(stock_code)
    target_date = _normalize_period(report_period)
    prefix = "sh" if get_prefix(code) == "sh" else "sz"
    url = "https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022"
    params = {
        "paperCode": f"{prefix}{code}", "source": "lrb",
        "type": "0", "page": "1", "num": "8",
    }
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=15)
    report_list = r.json().get("result", {}).get("data", {}).get("report_list", {}) or {}

    profit_row = {}
    for period in sorted(report_list.keys(), reverse=True)[:8]:
        pd_str = f"{period[:4]}-{period[4:6]}-{period[6:8]}"
        if pd_str == target_date:
            for it in report_list[period].get("data", []) or []:
                title = it.get("item_title", "")
                ak_field = _SINA_TO_AKSHARE.get(title)
                if ak_field and it.get("item_value") is not None:
                    try:
                        profit_row[ak_field] = float(it["item_value"])
                    except (ValueError, TypeError):
                        pass
            break

    if not profit_row:
        raise ValueError(f"新浪无 {code} {report_period} 利润表数据")

    return {
        "code": code,
        "name": code,
        "report_period": report_period,
        "target_date": target_date,
        "profit_row": profit_row,
        "balance_row": {},
        "cashflow_row": {},
        "profit_df": None,
        "balance_df": None,
    }
```

```python
# finagent/sources/registry.py 更新 financials 端点
from finagent.sources import akshare_src, tencent_src, eastmoney_src, sina_src

REGISTRY: dict[str, list[Callable]] = {
    "company_info": [akshare_src.company_info],
    "financials": [akshare_src.financials, sina_src.financials],
    "valuation": [tencent_src.valuation],
    "industry_ranking": [eastmoney_src.industry_ranking],
    "research_reports": [eastmoney_src.research_reports],
    "holder_change": [eastmoney_src.holder_change],
    "dividend_history": [eastmoney_src.dividend_history],
    "fund_flow": [eastmoney_src.fund_flow],
}
```

- [ ] **Step 4: Run test to verify it passes + 验证 fallback**

Run: `pytest tests/test_sources.py tests/test_registry.py -v`
Expected: PASS

追加一个 registry fallback 测试验证 akshare 失败时降级到 sina：

```python
# tests/test_registry.py 追加
def test_financials_fallback_akshare_to_sina(monkeypatch):
    from finagent.sources import registry
    monkeypatch.setattr(
        registry.akshare_src, "financials",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("akshare down"))
    )
    monkeypatch.setattr(
        registry.sina_src, "financials",
        lambda **kw: {"code": "002415", "profit_row": {}}
    )
    result = registry.fetch("financials", stock_code="002415", report_period="2024Q3")
    assert result["code"] == "002415"
```

Run: `pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/sources/sina_src.py finagent/sources/registry.py tests/test_sources.py tests/test_registry.py
git commit -m "feat: add sina financials fallback source"
```

---

### Task 15: `prompts.py` 加 SKILL_RECIPES（skill 层）

**Files:**
- Modify: `finagent/prompts.py`（追加 SKILL_RECIPES 拼入 RESEARCH_SYSTEM_PROMPT）
- Modify: `tests/test_prompts.py`（验证配方包含 8 工具指引）

**Interfaces:**
- Produces: `SKILL_RECIPES: str`（拼入 system prompt）

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py 追加
from finagent.prompts import SKILL_RECIPES, RESEARCH_SYSTEM_PROMPT


def test_skill_recipes_contains_tools():
    assert "get_company_info" in SKILL_RECIPES
    assert "get_financials" in SKILL_RECIPES
    assert "get_valuation" in SKILL_RECIPES
    assert "get_industry_ranking" in SKILL_RECIPES
    assert "get_research_reports" in SKILL_RECIPES
    assert "get_holder_change" in SKILL_RECIPES
    assert "get_dividend_history" in SKILL_RECIPES
    assert "get_fund_flow" in SKILL_RECIPES


def test_system_prompt_includes_recipes():
    assert SKILL_RECIPES in RESEARCH_SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py::test_skill_recipes_contains_tools -v`
Expected: FAIL — `ImportError: cannot import name 'SKILL_RECIPES'`

- [ ] **Step 3: Write minimal implementation**

```python
# finagent/prompts.py 追加（保留现有 RESEARCH_SYSTEM_PROMPT 内容）
SKILL_RECIPES = """
## 数据获取配方

根据用户问题类型，组合调用以下工具（不必每次全调，按问题侧重选取）：

**财报点评**（用户问某公司某期财报）：
1. get_company_info — 公司基本面（名称/行业/上市/主营）
2. get_financials — 财务三表 + 同比 + 近 8 期趋势（核心）
3. get_valuation — 当前估值水平（PE/PB/市值）
4. get_industry_ranking — 行业涨跌排名，定位行业冷暖
5. get_research_reports — 卖方研报观点（可选）
6. get_holder_change — 股东户数变化，判断筹码集中
7. get_dividend_history — 分红回报历史
8. get_fund_flow — 近 120 日主力资金动向

**优先级**：财务数据（2）是核心，必须先调。估值（3）与行业（4）提供横向参照。
跨工具数据矛盾时，以 get_financials 为准并标注差异。
东财源依赖网络，偶发超时属正常，工具已内置重试。
"""

RESEARCH_SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT + SKILL_RECIPES
```

**注意：** 若现有 `RESEARCH_SYSTEM_PROMPT` 是模块级常量，上述追加方式需调整为先存原文、再拼接。实现时确认 `prompts.py` 现有结构。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Final full test run**

Run: `pytest -q`
Expected: PASS（全量绿，8 工具 + 三层 + fallback 完整）

- [ ] **Step 6: Commit**

```bash
git add finagent/prompts.py tests/test_prompts.py
git commit -m "feat: add skill recipes to system prompt for tool composition"
```

---

## 自审记录

**Spec 覆盖检查：**
- 三层架构（skill/工具/DataSource）→ Task 15 / Task 5-7,13 / Task 4-12 覆盖
- 声明式 fallback registry + DI → Task 4 覆盖
- 8 端点（company_info/financials/valuation/industry/reports/holder/dividend/fundflow）→ Task 3,6,8,9,10,11,12 覆盖
- 3 基础设施（norm_ticker/get_prefix/em_get）→ Task 1,2 覆盖
- 工具层格式化保留 → Task 5 保留 `_format_*`
- 测试策略（mock 底层，DI 注入 registry）→ 各 Task TDD 覆盖
- 新浪备胎 → Task 14
- skill 配方 → Task 15

**Placeholder 扫描：** 无 TBD/TODO，每个步骤含真实代码。

**类型一致性：** `fetch(endpoint, *, registry=REGISTRY, **kwargs)` 签名贯穿 Task 4/5/7/13/14。源函数返回结构（dict/list）与工具层消费一致。

**已知偏差（已修正）：**
- spec 原 `get_industry_ranking(stock_code, top_n)` → 实际 `industry_comparison` 无 stock_code 参数，修正为 `get_industry_ranking(top_n)`。agent 需配合 `get_company_info` 定位公司所属行业。
- spec §9 阶段 1 financials 单源起步 → Task 4 REGISTRY 仅 akshare，Task 14 加 sina，符合。
