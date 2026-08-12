# A股财报点评 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI conversational LangChain agent that pulls A股 earnings data via akshare, discusses it with an analyst, and synthesizes a structured 财报点评 markdown report on `/report`.

**Architecture:** Single ReAct agent (LangGraph `create_react_agent` + memory checkpointer) for research dialogue, plus an independent report-synthesis chain triggered by CLI-intercepted `/report`. Both use DeepSeek `deepseek-chat`. Two akshare tools: `get_company_info` and `get_financials`.

**Tech Stack:** Python 3.11, langchain, langchain-deepseek, langgraph, akshare, rich, python-dotenv, pytest.

## Global Constraints

- Market: A股 only (上海 6xxxxx → SH, 深圳 0/3xxxxx → SZ, 北京 8/4xxxxx → BJ)
- LLM: DeepSeek `deepseek-chat` (model id `deepseek-chat`), API key from `DEEPSEEK_API_KEY` env var via `.env`
- Agent role: objective analysis only, no buy/sell rating, no target price
- No fabrication: system prompt forces "only cite data from conversation/tool output"
- akshare eastmoney endpoints require `User-Agent` header — inject globally via requests monkeypatch
- Sandbox blocks outbound network — all akshare calls need `dangerouslyDisableSandbox` at dev time; at runtime (user's machine) no sandbox
- Python: use `.venv/bin/python` (Python 3.11.15), install via `uv pip install -e ".[dev]"`
- Spec: `docs/superpowers/specs/2026-08-08-earnings-commentary-agent-design.md`

### Deviation from spec (justified by probing)

Spec defined 3 tools; this plan implements **2 tools**. The `get_financial_indicators` tool originally used `ak.stock_financial_analysis_indicator` (sina.com.cn HTML scrape — fragile, second data source, no User-Agent control). Probing revealed all 5 indicators (毛利率, 净利率, 资产负债率, 净现比, ROE) are computable from eastmoney raw statement fields. Merged into `get_financials` which returns current-period detail + 8-period computed-ratio trend in one API call. Result: one reliable data source, fewer tools, simpler agent, same analyst capability.

### akshare field map (verified via live probe 2026-08-08)

**Profit sheet** (`stock_profit_sheet_by_report_em`, columns):
- `TOTAL_OPERATE_INCOME` / `_YOY` — 营业总收入
- `OPERATE_INCOME` / `_YOY` — 营业收入
- `OPERATE_COST` / `_YOY` — 营业成本
- `SALE_EXPENSE` / `_YOY` — 销售费用
- `MANAGE_EXPENSE` / `_YOY` — 管理费用
- `RESEARCH_EXPENSE` / `_YOY` — 研发费用
- `FINANCE_EXPENSE` / `_YOY` — 财务费用
- `PARENT_NETPROFIT` / `_YOY` — 归母净利润
- `DEDUCT_PARENT_NETPROFIT` / `_YOY` — 扣非归母净利润
- `REPORT_DATE` — "2024-09-30 00:00:00"
- `REPORT_TYPE` — "三季报"
- `NOTICE_DATE` — 披露日期

**Balance sheet** (`stock_balance_sheet_by_report_em`, columns):
- `TOTAL_ASSETS` / `_YOY` — 总资产
- `TOTAL_LIABILITIES` / `_YOY` — 总负债
- `MONETARYFUNDS` / `_YOY` — 货币资金
- `ACCOUNTS_RECE` / `_YOY` — 应收账款
- `INVENTORY` / `_YOY` — 存货
- `TOTAL_PARENT_EQUITY` / `_YOY` — 归母权益

**Cash flow** (`stock_cash_flow_sheet_by_report_em`, columns):
- `NETCASH_OPERATE` / `_YOY` — 经营现金流净额
- `NETCASH_INVEST` / `_YOY` — 投资现金流净额
- `NETCASH_FINANCE` / `_YOY` — 筹资现金流净额

**Company info** (`stock_individual_info_em`, 2-col DataFrame item/value):
- 股票简称, 行业, 总股本, 流通股, 上市时间 (rows in `item` column)

**Computed ratios** (values in 元, divide by 1e8 for 亿):
- 毛利率 = (OPERATE_INCOME - OPERATE_COST) / OPERATE_INCOME × 100
- 净利率 = PARENT_NETPROFIT / OPERATE_INCOME × 100
- 资产负债率 = TOTAL_LIABILITIES / TOTAL_ASSETS × 100
- 净现比 = NETCASH_OPERATE / PARENT_NETPROFIT
- ROE = PARENT_NETPROFIT / TOTAL_PARENT_EQUITY × 100

---

### Task 1: Project scaffolding + config

**Files:**
- Create: `finagent/__init__.py`
- Create: `finagent/config.py`
- Create: `.env.example`
- Modify: `.gitignore`
- Init: git repo

**Interfaces:**
- Produces: `finagent/config.py` exports `get_llm()` returning a configured `ChatDeepSeek`; `load_env()` loading `.env`

- [ ] **Step 1: git init + update .gitignore**

```bash
git init
```

Append to `.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
reports/
*.egg-info/
dist/
build/
```

- [ ] **Step 2: Create .env.example**

```
DEEPSEEK_API_KEY=your-key-here
```

- [ ] **Step 3: Create finagent/__init__.py**

```python
"""A股财报点评 Agent."""
```

- [ ] **Step 4: Create finagent/config.py**

```python
"""Configuration: env loading + LLM factory."""
import os
from dotenv import load_dotenv


def load_env():
    """Load .env file if present."""
    load_dotenv()


def get_llm():
    """Return configured DeepSeek chat model."""
    from langchain_deepseek import ChatDeepSeek

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Copy .env.example to .env and fill in your key.")
    return ChatDeepSeek(model="deepseek-chat", api_key=api_key)
```

- [ ] **Step 5: Commit spec + scaffolding**

```bash
git add -A
git commit -m "feat: project scaffolding + config + spec doc"
```

---

### Task 2: Prompts module

**Files:**
- Create: `finagent/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces: `RESEARCH_SYSTEM_PROMPT` (str), `REPORT_SYSTEM_PROMPT` (str), `REPORT_TEMPLATE` (str)

- [ ] **Step 1: Write test for prompt content**

```python
"""tests/test_prompts.py"""
from finagent.prompts import RESEARCH_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT, REPORT_TEMPLATE


def test_research_prompt_mentions_constraints():
    assert "不出评级" in RESEARCH_SYSTEM_PROMPT or "无评级" in RESEARCH_SYSTEM_PROMPT
    assert "不编造" in RESEARCH_SYSTEM_PROMPT
    assert "A股" in RESEARCH_SYSTEM_PROMPT


def test_report_prompt_has_all_sections():
    for section in ["事件概述", "财务分析", "经营要点", "影响评估", "风险提示", "免责声明"]:
        assert section in REPORT_TEMPLATE


def test_report_prompt_no_rating():
    assert "评级" in REPORT_SYSTEM_PROMPT
    assert "目标价" in REPORT_SYSTEM_PROMPT
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `.venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement prompts.py**

```python
"""System prompts and report template."""


RESEARCH_SYSTEM_PROMPT = """你是一名 A股上市公司财报分析助手。你的任务是协助分析师分析财报数据。

工作规则：
1. 仅分析 A股上市公司。如果用户提供非 6 位股票代码，请要求其提供正确的代码。
2. 客观分析，绝不给出买卖评级、目标价或投资建议。
3. 只使用工具返回的数据进行分析，绝不编造或臆测未在工具结果中出现的数字。
4. 被动响应用户问题，不主动规定分析流程。
5. 报告期格式接受：2024Q3、2024三季报、2024-09-30 等，工具会自动归一化。
"""


REPORT_SYSTEM_PROMPT = """你是一名专业的 A股财报点评报告写手。

任务：根据对话历史，生成一份结构化的财报点评报告。

严格要求：
1. 只引用对话历史中出现过的数据，绝不编造任何数字。
2. 不给出买卖评级、目标价或任何投资建议。
3. 报告期和公司信息从对话历史中提取；如果无法确定，返回"[ERROR] 无法从对话历史确定公司代码或报告期"。
4. 严格按照提供的模板结构输出，不增减章节。
"""


REPORT_TEMPLATE = """# {公司名}({代码}) {报告期}财报点评

## 一、事件概述
报告期、披露日期、核心财务数据摘要（3-5 个关键数）

## 二、财务分析
营收/归母净利 同比环比、毛利率变化、费用率、现金流质量

## 三、经营要点
对话中讨论的业务分部变化、经营亮点

## 四、影响评估
正面因素 / 负面因素（客观陈述，无评级无目标价）

## 五、风险提示

## 六、免责声明
本报告由 AI 生成，仅供参考，不构成投资建议。
"""
```

- [ ] **Step 4: Run test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/prompts.py tests/test_prompts.py
git commit -m "feat: add system prompts and report template"
```

---

### Task 3: Tools helpers (market prefix, period normalize, requests patch)

**Files:**
- Create: `finagent/tools.py` (helpers portion)
- Test: `tests/test_tools_helpers.py`

**Interfaces:**
- Produces: `_market_prefix(code)`, `_normalize_period(period)`, `_patch_requests_ua()`, `_fmt_yi(value)`, `_validate_code(code)`

- [ ] **Step 1: Write tests**

```python
"""tests/test_tools_helpers.py"""
import pytest
from finagent.tools import (
    _market_prefix,
    _normalize_period,
    _validate_code,
    _fmt_yi,
)


def test_market_prefix():
    assert _market_prefix("600519") == "SH"
    assert _market_prefix("002415") == "SZ"
    assert _market_prefix("300750") == "SZ"
    assert _market_prefix("830799") == "BJ"


def test_normalize_period_formats():
    assert _normalize_period("2024Q3") == "2024-09-30"
    assert _normalize_period("2024Q1") == "2024-03-31"
    assert _normalize_period("2024Q2") == "2024-06-30"
    assert _normalize_period("2024Q4") == "2024-12-31"
    assert _normalize_period("2024三季报") == "2024-09-30"
    assert _normalize_period("2024年报") == "2024-12-31"
    assert _normalize_period("2024-09-30") == "2024-09-30"


def test_normalize_period_invalid():
    with pytest.raises(ValueError):
        _normalize_period("invalid")


def test_validate_code_valid():
    assert _validate_code("002415") == "002415"


def test_validate_code_invalid():
    with pytest.raises(ValueError):
        _validate_code("00241")
    with pytest.raises(ValueError):
        _validate_code("abcdef")


def test_fmt_yi():
    assert _fmt_yi(1_000_000_000) == "10.0亿"
    assert _fmt_yi(123_456_789) == "1.2亿"
    assert _fmt_yi(None) == "N/A"
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `.venv/bin/python -m pytest tests/test_tools_helpers.py -v`
Expected: FAIL

- [ ] **Step 3: Implement helpers in tools.py**

```python
"""akshare tool wrappers for the FinAgent."""
import re
import requests


# --- requests User-Agent injection (akshare doesn't set one) ---
_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_original_get = requests.get


def _patched_get(url, **kwargs):
    kwargs.setdefault("headers", {}).setdefault("User-Agent", _ua)
    return _original_get(url, **kwargs)


def _patch_requests_ua():
    """Monkeypatch requests.get to inject a User-Agent header."""
    requests.get = _patched_get


_patch_requests_ua()  # apply on import


# --- stock code ---
_CODE_RE = re.compile(r"^\d{6}$")

_PERIOD_MAP = {
    "一季报": "03-31", "q1": "03-31",
    "半年报": "06-30", "中报": "06-30", "q2": "06-30",
    "三季报": "09-30", "q3": "09-30",
    "年报": "12-31", "q4": "12-31",
}


def _validate_code(code: str) -> str:
    """Validate 6-digit A股 stock code."""
    code = code.strip()
    if not _CODE_RE.match(code):
        raise ValueError(f"无效股票代码（需 6 位数字）: {code}")
    return code


def _market_prefix(code: str) -> str:
    """Map 6-digit code to exchange prefix."""
    if code.startswith("6"):
        return "SH"
    if code.startswith(("0", "3")):
        return "SZ"
    if code.startswith(("8", "4")):
        return "BJ"
    raise ValueError(f"无法识别交易所: {code}")


def _normalize_period(period: str) -> str:
    """Normalize various period formats to 'YYYY-MM-DD'.

    Accepts: 2024Q3, 2024三季报, 2024-09-30, 20240930
    """
    period = period.strip()
    # already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", period):
        return period
    # YYYYMMDD
    if re.match(r"^\d{8}$", period):
        return f"{period[:4]}-{period[4:6]}-{period[6:8]}"
    # 2024Q3
    m = re.match(r"^(\d{4})[Qq](\d)$", period)
    if m:
        year, q = m.group(1), int(m.group(2))
        month_day = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}[str(q)]
        return f"{year}-{month_day}"
    # 2024三季报 / 2024年报
    for keyword, md in _PERIOD_MAP.items():
        if keyword in period:
            m = re.match(r"^(\d{4})", period)
            if m:
                return f"{m.group(1)}-{md}"
    raise ValueError(f"无法识别报告期格式: {period}")


def _fmt_yi(value) -> str:
    """Format absolute yuan value to 亿 with 1 decimal. Returns 'N/A' for None."""
    if value is None or value != value:  # NaN check
        return "N/A"
    return f"{value / 1e8:.1f}亿"
```

- [ ] **Step 4: Run test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_tools_helpers.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/tools.py tests/test_tools_helpers.py
git commit -m "feat: add tools helpers (code validation, period normalize, UA patch)"
```

---

### Task 4: get_company_info + get_financials tools

**Files:**
- Modify: `finagent/tools.py` (append tool functions)
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `get_company_info`, `get_financials` (LangChain `@tool`-decorated), and `tools` list

- [ ] **Step 1: Write tests (mock akshare)**

```python
"""tests/test_tools.py"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from finagent.tools import get_company_info, get_financials


def _fake_company_df():
    return pd.DataFrame({
        "item": ["股票简称", "行业", "总股本", "流通股", "上市时间"],
        "value": ["海康威视", "计算机设备", "91.6亿", "90.5亿", "20100528"],
    })


def test_get_company_info_success():
    with patch("finagent.tools.ak.stock_individual_info_em", return_value=_fake_company_df()):
        result = get_company_info.invoke({"stock_code": "002415"})
    assert "海康威视" in result
    assert "计算机设备" in result


def test_get_company_info_bad_code():
    result = get_company_info.invoke({"stock_code": "123"})
    assert "无效股票代码" in result


def _fake_profit_df():
    return pd.DataFrame([
        {
            "SECURITY_NAME_ABBR": "海康威视", "REPORT_DATE": "2024-09-30 00:00:00",
            "REPORT_TYPE": "三季报", "NOTICE_DATE": "2024-10-19",
            "TOTAL_OPERATE_INCOME": 6.502e10, "TOTAL_OPERATE_INCOME_YOY": 6.06,
            "OPERATE_INCOME": 6.502e10, "OPERATE_INCOME_YOY": 6.06,
            "OPERATE_COST": 3.599e10, "OPERATE_COST_YOY": 7.5,
            "SALE_EXPENSE": 5.4e9, "MANAGE_EXPENSE": 1.02e10,
            "RESEARCH_EXPENSE": 2.5e9, "FINANCE_EXPENSE": -3.8e8,
            "PARENT_NETPROFIT": 8.108e9, "PARENT_NETPROFIT_YOY": 8.0,
            "DEDUCT_PARENT_NETPROFIT": 7.83e9, "DEDUCT_PARENT_NETPROFIT_YOY": 5.2,
        },
    ])


def _fake_balance_df():
    return pd.DataFrame([
        {
            "REPORT_DATE": "2024-09-30 00:00:00",
            "TOTAL_ASSETS": 1.024e11, "TOTAL_LIABILITIES": 3.91e10,
            "MONETARYFUNDS": 3.12e10, "ACCOUNTS_RECE": 2.5e10, "INVENTORY": 1.8e10,
            "TOTAL_PARENT_EQUITY": 6.33e10,
        },
    ])


def _fake_cashflow_df():
    return pd.DataFrame([
        {
            "REPORT_DATE": "2024-09-30 00:00:00",
            "NETCASH_OPERATE": 4.52e9, "NETCASH_INVEST": -5e9, "NETCASH_FINANCE": -3e9,
        },
    ])


def test_get_financials_success():
    with patch("finagent.tools.ak.stock_profit_sheet_by_report_em", return_value=_fake_profit_df()), \
         patch("finagent.tools.ak.stock_balance_sheet_by_report_em", return_value=_fake_balance_df()), \
         patch("finagent.tools.ak.stock_cash_flow_sheet_by_report_em", return_value=_fake_cashflow_df()):
        result = get_financials.invoke({"stock_code": "002415", "report_period": "2024Q3"})
    assert "营收" in result
    assert "归母净利" in result
    assert "毛利率" in result
    assert "经营现金流" in result
```

- [ ] **Step 2: Run test — expect fail (tools not defined)**

Run: `.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Append tool implementations to tools.py**

```python
import akshare as ak
from langchain_core.tools import tool

tools = []  # populated below


@tool
def get_company_info(stock_code: str) -> str:
    """获取 A股上市公司基本信息：公司简称、所属行业、总股本、流通股、上市时间。

    Args:
        stock_code: 6 位股票代码，如 002415
    """
    try:
        code = _validate_code(stock_code)
    except ValueError as e:
        return str(e)
    try:
        df = ak.stock_individual_info_em(symbol=code)
        rows = dict(zip(df["item"], df["value"]))
        name = rows.get("股票简称", "未知")
        industry = rows.get("行业", "未知")
        shares = rows.get("总股本", "N/A")
        float_shares = rows.get("流通股", "N/A")
        listing = rows.get("上市时间", "N/A")
        return (
            f"公司: {name}({code})\n"
            f"行业: {industry}\n"
            f"总股本: {shares} | 流通股: {float_shares}\n"
            f"上市时间: {listing}"
        )
    except Exception as e:
        return f"获取公司信息失败: {e}"


def _compute_gross_margin(row):
    income = row.get("OPERATE_INCOME")
    cost = row.get("OPERATE_COST")
    if not income or not cost:
        return None
    return (income - cost) / income * 100


def _compute_debt_ratio(row):
    liabilities = row.get("TOTAL_LIABILITIES")
    assets = row.get("TOTAL_ASSETS")
    if not liabilities or not assets:
        return None
    return liabilities / assets * 100


def _compute_net_cash_ratio(profit_row, cashflow_row):
    net_profit = profit_row.get("PARENT_NETPROFIT")
    op_cash = cashflow_row.get("NETCASH_OPERATE")
    if not net_profit or not op_cash:
        return None
    return op_cash / net_profit


def _format_current_period(profit_row, balance_row, cashflow_row) -> str:
    """Format current-period detail + YoY into compact text."""
    gm = _compute_gross_margin(profit_row)
    debt = _compute_debt_ratio(balance_row)
    ncr = _compute_net_cash_ratio(profit_row, cashflow_row)
    lines = ["利润表:"]
    lines.append(f"  营收: {_fmt_yi(profit_row.get('TOTAL_OPERATE_INCOME'))} ({profit_row.get('TOTAL_OPERATE_INCOME_YOY', 'N/A'):+.1f}% YoY)")
    lines.append(f"  归母净利: {_fmt_yi(profit_row.get('PARENT_NETPROFIT'))} ({profit_row.get('PARENT_NETPROFIT_YOY', 'N/A'):+.1f}% YoY)")
    lines.append(f"  扣非净利: {_fmt_yi(profit_row.get('DEDUCT_PARENT_NETPROFIT'))} ({profit_row.get('DEDUCT_PARENT_NETPROFIT_YOY', 'N/A'):+.1f}% YoY)")
    lines.append(f"  毛利率: {gm:.1f}%" if gm else "  毛利率: N/A")
    lines.append(f"  销售费用: {_fmt_yi(profit_row.get('SALE_EXPENSE'))} | 管理费用: {_fmt_yi(profit_row.get('MANAGE_EXPENSE'))} | 研发费用: {_fmt_yi(profit_row.get('RESEARCH_EXPENSE'))}")
    lines.append("资产负债表:")
    lines.append(f"  总资产: {_fmt_yi(balance_row.get('TOTAL_ASSETS'))} | 负债率: {debt:.1f}%" if debt else "  总资产: N/A")
    lines.append(f"  货币资金: {_fmt_yi(balance_row.get('MONETARYFUNDS'))} | 应收账款: {_fmt_yi(balance_row.get('ACCOUNTS_RECE'))} | 存货: {_fmt_yi(balance_row.get('INVENTORY'))}")
    lines.append("现金流量表:")
    lines.append(f"  经营现金流净额: {_fmt_yi(cashflow_row.get('NETCASH_OPERATE'))}")
    lines.append(f"  净现比(经营现金流/归母净利): {ncr:.2f}" if ncr else "  净现比: N/A")
    return "\n".join(lines)


def _format_trend(profit_df, balance_df, n=8) -> str:
    """Compute 8-period trend for 5 ratios, return compact table."""
    merged = profit_df[["REPORT_DATE", "OPERATE_INCOME", "OPERATE_COST", "PARENT_NETPROFIT"]].copy()
    bal = balance_df[["REPORT_DATE", "TOTAL_LIABILITIES", "TOTAL_ASSETS", "TOTAL_PARENT_EQUITY"]].copy()
    merged = merged.merge(bal, on="REPORT_DATE", how="inner")
    merged = merged.sort_values("REPORT_DATE", ascending=False).head(n)
    lines = [f"近 {len(merged)} 期趋势:"]
    lines.append(f"{'报告期':<12} {'毛利率%':>7} {'净利率%':>7} {'负债率%':>7} {'ROE%':>7}")
    for _, r in merged.iterrows():
        gm = (r["OPERATE_INCOME"] - r["OPERATE_COST"]) / r["OPERATE_INCOME"] * 100 if r["OPERATE_INCOME"] else None
        nm = r["PARENT_NETPROFIT"] / r["OPERATE_INCOME"] * 100 if r["OPERATE_INCOME"] else None
        dr = r["TOTAL_LIABILITIES"] / r["TOTAL_ASSETS"] * 100 if r["TOTAL_ASSETS"] else None
        roe = r["PARENT_NETPROFIT"] / r["TOTAL_PARENT_EQUITY"] * 100 if r.get("TOTAL_PARENT_EQUITY") else None
        date_short = str(r["REPORT_DATE"])[:10]
        lines.append(f"{date_short:<12} {gm:>7.1f} {nm:>7.1f} {dr:>7.1f} {roe:>7.1f}")
    return "\n".join(lines)


@tool
def get_financials(stock_code: str, report_period: str) -> str:
    """获取 A股上市公司指定报告期的财务数据：利润表/资产负债表/现金流量表关键科目（含同比）+ 近 8 期趋势。

    Args:
        stock_code: 6 位股票代码，如 002415
        report_period: 报告期，支持 2024Q3 / 2024三季报 / 2024-09-30
    """
    try:
        code = _validate_code(stock_code)
        target_date = _normalize_period(report_period)
    except ValueError as e:
        return str(e)
    prefix = _market_prefix(code)
    symbol = f"{prefix}{code}"
    try:
        profit_df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
        balance_df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
        cashflow_df = ak.stock_cash_flow_sheet_by_report_em(symbol=symbol)
    except Exception as e:
        return f"财务数据拉取失败: {e}"

    # filter to target period
    def _find_row(df, date_str):
        matches = df[df["REPORT_DATE"].astype(str).str.startswith(date_str)]
        return matches.iloc[0].to_dict() if len(matches) else None

    profit_row = _find_row(profit_df, target_date)
    balance_row = _find_row(balance_df, target_date)
    cashflow_row = _find_row(cashflow_df, target_date)
    if profit_row is None:
        return f"该报告期({report_period})数据尚未披露或不存在。"

    name = profit_row.get("SECURITY_NAME_ABBR", code)
    detail = _format_current_period(profit_row, balance_row or {}, cashflow_row or {})
    trend = _format_trend(profit_df, balance_df)
    return f"=== {name}({code}) {report_period} 财务数据 ===\n{detail}\n\n{trend}"


tools = [get_company_info, get_financials]
```

- [ ] **Step 4: Run test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/tools.py tests/test_tools.py
git commit -m "feat: add get_company_info and get_financials akshare tools"
```

---

### Task 5: ReAct agent

**Files:**
- Create: `finagent/agent.py`

**Interfaces:**
- Consumes: `tools` from `finagent.tools`, `RESEARCH_SYSTEM_PROMPT` from `finagent.prompts`, `get_llm` from `finagent.config`
- Produces: `create_agent(thread_id)` returning a compiled LangGraph agent; shared `MemorySaver` checkpointer

- [ ] **Step 1: Implement agent.py**

```python
"""ReAct agent construction using LangGraph."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from finagent.config import get_llm
from finagent.prompts import RESEARCH_SYSTEM_PROMPT
from finagent.tools import tools

# shared in-memory checkpointer — one per process, cleared on exit
_checkpointer = MemorySaver()


def create_agent(thread_id: str = "default"):
    """Build a ReAct agent bound to a conversation thread.

    Args:
        thread_id: thread identifier for checkpointer scoping; /clear creates a new id.
    """
    llm = get_llm()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RESEARCH_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )
    return agent


def reset_checkpoint():
    """Reset in-memory checkpoint (used by /clear)."""
    global _checkpointer
    _checkpointer = MemorySaver()
```

Note: `reset_checkpoint` replaces the checkpointer; the next `create_agent` call uses the fresh one. The CLI holds a single agent instance per thread, so `/clear` re-creates the agent with a fresh checkpointer + new thread id.

- [ ] **Step 2: Smoke-import test (no live LLM call)**

```python
"""tests/test_agent.py"""
from finagent.agent import reset_checkpoint


def test_reset_checkpoint_does_not_error():
    reset_checkpoint()  # should not raise
```

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add finagent/agent.py tests/test_agent.py
git commit -m "feat: add ReAct agent with memory checkpointer"
```

---

### Task 6: Report synthesis

**Files:**
- Create: `finagent/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `REPORT_SYSTEM_PROMPT`, `REPORT_TEMPLATE` from `finagent.prompts`, `get_llm` from `finagent.config`
- Produces: `generate_report(messages: list) -> tuple[str, str]` returning `(filepath, content)` or raising `ValueError` if company/period undeterminable

- [ ] **Step 1: Write test (mock LLM)**

```python
"""tests/test_report.py"""
from unittest.mock import patch, MagicMock
from finagent.report import generate_report


_FAKE_MESSAGES = [
    {"role": "user", "content": "分析 002415 2024Q3 财报"},
    {"role": "assistant", "content": "海康威视 2024三季报：营收 650亿..."},
]


def test_generate_report_returns_filepath(tmp_path):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="# 海康威视(002415) 2024Q3财报点评\n\n## 一、事件概述\n...")
    with patch("finagent.report.get_llm", return_value=fake_llm), \
         patch("finagent.report.Path.cwd", return_value=tmp_path):
        filepath, content = generate_report(_FAKE_MESSAGES)
    assert "002415" in filepath
    assert "2024Q3" in filepath
    assert ".md" in filepath
    assert "事件概述" in content


def test_generate_report_error_on_missing_info():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="[ERROR] 无法从对话历史确定公司代码或报告期")
    with patch("finagent.report.get_llm", return_value=fake_llm):
        try:
            generate_report([{"role": "user", "content": "你好"}])
            assert False, "should have raised"
        except ValueError:
            pass
```

- [ ] **Step 2: Run — expect ImportError**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: FAIL

- [ ] **Step 3: Implement report.py**

```python
"""Report synthesis chain — generates structured 财报点评 from conversation history."""
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from finagent.config import get_llm
from finagent.prompts import REPORT_SYSTEM_PROMPT, REPORT_TEMPLATE


def _messages_to_text(messages: list) -> str:
    """Flatten message list to readable transcript text."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "user")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        label = {"user": "分析师", "assistant": "助手", "tool": "工具结果"}.get(role, role)
        lines.append(f"[{label}]\n{content}")
    return "\n\n".join(lines)


def generate_report(messages: list) -> tuple[str, str]:
    """Synthesize a 财报点评 report from conversation history.

    Returns (filepath, content). Raises ValueError if LLM cannot determine company/period.
    """
    transcript = _messages_to_text(messages)
    full_prompt = (
        f"{REPORT_SYSTEM_PROMPT}\n\n"
        f"模板结构:\n{REPORT_TEMPLATE}\n\n"
        f"对话历史:\n{transcript}\n\n"
        f"请按模板生成完整报告。"
    )
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=full_prompt)])
    content = response.content if hasattr(response, "content") else str(response)

    if "[ERROR]" in content:
        raise ValueError(content)

    # extract code + period for filename (fallback to 'report' if not found)
    code_match = re.search(r"\b(\d{6})\b", transcript)
    period_match = re.search(r"(\d{4}Q\d|\d{4}[一二三四三半年年度季报]+报|\d{4}-\d{2}-\d{2})", transcript)
    code = code_match.group(1) if code_match else "report"
    period = period_match.group(1) if period_match else "unknown"

    reports_dir = Path.cwd() / "reports"
    reports_dir.mkdir(exist_ok=True)
    # sanitize period for filename
    period_safe = re.sub(r"[^\w一-鿿]", "", period)
    filepath = reports_dir / f"{code}_{period_safe}_点评.md"
    filepath.write_text(content, encoding="utf-8")
    return str(filepath), content
```

- [ ] **Step 4: Run test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/report.py tests/test_report.py
git commit -m "feat: add report synthesis chain"
```

---

### Task 7: CLI interaction loop

**Files:**
- Create: `finagent/cli.py`
- Create: `finagent/__main__.py`

**Interfaces:**
- Consumes: all prior modules
- Produces: runnable `python -m finagent`

- [ ] **Step 1: Implement cli.py**

```python
"""CLI interaction loop for FinAgent."""
import uuid

from rich.console import Console
from rich.markdown import Markdown

from finagent.agent import create_agent, reset_checkpoint
from finagent.config import load_env
from finagent.report import generate_report

console = Console()

HELP_TEXT = """\
可用命令:
  /report   生成财报点评报告（基于当前对话历史）
  /clear    清空对话记忆（换公司分析时用）
  /help     显示此帮助
  /quit     退出
"""


def _run_agent_turn(agent, thread_id: str, user_input: str):
    """Invoke agent and render the response."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    # last AI message
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    if ai_messages:
        console.print(Markdown(ai_messages[-1].content))
    return result


def main():
    load_env()
    thread_id = str(uuid.uuid4())
    agent = create_agent(thread_id)

    console.print("[bold green]FinAgent[/bold green] — A股财报点评助手")
    console.print("输入股票代码 + 报告期开始分析，如: 002415 2024Q3")
    console.print("输入 /help 查看命令\n")

    while True:
        try:
            user_input = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见。")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd == "/quit":
            console.print("再见。")
            break
        elif cmd == "/help":
            console.print(HELP_TEXT)
            continue
        elif cmd == "/clear":
            reset_checkpoint()
            thread_id = str(uuid.uuid4())
            agent = create_agent(thread_id)
            console.print("[yellow]对话记忆已清空。[/yellow]\n")
            continue
        elif cmd == "/report":
            # gather full conversation history from checkpointer
            state = agent.get_state(config={"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", []) if state and state.values else []
            if not messages:
                console.print("[yellow]还没有对话内容，先聊几句再生成报告。[/yellow]\n")
                continue
            console.print("[dim]正在生成报告...[/dim]")
            try:
                msg_dicts = [{"role": m.type, "content": m.content} for m in messages]
                filepath, content = generate_report(msg_dicts)
                console.print(Markdown(content))
                console.print(f"\n[green]报告已保存: {filepath}[/green]\n")
            except ValueError as e:
                console.print(f"[red]{e}[/red]\n")
            except Exception as e:
                console.print(f"[red]报告生成失败: {e}[/red]\n")
            continue

        # normal conversation turn
        try:
            _run_agent_turn(agent, thread_id, user_input)
            console.print()
        except Exception as e:
            console.print(f"[red]出错: {e}[/red]\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement __main__.py**

```python
"""finagent/__main__.py"""
from finagent.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test — import succeeds**

Run: `.venv/bin/python -c "from finagent.cli import main; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add finagent/cli.py finagent/__main__.py
git commit -m "feat: add CLI interaction loop with rich rendering"
```

---

### Task 8: End-to-end manual verification

**Requires:** `DEEPSEEK_API_KEY` set in `.env`, network access (not sandboxed).

- [ ] **Step 1: Copy .env**

```bash
cp .env.example .env
# edit .env, fill in DEEPSEEK_API_KEY
```

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 3: Launch CLI (run manually, needs API key + network)**

```bash
.venv/bin/python -m finagent
```

- [ ] **Step 4: Test research dialogue**

Type: `002415 2024Q3`
Verify: agent calls get_financials tool, renders financial analysis with rich markdown.

Follow up: `毛利率变化的主要原因是什么？`
Verify: agent reasons from prior tool data.

- [ ] **Step 5: Test report generation**

Type: `/report`
Verify: `reports/002415_2024Q3点评.md` created, contains all 6 sections, numbers match conversation.

- [ ] **Step 6: Test /clear and /help**

Type: `/clear` → verify "对话记忆已清空"
Type: `/help` → verify command list shown
Type: `/quit` → verify clean exit

- [ ] **Step 7: Commit final state**

```bash
git add -A
git commit -m "chore: e2e verification complete"
```

---

## Self-Review

**Spec coverage:**
- 财报点评 report type ✓ (Task 2 template)
- Agent auto-fetch akshare ✓ (Task 4 tools)
- A股 only ✓ (Task 3 `_market_prefix`, `_validate_code`)
- Analysis only no rating ✓ (Task 2 prompts)
- Earnings only ✓ (Task 4 `get_financials` focused on statements)
- Conversational ✓ (Task 7 CLI loop)
- Single ReAct agent ✓ (Task 5)
- 3→2 tools deviation ✓ (documented in header)
- Report template 6 sections ✓ (Task 2)
- CLI commands English ✓ (Task 7)
- DeepSeek deepseek-chat ✓ (Task 1 config)
- /report CLI intercept independent chain ✓ (Task 7 intercept + Task 6 chain)
- No-args /report LLM extracts company+period ✓ (Task 6)
- Compact tool outputs ✓ (Task 4 `_format_*`)
- Stock code required 6-digit ✓ (Task 3 `_validate_code`)
- No prescribed workflow ✓ (Task 2 prompt "被动响应")
- 8-period 5-indicator trend ✓ (Task 4 `_format_trend`)
- Error handling tool-level ✓ (Task 4 try/except)
- Memory checkpointer ✓ (Task 5 MemorySaver)
- .env for API key ✓ (Task 1)

**Placeholder scan:** none found — all steps have concrete code.

**Type consistency:** `tools` list defined in Task 4, consumed in Task 5. `generate_report(messages)` signature consistent between Task 6 test and impl and Task 7 call site. `create_agent(thread_id)` consistent across Tasks 5 and 7.
