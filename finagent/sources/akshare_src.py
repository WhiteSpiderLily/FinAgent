import os
import re

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("TQDM_DISABLE", "1")

import requests

_orig_get = requests.get


def _ua_get(url, **kwargs):
    kwargs.setdefault("headers", {}).setdefault(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    )
    return _orig_get(url, **kwargs)


requests.get = _ua_get

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
