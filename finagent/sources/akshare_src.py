import os
import re

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("TQDM_DISABLE", "1")

import requests

_AKSHARE_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_orig_session_init = requests.Session.__init__


def _session_init_with_ua(self, *args, **kwargs):
    """Patch Session.__init__ so all HTTP clients get a browser UA.

    akshare calls requests.get() internally without setting User-Agent.
    Some APIs reject the default python-requests/* UA, so we replace it
    with a browser string. Sessions with a custom UA are not affected.
    """
    _orig_session_init(self, *args, **kwargs)
    if self.headers.get("User-Agent", "").startswith("python-requests/"):
        self.headers["User-Agent"] = _AKSHARE_UA


requests.Session.__init__ = _session_init_with_ua

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


def industry_ranking_ths(top_n: int = 20) -> dict:
    df = ak.stock_board_industry_summary_ths()
    if df.empty:
        return {"top": [], "bottom": [], "total": 0}
    df = df.sort_values("涨跌幅", ascending=False).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        rows.append({
            "rank": i + 1,
            "name": r.get("板块", ""),
            "change_pct": r.get("涨跌幅", 0),
            "code": "",
            "up_count": r.get("上涨家数", 0),
            "down_count": r.get("下跌家数", 0),
            "leader": r.get("领涨股", ""),
            "leader_change": r.get("领涨股-涨跌幅", 0),
        })
    return {"top": rows[:top_n], "bottom": rows[-top_n:], "total": len(rows)}


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
