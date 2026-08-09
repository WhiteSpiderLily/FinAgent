"""tests/test_tools.py"""
from unittest.mock import patch

import pandas as pd

from finagent.sources import DataSourceError
from finagent.tools import (
    get_company_info,
    get_financials,
    get_valuation,
    _format_current_period,
    _format_trend,
    _fmt_yi,
    _fmt_yoy,
    _fmt_trend_val,
)


def test_get_company_info_success():
    fake_data = {
        "code": "002415", "name": "海康威视", "industry": "计算机设备",
        "listing": "2010-05-28", "main_biz": "视频监控产品",
    }
    with patch("finagent.tools.fetch", return_value=fake_data):
        result = get_company_info.invoke({"stock_code": "002415"})
    assert "海康威视" in result
    assert "计算机设备" in result


def test_get_company_info_fetch_error():
    with patch("finagent.tools.fetch", side_effect=DataSourceError("all failed")):
        result = get_company_info.invoke({"stock_code": "002415"})
    assert "失败" in result or "error" in result.lower()


def test_get_company_info_validation_error():
    with patch("finagent.tools.fetch", side_effect=ValueError("无效股票代码")):
        result = get_company_info.invoke({"stock_code": "123"})
    assert "无效" in result


def test_get_financials_success():
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
        result = get_financials.invoke({"stock_code": "002415", "report_period": "2024Q3"})
    assert "海康威视" in result
    assert "利润表" in result


def test_get_financials_fetch_error():
    with patch("finagent.tools.fetch", side_effect=DataSourceError("all failed")):
        result = get_financials.invoke({"stock_code": "002415", "report_period": "2024Q3"})
    assert "失败" in result or "error" in result.lower()


def test_get_financials_sina_fallback_no_trend():
    """Sina fallback returns profit_df=None; trend must degrade gracefully, not raise."""
    fake_data = {
        "code": "002415", "name": "002415", "report_period": "2024Q3",
        "target_date": "2024-09-30",
        "profit_row": {"TOTAL_OPERATE_INCOME": 100e8, "PARENT_NETPROFIT": 20e8,
                       "OPERATE_INCOME": 100e8, "OPERATE_COST": 60e8},
        "balance_row": {"TOTAL_ASSETS": 500e8, "TOTAL_LIABILITIES": 200e8},
        "cashflow_row": {},
        "profit_df": None,
        "balance_df": None,
    }
    with patch("finagent.tools.fetch", return_value=fake_data):
        result = get_financials.invoke({"stock_code": "002415", "report_period": "2024Q3"})
    assert "趋势数据不可用" in result
    assert "利润表" in result


# --- formatting helper tests ---

def test_fmt_yi():
    assert _fmt_yi(1_000_000_000) == "10.0亿"
    assert _fmt_yi(123_456_789) == "1.2亿"
    assert _fmt_yi(None) == "N/A"


def test_fmt_yoy_none():
    assert _fmt_yoy(None) == "N/A"
    assert _fmt_yoy(6.06) == "+6.1%"


def test_fmt_trend_val_none():
    assert _fmt_trend_val(None) == "    N/A"
    assert _fmt_trend_val(12.34).strip() == "12.3"


def test_format_current_period_handles_none_yoy():
    """None YoY values must not crash; output should contain 'N/A'."""
    profit_row = {
        "TOTAL_OPERATE_INCOME": 6.502e10,
        "TOTAL_OPERATE_INCOME_YOY": None,
        "OPERATE_INCOME": 6.502e10,
        "OPERATE_COST": 3.599e10,
        "SALE_EXPENSE": 5.4e9,
        "MANAGE_EXPENSE": 1.02e10,
        "RESEARCH_EXPENSE": 2.5e9,
        "PARENT_NETPROFIT": 8.108e9,
        "PARENT_NETPROFIT_YOY": None,
        "DEDUCT_PARENT_NETPROFIT": 7.83e9,
        "DEDUCT_PARENT_NETPROFIT_YOY": None,
    }
    balance_row = {
        "TOTAL_ASSETS": 1.024e11,
        "TOTAL_LIABILITIES": 3.91e10,
        "MONETARYFUNDS": 3.12e10,
        "ACCOUNTS_RECE": 2.5e10,
        "INVENTORY": 1.8e10,
    }
    cashflow_row = {"NETCASH_OPERATE": 4.52e9}
    out = _format_current_period(profit_row, balance_row, cashflow_row)
    assert "N/A" in out


def test_format_current_period_missing_yoy_key():
    """Missing YoY keys must not crash either."""
    profit_row = {
        "TOTAL_OPERATE_INCOME": 6.502e10,
        "OPERATE_INCOME": 6.502e10,
        "OPERATE_COST": 3.599e10,
        "SALE_EXPENSE": 5.4e9,
        "MANAGE_EXPENSE": 1.02e10,
        "RESEARCH_EXPENSE": 2.5e9,
        "PARENT_NETPROFIT": 8.108e9,
        "DEDUCT_PARENT_NETPROFIT": 7.83e9,
    }
    out = _format_current_period(profit_row, {}, {})
    assert "N/A" in out


def test_format_trend_handles_zero_income():
    """A period with OPERATE_INCOME=0 must not crash; trend should show N/A."""
    profit_df = pd.DataFrame([
        {"REPORT_DATE": "2024-09-30", "OPERATE_INCOME": 6.502e10, "OPERATE_COST": 3.599e10, "PARENT_NETPROFIT": 8.108e9},
        {"REPORT_DATE": "2024-06-30", "OPERATE_INCOME": 0, "OPERATE_COST": 0, "PARENT_NETPROFIT": 0},
    ])
    balance_df = pd.DataFrame([
        {"REPORT_DATE": "2024-09-30", "TOTAL_LIABILITIES": 3.91e10, "TOTAL_ASSETS": 1.024e11, "TOTAL_PARENT_EQUITY": 6.33e10},
        {"REPORT_DATE": "2024-06-30", "TOTAL_LIABILITIES": 3.0e10, "TOTAL_ASSETS": 9.0e10, "TOTAL_PARENT_EQUITY": 0},
    ])
    out = _format_trend(profit_df, balance_df)
    assert "N/A" in out


# --- get_valuation tests ---

def test_get_valuation_success():
    fake = {
        "code": "002415", "name": "海康威视", "price": 32.50,
        "pe_ttm": 25.0, "pb": 5.5, "mcap_yi": 3000.0,
        "float_mcap_yi": 2900.0, "turnover_pct": 1.2,
        "change_pct": 3.5, "limit_up": 35.0, "limit_down": 29.0,
        "is_stale": False,
    }
    with patch("finagent.tools.fetch", return_value=fake):
        result = get_valuation.invoke({"stock_code": "002415"})
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
        result = get_valuation.invoke({"stock_code": "002415"})
    assert "停牌" in result or "stale" in result.lower()


# --- eastmoney tools tests ---

def test_get_industry_ranking():
    fake = {"total": 2, "top": [{"rank": 1, "name": "电力设备", "change_pct": 3.5,
            "up_count": 200, "down_count": 50, "leader": "宁德时代"}],
            "bottom": [{"rank": 2, "name": "白酒", "change_pct": -1.2}]}
    from finagent.tools import get_industry_ranking
    with patch("finagent.tools.fetch", return_value=fake):
        result = get_industry_ranking.invoke({"top_n": 5})
    assert "电力设备" in result


def test_get_research_reports():
    fake = [{"title": "业绩超预期", "publishDate": "2024-10-20", "orgSName": "中信",
             "emRatingName": "买入", "predictThisYearEps": 1.5}]
    from finagent.tools import get_research_reports
    with patch("finagent.tools.fetch", return_value=fake):
        result = get_research_reports.invoke({"stock_code": "002415"})
    assert "业绩超预期" in result


def test_get_holder_change():
    fake = [{"date": "2024-09-30", "holder_num": 50000, "change_ratio": -3.8, "avg_shares": 1500}]
    from finagent.tools import get_holder_change
    with patch("finagent.tools.fetch", return_value=fake):
        result = get_holder_change.invoke({"stock_code": "002415"})
    assert "50000" in result
    assert "-3.8" in result


def test_get_dividend_history():
    fake = [{"date": "2024-06-20", "bonus_rmb": 0.5, "transfer_ratio": 0,
             "bonus_ratio": 0, "plan": "实施"}]
    from finagent.tools import get_dividend_history
    with patch("finagent.tools.fetch", return_value=fake):
        result = get_dividend_history.invoke({"stock_code": "002415"})
    assert "0.5" in result


def test_get_fund_flow():
    fake = [{"date": "2024-10-20", "main_net": 1e8, "super_net": 5e7},
            {"date": "2024-10-21", "main_net": -5e7, "super_net": -2e7}]
    from finagent.tools import get_fund_flow
    with patch("finagent.tools.fetch", return_value=fake):
        result = get_fund_flow.invoke({"stock_code": "002415"})
    assert "主力" in result
