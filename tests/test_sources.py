import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from finagent.sources import akshare_src
from finagent.sources import tencent_src


def _make_tencent_raw():
    vals = [""] * 60
    vals[1] = "海康威视"
    vals[2] = "002415"
    vals[3] = "51.20"
    vals[4] = "51.20"
    vals[5] = "50.50"
    vals[31] = "1.20"
    vals[32] = "2.40"
    vals[33] = "52.00"
    vals[34] = "49.00"
    vals[37] = "187040"
    vals[38] = "4.55"
    vals[39] = "300.45"
    vals[43] = "5.88"
    vals[44] = "360.00"
    vals[45] = "410.88"
    vals[46] = "11.51"
    vals[47] = "56.00"
    vals[48] = "45.00"
    vals[49] = "1.20"
    vals[52] = "300.00"
    return 'v_sz002415="' + "~".join(vals) + '";'


_TENCENT_RAW = _make_tencent_raw()


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
    assert result["is_stale"] is False


def test_valuation_stale_flag(monkeypatch):
    raw = _TENCENT_RAW.replace("~187040~", "~0~")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw.encode("gbk")
    monkeypatch.setattr(
        tencent_src.urllib.request, "urlopen", lambda req, timeout: mock_resp
    )
    result = tencent_src.valuation("002415")
    assert result["is_stale"] is True
    assert result["stale_reason"]


def _fake_company_df():
    return pd.DataFrame({
        "item": ["org_short_name_cn", "affiliate_industry", "total_shares", "float_shares", "listed_date"],
        "value": ["海康威视", {"ind_name": "计算机设备"}, "91.6亿", "90.5亿", "20100528"],
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


def test_normalize_period_formats():
    from finagent.sources.akshare_src import _normalize_period
    assert _normalize_period("2024Q3") == "2024-09-30"
    assert _normalize_period("2024Q1") == "2024-03-31"
    assert _normalize_period("2024Q2") == "2024-06-30"
    assert _normalize_period("2024Q4") == "2024-12-31"
    assert _normalize_period("2024三季报") == "2024-09-30"
    assert _normalize_period("2024年报") == "2024-12-31"
    assert _normalize_period("2024-09-30") == "2024-09-30"


def test_normalize_period_invalid():
    from finagent.sources.akshare_src import _normalize_period
    with pytest.raises(ValueError):
        _normalize_period("invalid")


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
    monkeypatch.setattr(eastmoney_src, "em_get", lambda *a, **kw: fake_resp)
    result = eastmoney_src.industry_ranking(top_n=5)
    assert result["total"] == 2
    assert result["top"][0]["name"] == "电力设备"
    assert result["bottom"][-1]["name"] == "白酒"


def test_industry_ranking_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": {"diff": []}}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda *a, **kw: fake_resp)
    result = eastmoney_src.industry_ranking()
    assert result["total"] == 0


def test_research_reports_success(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": [{"title": "买入", "orgSName": "中信"}], "TotalPage": 1}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda *a, **kw: fake_resp)
    result = eastmoney_src.research_reports("002415", max_pages=1)
    assert len(result) == 1
    assert result[0]["title"] == "买入"


def test_research_reports_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": [], "TotalPage": 1}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda *a, **kw: fake_resp)
    assert eastmoney_src.research_reports("002415", max_pages=1) == []


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


def test_dividend_history_success(monkeypatch):
    fake_data = [{
        "EX_DIVIDEND_DATE": "2024-06-20 00:00:00", "PRETAX_BONUS_RMB": 0.5,
        "TRANSFER_RATIO": 0, "BONUS_RATIO": 0, "ASSIGN_PROGRESS": "实施",
    }]
    monkeypatch.setattr(eastmoney_src, "eastmoney_datacenter", lambda **kw: fake_data)
    result = eastmoney_src.dividend_history("002415")
    assert result[0]["date"] == "2024-06-20"
    assert result[0]["bonus_rmb"] == 0.5


def test_fund_flow_success(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "data": {"klines": [
            "2024-10-20,1000000,200000,300000,400000,100000",
            "2024-10-21,-500000,100000,-200000,-300000,-100000",
        ]}
    }
    monkeypatch.setattr(eastmoney_src, "em_get", lambda *a, **kw: fake_resp)
    result = eastmoney_src.fund_flow("002415")
    assert len(result) == 2
    assert result[0]["date"] == "2024-10-20"
    assert result[0]["main_net"] == 1000000
    assert result[1]["main_net"] == -500000


def test_fund_flow_empty(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"data": {"klines": []}}
    monkeypatch.setattr(eastmoney_src, "em_get", lambda *a, **kw: fake_resp)
    assert eastmoney_src.fund_flow("002415") == []


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


def test_dividend_history_null_fields(monkeypatch):
    """东财返回 JSON null 时 (送转股无现金分红), row.get(key, default) 返回 None 非默认值。
    必须 or default 防 None 进格式化崩溃。"""
    fake_data = [{
        "EX_DIVIDEND_DATE": None, "PRETAX_BONUS_RMB": None,
        "TRANSFER_RATIO": None, "BONUS_RATIO": None, "ASSIGN_PROGRESS": None,
    }]
    monkeypatch.setattr(eastmoney_src, "eastmoney_datacenter", lambda **kw: fake_data)
    result = eastmoney_src.dividend_history("002415")
    assert result[0]["bonus_rmb"] == 0
    assert result[0]["transfer_ratio"] == 0
    assert result[0]["bonus_ratio"] == 0
    assert result[0]["plan"] == ""
    assert result[0]["date"] == ""


def test_holder_change_null_fields(monkeypatch):
    fake_data = [{
        "END_DATE": None, "HOLDER_NUM": None, "HOLDER_NUM_CHANGE": None,
        "HOLDER_NUM_RATIO": None, "AVG_FREE_SHARES": None,
    }]
    monkeypatch.setattr(eastmoney_src, "eastmoney_datacenter", lambda **kw: fake_data)
    result = eastmoney_src.holder_change("002415")
    assert result[0]["holder_num"] == 0
    assert result[0]["change_ratio"] == 0
    assert result[0]["avg_shares"] == 0
