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


def test_financials_fallback_akshare_to_sina():
    fn_a = MagicMock(side_effect=RuntimeError("akshare down"))
    fn_b = MagicMock(return_value={"code": "002415", "profit_row": {}})
    result = fetch("financials", registry={"financials": [fn_a, fn_b]},
                   stock_code="002415", report_period="2024Q3")
    assert result["code"] == "002415"
    fn_a.assert_called_once_with(stock_code="002415", report_period="2024Q3")
    fn_b.assert_called_once_with(stock_code="002415", report_period="2024Q3")
