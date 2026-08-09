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
