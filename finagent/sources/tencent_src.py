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

    def _f(i: int) -> float:
        v = vals[i]
        return float(v) if v else 0.0

    result = {
        "code": code,
        "name": vals[1],
        "price": _f(3),
        "last_close": _f(4),
        "open": _f(5),
        "change_amt": _f(31),
        "change_pct": _f(32),
        "high": _f(33),
        "low": _f(34),
        "amount_wan": _f(37),
        "turnover_pct": _f(38),
        "pe_ttm": _f(39),
        "amplitude_pct": _f(43),
        "float_mcap_yi": _f(44),
        "mcap_yi": _f(45),
        "pb": _f(46),
        "limit_up": _f(47),
        "limit_down": _f(48),
        "vol_ratio": _f(49),
        "pe_static": _f(52),
    }
    result["is_stale"] = (
        result["amount_wan"] == 0
        and result["price"] == result["last_close"]
        and result["price"] > 0
    )
    if result["is_stale"]:
        result["stale_reason"] = "成交量为 0（停牌 / 未开盘 / 废码）"
    return result
