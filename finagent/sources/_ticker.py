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
