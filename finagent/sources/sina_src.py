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
    r = requests.get(url=url, params=params, headers={"User-Agent": UA}, timeout=15)
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
