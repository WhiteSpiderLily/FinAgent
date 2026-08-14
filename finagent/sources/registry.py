from collections.abc import Callable


class DataSourceError(Exception):
    pass


from finagent.sources import akshare_src, eastmoney_src, sina_src, tencent_src

REGISTRY: dict[str, list[Callable]] = {
    "company_info": [akshare_src.company_info],
    "financials": [akshare_src.financials, sina_src.financials],
    "valuation": [tencent_src.valuation],
    "industry_ranking": [akshare_src.industry_ranking_ths, eastmoney_src.industry_ranking],
    "research_reports": [eastmoney_src.research_reports],
    "holder_change": [eastmoney_src.holder_change],
    "dividend_history": [eastmoney_src.dividend_history],
    "fund_flow": [eastmoney_src.fund_flow],
}


def fetch(endpoint, *, registry=REGISTRY, **kwargs):
    sources = registry[endpoint]
    last_err = None
    for fn in sources:
        try:
            return fn(**kwargs)
        except Exception as e:
            last_err = e
    raise DataSourceError(
        f"endpoint '{endpoint}' all {len(sources)} sources failed: {last_err}"
    ) from last_err
