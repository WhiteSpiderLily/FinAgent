from finagent.sources._emclient import em_get, UA, REPORT_API, eastmoney_datacenter
from finagent.sources._ticker import norm_ticker, get_prefix


def industry_ranking(top_n: int = 20) -> dict:
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "100", "po": "1", "np": "1",
        "fltt": "2", "invt": "2", "fid": "f3",
        "fs": "m:90+t:2",
        "fields": "f2,f3,f4,f12,f13,f14,f104,f105,f128,f136,f140,f141,f207",
    }
    r = em_get(url, params=params, headers={"User-Agent": UA}, timeout=15)
    items = r.json().get("data", {}).get("diff", [])
    if not items:
        return {"top": [], "bottom": [], "total": 0}
    rows = []
    for i, item in enumerate(items):
        rows.append({
            "rank": i + 1,
            "name": item.get("f14", ""),
            "change_pct": item.get("f3", 0),
            "code": item.get("f12", ""),
            "up_count": item.get("f104", 0),
            "down_count": item.get("f105", 0),
            "leader": item.get("f140", ""),
            "leader_change": item.get("f136", 0),
        })
    return {"top": rows[:top_n], "bottom": rows[-top_n:], "total": len(rows)}


def research_reports(stock_code: str, max_pages: int = 3) -> list[dict]:
    code = norm_ticker(stock_code, stock_only=True)
    all_records = []
    for page in range(1, max_pages + 1):
        params = {
            "industryCode": "*", "pageSize": "100", "industry": "*",
            "rating": "*", "ratingChange": "*",
            "beginTime": "2000-01-01", "endTime": "2030-01-01",
            "pageNo": str(page), "fields": "", "qType": "0",
            "orgCode": "", "code": code, "rcode": "",
            "p": str(page), "pageNum": str(page), "pageNumber": str(page),
        }
        r = em_get(REPORT_API, params=params,
                   headers={"Referer": "https://data.eastmoney.com/"}, timeout=30)
        d = r.json()
        rows = d.get("data") or []
        if not rows:
            break
        all_records.extend(rows)
        if page >= (d.get("TotalPage", 1) or 1):
            break
    return all_records


def holder_change(stock_code: str, page_size: int = 10) -> list[dict]:
    code = norm_ticker(stock_code)
    data = eastmoney_datacenter(
        report_name="RPT_HOLDERNUMLATEST",
        filter_str=f'(SECURITY_CODE="{code}")',
        page_size=page_size,
        sort_columns="END_DATE", sort_types="-1",
    )
    return [{
        "date": str(row.get("END_DATE") or "")[:10],
        "holder_num": row.get("HOLDER_NUM") or 0,
        "change_num": row.get("HOLDER_NUM_CHANGE") or 0,
        "change_ratio": row.get("HOLDER_NUM_RATIO") or 0,
        "avg_shares": row.get("AVG_FREE_SHARES") or 0,
    } for row in data]


def dividend_history(stock_code: str, page_size: int = 20) -> list[dict]:
    code = norm_ticker(stock_code)
    data = eastmoney_datacenter(
        report_name="RPT_SHAREBONUS_DET",
        filter_str=f'(SECURITY_CODE="{code}")',
        page_size=page_size,
        sort_columns="EX_DIVIDEND_DATE", sort_types="-1",
    )
    return [{
        "date": str(row.get("EX_DIVIDEND_DATE") or "")[:10],
        "bonus_rmb": row.get("PRETAX_BONUS_RMB") or 0,
        "transfer_ratio": row.get("TRANSFER_RATIO") or 0,
        "bonus_ratio": row.get("BONUS_RATIO") or 0,
        "plan": row.get("ASSIGN_PROGRESS") or "",
    } for row in data]


def fund_flow(stock_code: str) -> list[dict]:
    code = norm_ticker(stock_code)
    market_code = 1 if get_prefix(code) == "sh" else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": f"{market_code}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "lmt": "120",
    }
    r = em_get(url, params=params,
               headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/",
                        "Origin": "https://quote.eastmoney.com"}, timeout=15)
    klines = r.json().get("data", {}).get("klines", [])
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 6:
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]) if parts[1] != "-" else 0,
                "small_net": float(parts[2]) if parts[2] != "-" else 0,
                "mid_net": float(parts[3]) if parts[3] != "-" else 0,
                "large_net": float(parts[4]) if parts[4] != "-" else 0,
                "super_net": float(parts[5]) if parts[5] != "-" else 0,
            })
    return rows
