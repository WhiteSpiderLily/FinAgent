"""Tool wrappers for the FinAgent."""
from langchain_core.tools import tool

from finagent.sources import fetch, DataSourceError
from finagent.report import (
    generate_report_tool,
    read_report,
    update_section,
    delete_section,
)
from finagent.skills import read_skill_md, get_finagent_roots


def _fmt_yi(value) -> str:
    """Format absolute yuan value to 亿 with 1 decimal. Returns 'N/A' for None."""
    if value is None or value != value:  # NaN check
        return "N/A"
    return f"{value / 1e8:.1f}亿"


def _fmt_yoy(value) -> str:
    """Format YoY percentage; 'N/A' for None/missing."""
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def _fmt_trend_val(v) -> str:
    """Format a trend table value; 'N/A' for None."""
    return f"{v:>7.1f}" if isinstance(v, (int, float)) else f"{'N/A':>7}"


@tool
def get_company_info(stock_code: str) -> str:
    """获取 A股上市公司基本信息：公司简称、所属行业、上市时间、主营业务。

    Args:
        stock_code: 6 位股票代码，如 002415
    """
    try:
        data = fetch("company_info", stock_code=stock_code)
    except (DataSourceError, ValueError) as e:
        return f"获取公司信息失败: {e}"
    return (
        f"公司: {data['name']}({data['code']})\n"
        f"行业: {data['industry']}\n"
        f"上市时间: {data['listing']}\n"
        f"主营业务: {data['main_biz']}"
    )


def _compute_gross_margin(row):
    income = row.get("OPERATE_INCOME")
    cost = row.get("OPERATE_COST")
    if not income or not cost:
        return None
    return (income - cost) / income * 100


def _compute_debt_ratio(row):
    liabilities = row.get("TOTAL_LIABILITIES")
    assets = row.get("TOTAL_ASSETS")
    if not liabilities or not assets:
        return None
    return liabilities / assets * 100


def _compute_net_cash_ratio(profit_row, cashflow_row):
    net_profit = profit_row.get("PARENT_NETPROFIT")
    op_cash = cashflow_row.get("NETCASH_OPERATE")
    if not net_profit or not op_cash:
        return None
    return op_cash / net_profit


def _format_current_period(profit_row, balance_row, cashflow_row) -> str:
    """Format current-period detail + YoY into compact text."""
    gm = _compute_gross_margin(profit_row)
    debt = _compute_debt_ratio(balance_row)
    ncr = _compute_net_cash_ratio(profit_row, cashflow_row)
    lines = ["利润表:"]
    lines.append(f"  营收: {_fmt_yi(profit_row.get('TOTAL_OPERATE_INCOME'))} ({_fmt_yoy(profit_row.get('TOTAL_OPERATE_INCOME_YOY'))} YoY)")
    lines.append(f"  归母净利: {_fmt_yi(profit_row.get('PARENT_NETPROFIT'))} ({_fmt_yoy(profit_row.get('PARENT_NETPROFIT_YOY'))} YoY)")
    lines.append(f"  扣非净利: {_fmt_yi(profit_row.get('DEDUCT_PARENT_NETPROFIT'))} ({_fmt_yoy(profit_row.get('DEDUCT_PARENT_NETPROFIT_YOY'))} YoY)")
    lines.append(f"  毛利率: {gm:.1f}%" if gm else "  毛利率: N/A")
    lines.append(f"  销售费用: {_fmt_yi(profit_row.get('SALE_EXPENSE'))} | 管理费用: {_fmt_yi(profit_row.get('MANAGE_EXPENSE'))} | 研发费用: {_fmt_yi(profit_row.get('RESEARCH_EXPENSE'))}")
    lines.append("资产负债表:")
    lines.append(f"  总资产: {_fmt_yi(balance_row.get('TOTAL_ASSETS'))} | 负债率: {debt:.1f}%" if debt else "  总资产: N/A")
    lines.append(f"  货币资金: {_fmt_yi(balance_row.get('MONETARYFUNDS'))} | 应收账款: {_fmt_yi(balance_row.get('ACCOUNTS_RECE'))} | 存货: {_fmt_yi(balance_row.get('INVENTORY'))}")
    lines.append("现金流量表:")
    lines.append(f"  经营现金流净额: {_fmt_yi(cashflow_row.get('NETCASH_OPERATE'))}")
    lines.append(f"  净现比(经营现金流/归母净利): {ncr:.2f}" if ncr else "  净现比: N/A")
    return "\n".join(lines)


def _format_trend(profit_df, balance_df, n=8) -> str:
    """Compute 8-period trend for 5 ratios, return compact table."""
    merged = profit_df[["REPORT_DATE", "OPERATE_INCOME", "OPERATE_COST", "PARENT_NETPROFIT"]].copy()
    bal = balance_df[["REPORT_DATE", "TOTAL_LIABILITIES", "TOTAL_ASSETS", "TOTAL_PARENT_EQUITY"]].copy()
    merged = merged.merge(bal, on="REPORT_DATE", how="inner")
    merged = merged.sort_values("REPORT_DATE", ascending=False).head(n)
    lines = [f"近 {len(merged)} 期趋势:"]
    lines.append(f"{'报告期':<12} {'毛利率%':>7} {'净利率%':>7} {'负债率%':>7} {'ROE%':>7}")
    for _, r in merged.iterrows():
        gm = (r["OPERATE_INCOME"] - r["OPERATE_COST"]) / r["OPERATE_INCOME"] * 100 if r["OPERATE_INCOME"] else None
        nm = r["PARENT_NETPROFIT"] / r["OPERATE_INCOME"] * 100 if r["OPERATE_INCOME"] else None
        dr = r["TOTAL_LIABILITIES"] / r["TOTAL_ASSETS"] * 100 if r["TOTAL_ASSETS"] else None
        roe = r["PARENT_NETPROFIT"] / r["TOTAL_PARENT_EQUITY"] * 100 if r.get("TOTAL_PARENT_EQUITY") else None
        date_short = str(r["REPORT_DATE"])[:10]
        lines.append(f"{date_short:<12} {_fmt_trend_val(gm)} {_fmt_trend_val(nm)} {_fmt_trend_val(dr)} {_fmt_trend_val(roe)}")
    return "\n".join(lines)


@tool
def get_financials(stock_code: str, report_period: str) -> str:
    """获取 A股上市公司指定报告期的财务数据：利润表/资产负债表/现金流量表关键科目（含同比）+ 近 8 期趋势。

    Args:
        stock_code: 6 位股票代码，如 002415
        report_period: 报告期，支持 2024Q3 / 2024三季报 / 2024-09-30
    """
    try:
        data = fetch("financials", stock_code=stock_code, report_period=report_period)
    except (DataSourceError, ValueError) as e:
        return f"获取财务数据失败: {e}"
    detail = _format_current_period(
        data["profit_row"], data["balance_row"], data["cashflow_row"]
    )
    trend = (
        _format_trend(data["profit_df"], data["balance_df"])
        if data.get("profit_df") is not None
        else "（趋势数据不可用）"
    )
    return f"=== {data['name']}({data['code']}) {data['report_period']} 财务数据 ===\n{detail}\n\n{trend}"


@tool
def get_valuation(stock_code: str) -> str:
    """获取 A股实时估值：PE(TTM)/PB/总市值/流通市值/换手率/涨跌停/涨跌幅。

    Args:
        stock_code: 6 位股票代码，如 002415
    """
    try:
        d = fetch("valuation", stock_code=stock_code)
    except (DataSourceError, ValueError) as e:
        return str(e)
    lines = [f"{d['name']}({d['code']}) 实时估值:"]
    lines.append(f"  现价: {d['price']:.2f} ({d['change_pct']:+.2f}%)")
    lines.append(f"  PE(TTM): {d['pe_ttm']:.1f} | PB: {d['pb']:.2f}")
    lines.append(f"  总市值: {d['mcap_yi']:.1f}亿 | 流通: {d['float_mcap_yi']:.1f}亿")
    lines.append(f"  换手率: {d['turnover_pct']:.2f}%")
    lines.append(f"  涨停: {d['limit_up']:.2f} | 跌停: {d['limit_down']:.2f}")
    if d.get("is_stale"):
        lines.append(f"  ⚠️ 疑似停牌/废码: {d.get('stale_reason', '成交量为0')}")
    return "\n".join(lines)


@tool
def get_industry_ranking(top_n: int = 20) -> str:
    """获取全行业涨跌幅排名（东财行业板块），定位公司所在行业的热度。

    Args:
        top_n: 返回前 N 名（默认 20）
    """
    try:
        d = fetch("industry_ranking", top_n=top_n)
    except (DataSourceError, ValueError) as e:
        return str(e)
    lines = [f"行业涨跌幅排名（共 {d['total']} 个行业）:"]
    lines.append("涨幅前列:")
    for r in d["top"][:10]:
        lines.append(f"  {r['rank']}. {r['name']} {r['change_pct']:+.2f}% 涨{r['up_count']}跌{r['down_count']} 领涨:{r['leader']}")
    return "\n".join(lines)


@tool
def get_research_reports(stock_code: str, max_pages: int = 3) -> str:
    """获取近期卖方研报标题与摘要。

    Args:
        stock_code: 6 位股票代码
        max_pages: 最多拉取页数（默认 3）
    """
    try:
        reports = fetch("research_reports", stock_code=stock_code, max_pages=max_pages)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not reports:
        return "暂无研报覆盖。"
    lines = [f"共 {len(reports)} 篇研报:"]
    for r in reports[:10]:
        date = (r.get("publishDate") or "")[:10]
        org = r.get("orgSName", "")
        title = (r.get("title") or "")[:60]
        rating = r.get("emRatingName", "")
        lines.append(f"  {date} | {org} | {rating} | {title}")
    return "\n".join(lines)


@tool
def get_holder_change(stock_code: str, page_size: int = 10) -> str:
    """获取股东户数变化趋势，判断筹码集中/分散。

    Args:
        stock_code: 6 位股票代码
        page_size: 取最近 N 期（默认 10）
    """
    try:
        rows = fetch("holder_change", stock_code=stock_code, page_size=page_size)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not rows:
        return "暂无股东户数数据。"
    lines = ["报告期      股东数    环比变化  户均持股"]
    for r in rows[:8]:
        lines.append(f"  {r['date']}  {r['holder_num']:>8}  {r['change_ratio']:>+6.1f}%  {r['avg_shares']:>8.0f}")
    return "\n".join(lines)


@tool
def get_dividend_history(stock_code: str, page_size: int = 20) -> str:
    """获取分红送转历史。

    Args:
        stock_code: 6 位股票代码
        page_size: 取最近 N 期（默认 20）
    """
    try:
        rows = fetch("dividend_history", stock_code=stock_code, page_size=page_size)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not rows:
        return "暂无分红记录。"
    lines = ["日期        每股派息  转增  送股  进度"]
    for r in rows[:10]:
        lines.append(f"  {r['date']}  {r['bonus_rmb']:>6.3f}  {r['transfer_ratio']:>4}  {r['bonus_ratio']:>4}  {r['plan']}")
    return "\n".join(lines)


@tool
def get_fund_flow(stock_code: str) -> str:
    """获取近 120 日主力资金流向趋势（超大/大/中/小单日级）。

    Args:
        stock_code: 6 位股票代码
    """
    try:
        rows = fetch("fund_flow", stock_code=stock_code)
    except (DataSourceError, ValueError) as e:
        return str(e)
    if not rows:
        return "暂无资金流数据。"
    recent = rows[-20:]
    total_main = sum(r["main_net"] for r in recent)
    total_super = sum(r["super_net"] for r in recent)
    lines = [f"近 {len(rows)} 日资金流（近 20 日汇总）:"]
    lines.append(f"  主力净流入: {total_main / 1e8:+.2f}亿")
    lines.append(f"  超大单净额: {total_super / 1e8:+.2f}亿")
    lines.append("近 5 日:")
    for r in rows[-5:]:
        lines.append(f"  {r['date']}  主力 {r['main_net'] / 1e4:+.0f}万  超大 {r['super_net'] / 1e4:+.0f}万")
    return "\n".join(lines)


@tool
def load_skill(name: str) -> str:
    """加载指定 skill 的完整指令。

    可用 skill 列表见每轮 system-reminder 中的 catalog。激活后该 skill 的指令
    在后续对话中持续生效。可同时激活多个 skill。

    Args:
        name: skill 名称(catalog 中列出的 name)
    """
    try:
        return read_skill_md(name)
    except FileNotFoundError:
        return f"未找到 skill: {name}。请检查 system-reminder 中的可用列表或 /reload_skills 后重试。"


@tool
def read_file(path: str) -> str:
    """读取 .finagent/ 路径下的文件(沙箱内)。

    path 相对于 .finagent/ 根目录(如 'skills/news-radar/assets/tpl.md')。
    自动在 ./.finagent 和 ~/.finagent 两处查找,项目本地优先。禁止读取沙箱外文件。

    Args:
        path: 相对 .finagent/ 的路径
    """
    for root in get_finagent_roots():
        candidate = (root / path).resolve()
        # Reject escape: resolved path must live under this root
        try:
            candidate.relative_to(root)
        except ValueError:
            # Path escapes this root's sandbox
            continue
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError as e:
                return f"读取失败: {e}"
    return f"禁止访问: {path} (超出沙箱或未找到)"


tools = [
    get_company_info, get_financials, get_valuation,
    get_industry_ranking, get_research_reports,
    get_holder_change, get_dividend_history, get_fund_flow,
    generate_report_tool, read_report, update_section, delete_section,
    load_skill, read_file,
]
