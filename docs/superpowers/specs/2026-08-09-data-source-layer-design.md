# 数据源分层架构设计

**日期**：2026-08-09
**状态**：待审核
**作者**：FinAgent 团队

## 1. 背景与动机

### 1.1 现状

`finagent/tools.py` 用 langchain `@tool` 装饰器定义 2 个工具（`get_company_info`、`get_financials`），直接调用 akshare 获取 A 股财报数据。`agent.py` 直接 `import tools` 喂给 `create_langchain_agent`。

痛点：**扩展数据源（加 tushare/wind/底层 HTTP 接口）时代码改动大、缺乏统一降级策略、akshare 重度依赖东财 HTTP 有封 IP 风险**。

### 1.2 决策记录

| 议题 | 决策 | 理由 |
|------|------|------|
| 是否上 MCP | **否** | 真实痛点是"加新源代码"，MCP 解决的是进程间通信/外部客户端复用，与多源扩展正交。学习 MCP 作为独立 spike 另做，不绑定生产架构 |
| 是否做 DataSource 抽象 | **是** | 直接解决"加新源"痛点：加源 = 加函数 + 配置行，不改工具/agent/skill |
| 抽象形式 | **声明式 fallback registry**，非 Protocol 类层级 | 源之间是互补关系（各管不同端点），少量端点有多源 fallback。声明式 dict 比强制全实现接口更灵活，避免"接口只有一个实现"反模式 |
| 端点范围 | **8 个核心端点**（最小闭环） | 围绕财报点评任务，不贪多。47 个端点全量引入违反 YAGNI |
| 工具粒度 | **高层任务导向**，非裸端点 | 不暴露 47 个裸端点给 agent（选错风险 + token 爆炸），做成少量高层工具内部组合 |
| 策略归属 | **三层分离**：skill（领域知识）/ 工具（任务组合）/ DataSource（机制+降级） | 机制固化到代码/配置，领域知识进 prompt，可调参数外置配置 |
| 依赖注入 | **`fetch()` 的 registry 作默认参数注入**（`registry=REGISTRY`） | 松耦合：生产用全局，测试注入自定义 registry，无需 mock 模块级全局。工具层因 `@tool` 签名约束（参数暴露给 agent schema）无法注入 fetch，保持 import + mock，是框架限制下的合理 trade-off |

### 1.3 参考资源

- `docs/a-stock-data-instructions.md`（V3.6.0）：A股全栈数据工具包，47 端点实测可用，含可直接移植的 Python 函数实现与多源基础设施。

## 2. 三层架构

```
┌─────────────────────────────────────────────────────┐
│  Skill 层（领域知识 → system prompt）                 │
│   财报点评配方：基本面+估值+行业+研报+筹码+回报+资金    │
└──────────────────┬──────────────────────────────────┘
                   │ 调用高层任务工具
┌──────────────────▼──────────────────────────────────┐
│  工具层（8 个 @tool，任务导向）                        │
│   get_company_info / get_financials / get_valuation  │
│   get_industry_ranking / get_research_reports        │
│   get_holder_change / get_dividend_history           │
│   get_fund_flow                                      │
└──────────────────┬──────────────────────────────────┘
                   │ registry.fetch(endpoint, ...)
┌──────────────────▼──────────────────────────────────┐
│  DataSource 层（声明式 fallback registry）            │
│   REGISTRY = { endpoint: [source_fn, ...] }          │
│   fetch() 按列表顺序尝试，失败自动降级                 │
└─────────────────────────────────────────────────────┘
```

**职责分离**：
- **Skill 层**：告诉 agent "做什么分析用什么数据组合"，不含调用细节
- **工具层**：单个数据维度的获取 + 格式化，对 agent 隐藏多源细节
- **DataSource 层**：纯数据获取机制（归一化、路由、降级、限流、重试），无业务语义

## 3. 文件结构

```
finagent/
  sources/
    __init__.py
    _ticker.py      # norm_ticker(), get_prefix() — 跨源基础设施（从文档 §374/§410 移植）
    _emclient.py    # em_get() — 东财请求统一限流/重试（从文档 §540 移植）
    akshare_src.py  # akshare 源：company_info, financials（从 tools.py 迁移）
    tencent_src.py  # 腾讯源：valuation（从文档 §1.2 tencent_quote 移植）
    sina_src.py     # 新浪源：financials 备胎（从文档 §2193 sina_financial_report 移植）
    eastmoney_src.py # 东财源：industry/reports/holders/dividend/fund_flow（从文档移植）
    registry.py     # REGISTRY dict + fetch() — 声明式 fallback 核心
  tools.py          # 8 个 @tool，调 registry.fetch()，负责格式化输出
  agent.py          # 不变（import tools）
  prompts.py        # 改：RESEARCH_SYSTEM_PROMPT 加入端点组合配方（skill 层）
  config.py         # 不变
  report.py         # 不变
  tui.py            # 不变
  __init__.py
  __main__.py
```

## 4. DataSource 层设计

### 4.1 声明式 Registry

**不使用 Protocol 类层级**。理由：源之间是互补关系（各管不同端点），非全部实现同一接口的替代关系。声明式 dict 更灵活：加源 = 加函数 + 列表项。

```python
# finagent/sources/registry.py
from collections.abc import Callable

# 端点名 → fallback 顺序的源函数列表
REGISTRY: dict[str, list[Callable]] = {
    "company_info":     [akshare_src.company_info, tencent_src.company_info],
    "financials":       [akshare_src.financials, sina_src.financials],
    "valuation":        [tencent_src.valuation],
    "industry_ranking": [eastmoney_src.industry_ranking],
    "research_reports": [eastmoney_src.research_reports],
    "holder_change":    [eastmoney_src.holder_change],
    "dividend_history": [eastmoney_src.dividend_history],
    "fund_flow":        [eastmoney_src.fund_flow],
}


class DataSourceError(Exception):
    """所有源均失败时抛出。"""


def fetch(endpoint: str, *, registry: dict = REGISTRY, **kwargs):
    """按 registry 声明的顺序尝试源函数，首个成功即返回。

    registry 作为默认参数注入（DI）：生产用全局 REGISTRY，测试可传入
    自定义 registry 替换，无需 mock 模块级全局。

    Args:
        endpoint: registry 中的端点名。
        registry: 端点名 → 源函数列表的映射，默认全局 REGISTRY。
        **kwargs: 透传给源函数。

    Returns:
        源函数的返回值（结构化数据）。

    Raises:
        DataSourceError: 所有源均失败。
        KeyError: endpoint 未注册。
    """
    sources = registry[endpoint]
    last_err = None
    for fn in sources:
        try:
            return fn(**kwargs)
        except Exception as e:
            last_err = e
    raise DataSourceError(f"endpoint '{endpoint}' all {len(sources)} sources failed: {last_err}")
```

**加新源 = 两步**：
1. 在对应 `*_src.py` 实现函数（签名与端点约定一致）
2. 在 `REGISTRY[endpoint]` 列表插入该函数

不改工具、不改 agent、不改 skill。

### 4.2 源函数约定

每个源函数：
- 接收原始参数（`stock_code: str` 等）
- **内部自行调用 `_ticker.norm_ticker()` / `_ticker.get_prefix()` 做归一化**（入口统一）
- 调用底层 API（akshare / requests HTTP）
- 返回 **结构化数据**（dict / DataFrame），**不做格式化**（格式化是工具层职责）
- 失败时抛异常（由 `fetch()` 捕获降级）

**反例**：源函数不应返回 str 格式化文本。格式化逻辑属工具层，避免源函数重复格式化。

### 4.3 基础设施（跨源共享）

**`_ticker.py`**（从文档 §374/§410 移植）：
- `norm_ticker(code, stock_only=False) -> str`：归一化各种 ticker 写法为纯 6 位代码 + 市场标识
- `get_prefix(code) -> str`：市场前缀路由（含沪市 ETF/指数白名单，解决 `000001` 歧义）

**`_emclient.py`**（从文档 §540 移植）：
- `em_get(url, params, headers) -> Response`：东财请求统一限流（最小间隔 + 随机抖动）+ 会话复用 + 指数退避重试（429/5xx，403 不重试）

### 4.4 8 端点 fallback 配置

| 端点 | 源（fallback 顺序） | 底层 | 文档函数 |
|------|---------------------|------|----------|
| `company_info` | akshare → tencent | akshare `stock_individual_basic_info_xq` / 腾讯 HTTP | §1.2 |
| `financials` | akshare → sina | akshare `stock_*_sheet_by_report_em` / 新浪 `sina_financial_report` | §2193 |
| `valuation` | tencent | 腾讯 HTTP（不封 IP） | `tencent_quote` §616 |
| `industry_ranking` | eastmoney | 东财 datacenter（走 em_get） | `industry_comparison` §1525 |
| `research_reports` | eastmoney | 东财 reportapi（走 em_get） | `eastmoney_reports` §796 |
| `holder_change` | eastmoney | 东财 datacenter（走 em_get） | `holder_num_change` §1847 |
| `dividend_history` | eastmoney | 东财 datacenter（走 em_get） | `dividend_history` §1879 |
| `fund_flow` | eastmoney | 东财 datacenter（走 em_get） | `stock_fund_flow_120d` §1912 |

## 5. 工具层设计

8 个 `@tool` 装饰函数，位于 `finagent/tools.py`。每个工具：
1. 校验输入（`norm_ticker`）
2. 调 `registry.fetch(endpoint, ...)`
3. 格式化返回值为 agent 友好的文本
4. 异常兜底返回错误提示 str

```python
# finagent/tools.py（示意）
from langchain_core.tools import tool
from finagent.sources.registry import fetch, DataSourceError


@tool
def get_company_info(stock_code: str) -> str:
    """获取 A股上市公司基本信息：公司简称、所属行业、上市时间、主营业务。"""
    try:
        data = fetch("company_info", stock_code=stock_code)
        return _format_company_info(data)
    except (DataSourceError, ValueError) as e:
        return str(e)


@tool
def get_financials(stock_code: str, report_period: str) -> str:
    """获取指定报告期财务数据：三表关键科目（含同比）+ 近 8 期趋势。"""
    ...  # 同上模式


@tool
def get_valuation(stock_code: str) -> str:
    """获取实时估值：PE(TTM)、PB、总市值、流通市值、换手率、涨跌停价。"""
    ...


@tool
def get_industry_ranking(stock_code: str, top_n: int = 20) -> str:
    """获取公司所在行业的成分股排名（市值/涨跌幅），定位行业地位。"""
    ...


@tool
def get_research_reports(stock_code: str, max_pages: int = 3) -> str:
    """获取近期卖方研报标题与摘要，了解市场观点。"""
    ...


@tool
def get_holder_change(stock_code: str, page_size: int = 10) -> str:
    """获取股东户数变化趋势，判断筹码集中/分散。"""
    ...


@tool
def get_dividend_history(stock_code: str, page_size: int = 20) -> str:
    """获取分红历史：派息率、股息率、分红连续性。"""
    ...


@tool
def get_fund_flow(stock_code: str) -> str:
    """获取近 120 日主力资金流向趋势（超大/大/中/小单）。"""
    ...


tools = [
    get_company_info, get_financials, get_valuation,
    get_industry_ranking, get_research_reports,
    get_holder_change, get_dividend_history, get_fund_flow,
]
```

**格式化函数**（`_format_company_info` 等）保留在 `tools.py`，从现有 `tools.py` 的 `_format_current_period` / `_format_trend` 等迁移。

## 6. Skill 层设计

**实现方式**：写入 `finagent/prompts.py` 的 `RESEARCH_SYSTEM_PROMPT`（现有机制，不引入新 skill 框架）。

加入"财报点评数据配方"段落，指导 agent 何时调用哪些工具：

```python
# finagent/prompts.py（节选追加）
SKILL_RECIPES = """
## 数据获取配方

根据用户问题类型，组合调用以下工具：

**财报点评**（用户问某公司某期财报）：
1. get_company_info — 公司基本面
2. get_financials — 财务三表 + 趋势（核心）
3. get_valuation — 当前估值水平
4. get_industry_ranking — 行业地位参照
5. get_research_reports — 市场观点（可选）
6. get_holder_change — 筹码集中度变化
7. get_dividend_history — 分红回报历史
8. get_fund_flow — 主力资金动向

不必每次全调，按问题侧重选取。财务是核心，其余按需。
跨工具数据矛盾时，以 get_financials 为准并标注差异。
"""
```

**后续演进**：若引入 MCP 或独立 skill 机制，此段落可抽成独立 skill 文件。MVP 阶段嵌 prompt 足够。

## 7. 依赖变更

```toml
# pyproject.toml
dependencies = [
    "langchain>=0.3",
    "langchain-deepseek>=0.1",
    "langgraph>=0.2",
    "akshare>=1.14",
    "requests>=2.28",      # 已被 akshare 间接依赖，显式声明（tencent/eastmoney HTTP）
    "rich>=13.0",
    "python-dotenv>=1.0",
    "textual>=0.40",
]
```

**不新增**：`mcp` / `langchain-mcp-adapters`（MCP 否决）、`mootdx`（valuation 用腾讯 HTTP 不需要通达信协议）。

## 8. 测试策略

| 层 | 测试文件 | 策略 |
|----|----------|------|
| `_ticker.py` | `tests/test_ticker.py` | 纯函数：各种 ticker 写法归一化、前缀路由、歧义码处理 |
| `_emclient.py` | `tests/test_emclient.py` | mock requests：限流间隔、429 重试、403 不重试 |
| `*_src.py` | `tests/test_sources.py` | mock 底层（akshare / requests HTTP），测源函数返回结构化数据 |
| `registry.py` | `tests/test_registry.py` | 注入自定义 registry（DI）：验证 fallback 顺序、全失败抛 DataSourceError，无需 mock 全局 |
| `tools.py` | `tests/test_tools.py`（改） | mock registry.fetch，测格式化输出 + 错误兜底 |

**迁移现有测试**：
- `tests/test_tools.py` / `test_tools_helpers.py` → 拆分：纯逻辑测试迁 `test_sources.py`（mock 路径 `finagent.tools.ak` → `finagent.sources.akshare_src.ak`），格式化测试留 `test_tools.py`
- 调用方式 `.invoke({...})` → 直接函数调用（源函数非 langchain tool）

**CI 兼容**：所有测试 mock 底层 HTTP/akshare，无真实网络调用，CI（`.github/workflows/ci.yml`）不受影响。

## 9. 实施顺序

分 4 个阶段，每阶段可独立验证（测试通过）：

**阶段 1：基础设施 + 迁移现有（不改 agent 行为）**
1. 建 `sources/` 目录 + `_ticker.py` + `_emclient.py`
2. 迁移 `tools.py` 现有 akshare 逻辑到 `sources/akshare_src.py`（去 @tool，返回结构化数据）
3. 建 `registry.py` + `fetch()`（仅 company_info、financials 两个端点）
4. 改 `tools.py`：get_company_info / get_financials 改调 `registry.fetch`
5. 迁移测试
- **验证**：现有测试全过，agent 行为不变

**阶段 2：腾讯源（valuation）**
1. 从文档 §1.2 移植 `tencent_quote` 到 `sources/tencent_src.py`
2. 加 `valuation` 端点到 REGISTRY
3. 加 `get_valuation` 工具
4. 加测试
- **验证**：新测试过，agent 多一个估值工具

**阶段 3：东财源（5 端点）**
1. 从文档移植 `industry_comparison` / `eastmoney_reports` / `holder_num_change` / `dividend_history` / `stock_fund_flow_120d` 到 `sources/eastmoney_src.py`
2. 加 5 端点到 REGISTRY
3. 加 5 个工具 + 格式化
4. 加测试
- **验证**：新测试过，8 工具齐全

**阶段 4：Skill 配方 + 新浪备胎**
1. `prompts.py` 追加 `SKILL_RECIPES`
2. 从文档 §2193 移植 `sina_financial_report` 到 `sources/sina_src.py`，加 financials 备胎
3. 加测试
- **验证**：端到端跑通财报点评

## 10. YAGNI 边界（本次不做）

- **MCP 化**：已否决。学习 MCP 作为独立 spike
- **DataSource Protocol 类层级**：声明式 registry 已覆盖，Protocol 是过度抽象
- **多源抽象（DataSource 基类）**：同上
- **HTTP/SSE transport**：MCP 相关，不做
- **剩余 39 端点**：龙虎榜/涨停池/解禁/融资融券/公告/北向/iwencai 等，留第二梯队
- **mootdx（通达信协议）**：valuation 用腾讯 HTTP 够用，不引入通达信客户端
- **配置文件外置（YAML）**：REGISTRY 暂用 Python dict。端点/源稳定后再考虑外置配置
- **真正独立 skill 机制**：配方嵌 system prompt 够用
- **resources/prompts（MCP 其他 capability）**：不要

## 11. 风险与对策

| 风险 | 对策 |
|------|------|
| 东财封 IP（5 端点依赖东财） | 全部走 `_emclient.em_get()` 限流；financials/company_info 有非东财备胎 |
| 文档函数移植 bug | 阶段化迁移，每阶段独立测试；保留文档原函数对照 |
| agent 面前 8 工具选择困难 | skill 配方指导组合；工具 docstring 清晰 |
| fallback 静默吞错（源失败降级但用户不知） | `fetch()` 记录降级日志；工具层可选标注数据源 |
