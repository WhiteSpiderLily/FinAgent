# A股财报点评 Agent — 设计规格

## 背景与目标

从零构建 LangChain Agent，CLI 对话式交互，自动拉取 A股上市公司财报数据，协助分析师完成财报点评。分析师随问随答，Agent 按需拉数据分析，最后 `/report` 汇总成结构化 markdown 报告。

预期成果：一个可运行的 CLI 工具 `python -m finagent`，输入股票代码开始对话，随时 `/report` 生成点评报告文件。

## 任务边界

| 维度 | 范围 | 明确不做 |
|------|------|----------|
| 报告类型 | 财报点评（季报/半年报/年报） | 个股深度、行业报告 |
| 数据来源 | Agent 自动拉取（akshare） | 用户提供原料 |
| 市场 | 仅 A股 | 美股、港股 |
| 建议边界 | 客观分析、影响评估，不出评级 | 买卖评级、目标价 |
| 事件范围 | 仅财报发布 | 并购、政策、突发新闻 |
| 生成流程 | 对话式（多轮交互） | 一锅端、分段检查点 |

## 架构

单 ReAct agent（研究对话）+ 独立报告合成链（`/report` 触发）。两个独立 LLM 调用路径，共享 DeepSeek deepseek-chat。

```
finagent/
├── __init__.py
├── __main__.py         # python -m finagent 入口
├── cli.py              # 交互循环、命令分发、rich 渲染
├── agent.py            # ReAct agent 构建（LangGraph create_react_agent + checkpointer）
├── tools.py            # akshare 工具封装（@tool，3 个工具）
├── report.py           # 报告合成（CLI 拦截 /report 后独立调用，读对话历史）
└── prompts.py          # system prompt + 报告模板
```

### 研究对话数据流

```
用户输入 → agent 决策（直接答 / 调工具）
  → (调工具) akshare 拉数据 → 工具压缩成紧凑摘要 → agent 分析
  → rich 渲染回复
```

### 报告合成数据流（`/report`）

```
CLI 拦截 /report → report.py 读对话历史（checkpointer 中的 messages）
  → LLM 从历史提取公司代码 + 报告期（提取不出则报错提示）
  → 报告写手 prompt（带模板）+ 对话历史 → deepseek-chat 生成
  → markdown → reports/{code}_{period}_点评.md（覆盖写）
```

## 组件细节

### tools.py — 3 个 akshare 工具

通用规则：
- 所有工具 catch akshare 异常，返回错误描述字符串（不抛异常）
- 入口校验 6 位股票代码，无效直接返回提示
- 工具必须压缩 akshare 输出为紧凑文本，禁止直接返回大 DataFrame（防 token 爆炸）

**1. `get_company_info(stock_code: str) -> str`**
- akshare 接口：`stock_individual_info_em(symbol)`
- 返回：公司名、所属行业、主营业务、上市日期

**2. `get_financial_statements(stock_code: str, report_period: str) -> str`**
- akshare 接口：`stock_profit_sheet_by_report_em` + `stock_balance_sheet_by_report_em` + `stock_cash_flow_sheet_by_report_em`
- 每张表拉全部历史 → 期间归一化（Q1→03-31, Q2→06-30, Q3→09-30, Q4→12-31）→ 过滤当前期 + 去年同期（算同比）
- 接受格式：`2024Q3` / `2024三季报` / `2024-09-30`，内部归一化
- 返回紧凑文本，关键科目：
  - 利润表：营收、归母净利、扣非净利、毛利率、四费合计（含同比）
  - 资产负债表：总资产、负债率、货币资金、应收账款、存货
  - 现金流量表：经营现金流净额、净现比（经营现金流/归母净利）

**3. `get_financial_indicators(stock_code: str) -> str`**
- akshare 接口：`stock_financial_analysis_indicator(symbol)`
- 返回近 8 期（2 年）趋势：ROE、毛利率、净利率、资产负债率、经营现金流/营收

### agent.py — ReAct agent

- LangGraph `create_react_agent(model, tools, checkpointer=MemorySaver)`
- 内存级 checkpointer，进程退出即丢（首版不持久化）
- System prompt：角色=A股财报分析助手；约束=客观分析不出评级、只用工具数据不编造数字、仅 A股；不规定固定工作流（被动响应分析师）

### report.py — 报告合成

- 纯对话历史驱动，无工具访问（分析师须在 `/report` 前聊够数据）
- `/report` 无参数，LLM 从历史提取公司代码 + 报告期；提取不出则报错
- 防幻觉：system prompt 强制"只引用对话中出现过的数据，不编造数字"
- ponytail: 已知上限=单公司深度对话可能逼近 64K 上下文，靠紧凑工具输出缓解；超了再加历史摘要

### 报告模板（固定 6 章节）

```markdown
# {公司名}({代码}) {报告期}财报点评

## 一、事件概述
报告期、披露日期、核心财务数据摘要（3-5 个关键数）

## 二、财务分析
营收/归母净利 同比环比、毛利率变化、费用率、现金流质量

## 三、经营要点
对话中讨论的业务分部变化、经营亮点

## 四、影响评估
正面因素 / 负面因素（客观陈述，无评级无目标价）

## 五、风险提示

## 六、免责声明
本报告由 AI 生成，仅供参考，不构成投资建议。
```

### cli.py — 交互

| 输入 | 动作 |
|------|------|
| 自然语言 | 传给 agent，rich 渲染回复 |
| `/report` | 拦截 → 调 report.py 合成 |
| `/clear` | 重置 checkpointer（换公司分析） |
| `/help` | 显示命令 |
| `/quit` | 退出 |

渲染用 `rich`（markdown + 表格）。

## 错误处理

- 工具内部 catch akshare 异常 → 返回错误描述字符串，agent 自然告知用户
- 股票代码格式校验（6 位）在工具入口
- 报告期不存在（未披露）→ 工具返回"该报告期数据尚未披露"
- LLM 失败 → CLI 层 catch，打印错误，不崩

## 依赖

```
langchain
langchain-deepseek
langgraph
akshare
rich
python-dotenv
```

配置：`.env` 放 `DEEPSEEK_API_KEY`（gitignore）。

## 验证

1. `python -m finagent` 启动 CLI
2. 输入 `002415 2024Q3` → 确认 agent 调工具拉到数据，rich 渲染财务分析
3. 追问 `毛利率为什么变化` → 确认 agent 基于已有数据分析
4. `/report` → 确认 `reports/002415_2024Q3_点评.md` 生成，含 6 章节，数字与对话一致
5. `/clear` → 确认记忆重置
6. 单元测试：mock akshare 返回，测工具格式化输出（关键科目提取 + 同比计算）；mock 对话历史，测报告合成含全部章节

## ponytail 标记（已知简化与上限）

- 上下文：单公司深度对话靠紧凑工具输出缓解，未加历史摘要（超 64K 再加）
- 报告合成纯历史驱动，无结构化数据捕获（数字不准再加）
- 会话不持久化（需要跨会话再加）
