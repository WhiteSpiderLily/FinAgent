# 多空辩论 (Bull-Bear Debate) 设计

## 概述

为 FinAgent 增加多 agent 协作能力。主 agent 通过 `select_agent` 工具启动 subagent，subagent 继承主会话上下文（fork 式），独立运行至完成并返回结果。首个用例为多空辩论：bull（多方）与 bear（空方）两个 subagent 分别给出看多/看空分析。

## 范围

### 现在做

- Subagent 定义与生命周期管理（SubagentDef dataclass）
- `select_agent` 工具：fork 上下文 → 创建 subagent → 运行至完成 → 返回结果
- 工具权限管理：全局黑名单 + per-agent tools/disallowed_tools
- 上下文 fork：读主 agent state → 预填到 subagent
- Bull / Bear subagent 定义与 prompt
- system-reminder 注入可用 subagent 列表
- 主 agent prompt 追加多空辩论使用指引

### Deferred（到 market research 场景再做）

- ctrl+b 后台异步执行
- Task 管理工具集（list/get/create/update）
- 后台 agent 工具白名单

## 架构

### 文件变更

| 文件 | 变更类型 | 内容 |
|------|----------|------|
| `finagent/subagents.py` | 新增 | SubagentDef、bull/bear 定义、fork_and_run、resolve_tools、select_agent 工具、_AgentContext |
| `finagent/agent.py` | 改 | tools 列表追加 select_agent |
| `finagent/tui.py` | 改 | system-reminder 注入 subagent catalog、_start_stream 中更新 _ctx、action_interrupt 中调用 cancel_all_subagents |
| `finagent/prompts.py` | 改 | SKILL_RECIPES 追加多空辩论使用指引 |

### 协作模式

中心化模式。subagent 间不通信，不与用户对话。主 agent 通过 select_agent 工具启动 subagent，subagent 运行至完成后返回文本结果，主 agent 综合结果输出。

```
用户: "分析 002415 多空观点"
  ↓
主 agent (DeepSeek) 一轮返回两个 tool call:
  select_agent(name="bull", task="分析 002415 2024Q3 看多理由")
  select_agent(name="bear", task="分析 002415 2024Q3 看空理由")
  ↓
LangGraph ToolNode 并行执行:
  bull subagent: fork context → 10 turns max → 返回看多分析
  bear subagent: fork context → 10 turns max → 返回看空分析
  ↓
主 agent 收到两方结果，综合输出平衡总结
```

## 详细设计

### SubagentDef 数据结构

```python
@dataclass
class SubagentDef:
    name: str                              # 唯一标识
    description: str                       # system-reminder 中展示
    system_prompt: str                     # subagent 角色提示
    max_turns: int = 10                    # 安全阀上限，不强制跑满
    tools: list[str] | None = None         # None = 全量工具减黑名单
    disallowed_tools: list[str] | None = None
```

### 全局工具黑名单

所有 subagent 禁止使用的工具：

```python
GLOBAL_TOOL_BLACKLIST = frozenset({
    "generate_report_tool",   # subagent 不写报告
    "update_section",         # 不编辑报告
    "delete_section",         # 不删除报告段落
    "select_agent",           # 禁止递归 spawn sub-subagent
})
```

允许使用的工具（不在黑名单中）：get_company_info, get_financials, get_valuation, get_industry_ranking, get_research_reports, get_holder_change, get_dividend_history, get_fund_flow, read_report, read_file, load_skill。

### Bull / Bear 定义

```python
BULL_SYSTEM_PROMPT = """你是一名 A股多方分析师。从给定数据和上下文中找出支持看多的理由。

规则：
1. 聚焦正面信号：营收增长、利润扩张、估值偏低、资金流入、行业景气、筹码集中。
2. 只使用工具返回的数据和继承的对话上下文，绝不编造数字。
3. 不做买卖评级，不给目标价。
4. 优先使用上下文已有数据，仅在需要补充时调用工具。
5. 输出 3-5 个核心看多逻辑，每个附数据支撑。
"""

BEAR_SYSTEM_PROMPT = """你是一名 A股空方分析师。从给定数据和上下文中找出支持看空的理由。

规则：
1. 聚焦负面信号：利润下滑、毛利率收窄、估值偏高、债务风险、资金流出、竞争恶化。
2. 只使用工具返回的数据和继承的对话上下文，绝不编造数字。
3. 不做买卖评级，不给目标价。
4. 优先使用上下文已有数据，仅在需要补充时调用工具。
5. 输出 3-5 个核心看空逻辑，每个附数据支撑。
"""

SUBAGENTS = {
    "bull": SubagentDef(
        name="bull",
        description="多方分析师，从财务/估值/资金/行业角度给出看多理由",
        system_prompt=BULL_SYSTEM_PROMPT,
    ),
    "bear": SubagentDef(
        name="bear",
        description="空方分析师，从财务/估值/资金/行业角度给出看空理由",
        system_prompt=BEAR_SYSTEM_PROMPT,
    ),
}
```

### Tool 解析

```python
def resolve_tools(sub: SubagentDef) -> list:
    """计算 subagent 有效工具集：(sub.tools or all) - sub.disallowed_tools - GLOBAL_BLACKLIST"""
    from finagent.tools import tools as all_tools
    tool_map = {t.name: t for t in all_tools}

    if sub.tools:
        resolved = [tool_map[n] for n in sub.tools if n in tool_map]
    else:
        resolved = list(all_tools)

    if sub.disallowed_tools:
        resolved = [t for t in resolved if t.name not in set(sub.disallowed_tools)]

    return [t for t in resolved if t.name not in GLOBAL_TOOL_BLACKLIST]
```

bull/bear 不声明 tools/disabled_tools → 自动获得 all_tools - GLOBAL_BLACKLIST = 8 个金融数据工具 + read_report + load_skill + read_file = 11 个工具。

### Late-binding 上下文访问

select_agent 需要读取主 agent 的对话状态来 fork 上下文。但 agent 在 select_agent 之后创建（tools 先于 agent 构造）。使用 late-binding 容器解决：

```python
class _AgentContext:
    """可变容器，TUI 初始化后注入，select_agent 运行时读取。"""
    agent = None
    thread_id: str | None = None

    def get_messages(self) -> list:
        if not self.agent or not self.thread_id:
            return []
        state = self.agent.get_state(
            config={"configurable": {"thread_id": self.thread_id}}
        )
        return state.values.get("messages", [])

_ctx = _AgentContext()
```

### select_agent 工具

```python
@tool
def select_agent(name: str, task: str) -> str:
    """启动指定 subagent 执行任务。继承当前对话上下文，独立运行至完成。

    可用 subagent 见每轮 system-reminder。subagent 不可与用户对话。

    Args:
        name: subagent 名称
        task: 任务指令
    """
    sub = SUBAGENTS.get(name)
    if sub is None:
        return f"未知 subagent: {name}。可用: {list(SUBAGENTS.keys())}"

    context_messages = _ctx.get_messages()
    return fork_and_run(sub, task, context_messages)
```

### fork_and_run 执行

```python
import threading
from langgraph.errors import GraphRecursionError

# 取消信号：key = sub_thread_id, value = threading.Event
_cancel_tokens: dict[str, threading.Event] = {}


def cancel_all_subagents():
    """TUI Esc 中断时调用，信号所有运行中的 subagent 停止。"""
    for token in _cancel_tokens.values():
        token.set()


def fork_and_run(sub: SubagentDef, task: str, context_messages: list) -> str:
    """Fork 上下文 → 创建 subagent → 运行至完成 → 返回最终文本。

    使用 .stream() 而非 .invoke()，在每步之间检查取消信号。
    """
    from finagent.config import get_llm

    sub_tools = resolve_tools(sub)
    sub_checkpointer = MemorySaver()
    sub_agent = create_langchain_agent(
        model=get_llm(),
        tools=sub_tools,
        system_prompt=sub.system_prompt,
        checkpointer=sub_checkpointer,
    )

    # Fork: 预填主会话上下文
    sub_thread = str(uuid.uuid4())
    cancel_token = threading.Event()
    _cancel_tokens[sub_thread] = cancel_token

    sub_agent.update_state(
        config={"configurable": {"thread_id": sub_thread}},
        values={"messages": context_messages},
    )

    # 运行（流式，每步检查取消信号）
    try:
        for _event in sub_agent.stream(
            {"messages": [HumanMessage(content=task)]},
            config={
                "configurable": {"thread_id": sub_thread},
                "recursion_limit": sub.max_turns * 2,
            },
        ):
            if cancel_token.is_set():
                break
    except GraphRecursionError:
        pass  # 达到 max_turns 上限，继续提取已有输出
    finally:
        _cancel_tokens.pop(sub_thread, None)

    # 提取最后一条 AI 文本
    state = sub_agent.get_state(
        config={"configurable": {"thread_id": sub_thread}}
    )
    for msg in reversed(state.values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return f"({sub.name} 未产生输出)"
```

subagent 使用独立的 MemorySaver，函数返回后被 GC 回收，不持久化，不污染主会话。

### max_turns → recursion_limit

LangGraph create_agent 每轮 = 1 agent node + 1 tool node = 2 步。recursion_limit = max_turns * 2。agent 提前完成（无 tool call 的纯文本回复）时自动退出，不强制跑满。GraphRecursionError 时提取已有输出。`from langgraph.errors import GraphRecursionError`。

### TUI 集成

**_start_stream 中更新 _ctx（统一入口）：**

```python
async def _start_stream(self, user_input: str) -> None:
    _ctx.agent = self.agent
    _ctx.thread_id = self.thread_id
    # ... existing streaming logic
```

一处覆盖 init / clear / resume 全部场景。

**_build_user_message 追加 subagent catalog：**

```python
def render_subagent_catalog() -> str:
    """渲染可用 subagent 列表，同 skill catalog 模式。"""
    lines = [f"- {s.name}: {s.description}" for s in SUBAGENTS.values()]
    return "\n".join(lines)
```

```python
# Subagent catalog (every turn)
parts.append(
    f"<system-reminder>\n"
    f"可用 subagents (用 select_agent 工具启动):\n"
    f"{render_subagent_catalog()}\n"
    f"</system-reminder>"
)
```

**执行期间 TUI 显示：**

select_agent 被调用时显示 `🔧 select_agent ⏳`，完成时 `🔧 select_agent ✓`。subagent 内部 tool call 不可见。采用现有 tool call 渲染逻辑，无需改动。

**action_interrupt 中取消 subagent：**

```python
def action_interrupt(self) -> None:
    """Esc handler: cancel current streaming worker + running subagents."""
    from finagent.subagents import cancel_all_subagents
    cancel_all_subagents()
    if self._streaming_worker is not None and self._streaming_worker.is_running:
        self._streaming_worker.cancel()
        self._add_message("[已中断]", classes="message-queued")
```

### agent.py 变更

```python
from finagent.tools import tools
from finagent.subagents import select_agent

def create_agent():
    agent = create_langchain_agent(
        model=llm,
        tools=tools + [select_agent],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )
    return agent
```

### prompts.py 变更

SKILL_RECIPES 追加：

```
## 多空辩论

当用户要求多空分析/看多看空/正反方对比时：
1. select_agent(name="bull", task="<具体分析指令>")
2. select_agent(name="bear", task="<具体分析指令>")
3. 综合两方返回结果，给出平衡总结

两步应在同一轮发出（单轮多 tool call），确保并行执行。
```

### 并行执行

同一轮 AI message 中多个 tool call 由 LangGraph ToolNode 处理。`.astream()` 触发 async path，sync tool 经 asyncio.to_thread 包装 + asyncio.gather → 真并行。此机制已验证可行。

**但并行前提是 DeepSeek 在同一轮 emit 两个 tool call。** 若 DeepSeek 每轮只 emit 一个 → 自动顺序执行（2× 延迟）。select_agent 改 async def 不解决此问题——这是 LLM emit 行为，不是执行层问题。

实现时第一步验证 DeepSeek 多 tool call 行为。若不可靠，fallback：

```python
@tool
def run_debate(stock_code: str, report_period: str) -> str:
    """启动多空辩论：并行 spawn bull + bear，返回综合结果。"""
    import asyncio
    context = _ctx.get_messages()
    bull_task = asyncio.to_thread(fork_and_run, SUBAGENTS["bull"],
                                  f"分析 {stock_code} {report_period} 看多理由", context)
    bear_task = asyncio.to_thread(fork_and_run, SUBAGENTS["bear"],
                                  f"分析 {stock_code} {report_period} 看空理由", context)
    bull_result, bear_result = asyncio.gather(bull_task, bear_task)
    return f"## 多方观点\n{bull_result}\n\n## 空方观点\n{bear_result}"
```

run_debate 保证并行（不依赖 LLM emit 行为），但牺牲通用性（硬编码 bull+bear）。select_agent 保留给未来需要单 agent 的场景。

## 错误处理

| 场景 | 处理 |
|------|------|
| 未知 subagent name | 返回错误提示 + 可用列表 |
| GraphRecursionError (max_turns) | 提取已有输出，不 crash |
| subagent 内部工具报错 | 工具自身已有 try/except，返回错误文本给 subagent |
| DeepSeek API 错误 | 传播给主 agent tool 层，TUI 显示错误 |
| _ctx 未初始化 | get_messages() 返回空列表，subagent 无上下文运行 |

## 测试计划

- `test_resolve_tools`: 验证工具解析逻辑（黑名单/声明/disallowed 交互组合）
- `test_select_agent_unknown_name`: 返回错误提示
- `test_fork_and_run`: mock LLM，验证 context fork + 运行 + 最终文本提取
- `test_max_turns_enforcement`: 验证 recursion_limit = max_turns * 2
- `test_subagent_catalog_render`: 验证 system-reminder 格式
- `test_global_blacklist`: 验证黑名单工具被排除

## 已知限制

1. **Token 成本**：fork 全量上下文给每个 subagent，N 个 subagent = N× input tokens。
2. **subagent 可能重复取数**：prompt 已加"优先使用上下文已有数据"缓解。
3. **DeepSeek 编排可靠性**：DeepSeek 可能不在同一轮 emit 两个 tool call，导致顺序执行（2× 时间）。fallback 为 run_debate 单工具。实现时第一步验证。
4. **subagent 对话记录不持久化**：subagent 使用独立 MemorySaver，函数返回后 GC 回收。
5. **Esc 中断后 subagent 可能耗时 1 步才停止**：stream 在两个 graph step 之间检查取消信号，正在执行的 LLM 调用无法中途取消。
