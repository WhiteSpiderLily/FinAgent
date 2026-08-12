# Bull-Bear Debate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-agent orchestration to FinAgent: a `select_agent` tool that spawns subagents with forked context, enabling bull-bear debate as the first use case.

**Architecture:** Centralized orchestration — main agent calls `select_agent` tool, which forks conversation context into a fresh subagent with its own system prompt and tool subset. Subagent runs via `.stream()` to completion with cancellation support, returns final text. Bull/bear are built-in subagent definitions in a new `finagent/subagents.py` module.

**Tech Stack:** Python 3.12, LangGraph (create_agent), LangChain (@tool), DeepSeek (ChatDeepSeek), Textual TUI, pytest.

## Global Constraints

- Project virtualenv: `.venv` (activate with `source .venv/bin/activate`)
- Tests run sequentially: `pytest tests/ -v` (never parallel, per CLAUDE.md)
- No stubs — all code must be functional
- Existing test suite (184 tests) must remain green after each task
- LangGraph import: `from langgraph.errors import GraphRecursionError`
- LLM factory: `from finagent.config import get_llm` (returns ChatDeepSeek)
- Agent factory: `from langchain.agents import create_agent as create_langchain_agent`
- Checkpointer: `from langgraph.checkpoint.memory import MemorySaver`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `finagent/subagents.py` | **NEW** — SubagentDef, GLOBAL_TOOL_BLACKLIST, SUBAGENTS dict, BULL/BEAR prompts, resolve_tools, render_subagent_catalog, _AgentContext, _ctx, _cancel_tokens, cancel_all_subagents, fork_and_run, select_agent tool |
| `finagent/agent.py` | **MODIFY** — import select_agent, append to tools list |
| `finagent/prompts.py` | **MODIFY** — append debate recipe to SKILL_RECIPES |
| `finagent/tui.py` | **MODIFY** — _start_stream ctx update, _build_user_message catalog injection, action_interrupt cancel |
| `tests/test_subagents.py` | **NEW** — all tests for subagent module |

---

## Task 1: Core Definitions + Tool Resolution

**Files:**
- Create: `finagent/subagents.py`
- Test: `tests/test_subagents.py`

**Interfaces:**
- Consumes: `finagent.tools.tools` (the 14-tool list)
- Produces: `SubagentDef`, `GLOBAL_TOOL_BLACKLIST`, `SUBAGENTS`, `resolve_tools()`, `render_subagent_catalog()`

- [ ] **Step 1: Write failing tests for resolve_tools and render_subagent_catalog**

```python
# tests/test_subagents.py
"""Tests for subagent module."""
from finagent.subagents import (
    SubagentDef,
    GLOBAL_TOOL_BLACKLIST,
    SUBAGENTS,
    resolve_tools,
    render_subagent_catalog,
)


class TestResolveTools:
    """resolve_tools: (sub.tools or all) - sub.disallowed_tools - GLOBAL_BLACKLIST."""

    def test_default_returns_all_minus_blacklist(self):
        """No tools/disallowed declared → all tools minus blacklist."""
        sub = SubagentDef(name="test", description="t", system_prompt="t")
        resolved = resolve_tools(sub)
        resolved_names = {t.name for t in resolved}
        # Blacklist items must NOT appear
        assert "generate_report_tool" not in resolved_names
        assert "update_section" not in resolved_names
        assert "delete_section" not in resolved_names
        assert "select_agent" not in resolved_names
        # Non-blacklist tools MUST appear
        assert "get_company_info" in resolved_names
        assert "get_financials" in resolved_names
        assert "read_report" in resolved_names
        assert "load_skill" in resolved_names

    def test_explicit_tools_only(self):
        """Declared tools list → only those minus blacklist."""
        sub = SubagentDef(
            name="test", description="t", system_prompt="t",
            tools=["get_company_info", "generate_report_tool"],
        )
        resolved = resolve_tools(sub)
        resolved_names = {t.name for t in resolved}
        assert resolved_names == {"get_company_info"}

    def test_disallowed_tools_removed(self):
        """disallowed_tools removes from the default set."""
        sub = SubagentDef(
            name="test", description="t", system_prompt="t",
            disallowed_tools=["get_valuation", "load_skill"],
        )
        resolved = resolve_tools(sub)
        resolved_names = {t.name for t in resolved}
        assert "get_valuation" not in resolved_names
        assert "load_skill" not in resolved_names
        assert "get_company_info" in resolved_names

    def test_explicit_plus_disallowed(self):
        """tools + disallowed_tools: disallowed wins."""
        sub = SubagentDef(
            name="test", description="t", system_prompt="t",
            tools=["get_company_info", "get_financials", "get_valuation"],
            disallowed_tools=["get_valuation"],
        )
        resolved = resolve_tools(sub)
        resolved_names = {t.name for t in resolved}
        assert resolved_names == {"get_company_info", "get_financials"}

    def test_blacklist_always_wins_over_explicit(self):
        """Even if explicitly listed, blacklist items excluded."""
        sub = SubagentDef(
            name="test", description="t", system_prompt="t",
            tools=["get_company_info", "delete_section", "select_agent"],
        )
        resolved = resolve_tools(sub)
        resolved_names = {t.name for t in resolved}
        assert resolved_names == {"get_company_info"}


class TestRenderCatalog:
    def test_catalog_contains_all_subagents(self):
        """render_subagent_catalog lists every defined subagent."""
        catalog = render_subagent_catalog()
        assert "bull" in catalog
        assert "bear" in catalog
        # Each line format: "- name: description"
        for line in catalog.splitlines():
            assert line.startswith("- ")

    def test_catalog_includes_descriptions(self):
        catalog = render_subagent_catalog()
        assert "多方" in catalog or "看多" in catalog
        assert "空方" in catalog or "看空" in catalog


class TestSubagentDefs:
    def test_bull_exists(self):
        assert "bull" in SUBAGENTS
        assert SUBAGENTS["bull"].name == "bull"
        assert SUBAGENTS["bull"].system_prompt

    def test_bear_exists(self):
        assert "bear" in SUBAGENTS
        assert SUBAGENTS["bear"].name == "bear"
        assert SUBAGENTS["bear"].system_prompt

    def test_default_max_turns(self):
        assert SUBAGENTS["bull"].max_turns == 10
        assert SUBAGENTS["bear"].max_turns == 10

    def test_blacklist_contents(self):
        assert "generate_report_tool" in GLOBAL_TOOL_BLACKLIST
        assert "update_section" in GLOBAL_TOOL_BLACKLIST
        assert "delete_section" in GLOBAL_TOOL_BLACKLIST
        assert "select_agent" in GLOBAL_TOOL_BLACKLIST
        assert "read_report" not in GLOBAL_TOOL_BLACKLIST
        assert "load_skill" not in GLOBAL_TOOL_BLACKLIST
        assert "read_file" not in GLOBAL_TOOL_BLACKLIST
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_subagents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finagent.subagents'`

- [ ] **Step 3: Implement subagents.py core definitions**

```python
# finagent/subagents.py
"""Subagent orchestration: definitions, tool resolution, and select_agent tool."""
import threading
import uuid
from dataclasses import dataclass

from langchain_core.tools import tool


# ── Subagent definition ──────────────────────────────────────────────

@dataclass
class SubagentDef:
    """Definition for one spawnable subagent."""
    name: str
    description: str
    system_prompt: str
    max_turns: int = 10
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None


# ── Global tool blacklist (all subagents forbidden) ──────────────────

GLOBAL_TOOL_BLACKLIST = frozenset({
    "generate_report_tool",
    "update_section",
    "delete_section",
    "select_agent",
})


# ── Bull / Bear system prompts ───────────────────────────────────────

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


# ── Built-in subagent registry ───────────────────────────────────────

SUBAGENTS: dict[str, SubagentDef] = {
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


# ── Tool resolution ──────────────────────────────────────────────────

def resolve_tools(sub: SubagentDef) -> list:
    """Compute effective tool list: (sub.tools or all) - sub.disallowed_tools - GLOBAL_BLACKLIST."""
    from finagent.tools import tools as all_tools
    tool_map = {t.name: t for t in all_tools}

    if sub.tools:
        resolved = [tool_map[n] for n in sub.tools if n in tool_map]
    else:
        resolved = list(all_tools)

    if sub.disallowed_tools:
        disallowed = set(sub.disallowed_tools)
        resolved = [t for t in resolved if t.name not in disallowed]

    return [t for t in resolved if t.name not in GLOBAL_TOOL_BLACKLIST]


def render_subagent_catalog() -> str:
    """Render available subagent list for system-reminder injection."""
    lines = [f"- {s.name}: {s.description}" for s in SUBAGENTS.values()]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_subagents.py -v`
Expected: PASS — all 11 tests green

- [ ] **Step 5: Run full suite to check no regressions**

Run: `pytest tests/ -v`
Expected: PASS — all existing tests + new tests green

- [ ] **Step 6: Commit**

```bash
git add finagent/subagents.py tests/test_subagents.py
git commit -m "feat: subagent core definitions + tool resolution

SubagentDef dataclass, GLOBAL_TOOL_BLACKLIST, bull/bear system
prompts, resolve_tools, render_subagent_catalog. TDD with 11 tests."
```

---

## Task 2: Execution Engine — fork_and_run + select_agent + Cancellation

**Files:**
- Modify: `finagent/subagents.py` (append execution code)
- Test: `tests/test_subagents.py` (append execution tests)

**Interfaces:**
- Consumes: `SubagentDef`, `resolve_tools` from Task 1; `finagent.config.get_llm`; `langchain.agents.create_agent as create_langchain_agent`; `langgraph.checkpoint.memory.MemorySaver`; `langgraph.errors.GraphRecursionError`
- Produces: `_AgentContext`, `_ctx`, `cancel_all_subagents()`, `fork_and_run(sub, task, context_messages) -> str`, `select_agent` (LangChain tool)

- [ ] **Step 1: Write failing tests for select_agent and fork_and_run**

Append to `tests/test_subagents.py`:

```python
import threading
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphRecursionError


class TestSelectAgentUnknownName:
    """select_agent returns error for unknown subagent name."""

    def test_unknown_name_returns_error(self):
        from finagent.subagents import select_agent
        result = select_agent.invoke({"name": "nonexistent", "task": "test"})
        assert "未知" in result or "nonexistent" in result
        assert "bull" in result  # available names listed

    def test_known_name_does_not_return_error(self):
        """Known name should not return the 'unknown' error message."""
        from finagent.subagents import select_agent
        with patch("finagent.subagents.fork_and_run", return_value="mocked result"):
            with patch("finagent.subagents._ctx") as mock_ctx:
                mock_ctx.get_messages.return_value = []
                result = select_agent.invoke({"name": "bull", "task": "test"})
                assert result == "mocked result"


class TestCancelAllSubagents:
    """cancel_all_subagents signals all registered tokens."""

    def test_cancel_all_sets_all_tokens(self):
        from finagent.subagents import _cancel_tokens, cancel_all_subagents
        t1 = threading.Event()
        t2 = threading.Event()
        _cancel_tokens["a"] = t1
        _cancel_tokens["b"] = t2
        try:
            cancel_all_subagents()
            assert t1.is_set()
            assert t2.is_set()
        finally:
            _cancel_tokens.pop("a", None)
            _cancel_tokens.pop("b", None)


class TestForkAndRun:
    """fork_and_run: fork context → create subagent → run → return text.

    Mock targets: finagent.subagents.get_llm and
    finagent.subagents.create_langchain_agent — these are module-level
    names in subagents.py, so patching them intercepts the call.
    """

    @patch("finagent.subagents.get_llm")
    @patch("finagent.subagents.create_langchain_agent")
    def test_forks_context_messages(self, mock_create, mock_llm):
        """Subagent is pre-seeded with context_messages via update_state."""
        from finagent.subagents import fork_and_run, SubagentDef

        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_agent.stream.return_value = iter([])
        mock_state = MagicMock()
        mock_state.values = {"messages": [AIMessage(content="看多分析结果")]}
        mock_agent.get_state.return_value = mock_state

        sub = SubagentDef(name="test", description="t", system_prompt="t", max_turns=5)
        context = [HumanMessage(content="之前的对话")]
        result = fork_and_run(sub, "分析任务", context)

        # Verify update_state called with context messages
        update_calls = mock_agent.update_state.call_args_list
        assert len(update_calls) == 1
        passed_values = update_calls[0].kwargs.get("values") or update_calls[0][1].get("values")
        assert passed_values["messages"] is context

    @patch("finagent.subagents.get_llm")
    @patch("finagent.subagents.create_langchain_agent")
    def test_recursion_limit_is_max_turns_times_two(self, mock_create, mock_llm):
        """recursion_limit config = max_turns * 2."""
        from finagent.subagents import fork_and_run, SubagentDef

        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_agent.stream.return_value = iter([])
        mock_state = MagicMock()
        mock_state.values = {"messages": [AIMessage(content="done")]}
        mock_agent.get_state.return_value = mock_state

        sub = SubagentDef(name="test", description="t", system_prompt="t", max_turns=7)
        fork_and_run(sub, "task", [])

        stream_kwargs = mock_agent.stream.call_args
        config = stream_kwargs.kwargs.get("config") or stream_kwargs[1].get("config")
        assert config["recursion_limit"] == 14  # 7 * 2

    @patch("finagent.subagents.get_llm")
    @patch("finagent.subagents.create_langchain_agent")
    def test_returns_last_ai_message_text(self, mock_create, mock_llm):
        """Extracts last AIMessage with content from final state."""
        from finagent.subagents import fork_and_run, SubagentDef

        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_agent.stream.return_value = iter([])
        mock_state = MagicMock()
        mock_state.values = {"messages": [
            HumanMessage(content="task"),
            AIMessage(content="第一轮"),
            AIMessage(content="最终结论"),
        ]}
        mock_agent.get_state.return_value = mock_state

        sub = SubagentDef(name="test", description="t", system_prompt="t")
        result = fork_and_run(sub, "task", [])
        assert result == "最终结论"

    @patch("finagent.subagents.get_llm")
    @patch("finagent.subagents.create_langchain_agent")
    def test_returns_fallback_when_no_ai_output(self, mock_create, mock_llm):
        """No AIMessage with content → fallback string."""
        from finagent.subagents import fork_and_run, SubagentDef

        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_agent.stream.return_value = iter([])
        mock_state = MagicMock()
        mock_state.values = {"messages": [HumanMessage(content="task")]}
        mock_agent.get_state.return_value = mock_state

        sub = SubagentDef(name="test", description="t", system_prompt="t")
        result = fork_and_run(sub, "task", [])
        assert "未产生输出" in result

    @patch("finagent.subagents.get_llm")
    @patch("finagent.subagents.create_langchain_agent")
    def test_graph_recursion_error_recovers(self, mock_create, mock_llm):
        """GraphRecursionError → still extracts partial output."""
        from finagent.subagents import fork_and_run, SubagentDef

        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_agent.stream.side_effect = GraphRecursionError("limit")
        mock_state = MagicMock()
        mock_state.values = {"messages": [AIMessage(content="部分结果")]}
        mock_agent.get_state.return_value = mock_state

        sub = SubagentDef(name="test", description="t", system_prompt="t")
        result = fork_and_run(sub, "task", [])
        assert result == "部分结果"

    @patch("finagent.subagents.get_llm")
    @patch("finagent.subagents.create_langchain_agent")
    def test_cancellation_breaks_stream(self, mock_create, mock_llm):
        """Cancel signal set mid-stream → loop breaks, partial output returned.

        Uses a generator that sets the cancel token after the first yield.
        The for-loop in fork_and_run checks cancel_token AFTER consuming
        each event. If the check works, only 2 events are consumed (the
        second triggers break). Without the check, all 3 are consumed.
        """
        from finagent.subagents import fork_and_run, SubagentDef, _cancel_tokens

        mock_agent = MagicMock()
        mock_create.return_value = mock_agent

        consumed = []

        def fake_stream(*args, **kwargs):
            consumed.append(1)
            yield {"s": 1}
            # Set cancel token after first yield — runs when generator resumes
            for token in _cancel_tokens.values():
                token.set()
            consumed.append(2)
            yield {"s": 2}
            consumed.append(3)
            yield {"s": 3}

        mock_agent.stream.side_effect = fake_stream
        mock_state = MagicMock()
        mock_state.values = {"messages": [AIMessage(content="partial")]}
        mock_agent.get_state.return_value = mock_state

        sub = SubagentDef(name="test", description="t", system_prompt="t")
        result = fork_and_run(sub, "task", [])

        # Cancel check should prevent consuming all 3 events
        assert 3 not in consumed, "Cancel check failed — all events consumed"
        assert result == "partial"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_subagents.py::TestSelectAgentUnknownName tests/test_subagents.py::TestCancelAllSubagents tests/test_subagents.py::TestForkAndRun -v`
Expected: FAIL — `ImportError: cannot import name 'select_agent'` / `'fork_and_run'` / `'cancel_all_subagents'` / `'_ctx'`

- [ ] **Step 3: Implement execution engine in subagents.py**

Append to `finagent/subagents.py`:

```python
# ── Module-level imports for execution (patchable by tests) ──────────

from finagent.config import get_llm
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError


# ── Late-binding context access ──────────────────────────────────────

class _AgentContext:
    """Mutable container — TUI populates after init, select_agent reads at call time."""
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


# ── Cancellation ─────────────────────────────────────────────────────

_cancel_tokens: dict[str, threading.Event] = {}


def cancel_all_subagents():
    """Signal all running subagents to stop. Called from TUI Esc handler."""
    for token in _cancel_tokens.values():
        token.set()


# ── Fork + run ───────────────────────────────────────────────────────

def fork_and_run(sub: SubagentDef, task: str, context_messages: list) -> str:
    """Fork context into subagent, run to completion, return final AI text.

    Uses .stream() to check cancellation between graph steps.
    """
    sub_tools = resolve_tools(sub)
    sub_checkpointer = MemorySaver()
    sub_agent = create_langchain_agent(
        model=get_llm(),
        tools=sub_tools,
        system_prompt=sub.system_prompt,
        checkpointer=sub_checkpointer,
    )

    # Fork: pre-seed main conversation context
    sub_thread = str(uuid.uuid4())
    cancel_token = threading.Event()
    _cancel_tokens[sub_thread] = cancel_token

    sub_agent.update_state(
        config={"configurable": {"thread_id": sub_thread}},
        values={"messages": context_messages},
    )

    # Run (stream to check cancellation between steps)
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
        pass  # hit max_turns, extract whatever we have
    finally:
        _cancel_tokens.pop(sub_thread, None)

    # Extract last AIMessage with content
    state = sub_agent.get_state(
        config={"configurable": {"thread_id": sub_thread}}
    )
    for msg in reversed(state.values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return f"({sub.name} 未产生输出)"


# ── select_agent tool ───────────────────────────────────────────────

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_subagents.py -v`
Expected: PASS — all tests green

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -v`
Expected: PASS — no regressions

- [ ] **Step 6: Commit**

```bash
git add finagent/subagents.py tests/test_subagents.py
git commit -m "feat: fork_and_run + select_agent tool + cancellation

Late-binding _AgentContext, threading.Event cancel tokens,
.stream() with per-step cancel check, GraphRecursionError
recovery. 9 execution tests."
```

---

## Task 3: Agent + Prompt Integration

**Files:**
- Modify: `finagent/agent.py` (append select_agent to tools)
- Modify: `finagent/prompts.py` (append debate recipe to SKILL_RECIPES)
- Test: `tests/test_subagents.py` (append integration test)

**Interfaces:**
- Consumes: `select_agent` tool from Task 2
- Produces: main agent with select_agent in tool list; updated RESEARCH_SYSTEM_PROMPT with debate guidance

- [ ] **Step 1: Write failing test for agent wiring**

Append to `tests/test_subagents.py`:

```python
class TestAgentIntegration:
    """Verify select_agent is wired into agent.py's create_agent."""

    def test_create_agent_references_select_agent(self):
        """agent.py create_agent source includes select_agent in tools.

        Uses source inspection — can't call create_agent() without real API key.
        Before the edit: 'select_agent' not in source → fails.
        After the edit: 'select_agent' in source → passes.
        """
        import inspect
        from finagent.agent import create_agent
        source = inspect.getsource(create_agent)
        assert "select_agent" in source
```

- [ ] **Step 2: Write failing test for prompt content**

Append to `tests/test_subagents.py`:

```python
class TestPromptIntegration:
    """Verify RESEARCH_SYSTEM_PROMPT contains debate guidance."""

    def test_prompt_contains_debate_recipe(self):
        from finagent.prompts import RESEARCH_SYSTEM_PROMPT
        assert "多空辩论" in RESEARCH_SYSTEM_PROMPT
        assert "select_agent" in RESEARCH_SYSTEM_PROMPT
        assert "bull" in RESEARCH_SYSTEM_PROMPT
        assert "bear" in RESEARCH_SYSTEM_PROMPT
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_subagents.py::TestAgentIntegration tests/test_subagents.py::TestPromptIntegration -v`
Expected: FAIL — "多空辩论" not in RESEARCH_SYSTEM_PROMPT; "select_agent" not in create_agent source

- [ ] **Step 4: Update prompts.py — append debate recipe**

In `finagent/prompts.py`, the `SKILL_RECIPES` string ends at line 30 with `编辑是覆盖写，无版本保留。` followed by `"""`. Insert the debate recipe between line 29 and the closing `"""` on line 30.

Use Edit tool to replace:

```
old_string: "编辑是覆盖写，无版本保留。\n\"\"\""

new_string: "编辑是覆盖写，无版本保留。\n\n**多空辩论**（用户要求多空分析/看多看空/正反方对比）：\n1. select_agent(name=\"bull\", task=\"<具体分析指令，包含股票代码和报告期>\")\n2. select_agent(name=\"bear\", task=\"<具体分析指令，包含股票代码和报告期>\")\n3. 综合两方返回结果，给出平衡总结\n两步应在同一轮发出（单轮多 tool call），确保并行执行。\n\"\"\""
```

The resulting SKILL_RECIPES tail becomes:

```python
段落标题精确匹配现有标题（含"一、"编号）；新增段直接 update_section 即可。
编辑是覆盖写，无版本保留。

**多空辩论**（用户要求多空分析/看多看空/正反方对比）：
1. select_agent(name="bull", task="<具体分析指令，包含股票代码和报告期>")
2. select_agent(name="bear", task="<具体分析指令，包含股票代码和报告期>")
3. 综合两方返回结果，给出平衡总结
两步应在同一轮发出（单轮多 tool call），确保并行执行。
"""
```

- [ ] **Step 5: Update agent.py — two surgical edits**

`finagent/agent.py` currently has 3 functions: `create_agent` (line 13), `reset_checkpoint` (line 25), `create_agent_with_history` (line 31). **Do NOT rewrite the file** — both other functions are imported by `tui.py:16`. Make two surgical edits:

**Edit 1: Add import.** After the existing `from finagent.tools import tools` line (line 7), add:

```
old_string: "from finagent.tools import tools"

new_string: "from finagent.tools import tools\nfrom finagent.subagents import select_agent"
```

**Edit 2: Append select_agent to tools list.** In the `create_agent` function body, change the tools argument:

```
old_string: "        tools=tools,"

new_string: "        tools=tools + [select_agent],"
```

The resulting `create_agent` function becomes (other two functions unchanged):

```python
def create_agent():
    """Build a ReAct agent bound to a conversation thread."""
    llm = get_llm()
    agent = create_langchain_agent(
        model=llm,
        tools=tools + [select_agent],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )
    return agent
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_subagents.py::TestAgentIntegration tests/test_subagents.py::TestPromptIntegration -v`
Expected: PASS

- [ ] **Step 7: Run full suite**

Run: `pytest tests/ -v`
Expected: PASS — no regressions

- [ ] **Step 8: Commit**

```bash
git add finagent/agent.py finagent/prompts.py tests/test_subagents.py
git commit -m "feat: wire select_agent into main agent + debate prompt guidance

agent.py: tools + [select_agent]. prompts.py: SKILL_RECIPES
appends multi-tool-call debate recipe."
```

---

## Task 4: TUI Integration

**Files:**
- Modify: `finagent/tui.py` (3 changes: _start_stream ctx, _build_user_message catalog, action_interrupt cancel)
- Test: `tests/test_subagents.py` (append TUI integration tests)

**Interfaces:**
- Consumes: `_ctx` from Task 2, `render_subagent_catalog` from Task 1, `cancel_all_subagents` from Task 2
- Produces: TUI that injects subagent catalog every turn, updates _ctx on stream start, cancels subagents on Esc

- [ ] **Step 1: Write import-verification test**

The TUI modifications (catalog injection, _ctx update, Esc cancel) are method edits on `FinAgentApp` — they require a full Textual app lifecycle to test meaningfully, which is out of scope for unit tests. Instead, verify the import chain works and the edit targets are findable.

Append to `tests/test_subagents.py`:

```python
class TestTUIIntegration:
    """Verify TUI import chain and edit targets.

    TUI method changes (_start_stream ctx, _build_user_message catalog,
    action_interrupt cancel) are verified by: (1) this import test,
    (2) full suite passing without import errors after edit, (3) manual
    testing per the verification checklist.
    """

    def test_tui_imports_subagent_symbols(self):
        """tui.py can import _ctx, cancel_all_subagents, render_subagent_catalog.

        This verifies the import line added in Step 3 works.
        Before the edit: ImportError if tui.py tries to use these.
        After the edit: import succeeds, objects are callable.
        """
        from finagent.subagents import _ctx, cancel_all_subagents, render_subagent_catalog
        assert callable(cancel_all_subagents)
        assert hasattr(_ctx, 'agent')
        assert hasattr(_ctx, 'thread_id')
        assert hasattr(_ctx, 'get_messages')
        catalog = render_subagent_catalog()
        assert "bull" in catalog and "bear" in catalog
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_subagents.py::TestTUIIntegration -v`
Expected: PASS — verifies the symbols the TUI edit will import

- [ ] **Step 3: Modify tui.py — add imports**

At the top of `finagent/tui.py`, add import alongside existing subagent-related imports. After the existing `from finagent.skills import ...` line:

```python
from finagent.subagents import _ctx, cancel_all_subagents, render_subagent_catalog
```

- [ ] **Step 4: Modify tui.py — _start_stream ctx update**

In `_start_stream` method, add two lines at the very start of the method body, before `self._set_status("思考中...")`:

```python
    @work
    async def _start_stream(self, user_input: str) -> None:
        """Stream agent response..."""
        _ctx.agent = self.agent
        _ctx.thread_id = self.thread_id
        self._set_status("思考中...")
        # ... rest of existing method unchanged
```

- [ ] **Step 5: Modify tui.py — _build_user_message catalog injection**

In `_build_user_message` method, after the skill catalog `parts.append(...)` block and before `return "\n\n".join(parts)`, add:

```python
        # Subagent catalog (every turn)
        parts.append(
            f"<system-reminder>\n"
            f"可用 subagents (用 select_agent 工具启动):\n"
            f"{render_subagent_catalog()}\n"
            f"</system-reminder>"
        )
```

- [ ] **Step 6: Modify tui.py — action_interrupt cancel**

In `action_interrupt` method, add `cancel_all_subagents()` call at the start:

```python
    def action_interrupt(self) -> None:
        """Esc handler: cancel current streaming worker + running subagents."""
        cancel_all_subagents()
        if self._streaming_worker is not None and self._streaming_worker.is_running:
            self._streaming_worker.cancel()
            self._add_message("[已中断]", classes="message-queued")
```

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS — all tests green including existing TUI tests

- [ ] **Step 8: Commit**

```bash
git add finagent/tui.py tests/test_subagents.py
git commit -m "feat: TUI integration — subagent catalog, ctx update, Esc cancel

_start_stream: populate _ctx (covers init/clear/resume).
_build_user_message: inject subagent catalog system-reminder.
action_interrupt: cancel_all_subagents() before worker cancel."
```

---

## Task 5 (Conditional): run_debate Fallback

**When to run:** Only if manual testing (Task 4 verification) shows DeepSeek does NOT emit multiple tool calls in a single turn.

**How to verify:** After Task 4, launch the TUI (`python -m finagent`), type a debate request like "分析 002415 多空观点", and observe whether the main agent emits both `select_agent(bull)` and `select_agent(bear)` in one turn (parallel) or two turns (sequential).

If sequential, implement `run_debate` as a single tool that guarantees parallel execution.

**Files:**
- Modify: `finagent/subagents.py` (add run_debate tool)
- Modify: `finagent/agent.py` (add run_debate to tools list, may remove select_agent if debate is the only use case)
- Modify: `finagent/prompts.py` (update recipe to use run_debate instead of two select_agent calls)
- Test: `tests/test_subagents.py`

- [ ] **Step 1: Write failing test for run_debate**

Append to `tests/test_subagents.py`:

```python
class TestRunDebate:
    """run_debate spawns bull + bear in parallel, returns combined result."""

    def test_run_debate_parallel(self):
        """Both subagents called, results combined."""
        from finagent.subagents import run_debate

        with patch("finagent.subagents.fork_and_run") as mock_fork:
            mock_fork.side_effect = lambda sub, task, ctx: f"[{sub.name}]"
            with patch("finagent.subagents._ctx") as mock_ctx:
                mock_ctx.get_messages.return_value = []
                result = run_debate.invoke({"stock_code": "002415", "report_period": "2024Q3"})
                assert "[bull]" in result
                assert "[bear]" in result
                assert "多方" in result or "空方" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_subagents.py::TestRunDebate -v`
Expected: FAIL — `ImportError: cannot import name 'run_debate'`

- [ ] **Step 3: Implement run_debate in subagents.py**

Append to `finagent/subagents.py`:

```python
@tool
def run_debate(stock_code: str, report_period: str) -> str:
    """启动多空辩论：并行 spawn bull + bear subagent，返回综合结果。

    Args:
        stock_code: 6 位股票代码
        report_period: 报告期（如 2024Q3）
    """
    import asyncio
    context = _ctx.get_messages()
    bull_task = asyncio.to_thread(
        fork_and_run, SUBAGENTS["bull"],
        f"分析 {stock_code} {report_period} 看多理由", context,
    )
    bear_task = asyncio.to_thread(
        fork_and_run, SUBAGENTS["bear"],
        f"分析 {stock_code} {report_period} 看空理由", context,
    )
    bull_result, bear_result = asyncio.run(asyncio.gather(bull_task, bear_task))
    return f"## 多方观点\n{bull_result}\n\n## 空方观点\n{bear_result}"
```

- [ ] **Step 4: Wire run_debate into agent.py**

In `finagent/agent.py`, update import and tools list:

```python
from finagent.subagents import select_agent, run_debate
# ...
        tools=tools + [select_agent, run_debate],
```

- [ ] **Step 5: Update prompts.py recipe**

Change the debate recipe from two select_agent calls to a single run_debate call:

```
**多空辩论**（用户要求多空分析/看多看空/正反方对比）：
调用 run_debate(stock_code="<6位代码>", report_period="<报告期>")，工具内部并行启动多空双方 subagent。
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add finagent/subagents.py finagent/agent.py finagent/prompts.py tests/test_subagents.py
git commit -m "feat: run_debate fallback for guaranteed parallel execution

Single tool spawns bull+bear via asyncio.gather. Used when
DeepSeek doesn't emit multi-tool-call in one turn."
```

---

## Verification Checklist (After All Tasks)

- [ ] `pytest tests/ -v` — all tests pass, no regressions
- [ ] `python -m finagent` — TUI starts without import errors
- [ ] Type a stock analysis question — normal single-agent flow works
- [ ] Type "分析 002415 多空观点" — select_agent spawns bull + bear
- [ ] Press Esc during subagent execution — subagents stop within 1 step
- [ ] `/clear` then type again — _ctx updates correctly
- [ ] `--resume <session>` — _ctx updates correctly on resume
