# TUI 会话恢复过滤 + ReAct 中间过程折叠 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 FinAgent TUI 中实现两个 feature：(1) 恢复会话时过滤 system-reminder；(2) ReAct 中间过程折叠（实时 + 恢复），类似 Claude Code 的"thought for Xs"体验。

**Architecture:** 改动集中在 `session.py`（格式扩展）和 `tui.py`（折叠 + 过滤）。session 格式新增 `type: "turn"` 元数据行存储每轮时长和消息索引范围。TUI 在流结束时将中间 widget（工具调用 + 过渡文本）包进 Textual `Collapsible` widget，恢复时按 turn 分组渲染。

**Tech Stack:** Python 3.11+, Textual 8.2.8 (`Collapsible` widget), LangGraph, LangChain messages, pytest + pytest-asyncio

## Global Constraints

- 项目环境是 `.venv`，测试用 `.venv/bin/pytest` 或 `.venv/bin/python -m pytest`
- 测试时避免并行（可能导致内存爆炸）
- 不需要向后兼容 shim——旧 session 返回空 turns，平铺降级
- `_build_user_message()` 不改——reminder 仍需注入给 LLM
- 禁止 stub

---

## File Structure

| 文件 | 职责 | 改动类型 |
|------|------|----------|
| `finagent/session.py` | session JSONL 序列化/反序列化 | 修改：`write_session` 加 turns 参数，`load_session` 返回 3 元组 |
| `finagent/tui.py` | Textual TUI 主逻辑 | 修改：折叠 + 过滤 + 分组恢复 |
| `tests/test_session.py` | session 测试 | 修改：适配 3 元组 + 新增 turn round-trip 测试 |
| `tests/test_tui.py` | TUI 测试 | 修改：适配 mock + 新增折叠/过滤测试 |

---

### Task 1: Session 格式扩展 + 调用点适配

**Files:**
- Modify: `finagent/session.py` (`write_session`, `load_session`)
- Modify: `finagent/tui.py:89` (`load_session` 调用)
- Modify: `tests/test_session.py:98,112,127` (`load_session` 解包)
- Modify: `tests/test_tui.py:668` (mock `return_value`)

**Interfaces:**
- Produces: `write_session(session_id, messages, cumulative_tokens, turns=None)` — `turns` 是 `list[dict]`，每项 `{"type":"turn","duration_s":float,"interrupted":bool,"msg_start":int,"msg_end":int}`
- Produces: `load_session(session_id) -> (messages, cumulative_tokens, turns)` — turns 为 `list[dict]`，旧文件/缺失文件返回 `[]`

- [ ] **Step 1: 写 turn round-trip 失败测试**

在 `tests/test_session.py` 末尾追加：

```python
def test_write_load_session_with_turns(tmp_path, monkeypatch):
    """write_session with turns data round-trips through load_session."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    messages = [
        HumanMessage(content="hi", id="m1"),
        AIMessage(content="hello", id="m2"),
    ]
    turns = [
        {"type": "turn", "duration_s": 5.2, "interrupted": False, "msg_start": 0, "msg_end": 2},
    ]
    write_session("turn-session", messages, 100, turns=turns)
    loaded, tokens, loaded_turns = load_session("turn-session")
    assert len(loaded) == 2
    assert tokens == 100
    assert len(loaded_turns) == 1
    assert loaded_turns[0]["duration_s"] == 5.2
    assert loaded_turns[0]["interrupted"] is False
    assert loaded_turns[0]["msg_start"] == 0
    assert loaded_turns[0]["msg_end"] == 2


def test_load_session_old_format_returns_empty_turns(tmp_path, monkeypatch):
    """Session file without turn lines returns empty turns list."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"type": "human", "content": "old"}\n'
        '{"type": "meta", "cumulative_tokens": 0}\n',
        encoding="utf-8",
    )
    messages, tokens, turns = load_session("old")
    assert len(messages) == 1
    assert turns == []


def test_load_session_missing_file_returns_empty_turns(tmp_path, monkeypatch):
    """Missing session file returns empty messages, zero tokens, empty turns."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    messages, tokens, turns = load_session("nonexistent")
    assert messages == []
    assert tokens == 0
    assert turns == []
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_session.py::test_write_load_session_with_turns tests/test_session.py::test_load_session_old_format_returns_empty_turns tests/test_session.py::test_load_session_missing_file_returns_empty_turns -v`
Expected: FAIL — `ValueError: not enough values to unpack` (2-tuple vs 3-tuple)

- [ ] **Step 3: 修改 `write_session` 加 turns 参数**

在 `finagent/session.py` 的 `write_session` 函数中：

```python
def write_session(session_id: str, messages: list, cumulative_tokens: int, turns: list[dict] | None = None) -> None:
    """Overwrite session file with full message snapshot + meta line."""
    lines = []
    for msg in messages:
        lines.append(json.dumps(serialize_message(msg), ensure_ascii=False))
    if turns:
        for turn in turns:
            lines.append(json.dumps(turn, ensure_ascii=False))
    lines.append(json.dumps(
        {"type": "meta", "cumulative_tokens": cumulative_tokens},
        ensure_ascii=False,
    ))
    atomic_write(session_path(session_id), "\n".join(lines) + "\n")
```

- [ ] **Step 4: 修改 `load_session` 返回 3 元组**

```python
def load_session(session_id: str) -> tuple[list, int, list]:
    """Read JSONL, return (messages, cumulative_tokens, turns).

    Skips malformed lines. Returns ([], 0, []) if file doesn't exist.
    """
    path = session_path(session_id)
    if not path.exists():
        return [], 0, []
    messages = []
    cumulative_tokens = 0
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "meta":
            cumulative_tokens = data.get("cumulative_tokens", 0)
        elif data.get("type") == "turn":
            turns.append(data)
        else:
            try:
                messages.append(deserialize_message(data))
            except (KeyError, ValueError):
                continue
    return messages, cumulative_tokens, turns
```

- [ ] **Step 5: 更新 `tui.py:89` 的 `load_session` 调用**

在 `finagent/tui.py` 的 `FinAgentApp.__init__` 中：

```python
if resume_session_id:
    messages, tokens, turns = load_session(resume_session_id)
    self.thread_id = resume_session_id
    self._cumulative_input_tokens = tokens
    self.agent = create_agent_with_history(self.thread_id, messages)
    self._resume_messages = messages
```

- [ ] **Step 6: 更新 `tests/test_session.py` 现有 3 处 2 元组解包为 3 元组**

```python
# Line 98: test_write_and_load_session
loaded, tokens, _turns = load_session("test-session")

# Line 112: test_load_session_missing_file
messages, tokens, _turns = load_session("nonexistent")

# Line 127: test_load_session_malformed_line_skipped
messages, tokens, _turns = load_session("bad")
```

- [ ] **Step 7: 更新 `tests/test_tui.py:668` mock 返回值**

```python
patch("finagent.tui.load_session", return_value=([], 0, [])):
```

- [ ] **Step 8: 运行全部 session + tui 测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_session.py tests/test_tui.py -v`
Expected: ALL PASS

- [ ] **Step 9: 提交**

```bash
git add finagent/session.py finagent/tui.py tests/test_session.py tests/test_tui.py
git commit -m "feat: session format supports turn metadata

write_session accepts turns param, load_session returns 3-tuple.
Old files without turn lines return empty turns list."
```

---

### Task 2: Reminder 过滤 helper + 恢复时过滤

**Files:**
- Modify: `finagent/tui.py` (新增 helper 函数 + `on_mount` 改动)
- Test: `tests/test_tui.py`

**Interfaces:**
- Produces: `strip_system_reminders(content: str) -> str` — 剥 `<system-reminder>` 块，返回清理后文本

- [ ] **Step 1: 写 `strip_system_reminders` 失败测试**

在 `tests/test_tui.py` 中追加：

```python
from finagent.tui import strip_system_reminders


def test_strip_system_reminders_removes_blocks():
    content = "分析002415\n\n<system-reminder>\n记忆内容\n</system-reminder>"
    result = strip_system_reminders(content)
    assert "<system-reminder>" not in result
    assert "分析002415" in result
    assert "记忆内容" not in result


def test_strip_system_reminders_multiple_blocks():
    content = (
        "原文\n\n"
        "<system-reminder>\n块1\n</system-reminder>\n\n"
        "<system-reminder>\n块2\n</system-reminder>"
    )
    result = strip_system_reminders(content)
    assert result.strip() == "原文"


def test_strip_system_reminders_no_blocks():
    content = "纯文本无reminder"
    assert strip_system_reminders(content) == "纯文本无reminder"


def test_strip_system_reminders_only_reminders():
    content = "<system-reminder>\n全部是reminder\n</system-reminder>"
    assert strip_system_reminders(content).strip() == ""
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_strip_system_reminders_removes_blocks -v`
Expected: FAIL — `ImportError: cannot import name 'strip_system_reminders'`

- [ ] **Step 3: 实现 `strip_system_reminders`**

在 `finagent/tui.py` 模块级（`parse_command` 函数之后）添加：

```python
import re

_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def strip_system_reminders(content: str) -> str:
    """Remove <system-reminder> blocks from message content."""
    return _REMINDER_RE.sub("", content)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_strip_system_reminders_removes_blocks tests/test_tui.py::test_strip_system_reminders_multiple_blocks tests/test_tui.py::test_strip_system_reminders_no_blocks tests/test_tui.py::test_strip_system_reminders_only_reminders -v`
Expected: ALL PASS

- [ ] **Step 5: 更新 `on_mount` 恢复循环使用过滤**

在 `finagent/tui.py` 的 `on_mount` 方法中，修改 `_resume_messages` 渲染循环：

```python
if self._resume_messages:
    self._add_message("— 恢复历史会话 —", classes="message-queued")
    for msg in self._resume_messages:
        if isinstance(msg, HumanMessage) and msg.content:
            cleaned = strip_system_reminders(msg.content).strip()
            if cleaned:
                self._add_message(f"> {cleaned}", classes="message-user")
        elif isinstance(msg, AIMessage) and msg.content:
            self._add_message(msg.content)
        elif isinstance(msg, ToolMessage):
            self._add_message(f"🔧 {msg.name} ✓", classes="message-tool")
    self._add_message("— 继续对话 —", classes="message-queued")
    self._resume_messages = None
```

注意：AIMessage 有 tool_calls 但 content 为空时，`msg.content` 为 `""`，falsy，跳过。不需要额外判断。

- [ ] **Step 6: 运行全部 TUI 测试**

Run: `.venv/bin/python -m pytest tests/test_tui.py -v`
Expected: ALL PASS

- [ ] **Step 7: 提交**

```bash
git add finagent/tui.py tests/test_tui.py
git commit -m "feat: filter system-reminder blocks from restored sessions

strip_system_reminders() removes <system-reminder> blocks from
HumanMessage content before display. Empty AIMessages skipped."
```

---

### Task 3: 轮次计时 + widget 追踪 + turns 累积

**Files:**
- Modify: `finagent/tui.py` (`__init__`, `_do_clear`, `_start_stream`, `_handle_update_msg`)

**Interfaces:**
- Produces: `self._turns: list[dict]` — 累积器，每轮 `_start_stream` 结束后追加一条 turn 元数据
- Consumes: Task 1 的 `write_session(..., turns=self._turns)`

- [ ] **Step 1: 写 turns 累积失败测试**

在 `tests/test_tui.py` 中追加：

```python
@pytest.mark.asyncio
async def test_turns_accumulated_after_stream():
    """After a stream with tool calls, self._turns has correct metadata."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    async def fake_astream(*args, **kwargs):
        yield ("updates", {
            "agent": {"messages": [AIMessage(
                content="", tool_calls=[{"name": "get_financials", "args": {}, "id": "tc1"}]
            )]}
        })
        yield ("updates", {
            "tools": {"messages": [ToolMessage(
                content="data", tool_call_id="tc1", name="get_financials"
            )]}
        })
        yield ("messages", (AIMessageChunk(content="最终回答"), {}))

    fake_agent = MagicMock()
    fake_agent.astream = fake_astream

    fake_state = MagicMock()
    fake_state.values = {"messages": [HumanMessage(content="hi", id="m1")]}

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        app.agent.get_state = MagicMock(return_value=fake_state)
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "test"
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

    assert len(app._turns) == 1
    turn = app._turns[0]
    assert turn["type"] == "turn"
    assert isinstance(turn["duration_s"], float)
    assert turn["interrupted"] is False
    assert isinstance(turn["msg_start"], int)
    assert isinstance(turn["msg_end"], int)


@pytest.mark.asyncio
async def test_do_clear_resets_turns():
    """_do_clear resets self._turns to empty list."""
    with patch("finagent.tui.create_agent", return_value=MagicMock()):
        app = FinAgentApp()
        app._turns = [{"fake": "turn"}]
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "/clear"
            await pilot.press("enter")
            await pilot.pause()
    assert app._turns == []
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_turns_accumulated_after_stream tests/test_tui.py::test_do_clear_resets_turns -v`
Expected: FAIL — `AttributeError: 'FinAgentApp' object has no attribute '_turns'`

- [ ] **Step 3: 在 `__init__` 和 `_do_clear` 中维护 `self._turns`**

在 `finagent/tui.py` 的 `FinAgentApp.__init__` 中：

新会话分支（else 块）加：
```python
self._turns = []
```

恢复会话分支（if 块），`load_session` 已返回 3 元组（Task 1），加：
```python
self._turns = turns
```

在 `_do_clear` 方法中加：
```python
self._turns = []
```

- [ ] **Step 4: 在 `_start_stream` 中加计时和 widget 追踪**

在 `_start_stream` 方法开头（`self._set_status("思考中...")` 之前）加：

```python
turn_start = time.monotonic()
turn_widgets: list[Static] = []

# Capture msg_start before streaming
try:
    pre_state = self.agent.get_state(
        config={"configurable": {"thread_id": self.thread_id}}
    )
    msg_start = len(pre_state.values.get("messages", []))
except Exception:
    msg_start = 0
```

在创建初始 reply_widget 后（`reply_widget = self._add_message("")` 行之后）加：
```python
turn_widgets.append(reply_widget)
```

在 streaming loop 的 messages mode 中，创建新 reply_widget 时（`reply_widget = self._add_message("")`）后加：
```python
turn_widgets.append(reply_widget)
```

- [ ] **Step 5: 更新 `_handle_update_msg` 接收 `turn_widgets` 参数**

修改方法签名：

```python
def _handle_update_msg(self, msg, reply_widget, buffer, tool_widgets, tool_names, turn_widgets):
```

在 tool widget 创建处（`tool_widgets[tc_id] = self._add_message(...)` 行之后）加：

```python
turn_widgets.append(tool_widgets[tc_id])
```

更新 `_start_stream` 中的调用：

```python
nr, lai, reply_widget, buffer = self._handle_update_msg(
    msg, reply_widget, buffer, tool_widgets, tool_names, turn_widgets
)
```

- [ ] **Step 6: 在 `_start_stream` finally 块中记录 turn 元数据并传给 `write_session`**

修改 finally 块的 session 持久化部分：

```python
# Session persistence (best-effort)
try:
    state = self.agent.get_state(
        config={"configurable": {"thread_id": self.thread_id}}
    )
    msgs = state.values.get("messages", [])
    msg_end = len(msgs)
    self._turns.append({
        "type": "turn",
        "duration_s": time.monotonic() - turn_start,
        "interrupted": current.is_cancelled,
        "msg_start": msg_start,
        "msg_end": msg_end,
    })
    write_session(self.thread_id, msgs, self._cumulative_input_tokens, self._turns)
    self._run_extraction(msgs)
except Exception:
    pass  # persistence is best-effort; don't crash UI
```

- [ ] **Step 7: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_turns_accumulated_after_stream tests/test_tui.py::test_do_clear_resets_turns -v`
Expected: ALL PASS

- [ ] **Step 8: 运行全部 TUI 测试确认无回归**

Run: `.venv/bin/python -m pytest tests/test_tui.py -v`
Expected: ALL PASS

- [ ] **Step 9: 提交**

```bash
git add finagent/tui.py tests/test_tui.py
git commit -m "feat: track turn timing and widget list for collapse

_start_stream records turn_start, msg_start/msg_end, interrupted.
_handle_update_msg appends tool widgets to turn_widgets list.
self._turns accumulates across the session lifecycle."
```

---

### Task 4: 实时折叠逻辑

**Files:**
- Modify: `finagent/tui.py` (`_start_stream` finally 块)
- Modify: `tests/test_tui.py` (更新受影响的流式测试 + 新增折叠测试)

**Interfaces:**
- Consumes: Task 3 的 `turn_widgets`, `turn_start`, `msg_start`
- Produces: DOM 中 `Collapsible` widget 包裹中间过程

- [ ] **Step 1: 更新受影响的流式测试适配 Collapsible**

`test_streaming_shows_tool_progress` 和 `test_tool_call_interleaved_between_text_segments` 需要修改 widget 查找方式——从 `chat.children` 改为 `chat.query(Static)` 递归查找（折叠后部分 Static 在 Collapsible 内部）。

修改 `test_streaming_shows_tool_progress`：

```python
chat = app.query_one("#chat-view")
messages = [str(c.content) for c in chat.query(Static)]
# tool progress line with tool name + checkmark
assert any("get_financials" in m and "✓" in m for m in messages)
```

修改 `test_tool_call_interleaved_between_text_segments`：

```python
chat = app.query_one("#chat-view")
messages = [str(c.content) for c in chat.query(Static)]
```

注意：折叠后中间文本"前半段"在 Collapsible 内，"后半段"在 Collapsible 外。顺序检查需要适配——工具行在 Collapsible 内，后段文本在外。改为验证内容存在性而非顺序：

```python
# 需求1: 单个工具 widget，⏳ 被替换为 ✓
tool_msgs = [m for m in messages if "get_financials" in m]
assert len(tool_msgs) == 1, f"expected 1 tool widget, got {tool_msgs}"
assert "✓" in tool_msgs[0]
assert "⏳" not in tool_msgs[0]
# 需求2: 两段文本都存在（折叠后顺序不保证）
assert any("前半段" in m for m in messages)
assert any("后半段" in m for m in messages)
# 需求3: 结构验证——"前半段"在 Collapsible 内，"后半段"在 Collapsible 外
from textual.widgets import Collapsible
collapsibles = list(chat.query(Collapsible))
if collapsibles:
    inside = [str(c.content) for c in collapsibles[0].query(Static)]
    outside = [str(c.content) for c in chat.children if isinstance(c, Static)]
    assert any("前半段" in m for m in inside), "前半段 should be inside Collapsible"
    assert any("后半段" in m for m in outside), "后半段 should be outside Collapsible"
```

- [ ] **Step 2: 写折叠行为失败测试**

在 `tests/test_tui.py` 中追加：

```python
@pytest.mark.asyncio
async def test_collapse_creates_collapsible_with_tools():
    """After stream with tool calls, a Collapsible widget exists in DOM."""
    from textual.widgets import Collapsible
    from langchain_core.messages import AIMessage, ToolMessage

    async def fake_astream(*args, **kwargs):
        yield ("updates", {
            "agent": {"messages": [AIMessage(
                content="", tool_calls=[{"name": "get_financials", "args": {}, "id": "tc1"}]
            )]}
        })
        yield ("updates", {
            "tools": {"messages": [ToolMessage(
                content="data", tool_call_id="tc1", name="get_financials"
            )]}
        })
        yield ("messages", (AIMessageChunk(content="最终回答"), {}))

    fake_agent = MagicMock()
    fake_agent.astream = fake_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "test"
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            chat = app.query_one("#chat-view")
            collapsibles = list(chat.query(Collapsible))
            assert len(collapsibles) >= 1
            c = collapsibles[0]
            assert "思考了" in c.title
            assert c.collapsed is True


@pytest.mark.asyncio
async def test_no_collapse_without_tools():
    """Stream without tool calls does not create a Collapsible."""
    from textual.widgets import Collapsible

    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="简单回答"), {}))

    fake_agent = MagicMock()
    fake_agent.astream = fake_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "test"
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            chat = app.query_one("#chat-view")
            assert len(list(chat.query(Collapsible))) == 0
```

- [ ] **Step 3: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_collapse_creates_collapsible_with_tools tests/test_tui.py::test_no_collapse_without_tools -v`
Expected: FAIL — 无 Collapsible 创建

- [ ] **Step 4: 添加顶部 import + 实现 `_static_text` helper**

在 `finagent/tui.py` 顶部修改 import 行：

```python
from textual.widgets import Collapsible, Static, TextArea
```

在模块级添加（`strip_system_reminders` 之后）：

```python
def _static_text(w: Static) -> str:
    """Extract text content from a Static widget for copying into Collapsible.

    Uses public `content` attribute (str), verified on Textual 8.2.8.
    """
    return w.content if isinstance(w.content, str) else str(w.content)
```

- [ ] **Step 5: 在 `_start_stream` finally 块中实现折叠逻辑**

在 finally 块中，`self._refresh_status_bar()` 之前，加入折叠逻辑：

```python
# Collapse intermediate widgets (best-effort)
try:
    had_tools = any(
        w.has_class("message-tool") for w in turn_widgets if w.is_mounted
    )
    if had_tools:
        mounted = [w for w in turn_widgets if w.is_mounted]
        reply_mounted = [
            w for w in mounted if not w.has_class("message-tool")
        ]
        final_widget = reply_mounted[-1] if reply_mounted else None
        final_has_content = (
            reply_widget is not None and bool(buffer)
        )

        if final_has_content and final_widget is not None:
            intermediate = [w for w in mounted if w is not final_widget]
        else:
            intermediate = mounted

        # Capture content before removing
        contents = [_static_text(w) for w in intermediate]

        # Remove intermediate widgets from DOM
        for w in intermediate:
            await w.remove()

        # Build collapsible title
        duration = time.monotonic() - turn_start
        title = f"思考了 {duration:.0f}s"
        if current.is_cancelled:
            title += " [已中断]"

        collapsible = Collapsible(
            *[Static(c) for c in contents],
            title=title,
            collapsed=True,
        )

        chat_view = self.query_one("#chat-view")
        if final_has_content and final_widget is not None and final_widget.is_mounted:
            await chat_view.mount(collapsible, before=final_widget)
        else:
            await chat_view.mount(collapsible)
except Exception:
    pass  # collapse is best-effort; don't break the finally block
```

注意：`buffer` 和 `reply_widget` 变量在 try 块中定义，在 finally 块中可访问（Python 作用域）。如果 try 块在到达 buffer 赋值前异常退出，`buffer` 未定义——需在 `_start_stream` 开头初始化 `buffer = ""` 和 `reply_widget = None`（当前代码已有这些初始化）。

- [ ] **Step 6: 运行折叠测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_collapse_creates_collapsible_with_tools tests/test_tui.py::test_no_collapse_without_tools -v`
Expected: ALL PASS

- [ ] **Step 7: 运行全部 TUI 测试确认无回归**

Run: `.venv/bin/python -m pytest tests/test_tui.py -v`
Expected: ALL PASS

- [ ] **Step 8: 提交**

```bash
git add finagent/tui.py tests/test_tui.py
git commit -m "feat: collapse ReAct intermediate steps at stream end

Tool calls and intermediate text wrapped in Collapsible widget.
Final answer stays visible. Interrupted turns show [已中断] suffix.
No collapse when turn has no tool calls."
```

---

### Task 5: 恢复时按 turn 分组折叠显示

**Files:**
- Modify: `finagent/tui.py` (`on_mount` 恢复循环 + `__init__` 存 turns)
- Test: `tests/test_tui.py`

**Interfaces:**
- Consumes: Task 1 的 `load_session` 3 元组 turns, Task 2 的 `strip_system_reminders`, Textual `Collapsible`
- Consumes: Task 3 的 `self._turns` (恢复时从 `load_session` 获取)

- [ ] **Step 1: 写恢复分组显示失败测试**

在 `tests/test_tui.py` 中追加：

```python
@pytest.mark.asyncio
async def test_restore_with_turns_shows_collapsibles():
    """Restored session with turn data renders Collapsibles per turn."""
    from textual.widgets import Collapsible
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    messages = [
        HumanMessage(content="分析002415", id="m1"),
        AIMessage(content="", tool_calls=[{"name": "get_financials", "args": {}, "id": "tc1", "type": "tool_call"}], id="m2"),
        ToolMessage(content="营收650亿", tool_call_id="tc1", name="get_financials", id="m3"),
        AIMessage(content="海康威视2024Q3营收同比增长。", id="m4"),
    ]
    turns = [
        {"type": "turn", "duration_s": 8.0, "interrupted": False, "msg_start": 0, "msg_end": 4},
    ]

    with patch("finagent.tui.create_agent_with_history", return_value=MagicMock()), \
         patch("finagent.tui.load_session", return_value=(messages, 500, turns)):
        app = FinAgentApp(resume_session_id="test-restore")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()
            chat = app.query_one("#chat-view")
            collapsibles = list(chat.query(Collapsible))
            assert len(collapsibles) >= 1
            assert "思考了" in collapsibles[0].title
            # Final answer visible outside collapsible
            statics = [str(c.content) for c in chat.query(Static)]
            assert any("海康威视" in m for m in statics)


@pytest.mark.asyncio
async def test_restore_without_turns_flat_display():
    """Restored session without turn data renders flat (no Collapsibles)."""
    from textual.widgets import Collapsible
    from langchain_core.messages import HumanMessage, AIMessage

    messages = [
        HumanMessage(content="hello", id="m1"),
        AIMessage(content="world", id="m2"),
    ]

    with patch("finagent.tui.create_agent_with_history", return_value=MagicMock()), \
         patch("finagent.tui.load_session", return_value=(messages, 0, [])):
        app = FinAgentApp(resume_session_id="test-flat")
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one("#chat-view")
            assert len(list(chat.query(Collapsible))) == 0


@pytest.mark.asyncio
async def test_restore_strips_reminders():
    """Restored HumanMessages have system-reminder blocks removed."""
    from langchain_core.messages import HumanMessage

    dirty_content = "分析002415\n\n<system-reminder>\n秘密\n</system-reminder>"
    messages = [HumanMessage(content=dirty_content, id="m1")]

    with patch("finagent.tui.create_agent_with_history", return_value=MagicMock()), \
         patch("finagent.tui.load_session", return_value=(messages, 0, [])):
        app = FinAgentApp(resume_session_id="test-strip")
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one("#chat-view")
            texts = [str(c.content) for c in chat.query(Static)]
            assert any("分析002415" in t for t in texts)
            assert not any("秘密" in t for t in texts)
            assert not any("<system-reminder>" in t for t in texts)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_restore_with_turns_shows_collapsibles tests/test_tui.py::test_restore_without_turns_flat_display tests/test_tui.py::test_restore_strips_reminders -v`
Expected: 部分 FAIL — `test_restore_strips_reminders` 可能通过（Task 2 已实现），`test_restore_with_turns_shows_collapsibles` FAIL（无分组逻辑）

- [ ] **Step 3: 实现 `_find_final_answer` helper**

在 `finagent/tui.py` 模块级添加：

```python
def _find_final_answer(messages: list):
    """Find last AIMessage with non-empty content and no tool_calls.

    Returns None if no such message exists.
    """
    for msg in reversed(messages):
        if (isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", None)):
            return msg
    return None
```

- [ ] **Step 4: 重写 `on_mount` 恢复循环**

将 `on_mount` 中的恢复循环替换为分组逻辑：

```python
if self._resume_messages:
    self._add_message("— 恢复历史会话 —", classes="message-queued")

    if self._turns:
        # Group messages by turn metadata
        # Sanity check: if indices are out of bounds, fall back to flat display
        max_end = max((t.get("msg_end", 0) for t in self._turns), default=0)
        if max_end > len(self._resume_messages):
            # Index drift (corrupted session or skipped lines) — flat fallback
            self._turns = []  # fall through to flat display below

    if self._turns:
        for turn in self._turns:
            start = turn.get("msg_start", 0)
            end = turn.get("msg_end", 0)
            turn_msgs = self._resume_messages[start:end]

            # Display HumanMessage (filtered)
            for msg in turn_msgs:
                if isinstance(msg, HumanMessage) and msg.content:
                    cleaned = strip_system_reminders(msg.content).strip()
                    if cleaned:
                        self._add_message(f"> {cleaned}", classes="message-user")

            # Check if turn had tool calls
            has_tools = any(
                getattr(m, "tool_calls", None)
                for m in turn_msgs
                if hasattr(m, "tool_calls")
            )

            if has_tools:
                # Find final answer
                final = _find_final_answer(turn_msgs)

                # Collect intermediate content as (text, is_tool) tuples
                intermediate = []
                for msg in turn_msgs:
                    if isinstance(msg, ToolMessage):
                        intermediate.append((f"🔧 {msg.name} ✓", True))
                    elif isinstance(msg, AIMessage) and msg.content and msg is not final:
                        if not getattr(msg, "tool_calls", None):
                            intermediate.append((msg.content, False))

                # Build title
                duration = turn.get("duration_s", 0)
                title = f"思考了 {duration:.0f}s"
                if turn.get("interrupted"):
                    title += " [已中断]"

                collapsible = Collapsible(
                    *[Static(text, classes="message-tool" if is_tool else "")
                      for text, is_tool in intermediate],
                    title=title,
                    collapsed=True,
                )
                chat_view = self.query_one("#chat-view")
                chat_view.mount(collapsible)
                chat_view.scroll_end(animate=False)

                # Show final answer after collapsible
                if final:
                    self._add_message(final.content)
            else:
                # No tools: show final answer directly
                final = _find_final_answer(turn_msgs)
                if final:
                    self._add_message(final.content)
    else:
        # Old session without turns: flat display (Feature 1 filtering still applies)
        for msg in self._resume_messages:
            if isinstance(msg, HumanMessage) and msg.content:
                cleaned = strip_system_reminders(msg.content).strip()
                if cleaned:
                    self._add_message(f"> {cleaned}", classes="message-user")
            elif isinstance(msg, AIMessage) and msg.content:
                self._add_message(msg.content)
            elif isinstance(msg, ToolMessage):
                self._add_message(f"🔧 {msg.name} ✓", classes="message-tool")

    self._add_message("— 继续对话 —", classes="message-queued")
    self._resume_messages = None
```

**注意**：`else` 分支（非恢复会话，显示"输入股票代码 + 报告期开始分析。输入 /help 查看命令。"）保持不变。

**中断双重标记说明**：`action_interrupt()` 已有 `_add_message("[已中断]")`（即时用户反馈）。折叠标题也会追加 `[已中断]`。两种取消路径行为不同：
- `/clear` → `_do_clear` 把 `self._streaming_worker = None` → finally 块 early return（`current is not self._streaming_worker`）→ 不折叠，随后 `remove_children()` 清空。正确。
- Esc → `action_interrupt` 只 `.cancel()` 不改 `self._streaming_worker` → finally 块正常执行 → 折叠 + 标题 `[已中断]`。同时 `action_interrupt` 的 `[已中断]` 行也显示。两处标记并存：即时行是响应反馈，折叠标题是历史摘要。可接受。

- [ ] **Step 5: 运行恢复测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_restore_with_turns_shows_collapsibles tests/test_tui.py::test_restore_without_turns_flat_display tests/test_tui.py::test_restore_strips_reminders -v`
Expected: ALL PASS

- [ ] **Step 6: 运行全部测试套件确认无回归**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: 提交**

```bash
git add finagent/tui.py tests/test_tui.py
git commit -m "feat: restore sessions with turn-grouped collapse display

on_mount groups restored messages by turn metadata. Each turn with
tool calls renders a Collapsible for intermediate content + final
answer outside. Old sessions without turns fall back to flat display."
```

---

## Self-Review

### Spec coverage

| Spec 条目 | 对应 Task |
|-----------|----------|
| §1.1 turn 元数据行格式 | Task 1 |
| §1.2 write_session 签名 | Task 1 |
| §1.3 load_session 3 元组 + 三分支解析 | Task 1 |
| §1.4 旧 session 空 turns | Task 1 (test_load_session_old_format_returns_empty_turns) |
| §1.5 受影响调用点 | Task 1 Steps 5-7 |
| §1.6 索引偏移降级 | Task 1 (load_session skip 行为不变) |
| §2 Feature 1: reminder 过滤 | Task 2 |
| §3.1 计时 | Task 3 |
| §3.2 widget 追踪 | Task 3 |
| §3.3 流结束折叠 | Task 4 |
| §3.4 运行中行为 | Task 4 (折叠只在 finally 发生) |
| §3.5 Collapsible 内容 | Task 4 |
| §4 Feature 2: 恢复分组 | Task 5 |
| §4.2 最终回答判定 | Task 5 (_find_final_answer) |
| §5 self._turns 生命周期 | Task 3 |

### Placeholder scan

无 TBD/TODO。所有步骤含具体代码。

### Type consistency

- `turn_widgets: list[Static]` — Task 3 定义，Task 4 消费 ✓
- `self._turns: list[dict]` — Task 3 定义，Task 1 序列化，Task 5 恢复 ✓
- `_handle_update_msg` 签名加 `turn_widgets` — Task 3 定义 ✓
- `_find_final_answer(messages) -> AIMessage | None` — Task 5 定义 ✓
- `_static_text(w: Static) -> str` — Task 4 定义 ✓
- `strip_system_reminders(content: str) -> str` — Task 2 定义 ✓
