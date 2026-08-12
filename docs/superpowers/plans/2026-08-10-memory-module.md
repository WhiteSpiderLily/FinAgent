# FinAgent 记忆模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three-layer memory system (session persistence, static memory injection, auto-memory with LLM-driven extraction + governance) to FinAgent.

**Architecture:** Three new modules (`session.py`, `memory.py`, `governance.py`) with clear single responsibilities. TUI integration via mtime-based conditional injection in `<system-reminder>` blocks, async workers for LLM tasks, and `--resume` CLI flag for session restoration. All file writes are atomic (temp + rename). Extraction and governance share an `asyncio.Lock` for mutual exclusion.

**Tech Stack:** Python 3.11, LangGraph (MemorySaver checkpointer), LangChain (create_agent, message types), Textual 8.x (TUI, @work), DeepSeek API, pytest + pytest-asyncio

## Global Constraints

- Python 3.11, virtualenv at `.venv`
- Run tests: `.venv/bin/python -m pytest tests/ -v` (no parallel)
- LangGraph message types: `HumanMessage`, `AIMessage`, `ToolMessage` from `langchain_core.messages`
- DeepSeek model via `get_llm()` from `finagent.config`
- All file writes use atomic write (temp + rename on same filesystem)
- Prompt templates with literal JSON braces use `.replace()` not `.format()`
- `multiprocessing.set_start_method("fork", force=True)` must stay before `app.run()`
- Memory files: `memory.md` ≤ 200 lines / 25KB; detail docs unlimited
- Commit messages: `feat:`, `fix:`, `test:` prefix, Chinese description OK

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `finagent/session.py` | Create | JSONL serialize/deserialize, write/load, count |
| `finagent/memory.py` | Create | MemoryLoader: mtime tracking + conditional injection |
| `finagent/governance.py` | Create | extract_from_turn + run_governance + check_governance_needed |
| `finagent/agent.py` | Modify | Add `create_agent_with_history()` |
| `finagent/tui.py` | Modify | Memory injection, session persistence, governance workers, main() CLI |
| `tests/test_session.py` | Create | Session module tests |
| `tests/test_memory.py` | Create | MemoryLoader tests |
| `tests/test_governance.py` | Create | Governance module tests |
| `tests/test_agent_memory.py` | Create | Agent history pre-seed test |
| `tests/conftest.py` | Create | Autouse fixture: mock persistence in TUI tests |

---

### Task 1: Session Module (`finagent/session.py`)

**Files:**
- Create: `finagent/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: `serialize_message(msg) -> dict`, `deserialize_message(data) -> Message`, `write_session(session_id, messages, cumulative_tokens) -> None`, `load_session(session_id) -> tuple[list, int]`, `count_sessions() -> int`, `atomic_write(path, content) -> None`, `SESSIONS_DIR: Path`

- [ ] **Step 1: Write failing tests for serialize/deserialize**

Create `tests/test_session.py`:

```python
"""Tests for session serialization."""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from finagent.session import serialize_message, deserialize_message


def test_serialize_human_message():
    msg = HumanMessage(content="hello", id="msg-1")
    data = serialize_message(msg)
    assert data["type"] == "human"
    assert data["content"] == "hello"
    assert data["id"] == "msg-1"


def test_serialize_ai_message_with_tool_calls():
    msg = AIMessage(
        content="分析完成",
        tool_calls=[{"name": "get_financials", "args": {"code": "000001"}, "id": "tc1", "type": "tool_call"}],
        usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
        id="msg-2",
    )
    data = serialize_message(msg)
    assert data["type"] == "ai"
    assert data["content"] == "分析完成"
    assert data["id"] == "msg-2"
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["name"] == "get_financials"
    assert data["usage_metadata"]["input_tokens"] == 500


def test_serialize_tool_message():
    msg = ToolMessage(content="营收650亿", tool_call_id="tc1", name="get_financials", id="msg-3")
    data = serialize_message(msg)
    assert data["type"] == "tool"
    assert data["content"] == "营收650亿"
    assert data["tool_call_id"] == "tc1"
    assert data["name"] == "get_financials"
    assert data["id"] == "msg-3"


def test_serialize_message_without_id():
    """Messages without id should still serialize (id omitted)."""
    msg = HumanMessage(content="no id")
    data = serialize_message(msg)
    assert data["type"] == "human"
    assert "id" not in data or data["id"] is None


def test_deserialize_roundtrip_human():
    original = HumanMessage(content="test", id="r-1")
    data = serialize_message(original)
    restored = deserialize_message(data)
    assert isinstance(restored, HumanMessage)
    assert restored.content == "test"
    assert restored.id == "r-1"


def test_deserialize_roundtrip_ai():
    original = AIMessage(
        content="reply",
        tool_calls=[{"name": "tool", "args": {}, "id": "t1", "type": "tool_call"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        id="r-2",
    )
    data = serialize_message(original)
    restored = deserialize_message(data)
    assert isinstance(restored, AIMessage)
    assert restored.content == "reply"
    assert restored.id == "r-2"
    assert len(restored.tool_calls) == 1


def test_deserialize_roundtrip_tool():
    original = ToolMessage(content="data", tool_call_id="t1", name="tool", id="r-3")
    data = serialize_message(original)
    restored = deserialize_message(data)
    assert isinstance(restored, ToolMessage)
    assert restored.content == "data"
    assert restored.tool_call_id == "t1"
    assert restored.name == "tool"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finagent.session'`

- [ ] **Step 3: Implement serialize/deserialize**

Create `finagent/session.py`:

```python
"""Session persistence: JSONL serialization for LangGraph messages."""
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ponytail: CWD-relative. If --resume run from different dir, session not found.
# Acceptable for TUI app always run from project root.
SESSIONS_DIR = Path(".finagent/sessions")


def atomic_write(path: Path, content: str) -> None:
    """Write to temp file then rename. Atomic on same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def serialize_message(msg) -> dict:
    """Convert LangGraph message to JSON dict."""
    data = {
        "type": msg.type,
        "content": msg.content,
        "ts": datetime.now().isoformat(),
    }
    msg_id = getattr(msg, "id", None)
    if msg_id is not None:
        data["id"] = msg_id
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = tool_calls
    usage = getattr(msg, "usage_metadata", None)
    if usage:
        data["usage_metadata"] = usage
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id is not None:
        data["tool_call_id"] = tool_call_id
    name = getattr(msg, "name", None)
    if name is not None:
        data["name"] = name
    return data


def deserialize_message(data: dict):
    """Reverse of serialize_message. JSON dict -> LangGraph message."""
    msg_type = data["type"]
    kwargs = {"content": data["content"]}
    if "id" in data and data["id"] is not None:
        kwargs["id"] = data["id"]
    if msg_type == "human":
        return HumanMessage(**kwargs)
    elif msg_type == "ai":
        if "tool_calls" in data:
            kwargs["tool_calls"] = data["tool_calls"]
        if "usage_metadata" in data:
            kwargs["usage_metadata"] = data["usage_metadata"]
        return AIMessage(**kwargs)
    elif msg_type == "tool":
        kwargs["tool_call_id"] = data["tool_call_id"]
        kwargs["name"] = data.get("name", "")
        return ToolMessage(**kwargs)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Write failing tests for write/load/count**

Append to `tests/test_session.py`:

```python
import pytest
from unittest.mock import patch

from finagent.session import write_session, load_session, count_sessions


def test_write_and_load_session(tmp_path, monkeypatch):
    """Round-trip: write messages, load them back."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    messages = [
        HumanMessage(content="分析000001", id="m1"),
        AIMessage(content="营收同比+8%", usage_metadata={"input_tokens": 500}, id="m2"),
        ToolMessage(content="数据", tool_call_id="tc1", name="get_financials", id="m3"),
    ]
    write_session("test-session", messages, 500)
    loaded, tokens = load_session("test-session")
    assert len(loaded) == 3
    assert isinstance(loaded[0], HumanMessage)
    assert loaded[0].content == "分析000001"
    assert isinstance(loaded[1], AIMessage)
    assert loaded[1].content == "营收同比+8%"
    assert isinstance(loaded[2], ToolMessage)
    assert loaded[2].name == "get_financials"
    assert tokens == 500


def test_load_session_missing_file(tmp_path, monkeypatch):
    """Missing session file returns empty list + zero tokens."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    messages, tokens = load_session("nonexistent")
    assert messages == []
    assert tokens == 0


def test_load_session_malformed_line_skipped(tmp_path, monkeypatch):
    """Malformed JSON lines are skipped, valid ones still load."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"type": "human", "content": "good"}\n'
        'NOT VALID JSON\n'
        '{"type": "meta", "cumulative_tokens": 42}\n',
        encoding="utf-8",
    )
    messages, tokens = load_session("bad")
    assert len(messages) == 1
    assert messages[0].content == "good"
    assert tokens == 42


def test_write_session_creates_parent_dir(tmp_path, monkeypatch):
    """Sessions dir is created if it doesn't exist."""
    target = tmp_path / "deep" / "nested"
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", target)
    write_session("s1", [], 0)
    assert (target / "s1.jsonl").exists()


def test_count_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    assert count_sessions() == 0
    (tmp_path / "a.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "not_json.txt").write_text("nope", encoding="utf-8")
    assert count_sessions() == 2
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_session.py -k "write or load or count" -v`
Expected: FAIL with `ImportError: cannot import name 'write_session'`

- [ ] **Step 7: Implement write/load/count**

Append to `finagent/session.py`:

```python
def session_path(session_id: str) -> Path:
    """Return path for a session JSONL file."""
    return SESSIONS_DIR / f"{session_id}.jsonl"


def write_session(session_id: str, messages: list, cumulative_tokens: int) -> None:
    """Overwrite session file with full message snapshot + meta line."""
    lines = []
    for msg in messages:
        lines.append(json.dumps(serialize_message(msg), ensure_ascii=False))
    lines.append(json.dumps(
        {"type": "meta", "cumulative_tokens": cumulative_tokens},
        ensure_ascii=False,
    ))
    atomic_write(session_path(session_id), "\n".join(lines) + "\n")


def load_session(session_id: str) -> tuple[list, int]:
    """Read JSONL, return (messages, cumulative_tokens).

    Skips malformed lines. Returns ([], 0) if file doesn't exist.
    """
    path = session_path(session_id)
    if not path.exists():
        return [], 0
    messages = []
    cumulative_tokens = 0
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
        else:
            try:
                messages.append(deserialize_message(data))
            except (KeyError, ValueError):
                continue
    return messages, cumulative_tokens


def count_sessions() -> int:
    """Count .jsonl files in sessions dir."""
    if not SESSIONS_DIR.exists():
        return 0
    return len(list(SESSIONS_DIR.glob("*.jsonl")))
```

- [ ] **Step 8: Run all session tests**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: PASS (12 tests)

- [ ] **Step 9: Commit**

```bash
git add finagent/session.py tests/test_session.py
git commit -m "feat: add session persistence module (JSONL serialize/deserialize)"
```

---

### Task 2: Memory Loader (`finagent/memory.py`)

**Files:**
- Create: `finagent/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: nothing (reads filesystem)
- Produces: `MemoryLoader` class with `get_injectable() -> str | None` and `reset() -> None`
- `MEMORY_FILES: list[tuple[str, Path]]` constant

- [ ] **Step 1: Write failing tests**

Create `tests/test_memory.py`:

```python
"""Tests for MemoryLoader mtime-based injection."""
from pathlib import Path

from finagent.memory import MemoryLoader


def test_get_injectable_first_call_returns_all(tmp_path):
    """First call returns content from all existing files."""
    user_file = tmp_path / "user.md"
    user_file.write_text("# 用户偏好\n喜欢简洁报告", encoding="utf-8")
    proj_file = tmp_path / "proj.md"
    proj_file.write_text("# 项目规则\nA股only", encoding="utf-8")
    mem_file = tmp_path / "memory.md"
    mem_file.write_text("# 摘要\n- 偏好: 简洁", encoding="utf-8")

    loader = MemoryLoader(files=[
        ("用户级", user_file),
        ("项目级", proj_file),
        ("自动记忆", mem_file),
    ])
    result = loader.get_injectable()
    assert result is not None
    assert "用户偏好" in result
    assert "项目规则" in result
    assert "摘要" in result


def test_get_injectable_no_changes_returns_none(tmp_path):
    """Second call with no file changes returns None."""
    f = tmp_path / "f.md"
    f.write_text("content", encoding="utf-8")
    loader = MemoryLoader(files=[("test", f)])
    first = loader.get_injectable()
    assert first is not None
    second = loader.get_injectable()
    assert second is None


def test_get_injectable_file_changed_returns_new_content(tmp_path):
    """When a file changes, get_injectable returns its new content."""
    f = tmp_path / "f.md"
    f.write_text("old", encoding="utf-8")
    loader = MemoryLoader(files=[("test", f)])
    loader.get_injectable()  # mark as seen
    f.write_text("new content", encoding="utf-8")
    result = loader.get_injectable()
    assert result is not None
    assert "new content" in result


def test_get_injectable_nonexistent_file_skipped(tmp_path):
    """Non-existent files are silently skipped."""
    missing = tmp_path / "nope.md"
    loader = MemoryLoader(files=[("missing", missing)])
    result = loader.get_injectable()
    assert result is None


def test_reset_clears_tracking(tmp_path):
    """After reset, next call returns content again."""
    f = tmp_path / "f.md"
    f.write_text("content", encoding="utf-8")
    loader = MemoryLoader(files=[("test", f)])
    loader.get_injectable()  # mark seen
    assert loader.get_injectable() is None  # no change
    loader.reset()
    result = loader.get_injectable()
    assert result is not None
    assert "content" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finagent.memory'`

- [ ] **Step 3: Implement MemoryLoader**

Create `finagent/memory.py`:

```python
"""Memory file loading with mtime-based conditional injection."""
from pathlib import Path

# Default file list: (label, path) pairs. Order determines injection order.
DEFAULT_MEMORY_FILES = [
    ("用户级长期记忆", Path.home() / ".finagent" / "finagent.md"),
    ("项目级长期记忆", Path(".finagent") / "finagent.md"),
    ("自动记忆摘要", Path(".finagent") / "memory" / "memory.md"),
]


class MemoryLoader:
    """Tracks file mtimes, returns changed content for injection.

    ponytail: mtime resolution depends on filesystem. APFS = nanosecond
    (fine locally). ext4/network mounts may be second-coarse — rapid
    edit+send within same second could miss a change.
    """

    def __init__(self, files: list[tuple[str, Path]] | None = None):
        self._files = files if files is not None else DEFAULT_MEMORY_FILES
        self._last_mtimes: dict[Path, float | None] = {}
        for _label, path in self._files:
            self._last_mtimes[path] = None

    def get_injectable(self) -> str | None:
        """Return memory content if any tracked file changed since last check.

        - First call: all existing files' content returned
        - Subsequent calls: only changed files' content returned
        - No changes: returns None
        - Non-existent files: skipped
        """
        changed_sections = []
        for label, path in self._files:
            if not path.exists():
                self._last_mtimes[path] = None
                continue
            current_mtime = path.stat().st_mtime
            last_mtime = self._last_mtimes.get(path)
            if last_mtime is not None and current_mtime == last_mtime:
                continue
            self._last_mtimes[path] = current_mtime
            content = path.read_text(encoding="utf-8").strip()
            if content:
                changed_sections.append(f"## {label}\n{content}")
        if not changed_sections:
            return None
        return "\n\n".join(changed_sections)

    def reset(self) -> None:
        """Clear mtime tracking so next call re-reads all files."""
        self._last_mtimes = {p: None for p in self._last_mtimes}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/memory.py tests/test_memory.py
git commit -m "feat: add MemoryLoader with mtime-based conditional injection"
```

---

### Task 3: Governance — Extract + Check (`finagent/governance.py`)

**Files:**
- Create: `finagent/governance.py`
- Test: `tests/test_governance.py`

**Interfaces:**
- Consumes: `atomic_write` from `finagent.session`, `count_sessions` from `finagent.session`, `get_llm` from `finagent.config`
- Produces: `extract_from_turn(messages: list) -> None` (async), `check_governance_needed() -> bool`, `_memory_lock: asyncio.Lock`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governance.py`:

```python
"""Tests for governance: extraction and check."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from finagent.governance import check_governance_needed


def test_check_needed_true(tmp_path, monkeypatch):
    """Returns True when >24h and >=5 sessions since last governance."""
    import datetime
    # count_sessions() reads finagent.session.SESSIONS_DIR, not governance's
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path / "memory")
    # Create 5 session files
    for i in range(5):
        (tmp_path / f"s{i}.jsonl").write_text("{}", encoding="utf-8")
    # Last governance was 2 days ago, processed 0 sessions
    lg = tmp_path / "memory" / ".last_governance"
    lg.parent.mkdir(parents=True)
    old_time = (datetime.datetime.now() - datetime.timedelta(hours=48)).isoformat()
    lg.write_text(json.dumps({"timestamp": old_time, "processed_sessions": 0}), encoding="utf-8")
    assert check_governance_needed() is True


def test_check_needed_false_recent_governance(tmp_path, monkeypatch):
    """Returns False when governance ran <24h ago."""
    import datetime
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path / "memory")
    for i in range(10):
        (tmp_path / f"s{i}.jsonl").write_text("{}", encoding="utf-8")
    lg = tmp_path / "memory" / ".last_governance"
    lg.parent.mkdir(parents=True)
    recent = (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat()
    lg.write_text(json.dumps({"timestamp": recent, "processed_sessions": 0}), encoding="utf-8")
    assert check_governance_needed() is False


def test_check_needed_false_few_sessions(tmp_path, monkeypatch):
    """Returns False when <5 new sessions."""
    import datetime
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path / "memory")
    (tmp_path / "s0.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "s1.jsonl").write_text("{}", encoding="utf-8")
    lg = tmp_path / "memory" / ".last_governance"
    lg.parent.mkdir(parents=True)
    old_time = (datetime.datetime.now() - datetime.timedelta(hours=48)).isoformat()
    lg.write_text(json.dumps({"timestamp": old_time, "processed_sessions": 0}), encoding="utf-8")
    assert check_governance_needed() is False


def test_check_needed_no_last_governance(tmp_path, monkeypatch):
    """No .last_governance file: needs >=5 sessions AND effectively treats
    processed_sessions as 0."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path / "memory")
    for i in range(6):
        (tmp_path / f"s{i}.jsonl").write_text("{}", encoding="utf-8")
    assert check_governance_needed() is True


@pytest.mark.asyncio
async def test_extract_from_turn_with_findings(tmp_path, monkeypatch):
    """Extract writes to detail files when LLM returns findings."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.governance.SESSIONS_DIR", tmp_path / "sessions")

    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "preference": ["报告用简洁格式"],
        "project": [],
        "feedback": [],
        "reference": [],
    })
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import extract_from_turn
        messages = [
            HumanMessage(content="以后报告用简洁格式"),
            AIMessage(content="好的，我会用简洁格式"),
        ]
        await extract_from_turn(messages)

    pref = (tmp_path / "preference.md").read_text(encoding="utf-8")
    assert "简洁格式" in pref
    mem = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert "preference" in mem


@pytest.mark.asyncio
async def test_extract_from_turn_no_findings(tmp_path, monkeypatch):
    """When LLM returns all empty lists, no files are written."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")

    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "preference": [], "project": [], "feedback": [], "reference": [],
    })
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import extract_from_turn
        # Use >=50 chars so the length filter doesn't skip before calling LLM
        messages = [
            HumanMessage(content="请帮我分析一下这个公司的财务数据，我需要详细的财报点评"),
            AIMessage(content="好的，我来为您分析这家公司的财务状况和经营情况"),
        ]
        await extract_from_turn(messages)

    assert not (tmp_path / "preference.md").exists()
    assert not (tmp_path / "memory.md").exists()


@pytest.mark.asyncio
async def test_extract_skips_short_turns(tmp_path, monkeypatch):
    """Turns with <50 chars of text are skipped (no LLM call)."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.governance.SESSIONS_DIR", tmp_path / "sessions")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import extract_from_turn
        messages = [HumanMessage(content="ok"), AIMessage(content="sure")]
        await extract_from_turn(messages)

    mock_llm.ainvoke.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_governance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finagent.governance'`

- [ ] **Step 3: Implement governance module (extract + check)**

Create `finagent/governance.py`:

```python
"""Memory governance: per-turn extraction + periodic maintenance."""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

from finagent.config import get_llm
from finagent.session import atomic_write, count_sessions

MEMORY_DIR = Path(".finagent/memory")

# Module-level lock ensures extraction and governance never write concurrently
_memory_lock = asyncio.Lock()

MEMORY_MD_MAX_LINES = 200
MEMORY_MD_MAX_BYTES = 25_600  # 25KB

EXTRACT_PROMPT = """分析以下对话轮次，提取适合长期记忆的内容。

只提取明确、持久的信息。不确定的不提取。
分类写入：
- preference: 用户明确表达的偏好（格式、风格、工作方式）
- project: 项目规则、约束、技术决策
- feedback: 用户对 agent 行为的纠正/指导
- reference: 外部信息来源（URL、文档路径、工具用法）

输出 JSON，每类一个列表。无内容则空列表。
{"preference": [...], "project": [...], "feedback": [...], "reference": [...]}

对话：
{messages}
"""


def check_governance_needed() -> bool:
    """Check if governance should run: >24h since last AND >=5 new sessions."""
    lg_path = MEMORY_DIR / ".last_governance"
    if lg_path.exists():
        try:
            data = json.loads(lg_path.read_text(encoding="utf-8"))
            last_time = datetime.fromisoformat(data["timestamp"])
            processed = data.get("processed_sessions", 0)
        except (json.JSONDecodeError, KeyError, ValueError):
            return False
    else:
        last_time = datetime.min
        processed = 0

    hours_elapsed = (datetime.now() - last_time).total_seconds() / 3600
    if hours_elapsed < 24:
        return False

    new_sessions = count_sessions() - processed
    return new_sessions >= 5


async def extract_from_turn(messages: list) -> None:
    """Per-turn extraction. LLM analyzes the latest turn, appends to detail + memory.md.

    Uses .replace() for prompt templating (template contains literal JSON braces).
    Only passes the latest user msg + AI reply (not full tool call chain).
    Skips turns with <50 chars of text.
    """
    # Extract only the latest user turn + AI reply
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_user_idx = i
            break
    if last_user_idx is None:
        return

    turn_msgs = messages[last_user_idx:]
    lines = []
    for m in turn_msgs:
        if isinstance(m, HumanMessage) and m.content:
            lines.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"助手: {m.content}")
    turn_text = "\n".join(lines)

    if len(turn_text) < 50:
        return

    prompt = EXTRACT_PROMPT.replace("{messages}", turn_text)
    llm = get_llm()

    async with _memory_lock:
        response = await llm.ainvoke(prompt)
        content = response.content
        # Extract JSON from response (may have markdown fences)
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            return
        try:
            findings = json.loads(content[start:end])
        except json.JSONDecodeError:
            return

        has_content = False
        for category in ("preference", "project", "feedback", "reference"):
            items = findings.get(category, [])
            if not items:
                continue
            has_content = True
            detail_path = MEMORY_DIR / f"{category}.md"
            existing = detail_path.read_text(encoding="utf-8") if detail_path.exists() else ""
            new_section = "\n".join(f"- {item}" for item in items)
            atomic_write(detail_path, existing + new_section + "\n")

        if has_content:
            _append_memory_md(findings)


def _append_memory_md(findings: dict) -> None:
    """Append summary lines to memory.md, enforcing size cap."""
    path = MEMORY_DIR / "memory.md"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    for category in ("preference", "project", "feedback", "reference"):
        items = findings.get(category, [])
        for item in items:
            content += f"- [{category}]({category}.md) — {item}\n"
    # Enforce line cap
    lines = content.split("\n")
    if len(lines) > MEMORY_MD_MAX_LINES:
        lines = lines[-MEMORY_MD_MAX_LINES:]
        content = "\n".join(lines)
    # Enforce byte cap
    while len(content.encode("utf-8")) > MEMORY_MD_MAX_BYTES and lines:
        lines.pop(0)
        content = "\n".join(lines)
    atomic_write(path, content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_governance.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/governance.py tests/test_governance.py
git commit -m "feat: add governance module — per-turn extraction + check_governance_needed"
```

---

### Task 4: Governance — run_governance

**Files:**
- Modify: `finagent/governance.py` (append run_governance)
- Test: `tests/test_governance.py` (append tests)

**Interfaces:**
- Consumes: `atomic_write` from `finagent.session`, existing module constants
- Produces: `run_governance() -> None` (async)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_governance.py`:

```python
@pytest.mark.asyncio
async def test_run_governance_rewrites_files(tmp_path, monkeypatch):
    """Governance reads existing memory files, LLM returns cleaned versions,
    all 5 files are written atomically."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()

    # Seed existing files
    (tmp_path / "preference.md").write_text("- old pref\n- dup pref\n", encoding="utf-8")
    (tmp_path / "memory.md").write_text("# old summary\n", encoding="utf-8")

    llm_output = """=== FILE: memory.md ===
# 记忆摘要

- [偏好](preference.md) — 简洁格式

=== FILE: preference.md ===
- 简洁报告格式
=== FILE: project.md ===
- A股only
=== FILE: feedback.md ===
=== FILE: reference.md ===
"""
    mock_response = MagicMock()
    mock_response.content = llm_output
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import run_governance
        await run_governance()

    pref = (tmp_path / "preference.md").read_text(encoding="utf-8")
    assert "简洁报告格式" in pref
    assert "old pref" not in pref
    mem = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert "记忆摘要" in mem
    proj = (tmp_path / "project.md").read_text(encoding="utf-8")
    assert "A股" in proj


@pytest.mark.asyncio
async def test_run_governance_updates_last_governance(tmp_path, monkeypatch):
    """After governance, .last_governance is updated with timestamp + session count."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "s0.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "sessions" / "s1.jsonl").write_text("{}", encoding="utf-8")

    mock_response = MagicMock()
    mock_response.content = "=== FILE: memory.md ===\nempty\n=== FILE: preference.md ===\n=== FILE: project.md ===\n=== FILE: feedback.md ===\n=== FILE: reference.md ===\n"
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import run_governance
        await run_governance()

    lg = json.loads((tmp_path / ".last_governance").read_text(encoding="utf-8"))
    assert "timestamp" in lg
    assert lg["processed_sessions"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_governance.py -k "run_governance" -v`
Expected: FAIL with `ImportError: cannot import name 'run_governance'`

- [ ] **Step 3: Implement run_governance**

Append to `finagent/governance.py`:

```python
GOVERNANCE_PROMPT = """你是记忆维护助手。以下是当前记忆文件内容。

任务：合并、去重、消解冲突、删除过期内容，输出整洁版本。

规则：
1. 重复内容合并为一条
2. 冲突内容保留最新，标注旧值已废弃
3. 明确过期的删除
4. 重新生成 memory.md 摘要 + 索引
5. memory.md 不超过 200 行 / 25KB。超限时优先压缩低价值条目
6. detail 文档无大小限制（不注入上下文，按需读取）
7. 用 === FILE: <name> === 分隔各文件输出

当前记忆：
{current_memory}
"""


async def run_governance() -> None:
    """Periodic maintenance. Reads memory/ files, LLM rewrites them cleanly.

    Does NOT read sessions — only compacts existing memory/ content.
    Stages all 5 files in a temp dir, then renames atomically.
    """
    # Read current memory files
    current_parts = []
    for name in ("memory.md", "preference.md", "project.md", "feedback.md", "reference.md"):
        path = MEMORY_DIR / name
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                current_parts.append(f"=== FILE: {name} ===\n{content}")

    current_memory = "\n\n".join(current_parts) if current_parts else "(空)"
    prompt = GOVERNANCE_PROMPT.replace("{current_memory}", current_memory)

    llm = get_llm()

    async with _memory_lock:
        response = await llm.ainvoke(prompt)
        output = response.content

        # Parse output: split by === FILE: <name> === markers
        files = {}
        current_name = None
        current_lines = []
        for line in output.split("\n"):
            if line.startswith("=== FILE: ") and line.endswith(" ==="):
                if current_name:
                    files[current_name] = "\n".join(current_lines).strip()
                current_name = line[len("=== FILE: "):-len(" ===")]
                current_lines = []
            else:
                current_lines.append(line)
        if current_name:
            files[current_name] = "\n".join(current_lines).strip()

        # Ensure all 5 files exist (missing sections default to empty)
        for name in ("memory.md", "preference.md", "project.md", "feedback.md", "reference.md"):
            if name not in files:
                files[name] = ""

        # Stage all files in temp subdir, then rename
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        staging = MEMORY_DIR / ".staging"
        staging.mkdir(exist_ok=True)
        for name, content in files.items():
            (staging / name).write_text(content + "\n", encoding="utf-8")
        # Atomic swap: rename all staged files
        for name in files:
            (staging / name).rename(MEMORY_DIR / name)
        staging.rmdir()

        # Update .last_governance
        lg_path = MEMORY_DIR / ".last_governance"
        lg_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_sessions": count_sessions(),
        }
        atomic_write(lg_path, json.dumps(lg_data, ensure_ascii=False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_governance.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/governance.py tests/test_governance.py
git commit -m "feat: add run_governance — periodic memory compaction with atomic batch write"
```

---

### Task 5: Agent History Pre-seed (`finagent/agent.py`)

**Files:**
- Modify: `finagent/agent.py` (append `create_agent_with_history`)
- Test: `tests/test_agent_memory.py`

**Interfaces:**
- Consumes: `create_agent()` from existing code
- Produces: `create_agent_with_history(thread_id: str, messages: list) -> agent`

- [ ] **Step 1: Write failing test**

Create `tests/test_agent_memory.py`:

```python
"""Test agent history pre-seeding via update_state."""
from langchain_core.messages import HumanMessage, AIMessage

from finagent.agent import create_agent_with_history


def test_create_agent_with_history_seeds_checkpoint():
    """update_state injects messages into checkpointer without executing graph."""
    messages = [
        HumanMessage(content="你好", id="m1"),
        AIMessage(content="你好！", id="m2"),
    ]
    agent = create_agent_with_history("test-thread", messages)

    # Verify messages are in the checkpoint
    state = agent.get_state(config={"configurable": {"thread_id": "test-thread"}})
    checkpoint_msgs = state.values.get("messages", [])
    assert len(checkpoint_msgs) >= 2
    assert checkpoint_msgs[0].content == "你好"
    assert checkpoint_msgs[1].content == "你好！"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agent_memory.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_agent_with_history'`

- [ ] **Step 3: Implement create_agent_with_history**

Append to `finagent/agent.py`:

```python
def create_agent_with_history(thread_id: str, messages: list):
    """Build agent, pre-seed checkpointer with message history.

    Uses update_state to write messages into checkpoint without
    triggering graph execution. The agent's add_messages reducer
    deduplicates by message id.
    """
    agent = create_agent()
    agent.update_state(
        {"configurable": {"thread_id": thread_id}},
        {"messages": messages},
    )
    return agent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_agent_memory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finagent/agent.py tests/test_agent_memory.py
git commit -m "feat: add create_agent_with_history for session resume"
```

---

### Task 6: TUI Memory Injection + Conftest

**Files:**
- Modify: `finagent/tui.py` (imports, `__init__`, `_build_user_message`)
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `MemoryLoader` from `finagent.memory`
- Produces: `FinAgentApp.__init__` now accepts `resume_session_id`, `self._memory_loader` attribute

- [ ] **Step 1: Add tui.py imports for persistence modules**

In `finagent/tui.py`, add to imports (after existing imports, around line 16-19):

```python
from finagent.memory import MemoryLoader
from finagent.session import write_session, load_session, count_sessions
from finagent.governance import extract_from_turn, run_governance, check_governance_needed
from finagent.agent import create_agent_with_history
```

These imports must exist before conftest.py patches them by name.

- [ ] **Step 2: Create conftest with autouse fixture**

Create `tests/conftest.py`:

```python
"""Shared test fixtures."""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_tui_persistence():
    """Auto-mock persistence layer in TUI tests.

    Patches at the tui.py import level so direct module tests are unaffected.
    """
    with patch("finagent.tui.write_session"), \
         patch("finagent.tui.extract_from_turn", new_callable=AsyncMock), \
         patch("finagent.tui.run_governance", new_callable=AsyncMock), \
         patch("finagent.tui.check_governance_needed", return_value=False):
        yield
```

- [ ] **Step 3: Write failing test for memory injection**

Append to `tests/test_tui.py`:

```python
@pytest.mark.asyncio
async def test_memory_injected_on_first_message(app, tmp_path, monkeypatch):
    """First user message in a session includes memory content."""
    from finagent.memory import MemoryLoader

    mem_file = tmp_path / "finagent.md"
    mem_file.write_text("# 项目记忆\nA股规则", encoding="utf-8")
    app._memory_loader = MemoryLoader(files=[("test", mem_file)])

    captured = []
    async def fake_astream(*args, **kwargs):
        captured.append(args[0])
        return
        yield

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "hello"
        await pilot.press("enter")
        await pilot.pause()

    assert captured
    content = captured[0]["messages"][0]["content"]
    assert "项目记忆" in content
    assert "<system-reminder>" in content


@pytest.mark.asyncio
async def test_memory_not_injected_when_unchanged(app, tmp_path):
    """After first injection, unchanged files don't re-inject."""
    from finagent.memory import MemoryLoader

    mem_file = tmp_path / "finagent.md"
    mem_file.write_text("memory content", encoding="utf-8")
    app._memory_loader = MemoryLoader(files=[("test", mem_file)])

    captured = []
    async def fake_astream(*args, **kwargs):
        captured.append(args[0])
        return
        yield

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        # First message: memory injected
        app.query_one("#input", ChatInput).text = "first"
        await pilot.press("enter")
        await pilot.pause()
        # Second message: no change, memory not re-injected
        app.query_one("#input", ChatInput).text = "second"
        await pilot.press("enter")
        await pilot.pause()

    first_content = captured[0]["messages"][0]["content"]
    second_content = captured[1]["messages"][0]["content"]
    assert "memory content" in first_content
    assert "memory content" not in second_content
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "memory_injected or memory_not_injected" -v`
Expected: FAIL (MemoryLoader not yet integrated in __init__/_build_user_message)

- [ ] **Step 5: Modify __init__ + _build_user_message**

Modify `__init__` (replace lines 83-94):

```python
def __init__(self, resume_session_id: str | None = None):
    super().__init__()
    load_env()
    self._memory_loader = MemoryLoader()
    if resume_session_id:
        messages, tokens = load_session(resume_session_id)
        self.thread_id = resume_session_id
        self._cumulative_input_tokens = tokens
        self.agent = create_agent_with_history(self.thread_id, messages)
    else:
        self.thread_id = str(uuid.uuid4())
        self.agent = create_agent()
        self._cumulative_input_tokens = 0
    self._queue: list[tuple[str, Static]] = []
    self._streaming_worker = None
    # Skill catalog (refreshed by /reload_skills and at startup)
    metas = scan_skills()
    self._skill_catalog_names: frozenset[str] = frozenset(metas.keys())
    self._skill_catalog: str = render_catalog(metas)
```

Modify `_build_user_message` (replace lines 158-170):

```python
def _build_user_message(self, user_input: str) -> str:
    """Wrap user input with memory + skill catalog as system-reminder suffixes."""
    parts = [user_input]

    # Memory injection (conditional on mtime change)
    memory_content = self._memory_loader.get_injectable()
    if memory_content:
        parts.append(f"<system-reminder>\n{memory_content}\n</system-reminder>")

    # Skill catalog (every turn)
    parts.append(
        f"<system-reminder>\n"
        f"可用 skills(用 load_skill 工具加载,或用户输入 /<name>):\n"
        f"{self._skill_catalog}\n"
        f"</system-reminder>"
    )

    return "\n\n".join(parts)
```

- [ ] **Step 6: Run memory injection tests**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "memory_injected or memory_not_injected" -v`
Expected: PASS

- [ ] **Step 7: Run full existing test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all existing tests still pass — conftest auto-mocks persistence)

- [ ] **Step 8: Commit**

```bash
git add finagent/tui.py tests/conftest.py tests/test_tui.py
git commit -m "feat: integrate MemoryLoader into TUI with conditional injection"
```

---

### Task 7: TUI Session Persistence

**Files:**
- Modify: `finagent/tui.py` (`_start_stream` finally block, `_do_clear`)

**Interfaces:**
- Consumes: `write_session` from `finagent.session`, `extract_from_turn` from `finagent.governance`

- [ ] **Step 1: Write failing test for session write after stream**

Append to `tests/test_tui.py`:

```python
@pytest.mark.asyncio
async def test_session_written_after_stream(app, tmp_path, monkeypatch):
    """After a successful stream, session JSONL is written."""
    from langchain_core.messages import AIMessage, HumanMessage
    from unittest.mock import patch, MagicMock

    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    # Also patch the import in tui to use this dir
    app.thread_id = "test-persist"

    fake_state = MagicMock()
    fake_state.values = {
        "messages": [HumanMessage(content="hi", id="m1")]
    }
    app.agent.get_state = MagicMock(return_value=fake_state)

    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="reply"), {}))
        ai = AIMessage(content="reply", usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
        yield ("updates", {"agent": {"messages": [ai]}})

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "hello"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    # write_session was called (mocked by conftest, check call args)
    # Verify via the mock
    from finagent.tui import write_session
    write_session.assert_called()


@pytest.mark.asyncio
async def test_do_clear_resets_memory_loader(app):
    """/clear resets MemoryLoader so next message re-injects memory."""
    from unittest.mock import MagicMock
    app._memory_loader = MagicMock()
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "/clear"
        await pilot.press("enter")
        await pilot.pause()
    app._memory_loader.reset.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "session_written or clear_resets_memory" -v`
Expected: FAIL (write_session not yet called in finally, reset not called in _do_clear)

- [ ] **Step 3: Modify _start_stream finally block**

In `finagent/tui.py`, modify the finally block of `_start_stream` (around line 248-262). After `self._streaming_worker = None` and before the queue flush, add session persistence:

```python
finally:
    current = get_current_worker()
    if self._streaming_worker is not current:
        # cancelled/superseded: don't clobber a newer worker's state
        return
    self._refresh_status_bar()
    self._streaming_worker = None

    # Session persistence (best-effort)
    try:
        state = self.agent.get_state(
            config={"configurable": {"thread_id": self.thread_id}}
        )
        msgs = state.values.get("messages", [])
        write_session(self.thread_id, msgs, self._cumulative_input_tokens)
        self._run_extraction(msgs)
    except Exception:
        pass  # persistence is best-effort; don't crash UI

    # remove queued message widgets and flush merged queue
    if self._queue:
        for _text, w in self._queue:
            w.remove()
        merged = "\n".join(t for t, _ in self._queue)
        self._queue.clear()
        self._add_message(f"> {merged}", classes="message-user")
        self._streaming_worker = self._start_stream(merged)
```

Note: `self.agent.get_state` is sync (MemorySaver is in-memory, fast).

- [ ] **Step 4: Add _run_extraction + _run_governance_check worker methods**

Add to `FinAgentApp` class (before `_do_report` or at end of class):

```python
@work
async def _run_extraction(self, messages: list) -> None:
    """Per-turn memory extraction. Async, non-blocking."""
    try:
        await extract_from_turn(messages)
    except Exception as e:
        self._set_status(f"记忆提取失败: {e}")
        self._refresh_status_bar()

@work
async def _run_governance_check(self) -> None:
    """Check if governance is needed, run if so."""
    if not check_governance_needed():
        return
    self._set_status("记忆治理中...")
    try:
        await run_governance()
    except Exception as e:
        self._add_message(f"治理失败: {e}", classes="message-error")
    finally:
        self._refresh_status_bar()
```

- [ ] **Step 5: Modify _do_clear to reset memory + check governance**

In `_do_clear` (around line 270-286), add after `self._refresh_status_bar()`:

```python
def _do_clear(self) -> None:
    """Clear conversation memory and chat view."""
    if self._streaming_worker is not None and self._streaming_worker.is_running:
        self._streaming_worker.cancel()
    self._streaming_worker = None
    reset_checkpoint()
    self.thread_id = str(uuid.uuid4())
    self._cumulative_input_tokens = 0
    self.agent = create_agent()
    set_current_report(None)
    self._queue.clear()
    self.query_one("#chat-view").remove_children()
    self._refresh_status_bar()
    self._memory_loader.reset()
    self._run_governance_check()
```

- [ ] **Step 6: Add governance check to on_mount**

In `on_mount` (around line 102-107), add at the end:

```python
def on_mount(self) -> None:
    self.query_one("#chat-view").can_focus = False
    self._add_message("输入股票代码 + 报告期开始分析。输入 /help 查看命令。")
    self.query_one("#input").focus()
    self._refresh_status_bar()
    self._run_governance_check()
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "session_written or clear_resets_memory" -v`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 9: Commit**

```bash
git add finagent/tui.py tests/test_tui.py
git commit -m "feat: add session persistence + governance workers to TUI"
```

---

### Task 8: CLI --resume Support

**Files:**
- Modify: `finagent/tui.py` (`main()` function)

**Interfaces:**
- Consumes: `FinAgentApp(resume_session_id=...)` from Task 6 changes

- [ ] **Step 1: Write failing test**

Append to `tests/test_tui.py`:

```python
def test_finagent_app_accepts_resume_id():
    """FinAgentApp.__init__ accepts resume_session_id parameter."""
    with patch("finagent.tui.create_agent", return_value=MagicMock()), \
         patch("finagent.tui.create_agent_with_history", return_value=MagicMock()), \
         patch("finagent.tui.load_session", return_value=([], 0)):
        app = FinAgentApp(resume_session_id="some-id")
        assert app.thread_id == "some-id"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "accepts_resume_id" -v`
Expected: FAIL or PASS (depends on whether __init__ change from Task 6 already covers this)

Note: This test may already pass from Task 6 changes. If so, skip to Step 3.

- [ ] **Step 3: Modify main() to add --resume CLI**

In `finagent/tui.py`, replace `main()` (lines 384-390):

```python
def main():
    # Textual replaces file descriptors (stdin/stdout/stderr). Python's default
    # 'spawn' start method needs to pass these fds to child processes, which fails
    # with "bad value(s) in fds_to_keep". 'fork' inherits fds without spawning.
    multiprocessing.set_start_method("fork", force=True)

    import argparse
    parser = argparse.ArgumentParser(description="FinAgent — A股财报点评助手")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复指定 session ID")
    args = parser.parse_args()

    app = FinAgentApp(resume_session_id=args.resume)
    app.run()

    # Print resume hint on exit
    if app.thread_id:
        print(f"\nResume this session with:\n"
              f"python -m finagent --resume {app.thread_id}")
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "accepts_resume_id" -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add finagent/tui.py tests/test_tui.py
git commit -m "feat: add --resume CLI flag for session restoration"
```

---

## Self-Review

**1. Spec coverage check:**

| Spec section | Task |
|---|---|
| session.py serialize/deserialize | Task 1 |
| session.py write/load/count | Task 1 |
| session.py atomic_write | Task 1 |
| session.py JSONL format + meta line | Task 1 |
| memory.py MemoryLoader | Task 2 |
| memory.py mtime conditional injection | Task 2 |
| governance.py extract_from_turn | Task 3 |
| governance.py check_governance_needed | Task 3 |
| governance.py run_governance | Task 4 |
| governance.py asyncio.Lock | Task 3 (module-level) |
| governance.py memory.md size cap | Task 3 (_append_memory_md) |
| governance.py .replace() templating | Task 3 + Task 4 |
| agent.py create_agent_with_history | Task 5 |
| tui.py MemoryLoader integration | Task 6 |
| tui.py _build_user_message memory block | Task 6 |
| tui.py __init__ resume support | Task 6 |
| tui.py session write in finally | Task 7 |
| tui.py _run_extraction worker | Task 7 |
| tui.py _run_governance_check worker | Task 7 |
| tui.py _do_clear memory reset | Task 7 |
| tui.py on_mount governance check | Task 7 |
| tui.py main() --resume CLI | Task 8 |
| tui.py main() multiprocessing fork | Task 8 |
| conftest.py autouse fixture | Task 6 |

All spec sections covered.

**2. Placeholder scan:** No TBD/TODO/vague items found. All steps have actual code.

**3. Type consistency check:**
- `write_session(session_id: str, messages: list, cumulative_tokens: int)` — consistent across session.py, tui.py
- `load_session(session_id: str) -> tuple[list, int]` — consistent across session.py, tui.py
- `extract_from_turn(messages: list) -> None` — consistent across governance.py, tui.py
- `run_governance() -> None` — consistent across governance.py, tui.py
- `check_governance_needed() -> bool` — consistent across governance.py, tui.py
- `MemoryLoader(files=...)` — consistent across memory.py, test usage
- `create_agent_with_history(thread_id: str, messages: list)` — consistent across agent.py, tui.py
