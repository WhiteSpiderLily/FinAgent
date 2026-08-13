# Memory Module v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicate memory records, fix memory.md responsibility drift, and block malformed writes via deterministic validation.

**Architecture:** Extract path adds `validate_and_dedup` (exact dedup + format check) and stops writing memory.md. Governance path gains async cap enforcement with 3x LLM compress retry then header-preserving truncation. memory.md becomes governance-only derived output.

**Tech Stack:** Python 3.9+, asyncio, langchain LLM, pytest

## Global Constraints

- `.replace()` for prompt templating (templates contain literal JSON braces, `.format()` would break)
- `removeprefix()` (Python 3.9+) for prefix stripping, NOT `lstrip()` (character set, over-dedups)
- `_memory_lock` guards all file writes in extract path; governance cap runs OUTSIDE lock, only stage+rename inside lock
- Sub-doc content read before lock for prompt injection; re-read inside lock for write base (avoids TOCTOU race between concurrent extracts)
- No backward compatibility shims
- Tests run sequentially (no parallel), project env is `.venv`

**Spec:** `docs/superpowers/specs/2026-08-13-memory-module-v2-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `finagent/governance.py` | Memory extraction + governance logic | Modify |
| `tests/test_governance.py` | Governance tests | Modify |

No new files created. All changes within existing two files.

---

### Task 1: validate_and_dedup — Pure Validation + Dedup Function

**Files:**
- Modify: `finagent/governance.py` (add function after `EXTRACT_PROMPT` block, before `check_governance_needed`)
- Test: `tests/test_governance.py`

**Interfaces:**
- Consumes: nothing (pure function)
- Produces: `validate_and_dedup(items: list, existing: str) -> list[str]`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_governance.py`:

```python
from finagent.governance import validate_and_dedup


def test_validate_rejects_non_str():
    items = [{"key": "val"}, 42, None, "valid"]
    assert validate_and_dedup(items, "") == ["valid"]


def test_validate_rejects_empty():
    items = ["", "   ", "\t\n", "valid"]
    assert validate_and_dedup(items, "") == ["valid"]


def test_validate_rejects_non_list():
    assert validate_and_dedup("not a list", "") == []
    assert validate_and_dedup({"a": 1}, "") == []
    assert validate_and_dedup(None, "") == []


def test_validate_dedup_exact():
    existing = "- 报告结构固定六段\n- 仅分析A股"
    items = ["报告结构固定六段", "新条目"]
    assert validate_and_dedup(items, existing) == ["新条目"]


def test_validate_dedup_strips_prefix():
    existing = "- foo\n- bar"
    items = ["foo", "bar", "baz"]
    assert validate_and_dedup(items, existing) == ["baz"]


def test_validate_dedup_no_overstrip():
    existing = "- foo"
    items = ["foo", "-- foo"]
    result = validate_and_dedup(items, existing)
    assert "-- foo" in result
    assert "foo" not in result


def test_validate_passes_new():
    assert validate_and_dedup(["新条目"], "") == ["新条目"]


def test_validate_mixed():
    existing = "- exists\n- also exists"
    items = [42, "", "exists", "also exists", "new1", "new1", "new2"]
    assert validate_and_dedup(items, existing) == ["new1", "new2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_governance.py -k validate -v`
Expected: FAIL with `ImportError: cannot import name 'validate_and_dedup'`

- [ ] **Step 3: Write minimal implementation**

Add to `finagent/governance.py` after the `EXTRACT_PROMPT` block (after line 37), before `check_governance_needed`:

```python
def validate_and_dedup(items: list, existing: str) -> list[str]:
    """Filter invalid items + exact dedup against existing content.

    Non-list items returns empty. Non-str items dropped. Empty items dropped.
    Exact duplicates (after stripping '- ' prefix from existing lines) dropped.
    Intra-batch duplicates also dropped.
    """
    if not isinstance(items, list):
        return []
    existing_set = {
        line.strip().removeprefix("- ").strip()
        for line in existing.split("\n")
        if line.strip()
    }
    result = []
    for item in items:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in existing_set:
            continue
        result.append(normalized)
        existing_set.add(normalized)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_governance.py -k validate -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/test_governance.py -v`
Expected: PASS (existing tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add finagent/governance.py tests/test_governance.py
git commit -m "feat: add validate_and_dedup for memory write validation"
```

---

### Task 2: Cap Helpers — _within_cap + _truncate_preserve_header

**Files:**
- Modify: `finagent/governance.py` (add new helper functions, do NOT delete old `_enforce_memory_md_cap` yet — it is replaced in Task 4)
- Test: `tests/test_governance.py`

**Interfaces:**
- Consumes: `MEMORY_MD_MAX_LINES`, `MEMORY_MD_MAX_BYTES` (module constants)
- Produces: `_within_cap(content: str) -> bool`, `_truncate_preserve_header(content: str) -> str`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_governance.py`:

```python
from finagent.governance import _within_cap, _truncate_preserve_header
from finagent.governance import MEMORY_MD_MAX_LINES


def test_within_cap_under():
    assert _within_cap("short content") is True


def test_within_cap_over_lines():
    content = "\n".join(f"line {i}" for i in range(MEMORY_MD_MAX_LINES + 1))
    assert _within_cap(content) is False


def test_within_cap_over_bytes():
    from finagent.governance import MEMORY_MD_MAX_BYTES
    content = "x" * (MEMORY_MD_MAX_BYTES + 1)
    assert _within_cap(content) is False


def test_truncate_preserve_header():
    header = "# Title\n\n"
    body = "\n".join(f"- item {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    content = header + body
    result = _truncate_preserve_header(content)
    lines = result.split("\n")
    assert lines[0] == "# Title"
    assert lines[1] == ""
    assert len(lines) <= MEMORY_MD_MAX_LINES


def test_truncate_no_blank_line():
    content = "\n".join(f"line {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    result = _truncate_preserve_header(content)
    assert len(result.split("\n")) <= MEMORY_MD_MAX_LINES


def test_truncate_header_over_cap():
    header = "\n".join(f"# header line {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    content = header + "\n\n- body item"
    result = _truncate_preserve_header(content)
    assert len(result.split("\n")) <= MEMORY_MD_MAX_LINES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_governance.py -k "within_cap or truncate" -v`
Expected: FAIL with `ImportError: cannot import name '_within_cap'`

- [ ] **Step 3: Write minimal implementation**

Add these functions to `finagent/governance.py`, right before the existing `_enforce_memory_md_cap` function (do NOT delete the old function yet — it is replaced by the async version in Task 4):

```python
def _within_cap(content: str) -> bool:
    """Check if content is within line + byte caps."""
    if len(content.split("\n")) > MEMORY_MD_MAX_LINES:
        return False
    if len(content.encode("utf-8")) > MEMORY_MD_MAX_BYTES:
        return False
    return True


def _truncate_preserve_header(content: str) -> str:
    """Truncate to cap, preserving header structure (up to first blank line).

    No blank line found: entire content treated as body.
    Header alone exceeds cap: hard-cap header.
    Body: keep tail (most recent entries).
    """
    lines = content.split("\n")

    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "" and i > 0:
            header_end = i + 1
            break

    header = lines[:header_end]
    body = lines[header_end:]

    if len(header) > MEMORY_MD_MAX_LINES:
        return "\n".join(header[:MEMORY_MD_MAX_LINES])

    max_body = MEMORY_MD_MAX_LINES - len(header)
    if len(body) > max_body:
        body = body[-max_body:]

    result = "\n".join(header + body)
    while len(result.encode("utf-8")) > MEMORY_MD_MAX_BYTES and body:
        body.pop(0)
        result = "\n".join(header + body)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_governance.py -k "within_cap or truncate" -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/governance.py tests/test_governance.py
git commit -m "feat: add _within_cap and _truncate_preserve_header helpers"
```

---

### Task 3: Extract Path — Inject Existing Memory + Dedup + Stop Writing memory.md

**Files:**
- Modify: `finagent/governance.py`:
  - `EXTRACT_PROMPT` constant: add `{existing_memory}` slot + "no new content" instruction
  - `extract_from_turn` function: inject existing memory, use `validate_and_dedup`, stop calling `_append_memory_md`
  - Delete `_append_memory_md` function entirely
- Test: `tests/test_governance.py`

**Interfaces:**
- Consumes: `validate_and_dedup` from Task 1
- Produces: modified `extract_from_turn` (same signature, different behavior: injects existing memory into prompt, re-reads sub-docs inside lock for write, writes sub-docs only, no memory.md)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_governance.py`:

```python
@pytest.mark.asyncio
async def test_extract_injects_existing_memory(tmp_path, monkeypatch):
    """Extract prompt includes existing memory content from sub-docs."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")
    (tmp_path / "project.md").write_text("- 已有项目记忆\n", encoding="utf-8")

    captured_prompt = []
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "preference": [], "project": [], "feedback": [], "reference": [],
    })

    async def capture_invoke(prompt):
        captured_prompt.append(prompt)
        return mock_response

    mock_llm = MagicMock()
    mock_llm.ainvoke = capture_invoke

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import extract_from_turn
        messages = [
            HumanMessage(content="请帮我分析一下这个公司的财务数据，我需要详细的财报点评"),
            AIMessage(content="好的，我来为您分析这家公司的财务状况和经营情况"),
        ]
        await extract_from_turn(messages)

    assert "已有项目记忆" in captured_prompt[0]


@pytest.mark.asyncio
async def test_extract_no_memory_md_write(tmp_path, monkeypatch):
    """Extraction never creates or writes memory.md."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")

    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "preference": ["新偏好"], "project": [], "feedback": [], "reference": [],
    })
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import extract_from_turn
        messages = [
            HumanMessage(content="以后报告用简洁格式，不要太长，关键数据突出即可"),
            AIMessage(content="好的，我会用简洁格式"),
        ]
        await extract_from_turn(messages)

    assert not (tmp_path / "memory.md").exists()


@pytest.mark.asyncio
async def test_extract_dedup_at_write(tmp_path, monkeypatch):
    """LLM returns existing item → not appended; new item → appended."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")
    (tmp_path / "preference.md").write_text("- 简洁格式\n", encoding="utf-8")

    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "preference": ["简洁格式", "新偏好"],
        "project": [], "feedback": [], "reference": [],
    })
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import extract_from_turn
        messages = [
            HumanMessage(content="以后报告用简洁格式，不要太长，关键数据突出即可"),
            AIMessage(content="好的，我会用简洁格式"),
        ]
        await extract_from_turn(messages)

    pref = (tmp_path / "preference.md").read_text(encoding="utf-8")
    assert pref.count("简洁格式") == 1
    assert "新偏好" in pref
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_governance.py -k "injects_existing or no_memory_md or dedup_at_write" -v`
Expected: FAIL (current code doesn't inject existing memory, does write memory.md)

- [ ] **Step 3: Update EXTRACT_PROMPT**

Replace the `EXTRACT_PROMPT` constant in `finagent/governance.py` with:

```python
EXTRACT_PROMPT = """分析以下对话轮次，提取适合长期记忆的内容。

以下是当前已有记忆（仅最近 50 行）：
{existing_memory}

只提取明确、持久的信息。不确定的不提取。
与已有记忆语义重复的、无关的，不需要输出。
没有值得记忆的内容就什么都不做，全部输出空列表。
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
```

- [ ] **Step 4: Update extract_from_turn**

Replace the entire `extract_from_turn` function in `finagent/governance.py` with:

```python
async def extract_from_turn(messages: list) -> None:
    """Per-turn extraction. LLM analyzes the latest turn, appends to detail files only.

    Injects tail 50 lines of each sub-doc into prompt for context.
    Uses validate_and_dedup to prevent duplicates and malformed writes.
    Does NOT write memory.md — that is governance-only.
    """
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

    existing_parts = []
    existing_contents = {}
    for category in ("preference", "project", "feedback", "reference"):
        path = MEMORY_DIR / f"{category}.md"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            existing_contents[category] = content
            tail_lines = content.strip().split("\n")[-50:]
            existing_parts.append(f"### {category}\n" + "\n".join(tail_lines))
        else:
            existing_contents[category] = ""
    existing_memory = "\n\n".join(existing_parts) if existing_parts else "(空)"

    prompt = EXTRACT_PROMPT.replace("{existing_memory}", existing_memory)
    prompt = prompt.replace("{messages}", turn_text)
    llm = get_llm()

    response = await llm.ainvoke(prompt)
    content = response.content
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        return
    try:
        findings = json.loads(content[start:end])
    except json.JSONDecodeError:
        return

    async with _memory_lock:
        for category in ("preference", "project", "feedback", "reference"):
            items = findings.get(category, [])
            if not items:
                continue
            detail_path = MEMORY_DIR / f"{category}.md"
            # Re-read inside lock to avoid TOCTOU: stale snapshot from
            # pre-lock read could lose concurrent writes from other extracts
            existing = detail_path.read_text(encoding="utf-8") if detail_path.exists() else ""
            new_items = validate_and_dedup(items, existing)
            if not new_items:
                continue
            new_section = "\n".join(f"- {item}" for item in new_items)
            atomic_write(detail_path, existing + new_section + "\n")
```

- [ ] **Step 5: Delete _append_memory_md**

Delete the entire `_append_memory_md` function from `finagent/governance.py`. This function is no longer called anywhere after Step 4.

- [ ] **Step 6: Modify existing test_extract_from_turn_with_findings**

In `tests/test_governance.py`, find `test_extract_from_turn_with_findings`. Replace the last two lines:

```python
    # OLD:
    mem = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert "preference" in mem
```

With:

```python
    # NEW:
    assert not (tmp_path / "memory.md").exists()
```

- [ ] **Step 7: Run all extract tests**

Run: `.venv/bin/pytest tests/test_governance.py -k extract -v`
Expected: PASS (all extract tests including modified + new)

- [ ] **Step 8: Run full test suite**

Run: `.venv/bin/pytest tests/test_governance.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add finagent/governance.py tests/test_governance.py
git commit -m "feat: inject existing memory into extract prompt, dedup at write, stop writing memory.md"
```

---

### Task 4: Async Cap Enforcement — _enforce_memory_md_cap with 3x Retry

**Files:**
- Modify: `finagent/governance.py`:
  - Add `COMPRESS_PROMPT` constant
  - Add `MAX_COMPRESS_RETRIES` constant
  - Add async `_enforce_memory_md_cap` function (replaces the gap left by Task 2 deletion of old sync version)
- Test: `tests/test_governance.py`

**Interfaces:**
- Consumes: `_within_cap`, `_truncate_preserve_header` from Task 2, `get_llm`
- Produces: `async _enforce_memory_md_cap(content: str) -> str`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_governance.py`:

```python
from finagent.governance import _enforce_memory_md_cap, MEMORY_MD_MAX_LINES


@pytest.mark.asyncio
async def test_enforce_cap_within(tmp_path, monkeypatch):
    """Content already within cap → returned unchanged, no LLM call."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        result = await _enforce_memory_md_cap("short content")

    assert result == "short content"
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_enforce_cap_llm_compress(tmp_path, monkeypatch):
    """Over-cap content → LLM compresses → returns compressed."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    over_content = "\n".join(f"line {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    compressed = "# Summary\n\n- item 1"

    mock_response = MagicMock()
    mock_response.content = compressed
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        result = await _enforce_memory_md_cap(over_content)

    assert result == compressed


@pytest.mark.asyncio
async def test_enforce_cap_retry_then_success(tmp_path, monkeypatch):
    """First 2 attempts fail (still over cap), 3rd succeeds."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    over_content = "# Title\n\n" + "\n".join(f"- item {i}" for i in range(MEMORY_MD_MAX_LINES + 50))

    call_count = 0

    async def flaky_invoke(prompt):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count < 3:
            resp.content = "\n".join(f"line {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
        else:
            resp.content = "# Title\n\n- compressed\n"
        return resp

    mock_llm = MagicMock()
    mock_llm.ainvoke = flaky_invoke

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        result = await _enforce_memory_md_cap(over_content)

    assert call_count == 3
    assert "compressed" in result


@pytest.mark.asyncio
async def test_enforce_cap_fallback(tmp_path, monkeypatch):
    """All 3 retries fail → truncate fallback."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    over_content = "# Title\n\n" + "\n".join(f"- item {i}" for i in range(MEMORY_MD_MAX_LINES + 50))

    mock_response = MagicMock()
    mock_response.content = "\n".join(f"line {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        result = await _enforce_memory_md_cap(over_content)

    # Fallback truncates the ORIGINAL content (preserves header), not the
    # last LLM output which had no header structure
    lines = result.split("\n")
    assert lines[0] == "# Title"
    assert len(lines) <= MEMORY_MD_MAX_LINES


@pytest.mark.asyncio
async def test_enforce_cap_exception_retry(tmp_path, monkeypatch):
    """LLM raises exception → retries, then falls back to truncate."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    over_content = "# Title\n\n" + "\n".join(f"- item {i}" for i in range(MEMORY_MD_MAX_LINES + 50))

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        result = await _enforce_memory_md_cap(over_content)

    # Fallback truncates original content (preserves header)
    lines = result.split("\n")
    assert lines[0] == "# Title"
    assert len(lines) <= MEMORY_MD_MAX_LINES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_governance.py -k enforce_cap -v`
Expected: FAIL with `ImportError: cannot import name '_enforce_memory_md_cap'`

- [ ] **Step 3: Write implementation**

Add to `finagent/governance.py`, after `_truncate_preserve_header` and before `GOVERNANCE_PROMPT`:

```python
COMPRESS_PROMPT = """压缩以下记忆摘要文件，控制在 {max_lines} 行 / {max_bytes} 字节以内。

保留索引结构和重要条目，删除低价值条目。
格式不变。

内容：
{content}
"""

MAX_COMPRESS_RETRIES = 3


async def _enforce_memory_md_cap(content: str) -> str:
    """Enforce memory.md size cap with LLM compress retry then fallback truncate.

    1. Within cap → return as-is.
    2. Over cap → LLM compress, up to 3 retries (including exceptions).
    3. All retries fail → _truncate_preserve_header on ORIGINAL content.
       Falls back to original (not last LLM output) to preserve header structure.
    """
    if _within_cap(content):
        return content

    original = content
    llm = get_llm()
    for _ in range(MAX_COMPRESS_RETRIES):
        try:
            prompt = COMPRESS_PROMPT.replace("{max_lines}", str(MEMORY_MD_MAX_LINES))
            prompt = prompt.replace("{max_bytes}", str(MEMORY_MD_MAX_BYTES))
            prompt = prompt.replace("{content}", content)
            response = await llm.ainvoke(prompt)
            compressed = response.content.strip()
            if _within_cap(compressed):
                return compressed
            content = compressed
        except Exception:
            continue

    return _truncate_preserve_header(original)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_governance.py -k enforce_cap -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add finagent/governance.py tests/test_governance.py
git commit -m "feat: async _enforce_memory_md_cap with 3x LLM compress retry + truncate fallback"
```

---

### Task 5: Governance Integration — Cap Outside Lock + GOVERNANCE_PROMPT Fix + End-to-End Test

**Files:**
- Modify: `finagent/governance.py`:
  - `run_governance` function: move `_enforce_memory_md_cap` call outside `_memory_lock`, await it
  - `GOVERNANCE_PROMPT` constant: fix item 6
  - Delete old sync `_enforce_memory_md_cap` (replaced by async version from Task 4)
- Test: `tests/test_governance.py`

**Interfaces:**
- Consumes: `_enforce_memory_md_cap` from Task 4
- Produces: modified `run_governance` (cap execution outside lock)

- [ ] **Step 1: Write failing test**

Add to `tests/test_governance.py`:

```python
@pytest.mark.asyncio
async def test_run_governance_caps_memory_md(tmp_path, monkeypatch):
    """Governance produces over-cap memory.md → compress → write within cap."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()

    over_memory = "# Title\n\n" + "\n".join(f"- item {i}" for i in range(MEMORY_MD_MAX_LINES + 50))

    governance_output = (
        f"=== FILE: memory.md ===\n{over_memory}\n"
        "=== FILE: preference.md ===\n=== FILE: project.md ===\n"
        "=== FILE: feedback.md ===\n=== FILE: reference.md ===\n"
    )
    compress_response = MagicMock()
    compress_response.content = "# Title\n\n- compressed\n"

    call_count = 0

    async def llm_invoke(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            resp = MagicMock()
            resp.content = governance_output
            return resp
        return compress_response

    mock_llm = MagicMock()
    mock_llm.ainvoke = llm_invoke

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import run_governance
        await run_governance()

    mem = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert len(mem.strip().split("\n")) <= MEMORY_MD_MAX_LINES
    assert "compressed" in mem
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_governance.py -k caps_memory_md -v`
Expected: FAIL (current `run_governance` calls old sync `_enforce_memory_md_cap` inside lock, needs async + outside lock)

- [ ] **Step 3: Delete old sync _enforce_memory_md_cap**

Delete the old sync `_enforce_memory_md_cap` function from `finagent/governance.py`. It is fully replaced by the async version added in Task 4. Do NOT delete the `_within_cap` or `_truncate_preserve_header` helpers from Task 2.

- [ ] **Step 4: Fix GOVERNANCE_PROMPT item 6**

In `finagent/governance.py`, in the `GOVERNANCE_PROMPT` constant, change:

```
6. detail 文档无大小限制（不注入上下文，按需读取）
```

To:

```
6. detail 文档无大小限制（提取时各注入尾部 50 行，按需读取）
```

- [ ] **Step 5: Update run_governance — cap outside lock**

In `finagent/governance.py`, find the `run_governance` function. The current code calls `_enforce_memory_md_cap` inside `async with _memory_lock:`. Move it outside:

Replace the section starting from `# Guard: if LLM returned prose without markers` through the end of the function with:

```python
    # Guard: if LLM returned prose without markers, don't wipe memory
    if not files:
        return

    # Ensure all 5 files exist (missing sections default to empty)
    for name in KNOWN_FILES:
        if name not in files:
            files[name] = ""

    # Enforce memory.md cap BEFORE acquiring lock (async, may call LLM)
    files["memory.md"] = await _enforce_memory_md_cap(files["memory.md"])

    async with _memory_lock:
        # Stage all files in temp subdir, then rename
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        staging = MEMORY_DIR / ".staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(exist_ok=True)
        for name in KNOWN_FILES:
            (staging / name).write_text(files[name] + "\n", encoding="utf-8")
        # Per-file atomic swap
        for name in KNOWN_FILES:
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

- [ ] **Step 6: Run cap integration test**

Run: `.venv/bin/pytest tests/test_governance.py -k caps_memory_md -v`
Expected: PASS

- [ ] **Step 7: Run full governance test suite**

Run: `.venv/bin/pytest tests/test_governance.py -v`
Expected: PASS (all tests)

- [ ] **Step 8: Run entire project test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS (no regressions)

- [ ] **Step 9: Commit**

```bash
git add finagent/governance.py tests/test_governance.py
git commit -m "feat: governance cap outside lock, fix GOVERNANCE_PROMPT item 6"
```
