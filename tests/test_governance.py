"""Tests for governance: extraction and check."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from finagent.governance import (
    check_governance_needed,
    validate_and_dedup,
    _enforce_memory_md_cap,
    MEMORY_MD_MAX_LINES,
)


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


def test_within_cap_under():
    from finagent.governance import _within_cap
    assert _within_cap("short content") is True


def test_within_cap_over_lines():
    from finagent.governance import _within_cap, MEMORY_MD_MAX_LINES
    content = "\n".join(f"line {i}" for i in range(MEMORY_MD_MAX_LINES + 1))
    assert _within_cap(content) is False


def test_within_cap_over_bytes():
    from finagent.governance import _within_cap, MEMORY_MD_MAX_BYTES
    content = "x" * (MEMORY_MD_MAX_BYTES + 1)
    assert _within_cap(content) is False


def test_truncate_preserve_header():
    from finagent.governance import _truncate_preserve_header, MEMORY_MD_MAX_LINES
    header = "# Title\n\n"
    body = "\n".join(f"- item {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    content = header + body
    result = _truncate_preserve_header(content)
    lines = result.split("\n")
    assert lines[0] == "# Title"
    assert lines[1] == ""
    assert len(lines) <= MEMORY_MD_MAX_LINES


def test_truncate_no_blank_line():
    from finagent.governance import _truncate_preserve_header, MEMORY_MD_MAX_LINES
    content = "\n".join(f"line {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    result = _truncate_preserve_header(content)
    assert len(result.split("\n")) <= MEMORY_MD_MAX_LINES


def test_truncate_header_over_cap():
    from finagent.governance import _truncate_preserve_header, MEMORY_MD_MAX_LINES
    header = "\n".join(f"# header line {i}" for i in range(MEMORY_MD_MAX_LINES + 50))
    content = header + "\n\n- body item"
    result = _truncate_preserve_header(content)
    assert len(result.split("\n")) <= MEMORY_MD_MAX_LINES


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


@pytest.mark.asyncio
async def test_extract_from_turn_with_findings(tmp_path, monkeypatch):
    """Extract writes to detail files when LLM returns findings."""
    monkeypatch.setattr("finagent.governance.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")

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
            HumanMessage(content="以后报告用简洁格式，不要太长，关键数据突出即可"),
            AIMessage(content="好的，我会用简洁格式，关键数据突出，不要太长"),
        ]
        await extract_from_turn(messages)

    pref = (tmp_path / "preference.md").read_text(encoding="utf-8")
    assert "简洁格式" in pref
    assert not (tmp_path / "memory.md").exists()


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
            AIMessage(content="好的，我会用简洁格式，关键数据突出，不要太长"),
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
            AIMessage(content="好的，我会用简洁格式，关键数据突出，不要太长"),
        ]
        await extract_from_turn(messages)

    pref = (tmp_path / "preference.md").read_text(encoding="utf-8")
    assert pref.count("简洁格式") == 1
    assert "新偏好" in pref


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
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path / "sessions")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()

    with patch("finagent.governance.get_llm", return_value=mock_llm):
        from finagent.governance import extract_from_turn
        messages = [HumanMessage(content="ok"), AIMessage(content="sure")]
        await extract_from_turn(messages)

    mock_llm.ainvoke.assert_not_called()


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
