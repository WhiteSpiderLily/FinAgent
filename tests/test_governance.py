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
