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


import pytest
from unittest.mock import patch

from finagent.session import write_session, load_session, count_sessions


def test_write_and_load_session(tmp_path, monkeypatch):
    """Round-trip: write messages, load them back."""
    monkeypatch.setattr("finagent.session.SESSIONS_DIR", tmp_path)
    messages = [
        HumanMessage(content="分析000001", id="m1"),
        AIMessage(content="营收同比+8%", usage_metadata={"input_tokens": 500, "output_tokens": 50, "total_tokens": 550}, id="m2"),
        ToolMessage(content="数据", tool_call_id="tc1", name="get_financials", id="m3"),
    ]
    write_session("test-session", messages, 500)
    loaded, tokens, _turns = load_session("test-session")
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
    messages, tokens, _turns = load_session("nonexistent")
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
    messages, tokens, _turns = load_session("bad")
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
