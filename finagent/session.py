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


def session_path(session_id: str) -> Path:
    """Return path for a session JSONL file."""
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise ValueError(f"invalid session id: {session_id!r}")
    return SESSIONS_DIR / f"{session_id}.jsonl"


def write_session(session_id: str, messages: list, cumulative_tokens: int, turns: list[dict] | None = None) -> None:
    """Overwrite session file with full message snapshot + optional turns + meta line."""
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


def count_sessions() -> int:
    """Count .jsonl files in sessions dir."""
    if not SESSIONS_DIR.exists():
        return 0
    return len(list(SESSIONS_DIR.glob("*.jsonl")))
