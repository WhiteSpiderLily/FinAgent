"""Deep Agent construction using Deep Agents framework."""
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent

from finagent.config import get_llm
from finagent.prompts import RESEARCH_SYSTEM_PROMPT
from finagent.tools import tools
from finagent.subagents import build_subagent_specs, FILESYSTEM_MIDDLEWARE

# shared in-memory checkpointer — one per process, cleared on exit
_checkpointer = MemorySaver()


def create_agent():
    """Build a Deep Agent with bull/bear subagents."""
    llm = get_llm()
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        subagents=build_subagent_specs(),
        middleware=[FILESYSTEM_MIDDLEWARE],
        checkpointer=_checkpointer,
    )
    return agent


def reset_checkpoint() -> None:
    """重置 checkpointer，使 /clear 后新对话从干净状态开始。"""
    global _checkpointer
    _checkpointer = MemorySaver()


# TODO(@2026-11): remove _strip_legacy_tool_messages once pre-migration
# sessions containing select_agent calls are aged out.
def _strip_legacy_tool_messages(messages: list, tool_name: str) -> list:
    """Remove tool_calls for tool_name and their ToolMessage responses.

    Pre-migration sessions may contain select_agent tool calls. DeepSeek API
    rejects tool_calls for tools not in the current tool set.

    For AIMessages with mixed tool_calls (some legacy, some valid), strips
    only the legacy entries — never leaves a dangling tool_call_id without
    a matching ToolMessage.
    """
    bad_ids = set()
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None) or []
        invalid_tcs = getattr(msg, "invalid_tool_calls", None) or []
        for tc in tcs + invalid_tcs:
            if tc.get("name") == tool_name:
                bad_ids.add(tc.get("id"))
    if not bad_ids:
        return messages

    result = []
    for msg in messages:
        tc_id = getattr(msg, "tool_call_id", None)
        if tc_id and tc_id in bad_ids:
            continue  # ToolMessage responding to legacy tool_call

        tcs = getattr(msg, "tool_calls", None) or []
        invalid_tcs = getattr(msg, "invalid_tool_calls", None) or []

        legacy_tcs = [tc for tc in tcs if tc.get("id") in bad_ids]
        legacy_invalid = [tc for tc in invalid_tcs if tc.get("id") in bad_ids]

        if not legacy_tcs and not legacy_invalid:
            result.append(msg)
            continue

        # Strip legacy entries from both lists
        kept_tcs = [tc for tc in tcs if tc.get("id") not in bad_ids]
        kept_invalid = [tc for tc in invalid_tcs if tc.get("id") not in bad_ids]

        if kept_tcs or kept_invalid or msg.content:
            kwargs = {"content": msg.content}
            if kept_tcs:
                kwargs["tool_calls"] = kept_tcs
            if kept_invalid:
                kwargs["invalid_tool_calls"] = kept_invalid
            new_msg = AIMessage(**kwargs)
            for attr in ("id", "usage_metadata"):
                val = getattr(msg, attr, None)
                if val is not None:
                    setattr(new_msg, attr, val)
            result.append(new_msg)
        # else: all calls were legacy + no content → drop entire message
    return result


def create_agent_with_history(thread_id: str, messages: list):
    """Build agent, pre-seed checkpointer with message history.

    Filters out select_agent tool calls/responses from pre-migration sessions.
    """
    cleaned = _strip_legacy_tool_messages(messages, "select_agent")
    agent = create_agent()
    agent.update_state(
        {"configurable": {"thread_id": thread_id}},
        {"messages": cleaned},
    )
    return agent
