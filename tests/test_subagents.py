"""Tests for subagent module — Deep Agents dict specs."""
from finagent.subagents import (
    build_subagent_specs,
    BULL_SYSTEM_PROMPT,
    BEAR_SYSTEM_PROMPT,
)


class TestSubagentSpecs:
    """build_subagent_specs: returns list of dicts matching Deep Agents schema."""

    def test_is_list_of_dicts(self):
        specs = build_subagent_specs()
        assert isinstance(specs, list)
        assert len(specs) == 2

    def test_bull_spec(self):
        specs = build_subagent_specs()
        bull = next(s for s in specs if s["name"] == "bull")
        assert bull["description"]
        assert bull["system_prompt"]
        assert isinstance(bull["tools"], list)
        assert len(bull["tools"]) > 0
        assert "middleware" in bull

    def test_bear_spec(self):
        specs = build_subagent_specs()
        bear = next(s for s in specs if s["name"] == "bear")
        assert bear["description"]
        assert bear["system_prompt"]
        assert isinstance(bear["tools"], list)
        assert "middleware" in bear

    def test_specs_have_filesystem_middleware(self):
        specs = build_subagent_specs()
        for spec in specs:
            mw = spec["middleware"]
            assert isinstance(mw, list)
            assert len(mw) > 0
            fs_tools = [t.name for t in mw[0].tools]
            assert fs_tools == ["read_file"]


class TestPrompts:
    def test_bull_no_context_inheritance_rule(self):
        """Bull prompt must NOT say 'use context data' — subagents have no context."""
        assert "上下文已有" not in BULL_SYSTEM_PROMPT
        assert "prior context" not in BULL_SYSTEM_PROMPT.lower()

    def test_bear_no_context_inheritance_rule(self):
        assert "上下文已有" not in BEAR_SYSTEM_PROMPT

    def test_bull_says_call_tools(self):
        assert "工具" in BULL_SYSTEM_PROMPT or "tool" in BULL_SYSTEM_PROMPT.lower()

    def test_bear_says_call_tools(self):
        assert "工具" in BEAR_SYSTEM_PROMPT or "tool" in BEAR_SYSTEM_PROMPT.lower()


def test_build_subagent_specs_structure():
    from finagent.subagents import build_subagent_specs
    specs = build_subagent_specs()
    assert len(specs) == 2
    for spec in specs:
        assert set(spec.keys()) >= {"name", "description", "system_prompt", "tools", "middleware"}
        assert spec["name"] in {"bull", "bear"}
        assert len(spec["tools"]) == 8
        # Custom middleware enforces SUBAGENT_PERMISSIONS via _permissions=
        from finagent.subagents import SUBAGENT_PERMISSIONS
        from deepagents.middleware.filesystem import _check_fs_permission
        assert _check_fs_permission(SUBAGENT_PERMISSIONS, "write", "/.finagent/reports/x.md") == "deny"

def test_deleted_symbols_gone():
    import pytest
    for name in ("GLOBAL_TOOL_BLACKLIST", "resolve_subagent_tools", "FILESYSTEM_MIDDLEWARE"):
        with pytest.raises((ImportError, AttributeError)):
            import finagent.subagents as s
            getattr(s, name)


class TestStripLegacyToolMessages:
    """_strip_legacy_tool_messages removes select_agent tool calls + responses."""

    def test_no_filter_when_clean(self):
        from finagent.agent import _strip_legacy_tool_messages
        from langchain_core.messages import HumanMessage, AIMessage
        msgs = [
            HumanMessage(content="分析600519"),
            AIMessage(content="好的"),
        ]
        result = _strip_legacy_tool_messages(msgs, ["select_agent"])
        assert len(result) == 2

    def test_strips_pure_select_agent_calls(self):
        """AIMessage with ONLY select_agent tool_calls → entire message dropped."""
        from finagent.agent import _strip_legacy_tool_messages
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        msgs = [
            HumanMessage(content="分析多空"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "select_agent", "args": {"name": "bull"},
                    "id": "tc1", "type": "tool_call",
                }],
            ),
            ToolMessage(content="bull result", tool_call_id="tc1", name="select_agent"),
            AIMessage(content="综合结论"),
        ]
        result = _strip_legacy_tool_messages(msgs, ["select_agent"])
        assert len(result) == 2
        assert result[0].content == "分析多空"
        assert result[1].content == "综合结论"

    def test_mixed_tool_calls_strips_select_agent_only(self):
        """AIMessage with get_financials + select_agent → keep AIMessage
        with only get_financials tool_call, drop select_agent ToolMessage.
        Must NOT leave dangling tool_call_id without ToolMessage."""
        from finagent.agent import _strip_legacy_tool_messages
        from langchain_core.messages import AIMessage
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_financials", "args": {"stock_code": "600519"},
                     "id": "tc_fin", "type": "tool_call"},
                    {"name": "select_agent", "args": {"name": "bull"},
                     "id": "tc_sel", "type": "tool_call"},
                ],
            ),
        ]
        result = _strip_legacy_tool_messages(msgs, ["select_agent"])
        assert len(result) == 1
        ai = result[0]
        assert isinstance(ai, AIMessage)
        # select_agent tool_call must be stripped from the kept AIMessage
        remaining_tcs = ai.tool_calls or []
        assert all(tc["name"] != "select_agent" for tc in remaining_tcs)
        assert len(remaining_tcs) == 1
        assert remaining_tcs[0]["name"] == "get_financials"

    def test_strips_invalid_tool_calls(self):
        """AIMessage with invalid_tool_calls for select_agent → dropped."""
        from finagent.agent import _strip_legacy_tool_messages
        from langchain_core.messages import AIMessage
        msgs = [
            AIMessage(
                content="",
                invalid_tool_calls=[{
                    "name": "select_agent",
                    "args": '{"name": "bull"}',
                    "id": "tc_inv",
                    "error": "Invalid tool name",
                    "type": "invalid_tool_call",
                }],
            ),
        ]
        result = _strip_legacy_tool_messages(msgs, ["select_agent"])
        assert len(result) == 0

    def test_keeps_content_when_all_tool_calls_legacy(self):
        """AIMessage with legacy tool_calls + non-empty content → kept as content-only."""
        from finagent.agent import _strip_legacy_tool_messages
        from langchain_core.messages import AIMessage
        msgs = [
            AIMessage(
                content="这是重要的分析文本",
                tool_calls=[{
                    "name": "select_agent", "args": {"name": "bull"},
                    "id": "tc1", "type": "tool_call",
                }],
            ),
        ]
        result = _strip_legacy_tool_messages(msgs, ["select_agent"])
        assert len(result) == 1
        assert result[0].content == "这是重要的分析文本"
        assert not (result[0].tool_calls or [])

    def test_strip_legacy_tool_messages_multiple_names(self):
        """Single scan must handle multiple legacy tool names."""
        from finagent.agent import _strip_legacy_tool_messages
        from langchain_core.messages import AIMessage, ToolMessage
        msgs = [
            AIMessage(content="hi", tool_calls=[{"name": "load_skill", "args": {}, "id": "t1"}]),
            ToolMessage(content="ok", tool_call_id="t1", name="load_skill"),
            AIMessage(content="hi2", tool_calls=[{"name": "select_agent", "args": {}, "id": "t2"}]),
            ToolMessage(content="ok2", tool_call_id="t2", name="select_agent"),
            AIMessage(content="kept"),
        ]
        cleaned = _strip_legacy_tool_messages(msgs, ["load_skill", "select_agent"])
        assert len(cleaned) == 3
        assert cleaned[0].content == "hi"
        assert cleaned[1].content == "hi2"
        assert cleaned[2].content == "kept"


class TestPromptIntegration:
    """Verify RESEARCH_SYSTEM_PROMPT uses task tool, not select_agent."""

    def test_prompt_uses_task_tool(self):
        from finagent.prompts import RESEARCH_SYSTEM_PROMPT
        assert "task" in RESEARCH_SYSTEM_PROMPT
        assert "subagent_type" in RESEARCH_SYSTEM_PROMPT
        assert "bull" in RESEARCH_SYSTEM_PROMPT
        assert "bear" in RESEARCH_SYSTEM_PROMPT

    def test_prompt_no_select_agent(self):
        from finagent.prompts import RESEARCH_SYSTEM_PROMPT
        assert "select_agent" not in RESEARCH_SYSTEM_PROMPT


class TestDeclarativePermissions:
    """Declarative FilesystemPermission rules + FilesystemBackend config."""

    def test_main_agent_permissions_rules(self):
        """Rules: allow read .finagent/** + write reports/**, deny everything else."""
        from finagent.subagents import MAIN_AGENT_PERMISSIONS
        from deepagents.middleware.filesystem import _check_fs_permission
        assert _check_fs_permission(MAIN_AGENT_PERMISSIONS, "read", "/etc/passwd") == "deny"
        assert _check_fs_permission(MAIN_AGENT_PERMISSIONS, "write", "/tmp/x") == "deny"
        assert _check_fs_permission(MAIN_AGENT_PERMISSIONS, "write",
                                    "/.finagent/reports/002415_2024Q3_点评.md") == "allow"
        assert _check_fs_permission(MAIN_AGENT_PERMISSIONS, "read", "/.finagent") == "allow"

    def test_subagent_permissions_readonly(self):
        from finagent.subagents import SUBAGENT_PERMISSIONS
        from deepagents.middleware.filesystem import _check_fs_permission
        assert _check_fs_permission(SUBAGENT_PERMISSIONS, "write",
                                    "/.finagent/reports/x.md") == "deny"
        assert _check_fs_permission(SUBAGENT_PERMISSIONS, "read", "/.finagent") == "allow"

    def test_backend_virtual_mode(self):
        from finagent.subagents import BACKEND
        from deepagents.backends import FilesystemBackend
        assert isinstance(BACKEND, FilesystemBackend)
        assert BACKEND.virtual_mode is True

    def test_permission_denial_returns_correct_toolmessage(self):
        """Integration: a denied read_file call must produce a ToolMessage with the
        verified format 'Error: permission denied for read on {path}'."""
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from langchain.tools import ToolRuntime
        from finagent.subagents import MAIN_AGENT_PERMISSIONS, BACKEND

        mw = FilesystemMiddleware(backend=BACKEND, _permissions=MAIN_AGENT_PERMISSIONS)
        read_tool = next(t for t in mw.tools if t.name == "read_file")
        # ponytail: read_file's denied path only touches runtime.tool_call_id;
        # call .func directly with a mock runtime — .invoke() needs framework injection.
        runtime = ToolRuntime(
            state=None, context=None, config={},
            stream_writer=lambda *a, **kw: None,
            tool_call_id="tc_test", store=None,
        )
        result = read_tool.func(file_path="/etc/passwd", runtime=runtime)
        text = result.content if hasattr(result, "content") else str(result)
        assert "permission denied" in text.lower()
        assert "/etc/passwd" in text or "etc/passwd" in text


def test_backend_no_double_finagent_prefix():
    """Regression: BACKEND root_dir must NOT be '.finagent' — that combined
    with agent paths like '.finagent/skills/...' caused on-disk double-prefix
    '.finagent/.finagent/skills/...'. root_dir='.' lets agent paths resolve
    correctly while permission rules enforce the .finagent/ boundary."""
    from finagent.subagents import BACKEND
    # Skill path as it appears in system prompt — must resolve to project's
    # actual .finagent/skills/earnings-review/skill.md, not a nested copy.
    r = BACKEND.read(".finagent/skills/earnings-review/skill.md")
    assert not r.error, f"expected skill readable at .finagent/skills/...; got {r.error}"
    assert "earnings-review" in r.file_data.get("content", "")



