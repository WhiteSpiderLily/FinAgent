"""Tests for subagent module — Deep Agents dict specs."""
from finagent.subagents import (
    GLOBAL_TOOL_BLACKLIST,
    build_subagent_specs,
    resolve_subagent_tools,
    BULL_SYSTEM_PROMPT,
    BEAR_SYSTEM_PROMPT,
    FILESYSTEM_MIDDLEWARE,
)


class TestResolveSubagentTools:
    """resolve_subagent_tools: all tools minus blacklist."""

    def test_excludes_report_tools(self):
        resolved = resolve_subagent_tools()
        names = {t.name for t in resolved}
        assert "generate_report_tool" not in names
        assert "update_section" not in names
        assert "delete_section" not in names

    def test_excludes_sandbox_reader(self):
        resolved = resolve_subagent_tools()
        names = {t.name for t in resolved}
        assert "read_sandbox_file" not in names

    def test_includes_financial_tools(self):
        resolved = resolve_subagent_tools()
        names = {t.name for t in resolved}
        assert "get_company_info" in names
        assert "get_financials" in names
        assert "get_valuation" in names
        assert "load_skill" in names
        assert "read_report" in names


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


class TestFilesystemMiddleware:
    def test_only_read_file(self):
        tools = [t.name for t in FILESYSTEM_MIDDLEWARE.tools]
        assert tools == ["read_file"]

    def test_no_execute(self):
        tools = [t.name for t in FILESYSTEM_MIDDLEWARE.tools]
        assert "execute" not in tools
        assert "write_file" not in tools
        assert "edit_file" not in tools
        assert "delete" not in tools




class TestStripLegacyToolMessages:
    """_strip_legacy_tool_messages removes select_agent tool calls + responses."""

    def test_no_filter_when_clean(self):
        from finagent.agent import _strip_legacy_tool_messages
        from langchain_core.messages import HumanMessage, AIMessage
        msgs = [
            HumanMessage(content="分析600519"),
            AIMessage(content="好的"),
        ]
        result = _strip_legacy_tool_messages(msgs, "select_agent")
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
        result = _strip_legacy_tool_messages(msgs, "select_agent")
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
        result = _strip_legacy_tool_messages(msgs, "select_agent")
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
        result = _strip_legacy_tool_messages(msgs, "select_agent")
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
        result = _strip_legacy_tool_messages(msgs, "select_agent")
        assert len(result) == 1
        assert result[0].content == "这是重要的分析文本"
        assert not (result[0].tool_calls or [])


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


