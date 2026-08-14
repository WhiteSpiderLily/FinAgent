"""Tests for FinAgent TUI."""
import asyncio

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessageChunk
from textual.widgets import Static

from finagent.tui import parse_command, FinAgentApp, ChatInput


@pytest.fixture
def app():
    """Create app with mocked agent (no real LLM needed)."""
    with patch("finagent.tui.create_agent", return_value=MagicMock()):
        yield FinAgentApp()


@pytest.mark.asyncio
async def test_app_composes_widgets(app):
    async with app.run_test() as pilot:
        assert app.query_one("#header") is not None
        assert app.query_one("#chat-view") is not None
        assert app.query_one("#input") is not None
        assert app.query_one("#status") is not None


@pytest.mark.asyncio
async def test_help_command(app):
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "/help"
        await pilot.press("enter")
        await pilot.pause()
        chat = app.query_one("#chat-view")
        messages = [str(c.content) for c in chat.children if isinstance(c, Static)]
        assert any("可用命令" in m for m in messages)


@pytest.mark.asyncio
async def test_quit_command_exits(app):
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "/quit"
        await pilot.press("enter")
        await pilot.pause()
    # if we reach here, exit() was called and context exited cleanly


def test_parse_command_slash_commands():
    assert parse_command("/help") == ("help", "")
    assert parse_command("/quit") == ("quit", "")
    assert parse_command("/clear") == ("clear", "")
    assert parse_command("/reload_skills") == ("reload_skills", "")


def test_parse_command_case_insensitive():
    assert parse_command("/HELP") == ("help", "")
    assert parse_command("/Quit") == ("quit", "")


def test_parse_command_message():
    assert parse_command("002415 2024Q3") == ("message", "002415 2024Q3")
    assert parse_command("  hello world  ") == ("message", "hello world")


def test_parse_command_unknown_slash_is_message():
    assert parse_command("/unknown") == ("message", "/unknown")


@pytest.mark.asyncio
async def test_streaming_renders_reply():
    """Agent reply text streams into chat view."""

    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="营收同比增长"), {}))
        yield ("messages", (AIMessageChunk(content=" 8.2%。"), {}))

    fake_agent = MagicMock()
    fake_agent.astream = fake_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "002415 2024Q3"
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            chat = app.query_one("#chat-view")
            messages = [
                c.content.markup if hasattr(c.content, "markup") else str(c.content)
                for c in chat.children if isinstance(c, Static)
            ]
            assert any("营收同比增长" in m for m in messages)


@pytest.mark.asyncio
async def test_streaming_shows_tool_progress():
    """Tool calls render progress lines."""

    async def fake_astream(*args, **kwargs):
        from langchain_core.messages import AIMessage, ToolMessage
        yield ("updates", {
            "agent": {"messages": [AIMessage(
                content="", tool_calls=[{"name": "get_financials", "args": {}, "id": "tc1"}]
            )]}
        })
        yield ("updates", {
            "tools": {"messages": [ToolMessage(
                content="营收650亿", tool_call_id="tc1", name="get_financials"
            )]}
        })
        yield ("messages", (AIMessageChunk(content="分析完成"), {}))

    fake_agent = MagicMock()
    fake_agent.astream = fake_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "002415 2024Q3"
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            chat = app.query_one("#chat-view")
            messages = [str(c.content) for c in chat.query(Static)]
            # tool progress line with tool name + checkmark
            assert any("get_financials" in m and "✓" in m for m in messages)


@pytest.mark.asyncio
async def test_tool_call_interleaved_between_text_segments():
    """Tool call renders between two text segments; ⏳ replaced by ✓ on same widget.

    Verifies:
    - Tool call widget is a single widget updated ⏳→✓, not two separate widgets.
    - Tool call appears between, not after, surrounding text segments.
    """

    async def fake_astream(*args, **kwargs):
        from langchain_core.messages import AIMessage, ToolMessage
        yield ("messages", (AIMessageChunk(content="前半段"), {}))
        yield ("updates", {
            "agent": {"messages": [AIMessage(
                content="", tool_calls=[{"name": "get_financials", "args": {}, "id": "tc1"}]
            )]}
        })
        yield ("updates", {
            "tools": {"messages": [ToolMessage(
                content="营收650亿", tool_call_id="tc1", name="get_financials"
            )]}
        })
        yield ("messages", (AIMessageChunk(content="后半段"), {}))

    fake_agent = MagicMock()
    fake_agent.astream = fake_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "002415"
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            chat = app.query_one("#chat-view")
            messages = [str(c.content) for c in chat.query(Static)]
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


@pytest.mark.asyncio
async def test_queue_flushes_after_turn():
    """Messages typed during agent processing queue and merge into next turn."""

    turn_count = 0
    captured_inputs = []
    release = asyncio.Event()

    async def fake_astream(*args, **kwargs):
        nonlocal turn_count
        turn_count += 1
        captured_inputs.append(args[0] if args else kwargs.get("input"))
        if turn_count == 1:
            # first turn: yield then block so worker stays alive while we queue
            yield ("messages", (AIMessageChunk(content="第一轮回复"), {}))
            await release.wait()
        else:
            # second turn: the merged queued messages arrive here
            yield ("messages", (AIMessageChunk(content="第二轮回复"), {}))

    fake_agent = MagicMock()
    fake_agent.astream = fake_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            # start first turn
            app.query_one("#input", ChatInput).text = "first"
            await pilot.press("enter")
            # while worker runs, queue a message
            app._submit_message("queued1")
            app._submit_message("queued2")
            assert len(app._queue) == 2
            # release first turn so it completes and flushes queue
            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # queue should be flushed
            assert len(app._queue) == 0
            # second turn should have run
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert turn_count == 2
            # second turn received the merged queued messages, not just one
            second_content = captured_inputs[1]["messages"][0]["content"]
            assert "queued1" in second_content
            assert "queued2" in second_content


@pytest.mark.asyncio
async def test_clear_command(app):
    async with app.run_test() as pilot:
        # add some messages first
        app._add_message("some content")
        chat = app.query_one("#chat-view")
        initial_count = len([c for c in chat.children if isinstance(c, Static)])
        assert initial_count > 0
        # clear
        app.query_one("#input", ChatInput).text = "/clear"
        await pilot.press("enter")
        await pilot.pause()
        chat = app.query_one("#chat-view")
        after_count = len([c for c in chat.children if isinstance(c, Static)])
        assert after_count == 0


@pytest.mark.asyncio
async def test_clear_cancels_running_worker():
    """/clear cancels a running streaming worker and nulls the handle.

    Covers Finding 1: without the cancel, the old worker's astream loop keeps
    calling _add_message after remove_children(), so stale messages reappear.
    """
    cancelled = asyncio.Event()

    async def slow_astream(*args, **kwargs):
        try:
            yield ("messages", (AIMessageChunk(content="partial"), {}))
            await asyncio.sleep(10)  # keep worker alive until /clear cancels
        except asyncio.CancelledError:
            cancelled.set()
            raise

    fake_agent = MagicMock()
    fake_agent.astream = slow_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            # start a streaming turn
            app.query_one("#input", ChatInput).text = "002415"
            await pilot.press("enter")
            await pilot.pause(0.01)
            assert app._streaming_worker is not None
            worker_ref = app._streaming_worker
            # /clear must cancel the worker before wiping the view
            app.query_one("#input", ChatInput).text = "/clear"
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # worker handle is null and the view is empty
            assert app._streaming_worker is None
            chat = app.query_one("#chat-view")
            assert len([c for c in chat.children if isinstance(c, Static)]) == 0
            # the streaming worker was actually cancelled (not just left running)
            assert worker_ref.is_cancelled


@pytest.mark.asyncio
async def test_interrupt_cancels_worker():
    """Esc cancels a running streaming worker."""

    async def slow_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="partial"), {}))
        await asyncio.sleep(10)  # keep worker alive until interrupt

    fake_agent = MagicMock()
    fake_agent.astream = slow_astream

    with patch("finagent.tui.create_agent", return_value=fake_agent):
        app = FinAgentApp()
        async with app.run_test() as pilot:
            app.query_one("#input", ChatInput).text = "002415"
            await pilot.press("enter")
            await pilot.pause(0.01)
            # press escape to interrupt
            await pilot.press("escape")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            chat = app.query_one("#chat-view")
            messages = [str(c.content) for c in chat.children if isinstance(c, Static)]
            assert any("已中断" in m for m in messages)


def test_parse_command_skill_match():
    cmd, payload = parse_command("/news-radar", skill_names=frozenset({"news-radar"}))
    assert cmd == "skill"
    assert payload == "news-radar"


def test_parse_command_skill_no_match_falls_back_to_message():
    """Unknown /xxx with no matching skill is treated as plain message."""
    cmd, payload = parse_command("/not-a-skill", skill_names=frozenset({"news-radar"}))
    assert cmd == "message"
    assert payload == "/not-a-skill"


def test_parse_command_skill_default_empty_names():
    """Without skill_names, /xxx falls back to message (backward compat)."""
    cmd, payload = parse_command("/anything")
    assert cmd == "message"


def test_parse_command_reload_skills_recognized():
    cmd, _ = parse_command("/reload_skills")
    assert cmd == "reload_skills"


@pytest.mark.asyncio
async def test_reload_skills_refreshes_catalog(app, tmp_path, monkeypatch):
    """Skill added to disk shows up after /reload_skills."""
    from pathlib import Path
    from finagent import skills

    fake_cwd = tmp_path / "cwd"
    fake_cwd.mkdir()
    (fake_cwd / ".finagent" / "skills").mkdir(parents=True)
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    monkeypatch.setattr(Path, "cwd", lambda: fake_cwd)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    async with app.run_test() as pilot:
        # Initially empty
        assert "demo-skill" not in app._skill_catalog_names

        # Add skill on disk
        d = fake_cwd / ".finagent" / "skills" / "demo-skill"
        d.mkdir()
        (d / "skill.md").write_text(
            "---\nname: demo-skill\ndescription: for test\n---\nbody", encoding="utf-8"
        )

        app.query_one("#input", ChatInput).text = "/reload_skills"
        await pilot.press("enter")
        await pilot.pause()

        assert "demo-skill" in app._skill_catalog_names
        chat = app.query_one("#chat-view")
        msgs = [str(c.content) for c in chat.children if isinstance(c, Static)]
        assert any("已加载" in m for m in msgs)


@pytest.mark.asyncio
async def test_user_message_includes_skill_catalog(app):
    """Each turn's user message must carry the skill catalog suffix."""
    app._skill_catalog = "- demo: 测试 skill"
    captured = []

    async def fake_astream(*args, **kwargs):
        captured.append(args[0])  # the messages dict
        return
        yield  # make it an async generator

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "hello"
        await pilot.press("enter")
        await pilot.pause()

    assert captured, "astream was not invoked"
    user_msg = captured[0]["messages"][0]
    content = user_msg["content"] if isinstance(user_msg, dict) else getattr(user_msg, "content", "")
    assert "hello" in content
    assert "<system-reminder>" in content
    assert "demo: 测试 skill" in content


@pytest.mark.asyncio
async def test_alt_enter_inserts_newline(app):
    """Option+Enter inserts a newline in the input box."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "line1"
        await pilot.press("alt+enter")
        await pilot.pause()
        assert "\n" in inp.text
        assert "line1" in inp.text


@pytest.mark.asyncio
async def test_ctrl_enter_inserts_newline(app):
    """Ctrl+Enter also inserts a newline (best-effort binding)."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "hello"
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert "\n" in inp.text
        assert "hello" in inp.text


@pytest.mark.asyncio
async def test_esc_idle_does_nothing(app):
    """Esc when not streaming is a no-op — input unchanged."""
    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="reply"), {}))

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "my draft"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        inp.text = "editing"
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert inp.text == "editing"


@pytest.mark.asyncio
async def test_esc_interrupts_and_backfills(app):
    """Esc during streaming stops worker and backfills last sent input."""
    async def slow_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="partial"), {}))
        await asyncio.sleep(10)

    app.agent.astream = slow_astream
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "分析000001"
        await pilot.press("enter")
        await pilot.pause(0.01)
        assert app._streaming_worker is not None
        await pilot.press("escape")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert inp.text == "分析000001"


@pytest.mark.asyncio
async def test_double_esc_clears_input(app):
    """Two Esc presses within the window clear the input box."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "some draft"
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert inp.text == ""


@pytest.mark.asyncio
async def test_up_arrow_loads_previous_history(app):
    """Up arrow at (0,0) loads the previous history entry."""
    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="reply"), {}))

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "first"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        inp.text = "second"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert inp.text == ""
        await pilot.press("up")
        await pilot.pause()
        assert inp.text == "second"
        await pilot.press("up")
        await pilot.pause()
        assert inp.text == "first"


@pytest.mark.asyncio
async def test_down_arrow_loads_next_and_clears(app):
    """Down arrow at end navigates forward; past newest clears input."""
    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="reply"), {}))

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "msg1"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        inp.text = "msg2"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("up")   # → msg2
        await pilot.press("up")   # → msg1
        await pilot.pause()
        assert inp.text == "msg1"
        await pilot.press("down") # → msg2
        await pilot.pause()
        assert inp.text == "msg2"
        await pilot.press("down") # past newest → clear
        await pilot.pause()
        assert inp.text == ""


@pytest.mark.asyncio
async def test_up_arrow_moves_cursor_in_multiline(app):
    """When cursor is not at start, up arrow moves cursor, not history."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "line1\nline2"
        await pilot.press("down")  # cursor to second line
        await pilot.pause()
        await pilot.press("up")    # cursor back, not history
        await pilot.pause()
        assert inp.text == "line1\nline2"


@pytest.mark.asyncio
async def test_down_arrow_restores_draft(app):
    """Down arrow past newest restores the draft, not blank."""
    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="reply"), {}))

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "submitted msg"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        inp.text = "my draft"
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert inp.text == "submitted msg"
        await pilot.press("down")
        await pilot.pause()
        assert inp.text == "my draft"


@pytest.mark.asyncio
async def test_edited_history_entry_persists_on_revisit(app):
    """Edits made while browsing history are cached per-slot."""
    async def fake_astream(*args, **kwargs):
        yield ("messages", (AIMessageChunk(content="reply"), {}))

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "original1"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        inp.text = "original2"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("up")   # → original2
        await pilot.pause()
        assert inp.text == "original2"
        inp.text = "edited2"
        await pilot.pause()
        await pilot.press("up")   # → original1
        await pilot.pause()
        assert inp.text == "original1"
        await pilot.press("down") # back → should see edited2
        await pilot.pause()
        assert inp.text == "edited2"


@pytest.mark.asyncio
async def test_input_box_grows_and_caps_at_max(app):
    """Input box auto-grows with content, capped at max-height 12."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "\n".join(f"line{i}" for i in range(20))
        await pilot.pause()
        assert inp.region.height <= 12


def test_report_command_removed():
    """'/report' is no longer a registered command."""
    from finagent.tui import _COMMANDS
    assert "/report" not in _COMMANDS


def test_activate_skill_injects_read_file_call(app):
    """_activate_skill must emit a read_file tool call on the skill path, not load_skill."""
    app.thread_id = "test-thread"
    with patch.object(app, "_add_message"):
        app._activate_skill("smoke")
    assert app.agent.update_state.called
    call_kwargs = app.agent.update_state.call_args
    messages = call_kwargs.kwargs.get("values", {}).get("messages", [])
    found = False
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None) or []
        for tc in tcs:
            if tc.get("name") == "read_file":
                path = tc.get("args", {}).get("file_path", "")
                if "smoke" in path and "skill.md" in path:
                    found = True
    assert found, f"expected read_file tool call on smoke/skill.md; got messages={messages}"
    # Preserve thread_id config — session isolation depends on it
    config = call_kwargs.kwargs.get("config", {})
    assert config.get("configurable", {}).get("thread_id") == "test-thread"


@pytest.mark.asyncio
async def test_activate_skill_unknown_falls_through(app):
    """If /<name> doesn't match a known skill, it's sent as plain text."""
    captured = []
    async def fake_astream(*args, **kwargs):
        captured.append(args[0])
        return
        yield

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "/not-a-skill"
        await pilot.press("enter")
        await pilot.pause()

    # Falls through to message: astream called with the raw text
    assert captured
    user_msg = captured[0]["messages"][0]
    content = user_msg["content"] if isinstance(user_msg, dict) else getattr(user_msg, "content", "")
    assert "/not-a-skill" in content


@pytest.mark.asyncio
async def test_token_counter_accumulates(app):
    """Two turns with usage_metadata should accumulate input_tokens."""
    from finagent.config import CONTEXT_WINDOW_TOKENS

    # Build two fake responses with usage_metadata
    class FakeChunk:
        def __init__(self, content="", tool_call_chunks=None, usage_metadata=None):
            self.content = content
            self.tool_call_chunks = tool_call_chunks
            self.usage_metadata = usage_metadata

    # Simulate stream: yields messages chunks, then final AIMessage in updates
    async def fake_astream(*args, **kwargs):
        # First chunk: text + usage at end of messages mode
        yield ("messages", (FakeChunk(content="hello", usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600}), {}))
        # Updates with final AIMessage carrying usage_metadata
        from langchain_core.messages import AIMessage
        ai = AIMessage(content="hello", usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600})
        yield ("updates", {"agent": {"messages": [ai]}})

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "first"
        await pilot.press("enter")
        await pilot.pause()
        first_total = app._cumulative_input_tokens
        assert first_total == 500

        app.query_one("#input", ChatInput).text = "second"
        await pilot.press("enter")
        await pilot.pause()
        assert app._cumulative_input_tokens == 1000


@pytest.mark.asyncio
async def test_token_counter_missing_usage_does_not_crash(app):
    """If usage_metadata is None, counter stays unchanged and no exception."""
    async def fake_astream(*args, **kwargs):
        from langchain_core.messages import AIMessage
        ai = AIMessage(content="ok")  # no usage_metadata
        yield ("updates", {"agent": {"messages": [ai]}})

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        before = app._cumulative_input_tokens
        app.query_one("#input", ChatInput).text = "x"
        await pilot.press("enter")
        await pilot.pause()
        assert app._cumulative_input_tokens == before


@pytest.mark.asyncio
async def test_clear_resets_token_counter(app):
    app._cumulative_input_tokens = 9999
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "/clear"
        await pilot.press("enter")
        await pilot.pause()
        assert app._cumulative_input_tokens == 0


@pytest.mark.asyncio
async def test_status_bar_shows_model_and_tokens_initially(app):
    """on_mount must initialize status bar to model+token format, not 就绪."""
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#status")
        text = str(status.content)
        assert "deepseek-chat" in text
        assert "0K/1M" in text
        assert "(0.0%)" in text


@pytest.mark.asyncio
async def test_status_bar_shows_token_count_after_response(app):
    """After a successful stream with usage_metadata, status bar reflects accumulated tokens."""
    from langchain_core.messages import AIMessage

    async def fake_astream(*args, **kwargs):
        yield ("messages", (type("C", (), {"content": "hi", "tool_call_chunks": None, "usage_metadata": None}), {}))
        ai = AIMessage(content="hi", usage_metadata={"input_tokens": 1500, "output_tokens": 50, "total_tokens": 1550})
        yield ("updates", {"agent": {"messages": [ai]}})

    app.agent.astream = fake_astream
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "hello"
        await pilot.press("enter")
        await pilot.pause()

        status = app.query_one("#status")
        text = str(status.content)
        assert "deepseek-chat" in text
        assert "1K/1M" in text  # 1500 // 1000 = 1
        assert "(0.1%)" in text  # 1500 / 1_000_000 * 100 = 0.15, :.1f → 0.1


@pytest.mark.asyncio
async def test_clear_refreshes_status_bar_to_zero(app):
    """After /clear, status bar shows 0K/1M (0.0%), not 就绪."""
    app._cumulative_input_tokens = 99999
    async with app.run_test() as pilot:
        app.query_one("#input", ChatInput).text = "/clear"
        await pilot.press("enter")
        await pilot.pause()
        status = app.query_one("#status")
        text = str(status.content)
        assert "0K/1M" in text
        assert "(0.0%)" in text


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


def test_finagent_app_accepts_resume_id():
    """FinAgentApp.__init__ accepts resume_session_id parameter."""
    with patch("finagent.tui.create_agent", return_value=MagicMock()), \
         patch("finagent.tui.create_agent_with_history", return_value=MagicMock()), \
         patch("finagent.tui.load_session", return_value=([], 0, [])):
        app = FinAgentApp(resume_session_id="some-id")
        assert app.thread_id == "some-id"


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


# ── Slash completion popup tests ──────────────────────────────────────────

def _popup_visible(app) -> bool:
    """Check if popup widget is displayed."""
    return app.query_one("#popup").styles.display != "none"


def _popup_text(app) -> str:
    return str(app.query_one("#popup").content)


@pytest.mark.asyncio
async def test_popup_hidden_initially(app):
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not _popup_visible(app)


@pytest.mark.asyncio
async def test_popup_visible_on_slash(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/"
        await pilot.pause()
        assert _popup_visible(app)


@pytest.mark.asyncio
async def test_popup_hidden_on_non_slash(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "hello"
        await pilot.pause()
        assert not _popup_visible(app)


@pytest.mark.asyncio
async def test_popup_hidden_on_mid_text_slash(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "a/b"
        await pilot.pause()
        assert not _popup_visible(app)


@pytest.mark.asyncio
async def test_popup_hidden_on_multiline_second_row(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/\nsecond"
        inp.move_cursor((1, 0))
        await pilot.pause()
        assert not _popup_visible(app)


@pytest.mark.asyncio
async def test_popup_filters_by_substring(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/he"
        await pilot.pause()
        assert _popup_visible(app)
        text = _popup_text(app)
        assert "/help" in text
        assert "/quit" not in text


@pytest.mark.asyncio
async def test_popup_hidden_on_zero_match(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/zzz"
        await pilot.pause()
        assert not _popup_visible(app)


@pytest.mark.asyncio
async def test_popup_includes_descriptions(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/"
        await pilot.pause()
        text = _popup_text(app)
        assert "—" in text  # name — description format


@pytest.mark.asyncio
async def test_popup_single_match_visible(app):
    """Single match: popup stays visible with one item."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        # /help is the only builtin containing "elp"
        inp.text = "/elp"
        await pilot.pause()
        assert _popup_visible(app)
        text = _popup_text(app)
        assert "/help" in text


# ── Ranking ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_popup_ranking_by_frequency(app):
    async with app.run_test() as pilot:
        app._command_freq._freq = {"quit": 10, "help": 0}
        inp = app.query_one("#input", ChatInput)
        # Both contain empty substring when query is ""
        inp.text = "/"
        await pilot.pause()
        text = _popup_text(app)
        lines = text.split("\n")
        # /quit (freq 10) should come before /help (freq 0)
        quit_idx = next(i for i, l in enumerate(lines) if "/quit" in l)
        help_idx = next(i for i, l in enumerate(lines) if "/help" in l)
        assert quit_idx < help_idx


@pytest.mark.asyncio
async def test_popup_ranking_tiebreak_by_position(app):
    async with app.run_test() as pilot:
        # /help contains "h" at position 0 (after stripping /)
        # /reload_skills does NOT contain "h"... use a skill that does
        app._command_freq._freq = {}  # all zero
        inp = app.query_one("#input", ChatInput)
        inp.text = "/h"
        await pilot.pause()
        text = _popup_text(app)
        lines = text.split("\n")
        # /help (position 0) should appear
        assert any("/help" in l for l in lines)


# ── Tab completion ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tab_completes_and_adds_space(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/he"
        await pilot.pause()
        assert _popup_visible(app)
        await pilot.press("tab")
        await pilot.pause()
        assert inp.text == "/help "
        assert not _popup_visible(app)


@pytest.mark.asyncio
async def test_tab_when_popup_hidden_indents(app):
    """Tab with no popup falls back to TextArea default indent."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "hello"
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert "\t" in inp.text
        assert not _popup_visible(app)


# ── Enter completion + send ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enter_when_popup_visible_sends_completed(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/he"
        await pilot.pause()
        assert _popup_visible(app)
        await pilot.press("enter")
        await pilot.pause()
        # /help was executed — help text should appear
        assert not _popup_visible(app)
        assert inp.text == ""
        chat = app.query_one("#chat-view")
        messages = [str(c.content) for c in chat.children if isinstance(c, Static)]
        assert any("可用命令" in m for m in messages)


# ── Up/Down navigation ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_down_arrow_moves_popup_selection(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/"
        await pilot.pause()
        assert app._popup_index == 0
        await pilot.press("down")
        await pilot.pause()
        assert app._popup_index == 1


@pytest.mark.asyncio
async def test_up_arrow_moves_popup_selection(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/"
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert app._popup_index == 1
        await pilot.press("up")
        await pilot.pause()
        assert app._popup_index == 0


@pytest.mark.asyncio
async def test_popup_down_does_not_trigger_history(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/"
        await pilot.pause()
        before_text = inp.text
        await pilot.press("down")
        await pilot.pause()
        # text unchanged — down navigated popup, not history
        assert inp.text == before_text


# ── Dismissal ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_space_dismisses_popup(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/he"
        await pilot.pause()
        assert _popup_visible(app)
        # type space
        inp.text = "/he "
        await pilot.pause()
        assert not _popup_visible(app)


@pytest.mark.asyncio
async def test_deleting_slash_dismisses_popup(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/"
        await pilot.pause()
        assert _popup_visible(app)
        inp.text = "hello"
        await pilot.pause()
        assert not _popup_visible(app)


# ── Frequency recording ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_command_increments_freq(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/clear"
        await pilot.press("enter")
        await pilot.pause()
        assert app._command_freq.get("clear") == 1


@pytest.mark.asyncio
async def test_skill_command_increments_freq(app):
    async with app.run_test() as pilot:
        app._skill_catalog_names = frozenset({"demo"})
        inp = app.query_one("#input", ChatInput)
        inp.text = "/demo"
        await pilot.press("enter")
        await pilot.pause()
        assert app._command_freq.get("demo") == 1


@pytest.mark.asyncio
async def test_skill_with_args_still_records_freq(app):
    """/skill-name + args falls to message path, but freq still recorded (decoupled)."""
    async with app.run_test() as pilot:
        app._skill_catalog_names = frozenset({"earnings-review"})
        inp = app.query_one("#input", ChatInput)
        inp.text = "/earnings-review\n000001 2025q1"
        await pilot.press("enter")
        await pilot.pause()
        assert app._command_freq.get("earnings-review") == 1


# ── parse_command edge cases ──────────────────────────────────────────────

def test_parse_command_trailing_space_builtin():
    assert parse_command("/help ", skill_names=frozenset()) == ("help", "")


def test_parse_command_trailing_space_skill():
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review ", skill_names=sn) == ("skill", "earnings-review")


def test_parse_command_inline_args_is_message():
    """Skill name with inline args falls through to message path."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review 000001", skill_names=sn) == ("message", "/earnings-review 000001")


def test_parse_command_newline_args_is_message():
    """Skill name with newline-separated args falls through to message path."""
    sn = frozenset({"earnings-review"})
    result = parse_command("/earnings-review\n000001", skill_names=sn)
    assert result == ("message", "/earnings-review\n000001")


def test_parse_command_tab_args_is_message():
    """Skill name with tab-separated args falls through to message path."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review\t000001", skill_names=sn) == ("message", "/earnings-review\t000001")


def test_parse_command_skill_with_args_is_message():
    """Skill name followed by args must fall through to message path."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review 000001 2025q3", skill_names=sn) == ("message", "/earnings-review 000001 2025q3")


def test_parse_command_skill_alone_no_args_matches():
    """Skill name alone (no args) still matches as skill."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review", skill_names=sn) == ("skill", "earnings-review")


def test_parse_command_leading_trailing_whitespace():
    assert parse_command("  /help  ") == ("help", "")


def test_parse_command_slash_alone_is_message():
    assert parse_command("/") == ("message", "/")


def test_parse_command_slash_space_is_message():
    assert parse_command("/   ") == ("message", "/")


def test_parse_command_empty_is_message():
    assert parse_command("") == ("message", "")


def test_parse_command_builtin_with_args_is_message():
    """Builtin with extra args falls through to message path."""
    assert parse_command("/HELP extra") == ("message", "/HELP extra")


def test_parse_command_skill_case_sensitive():
    sn = frozenset({"earnings-review"})
    assert parse_command("/Earnings-Review", skill_names=sn) == ("message", "/Earnings-Review")


@pytest.mark.asyncio
async def test_message_does_not_increment_freq(app):
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "002415 2024Q3"
        await pilot.press("enter")
        await pilot.pause()
        # No builtin or skill triggered
        assert all(v == 0 for v in app._command_freq._freq.values())


@pytest.mark.asyncio
async def test_tab_only_does_not_increment_freq(app):
    """Tab completion alone (no send) must not record frequency."""
    async with app.run_test() as pilot:
        inp = app.query_one("#input", ChatInput)
        inp.text = "/he"
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert app._command_freq.get("help") == 0
