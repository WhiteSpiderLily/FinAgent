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
    assert parse_command("/report") == ("report", "")


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
            messages = [str(c.content) for c in chat.children if isinstance(c, Static)]
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
            messages = [str(c.content) for c in chat.children if isinstance(c, Static)]
            # 需求1: 单个工具 widget，⏳ 被替换为 ✓
            tool_msgs = [m for m in messages if "get_financials" in m]
            assert len(tool_msgs) == 1, f"expected 1 tool widget, got {tool_msgs}"
            assert "✓" in tool_msgs[0]
            assert "⏳" not in tool_msgs[0]
            # 需求2: 工具调用夹在两段文本之间
            idx_before = next(i for i, m in enumerate(messages) if "前半段" in m)
            idx_tool = next(i for i, m in enumerate(messages) if "get_financials" in m)
            idx_after = next(i for i, m in enumerate(messages) if "后半段" in m)
            assert idx_before < idx_tool < idx_after
            assert idx_before != idx_after


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
async def test_report_command_shows_summary():
    fake_content = "# 海康威视(002415) 2024Q3财报点评\n\n## 一、事件概述\n..."
    with patch("finagent.tui.generate_report", return_value=("/tmp/report.md", fake_content)):
        with patch("finagent.tui.create_agent", return_value=MagicMock()):
            app = FinAgentApp()
            async with app.run_test() as pilot:
                app.query_one("#input", ChatInput).text = "/report"
                await pilot.press("enter")
                # /report runs in a thread worker now; wait for it to drain
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()
                chat = app.query_one("#chat-view")
                messages = [str(c.content) for c in chat.children if isinstance(c, Static)]
                assert any("海康威视" in m for m in messages)
                assert any("/tmp/report.md" in m for m in messages)


@pytest.mark.asyncio
async def test_report_command_injects_path_message():
    """/report 成功后向 agent checkpoint 注入报告路径 message。"""
    fake_path = "/tmp/reports/002415_2024Q3_点评.md"
    fake_content = "# 海康威视 2024Q3财报点评\n\n## 一、事件概述\n内容。"
    fake_state = MagicMock()
    fake_state.values = {"messages": [MagicMock(content="分析 002415")]}
    fake_agent = MagicMock()
    fake_agent.get_state.return_value = fake_state

    with patch("finagent.tui.generate_report", return_value=(fake_path, fake_content)):
        with patch("finagent.tui.create_agent", return_value=fake_agent):
            app = FinAgentApp()
            async with app.run_test() as pilot:
                app.query_one("#input", ChatInput).text = "/report"
                await pilot.press("enter")
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

    fake_agent.update_state.assert_called_once()
    call_kwargs = fake_agent.update_state.call_args.kwargs
    injected_msgs = call_kwargs.get("values", {}).get("messages", [])
    assert len(injected_msgs) == 1
    assert fake_path in injected_msgs[0].content
    # config carries thread_id so the injection lands in the right checkpoint
    assert call_kwargs.get("config", {}).get("configurable", {}).get("thread_id") == app.thread_id


@pytest.mark.asyncio
async def test_report_runs_offloaded_to_worker():
    """/report offloads generate_report to a worker so the UI never blocks.

    Covers Finding 2: generate_report (sync llm.invoke) must not run on the
    UI thread. We block the report on a threading.Event and confirm the event
    loop is still serviced while the worker sleeps.
    """
    import threading

    release = threading.Event()

    def blocking_report(_msgs):
        # simulate a sync 10-30s llm.invoke call
        release.wait(timeout=2)
        return ("/tmp/r.md", "# 标题\n")

    fake_state = MagicMock()
    fake_state.values = {"messages": [MagicMock(type="user", content="hi")]}
    fake_agent = MagicMock()
    fake_agent.get_state.return_value = fake_state

    with patch("finagent.tui.generate_report", side_effect=blocking_report):
        with patch("finagent.tui.create_agent", return_value=fake_agent):
            app = FinAgentApp()
            async with app.run_test() as pilot:
                app.query_one("#input", ChatInput).text = "/report"
                await pilot.press("enter")
                # while the blocking report runs in a worker, the UI loop is
                # still alive: status should already be updated to "正在生成报告..."
                await pilot.pause(0.05)
                status = str(app.query_one("#status").render() if hasattr(app.query_one("#status"), 'render') else app.query_one("#status").content)
                assert "正在生成报告" in status
                # release the worker and let it finish
                release.set()
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()
                chat = app.query_one("#chat-view")
                messages = [str(c.content) for c in chat.children if isinstance(c, Static)]
                assert any("标题" in m for m in messages)
                assert "就绪" in str(app.query_one("#status").content)


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
