"""FinAgent TUI — Textual terminal interface."""
import multiprocessing
import time
import uuid

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Static, TextArea
from textual.worker import get_current_worker

from langchain_core.messages import HumanMessage

from finagent.agent import create_agent, reset_checkpoint
from finagent.config import load_env
from finagent.report import generate_report, set_current_report

HELP_TEXT = """\
可用命令:
  /report   生成财报点评报告（基于当前对话历史）
  /clear    清空对话记忆（换公司分析时用）
  /help     显示此帮助
  /quit     退出
"""

_COMMANDS = {"/quit", "/help", "/clear", "/report"}


def parse_command(text: str) -> tuple[str, str]:
    """Parse user input into (command_name, payload).

    Returns ("message", text) for non-command input.
    For slash commands, returns (command_name_without_slash, "").
    """
    stripped = text.strip()
    lower = stripped.lower()
    if lower in _COMMANDS:
        return lower[1:], ""
    return "message", stripped


class ChatInput(TextArea):
    """TextArea-based input — Input widget has keyboard issues in Textual 8.x."""

    BINDINGS = [
        Binding("enter", "submit", "发送", priority=True),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit(self) -> None:
        text = self.text
        self.text = ""
        self.post_message(self.Submitted(text))


class FinAgentApp(App):
    CSS = """
    Screen { layout: vertical; }
    #header { height: 1; background: $boost; padding: 0 1; }
    #chat-view { height: 1fr; padding: 0 1; }
    #input { height: 3; }
    #status { height: 1; background: $boost; color: $text-muted; padding: 0 1; }
    .message-user { color: $primary; }
    .message-tool { color: $text-muted; }
    .message-error { color: $error; }
    .message-queued { color: $text-disabled; text-style: italic; }
    """

    BINDINGS = [Binding("escape", "interrupt", "中断")]

    def __init__(self):
        super().__init__()
        load_env()
        self.thread_id = str(uuid.uuid4())
        self.agent = create_agent()
        self._queue: list[tuple[str, Static]] = []
        self._streaming_worker = None

    def compose(self) -> ComposeResult:
        yield Static("FinAgent — A股财报点评助手", id="header")
        yield VerticalScroll(id="chat-view")
        yield ChatInput(id="input")
        yield Static("就绪", id="status")

    def on_mount(self) -> None:
        # chat-view must not steal keyboard focus from the input box
        self.query_one("#chat-view").can_focus = False
        self._add_message("输入股票代码 + 报告期开始分析。输入 /help 查看命令。")
        self.query_one("#input").focus()

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        cmd, payload = parse_command(text)
        if cmd == "quit":
            self.exit()
        elif cmd == "help":
            self._add_message(HELP_TEXT)
        elif cmd == "clear":
            self._do_clear()
        elif cmd == "report":
            self._do_report()
        else:
            self._submit_message(payload)

    def _add_message(self, content, classes: str = "") -> Static:
        """Mount a message widget into the chat view, scroll to bottom."""
        chat_view = self.query_one("#chat-view")
        widget = Static(content, classes=classes)
        chat_view.mount(widget)
        chat_view.scroll_end(animate=False)
        return widget

    def _set_status(self, text: str) -> None:
        self.query_one("#status").update(text)

    def _submit_message(self, text: str) -> None:
        """Send a message to the agent, or enqueue if agent is busy."""
        if self._streaming_worker is not None and self._streaming_worker.is_running:
            widget = self._add_message(f"[排队中] {text}", classes="message-queued")
            self._queue.append((text, widget))
        else:
            self._add_message(f"> {text}", classes="message-user")
            self._streaming_worker = self._start_stream(text)

    @work
    async def _start_stream(self, user_input: str) -> None:
        """Stream agent response: tool-call progress + token-by-token reply.

        Tool calls are interleaved with text: a tool call freezes the current
        reply widget, mounts a progress widget (⏳), and on completion updates
        the same widget to ✓ — not a separate line. Subsequent text starts a
        fresh reply widget, so calls render between text segments.
        """
        self._set_status("思考中...")
        reply_widget: Static | None = self._add_message("")
        buffer = ""
        tool_widgets: dict[str, Static] = {}
        tool_names: dict[str, str] = {}
        last_update = 0.0
        need_new_reply = False
        try:
            async for mode, data in self.agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": self.thread_id}},
                stream_mode=["messages", "updates"],
            ):
                if get_current_worker().is_cancelled:
                    break
                if mode == "messages":
                    chunk, _metadata = data
                    content = getattr(chunk, "content", "")
                    has_tool_calls = bool(getattr(chunk, "tool_call_chunks", None))
                    if content and not has_tool_calls:
                        if need_new_reply or reply_widget is None:
                            reply_widget = self._add_message("")
                            buffer = ""
                            need_new_reply = False
                        buffer += content
                        now = time.monotonic()
                        if now - last_update > 0.1:
                            reply_widget.update(buffer)
                            last_update = now
                elif mode == "updates":
                    for node, state in data.items():
                        for msg in state.get("messages", []):
                            tool_calls = getattr(msg, "tool_calls", None)
                            if tool_calls:
                                if reply_widget is not None and not buffer:
                                    reply_widget.remove()
                                    reply_widget = None
                                for tc in tool_calls:
                                    name = tc.get("name", "tool")
                                    tc_id = tc.get("id", "")
                                    tool_names[tc_id] = name
                                    w = self._add_message(
                                        f"🔧 {name} ⏳", classes="message-tool"
                                    )
                                    tool_widgets[tc_id] = w
                            elif hasattr(msg, "tool_call_id"):
                                w = tool_widgets.pop(msg.tool_call_id, None)
                                name = tool_names.pop(
                                    msg.tool_call_id, getattr(msg, "name", "tool")
                                )
                                if w is not None:
                                    w.update(f"🔧 {name} ✓")
                                need_new_reply = True
            if buffer and reply_widget is not None:
                reply_widget.update(buffer)
            elif reply_widget is not None:
                reply_widget.remove()
        except Exception as e:
            self._add_message(f"出错: {e}", classes="message-error")
        finally:
            current = get_current_worker()
            if self._streaming_worker is not current:
                # cancelled/superseded: don't clobber a newer worker's state
                return
            self._set_status("就绪")
            self._streaming_worker = None
            # remove queued message widgets and flush merged queue
            if self._queue:
                for _text, w in self._queue:
                    w.remove()
                merged = "\n".join(t for t, _ in self._queue)
                self._queue.clear()
                self._add_message(f"> {merged}", classes="message-user")
                self._streaming_worker = self._start_stream(merged)

    def action_interrupt(self) -> None:
        """Esc handler: cancel current streaming worker."""
        if self._streaming_worker is not None and self._streaming_worker.is_running:
            self._streaming_worker.cancel()
            self._add_message("[已中断]", classes="message-queued")

    def _do_clear(self) -> None:
        """Clear conversation memory and chat view."""
        # Cancel any running streaming worker first: otherwise the old worker
        # keeps calling _add_message after remove_children(), reappearing as
        # stale messages in the cleared view.
        if self._streaming_worker is not None and self._streaming_worker.is_running:
            self._streaming_worker.cancel()
        self._streaming_worker = None
        reset_checkpoint()
        self.thread_id = str(uuid.uuid4())
        self.agent = create_agent()
        # /clear 语义为切到新公司分析，丢弃上一 session 的报告路径
        set_current_report(None)
        self._queue.clear()
        self.query_one("#chat-view").remove_children()
        self._set_status("就绪")

    @work(thread=True)
    def _do_report(self) -> None:
        """Generate report from conversation history, show summary inline.

        Runs in a thread worker: generate_report() calls llm.invoke()
        synchronously (a 10-30s network call). @work(thread=True) keeps
        the event loop free; all UI mutations route through call_from_thread.
        """
        state = self.agent.get_state(
            config={"configurable": {"thread_id": self.thread_id}}
        )
        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            self.app.call_from_thread(
                self._add_message,
                "还没有对话内容，先聊几句再生成报告。",
                classes="message-error",
            )
            return
        self.app.call_from_thread(self._set_status, "正在生成报告...")
        try:
            filepath, content = generate_report(messages)
            # inject report path into agent checkpoint so subsequent edit
            # requests ("把风险提示改短") know which file to edit
            self.agent.update_state(
                config={"configurable": {"thread_id": self.thread_id}},
                values={"messages": [HumanMessage(
                    content=f"(系统通知) 报告已生成，路径: {filepath}。如需编辑报告，请直接告知具体修改。"
                )]},
            )
            # extract title (first markdown heading)
            title = "未命名报告"
            for line in content.split("\n"):
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break
            self.app.call_from_thread(
                self._add_message,
                f"报告已生成: {title}\n文件: {filepath}",
                classes="message-user",
            )
        except ValueError as e:
            self.app.call_from_thread(
                self._add_message, str(e), classes="message-error"
            )
        except Exception as e:
            self.app.call_from_thread(
                self._add_message, f"报告生成失败: {e}", classes="message-error"
            )
        finally:
            self.app.call_from_thread(self._set_status, "就绪")


def main():
    # Textual replaces file descriptors (stdin/stdout/stderr). Python's default
    # 'spawn' start method needs to pass these fds to child processes, which fails
    # with "bad value(s) in fds_to_keep". 'fork' inherits fds without spawning.
    multiprocessing.set_start_method("fork", force=True)
    app = FinAgentApp()
    app.run()
