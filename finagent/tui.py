"""FinAgent TUI — Textual terminal interface."""
import asyncio
import multiprocessing
import re
import time
import uuid
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Static, TextArea
from textual.worker import get_current_worker

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from finagent.agent import create_agent, create_agent_with_history, reset_checkpoint
from finagent.command_freq import CommandFreq
from finagent.config import load_env, MODEL_NAME, CONTEXT_WINDOW_TOKENS
from finagent.governance import extract_from_turn, run_governance, check_governance_needed
from finagent.input_history import InputHistory
from finagent.memory import MemoryLoader
from finagent.session import write_session, load_session, count_sessions
from finagent.skills import render_catalog, scan_skills

HELP_TEXT = """\
可用命令:
  /clear          清空对话记忆（换公司分析时用）
  /reload_skills  重新扫描 skill 目录，热更新可用列表
  /<skill-name>   激活指定 skill（列表见每轮 system-reminder）
  /help           显示此帮助
  /quit           退出
"""

_COMMANDS = {"/quit", "/help", "/clear", "/reload_skills"}

_BUILTIN_DESCRIPTIONS = {
    "quit": "退出",
    "help": "显示此帮助",
    "clear": "清空对话记忆（换公司分析时用）",
    "reload_skills": "重新扫描 skill 目录，热更新可用列表",
}

INPUT_HISTORY_PATH = Path.home() / ".finagent" / "input_history.json"
COMMAND_FREQ_PATH = Path.cwd() / ".finagent.json"

_DOUBLE_ESC_WINDOW = 0.3  # seconds


def parse_command(text: str, skill_names: frozenset[str] = frozenset()) -> tuple[str, str]:
    """Parse user input into (command_name, payload).

    Returns ("message", text) for non-command input.
    For slash commands, returns (command_name_without_slash, "").
    For /<skill-name> matching an active skill, returns ("skill", skill_name).

    Only matches when input is exactly /name (single token, no args).
    Any args after the name cause the entire input to fall through to message.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "message", stripped
    tokens = stripped[1:].split()
    if not tokens or len(tokens) > 1:
        return "message", stripped
    cmd = tokens[0].lower()
    if "/" + cmd in _COMMANDS:
        return cmd, ""
    if tokens[0] in skill_names:
        return "skill", tokens[0]
    return "message", stripped


_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def strip_system_reminders(content: str) -> str:
    """Remove <system-reminder> blocks from message content."""
    return _REMINDER_RE.sub("", content)


def _static_text(w: Static) -> str:
    """Extract text content from a Static widget for copying into Collapsible."""
    return w.content if isinstance(w.content, str) else str(w.content)


def _find_final_answer(messages: list):
    """Find last AIMessage with non-empty content and no tool_calls.

    Returns None if no such message exists.
    """
    for msg in reversed(messages):
        if (isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", None)):
            return msg
    return None


class ChatInput(TextArea):
    """TextArea-based input — Input widget has keyboard issues in Textual 8.x."""

    BINDINGS = [
        Binding("enter", "submit", "发送", priority=True),
        Binding("alt+enter", "newline", "换行", priority=True),
        Binding("ctrl+enter", "newline", "换行", priority=True),
        Binding("up", "history_up", "上一条", priority=True),
        Binding("down", "history_down", "下一条", priority=True),
        Binding("tab", "tab_complete", "补全", priority=True),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_newline(self) -> None:
        self.insert("\n")

    def action_history_up(self) -> None:
        app = self.app
        if not isinstance(app, FinAgentApp):
            return
        if app._popup_visible:
            app._popup_move(-1)
            return
        text = app.history_up(self.cursor_at_start_of_text, self.text)
        if text is not None:
            self.text = text
            self._cursor_to_end()
        else:
            self.action_cursor_up()

    def action_history_down(self) -> None:
        app = self.app
        if not isinstance(app, FinAgentApp):
            return
        if app._popup_visible:
            app._popup_move(1)
            return
        text = app.history_down(self.text)
        if text is not None:
            self.text = text
            if text:
                self._cursor_to_end()
        elif not self.cursor_at_end_of_text:
            self.action_cursor_down()

    def action_tab_complete(self) -> None:
        app = self.app
        if not isinstance(app, FinAgentApp):
            return
        if app._popup_visible:
            app._complete(self, send=False)
        else:
            self.insert("\t")

    def action_submit(self) -> None:
        app = self.app
        if isinstance(app, FinAgentApp) and app._popup_visible:
            app._complete(self, send=True)
            return
        text = self.text
        self.text = ""
        self.post_message(self.Submitted(text))

    def _cursor_to_end(self) -> None:
        lines = self.text.split("\n")
        self.move_cursor((len(lines) - 1, len(lines[-1])))


class FinAgentApp(App):
    CSS = """
    Screen { layout: vertical; }
    #header { height: 1; background: $boost; padding: 0 1; }
    #chat-view { height: 1fr; padding: 0 1; }
    #input { height: auto; min-height: 3; max-height: 12; }
    #popup { height: auto; max-height: 8; background: $boost; padding: 0 1; display: none; }
    #status { height: 1; background: $boost; color: $text-muted; padding: 0 1; }
    .message-user { color: $primary; }
    .message-tool { color: $text-muted; }
    .message-error { color: $error; }
    .message-queued { color: $text-disabled; text-style: italic; }
    """

    BINDINGS = [Binding("escape", "interrupt", "中断")]

    def __init__(self, resume_session_id: str | None = None):
        super().__init__()
        load_env()
        self._memory_loader = MemoryLoader()
        if resume_session_id:
            messages, tokens, turns = load_session(resume_session_id)
            self.thread_id = resume_session_id
            self._cumulative_input_tokens = tokens
            self.agent = create_agent_with_history(self.thread_id, messages)
            self._resume_messages = messages
            self._turns = turns
        else:
            self.thread_id = str(uuid.uuid4())
            self.agent = create_agent()
            self._cumulative_input_tokens = 0
            self._resume_messages = None
            self._turns = []
        self._queue: list[tuple[str, Static]] = []
        self._streaming_worker = None
        self._input_history = InputHistory(INPUT_HISTORY_PATH)
        self._command_freq = CommandFreq(COMMAND_FREQ_PATH)
        self._last_esc_time = 0.0
        self._hist_index = -1
        self._draft = ""
        self._edits: dict[int, str] = {}
        # Slash completion popup state
        self._popup_visible = False
        self._popup_items: list[tuple[str, str]] = []
        self._popup_index = 0
        # Skill catalog (refreshed by /reload_skills and at startup)
        self._skill_catalog_names: frozenset[str] = frozenset()
        self._skill_catalog: str = ""
        self._skill_descriptions: dict[str, str] = {}
        self._apply_skill_catalog(scan_skills())

    def compose(self) -> ComposeResult:
        yield Static("FinAgent — A股财报点评助手", id="header")
        yield VerticalScroll(id="chat-view")
        yield ChatInput(id="input")
        yield Static("", id="popup")
        yield Static("就绪", id="status")

    def on_mount(self) -> None:
        # chat-view must not steal keyboard focus from the input box
        self.query_one("#chat-view").can_focus = False
        # Render resumed conversation history into chat view
        if self._resume_messages:
            self._add_message("— 恢复历史会话 —", classes="message-queued")

            if self._turns:
                # Sanity check: indices out of bounds → flat fallback
                max_end = max((t.get("msg_end", 0) for t in self._turns), default=0)
                if max_end > len(self._resume_messages):
                    self._turns = []

            if self._turns:
                for turn in self._turns:
                    start = turn.get("msg_start", 0)
                    end = turn.get("msg_end", 0)
                    turn_msgs = self._resume_messages[start:end]

                    for msg in turn_msgs:
                        if isinstance(msg, HumanMessage) and msg.content:
                            cleaned = strip_system_reminders(msg.content).strip()
                            if cleaned:
                                self._add_message(f"> {cleaned}", classes="message-user")

                    has_tools = any(
                        getattr(m, "tool_calls", None)
                        for m in turn_msgs
                        if hasattr(m, "tool_calls")
                    )

                    if has_tools:
                        final = _find_final_answer(turn_msgs)
                        intermediate = []
                        for msg in turn_msgs:
                            if isinstance(msg, ToolMessage):
                                intermediate.append((f"🔧 {msg.name} ✓", True))
                            elif isinstance(msg, AIMessage) and msg.content and msg is not final:
                                if not getattr(msg, "tool_calls", None):
                                    intermediate.append((msg.content, False))

                        duration = turn.get("duration_s", 0)
                        title = f"思考了 {duration:.0f}s"
                        if turn.get("interrupted"):
                            title += " [已中断]"

                        collapsible = Collapsible(
                            *[Static(text, classes="message-tool" if is_tool else "")
                              for text, is_tool in intermediate],
                            title=title,
                            collapsed=True,
                        )
                        chat_view = self.query_one("#chat-view")
                        chat_view.mount(collapsible)
                        chat_view.scroll_end(animate=False)

                        if final:
                            self._add_message(final.content)
                    else:
                        final = _find_final_answer(turn_msgs)
                        if final:
                            self._add_message(final.content)
            else:
                for msg in self._resume_messages:
                    if isinstance(msg, HumanMessage) and msg.content:
                        cleaned = strip_system_reminders(msg.content).strip()
                        if cleaned:
                            self._add_message(f"> {cleaned}", classes="message-user")
                    elif isinstance(msg, AIMessage) and msg.content:
                        self._add_message(msg.content)
                    elif isinstance(msg, ToolMessage):
                        self._add_message(f"🔧 {msg.name} ✓", classes="message-tool")

            self._add_message("— 继续对话 —", classes="message-queued")
            self._resume_messages = None
        else:
            self._add_message("输入股票代码 + 报告期开始分析。输入 /help 查看命令。")
        self.query_one("#input").focus()
        self._refresh_status_bar()
        self._run_governance_check()

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self._input_history.append(text)
        self._input_history.save()
        self._hist_index = -1
        self._edits.clear()
        self._draft = ""
        cmd, payload = parse_command(text, skill_names=self._skill_catalog_names)
        if cmd == "quit":
            self.exit()
        elif cmd == "help":
            self._add_message(HELP_TEXT)
        elif cmd == "clear":
            self._do_clear()
        elif cmd == "reload_skills":
            self._do_reload_skills()
        elif cmd == "skill":
            self._activate_skill(payload)
        else:
            self._submit_message(payload)
        self._record_freq(text)

    def _record_freq(self, text: str) -> None:
        """Extract command/skill name from input for frequency tracking.

        Decoupled from routing — records even when args are present
        and the input falls through to the message path.
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return
        tokens = stripped[1:].split()
        if not tokens:
            return
        first = tokens[0].lower()
        if "/" + first in _COMMANDS:
            self._command_freq.increment(first)
        elif tokens[0] in self._skill_catalog_names:
            self._command_freq.increment(tokens[0])
        self._command_freq.save()

    def _add_message(self, content, classes: str = "") -> Static:
        """Mount a message widget into the chat view, scroll to bottom."""
        chat_view = self.query_one("#chat-view")
        widget = Static(content, classes=classes)
        chat_view.mount(widget)
        chat_view.scroll_end(animate=False)
        return widget

    def _set_status(self, text: str) -> None:
        self.query_one("#status").update(text)

    def _refresh_status_bar(self) -> None:
        """Show model name + cumulative token usage in the status bar."""
        pct = self._cumulative_input_tokens * 100 / CONTEXT_WINDOW_TOKENS
        k_tokens = self._cumulative_input_tokens // 1000
        window_m = CONTEXT_WINDOW_TOKENS // 1_000_000
        self.query_one("#status").update(
            f"{MODEL_NAME} | {k_tokens}K/{window_m}M ({pct:.1f}%)"
        )

    # ── Slash completion popup ────────────────────────────────────────────

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Refresh popup on every text change in the input box."""
        self._refresh_popup()

    def _apply_skill_catalog(self, metas: dict) -> None:
        """Update all skill-derived state from scan_skills() result."""
        self._skill_catalog_names = frozenset(metas.keys())
        self._skill_catalog = render_catalog(metas)
        self._skill_descriptions = {
            name: meta.description for name, meta in metas.items()
        }

    def _refresh_popup(self) -> None:
        """Recompute popup items based on current input text and cursor."""
        inp = self.query_one("#input", ChatInput)
        text = inp.text
        if not text.startswith("/") or not inp.cursor_at_first_line:
            self._popup_hide()
            return
        query = text[1:]
        if " " in query:
            self._popup_hide()
            return
        items = self._match_items(query)
        if not items:
            self._popup_hide()
            return
        self._popup_items = items
        self._popup_index = min(self._popup_index, len(items) - 1)
        self._popup_show()

    def _match_items(self, query: str) -> list[tuple[str, str]]:
        """Return (name, description) pairs matching query by substring,
        ranked by frequency desc, substring position asc, name asc."""
        candidates: list[tuple[str, str]] = []
        for name, desc in _BUILTIN_DESCRIPTIONS.items():
            if query in name:
                candidates.append((name, desc))
        for name, desc in self._skill_descriptions.items():
            if query in name:
                candidates.append((name, desc))
        if not candidates:
            return []
        candidates.sort(key=lambda item: (
            -self._command_freq.get(item[0]),
            item[0].find(query),
            item[0],
        ))
        return candidates

    def _popup_show(self) -> None:
        self._popup_visible = True
        popup = self.query_one("#popup")
        popup.update(self._render_popup())
        popup.styles.display = "block"

    def _popup_hide(self) -> None:
        self._popup_visible = False
        self._popup_items = []
        self._popup_index = 0
        self.query_one("#popup").styles.display = "none"

    def _render_popup(self) -> str:
        lines = []
        for i, (name, desc) in enumerate(self._popup_items):
            line = f"/{name} — {desc}"
            if i == self._popup_index:
                line = f"[reverse]{line}[/]"
            lines.append(line)
        return "\n".join(lines)

    def _popup_move(self, delta: int) -> None:
        if not self._popup_items:
            return
        n = len(self._popup_items)
        self._popup_index = (self._popup_index + delta) % n
        self.query_one("#popup").update(self._render_popup())

    def _complete(self, inp: ChatInput, send: bool) -> None:
        """Replace /token in input with selected item, optionally send."""
        if not self._popup_items:
            return
        name, _desc = self._popup_items[self._popup_index]
        text = inp.text
        end_col = next(
            (i for i, ch in enumerate(text) if ch.isspace()), len(text)
        )
        replacement = "/" + name + (" " if not send else "")
        inp.replace(replacement, (0, 0), (0, end_col))
        inp.move_cursor((0, len(replacement)))
        self._popup_hide()
        if send:
            submitted_text = inp.text
            inp.text = ""
            inp.post_message(ChatInput.Submitted(submitted_text))

    def _submit_message(self, text: str) -> None:
        """Send a message to the agent, or enqueue if agent is busy."""
        if self._streaming_worker is not None and self._streaming_worker.is_running:
            widget = self._add_message(f"[排队中] {text}", classes="message-queued")
            self._queue.append((text, widget))
        else:
            self._add_message(f"> {text}", classes="message-user")
            self._streaming_worker = self._start_stream(text)

    def _build_user_message(self, user_input: str) -> str:
        """Wrap user input with memory + skill catalog as system-reminder suffixes."""
        parts = [user_input]

        # Memory injection (conditional on mtime change)
        memory_content = self._memory_loader.get_injectable()
        if memory_content:
            parts.append(f"<system-reminder>\n{memory_content}\n</system-reminder>")

        # Skill catalog (every turn)
        parts.append(
            f"<system-reminder>\n"
            f"可用 skills(用 read_file 工具加载 .finagent/skills/<name>/skill.md,或用户输入 /<name>):\n"
            f"{self._skill_catalog}\n"
            f"</system-reminder>"
        )

        return "\n\n".join(parts)

    def _handle_update_msg(self, msg, reply_widget, buffer, tool_widgets, tool_names, turn_widgets):
        """Process one message from updates stream.

        Returns (need_new_reply, last_ai_msg, reply_widget, buffer).
        reply_widget may be modified (empty widget removed on tool_call);
        buffer is returned unchanged.
        """
        need_new_reply = False
        last_ai_msg = None

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            if reply_widget is not None and not buffer:
                reply_widget.remove()
                reply_widget = None
            for tc in tool_calls:
                name = tc.get("name", "tool")
                tc_id = tc.get("id", "")
                tool_names[tc_id] = name
                tool_widgets[tc_id] = self._add_message(
                    f"🔧 {name} ⏳", classes="message-tool"
                )
                turn_widgets.append(tool_widgets[tc_id])
        elif hasattr(msg, "tool_call_id"):
            w = tool_widgets.pop(msg.tool_call_id, None)
            name = tool_names.pop(msg.tool_call_id, getattr(msg, "name", "tool"))
            if w is not None:
                w.update(f"🔧 {name} ✓")
            need_new_reply = True

        if isinstance(msg, AIMessage) and getattr(msg, "usage_metadata", None):
            last_ai_msg = msg

        return need_new_reply, last_ai_msg, reply_widget, buffer

    @work
    async def _start_stream(self, user_input: str) -> None:
        """Stream agent response: tool-call progress + token-by-token reply.

        Tool calls are interleaved with text: a tool call freezes the current
        reply widget, mounts a progress widget (⏳), and on completion updates
        the same widget to ✓ — not a separate line. Subsequent text starts a
        fresh reply widget, so calls render between text segments.
        """
        turn_start = time.monotonic()
        turn_widgets: list[Static] = []

        # Capture msg_start before streaming
        try:
            pre_state = self.agent.get_state(
                config={"configurable": {"thread_id": self.thread_id}}
            )
            msg_start = len(pre_state.values.get("messages", []))
        except Exception:
            msg_start = 0

        self._set_status("思考中...")
        reply_widget: Static | None = self._add_message("")
        turn_widgets.append(reply_widget)
        buffer = ""
        tool_widgets: dict[str, Static] = {}
        tool_names: dict[str, str] = {}
        last_update = 0.0
        need_new_reply = False
        last_ai_msg = None
        try:
            async for mode, data in self.agent.astream(
                {"messages": [{"role": "user", "content": self._build_user_message(user_input)}]},
                config={"configurable": {"thread_id": self.thread_id}},
                stream_mode=["messages", "updates"],
            ):
                if get_current_worker().is_cancelled:
                    break
                if mode == "messages":
                    chunk, _metadata = data
                    # Deep Agents: task tool's ToolMessage carries subagent
                    # analysis text in messages stream. Skip non-AIMessageChunk
                    # to only render coordinator streaming tokens.
                    if not isinstance(chunk, AIMessageChunk):
                        continue
                    content = getattr(chunk, "content", "")
                    has_tool_calls = bool(getattr(chunk, "tool_call_chunks", None))
                    if content and not has_tool_calls:
                        if need_new_reply or reply_widget is None:
                            reply_widget = self._add_message("")
                            turn_widgets.append(reply_widget)
                            buffer = ""
                            need_new_reply = False
                        buffer += content
                        now = time.monotonic()
                        if now - last_update > 0.1:
                            reply_widget.update(buffer)
                            last_update = now
                elif mode == "updates":
                    for node, state in data.items():
                        if state is None:
                            continue
                        for msg in state.get("messages", []):
                            nr, lai, reply_widget, buffer = self._handle_update_msg(
                                msg, reply_widget, buffer, tool_widgets, tool_names, turn_widgets
                            )
                            need_new_reply = need_new_reply or nr
                            if lai:
                                last_ai_msg = lai
            if buffer and reply_widget is not None:
                reply_widget.update(buffer)
            elif reply_widget is not None:
                reply_widget.remove()
            if last_ai_msg is not None:
                usage = getattr(last_ai_msg, "usage_metadata", None) or {}
                input_tokens = usage.get("input_tokens")
                if isinstance(input_tokens, int):
                    self._cumulative_input_tokens += input_tokens
        except Exception as e:
            self._add_message(f"出错: {e}", classes="message-error")
        finally:
            current = get_current_worker()
            if self._streaming_worker is not current:
                # cancelled/superseded: don't clobber a newer worker's state
                return
            # Collapse intermediate widgets (best-effort)
            try:
                # Yield so pending mounts/removes settle before is_mounted check
                await asyncio.sleep(0)
                had_tools = any(
                    w.has_class("message-tool") for w in turn_widgets if w.is_mounted
                )
                if had_tools:
                    mounted = [w for w in turn_widgets if w.is_mounted]
                    reply_mounted = [
                        w for w in mounted if not w.has_class("message-tool")
                    ]
                    final_widget = reply_mounted[-1] if reply_mounted else None
                    final_has_content = (
                        reply_widget is not None and bool(buffer)
                    )

                    if final_has_content and final_widget is not None:
                        intermediate = [w for w in mounted if w is not final_widget]
                    else:
                        intermediate = mounted

                    contents = [_static_text(w) for w in intermediate]
                    for w in intermediate:
                        await w.remove()

                    duration = time.monotonic() - turn_start
                    title = f"思考了 {duration:.0f}s"
                    if current.is_cancelled:
                        title += " [已中断]"

                    collapsible = Collapsible(
                        *[Static(c) for c in contents],
                        title=title,
                        collapsed=True,
                    )

                    chat_view = self.query_one("#chat-view")
                    if final_has_content and final_widget is not None and final_widget.is_mounted:
                        await chat_view.mount(collapsible, before=final_widget)
                    else:
                        await chat_view.mount(collapsible)
            except Exception:
                pass  # collapse is best-effort; don't break the finally block
            self._refresh_status_bar()
            self._streaming_worker = None

            # Session persistence (best-effort)
            try:
                state = self.agent.get_state(
                    config={"configurable": {"thread_id": self.thread_id}}
                )
                msgs = state.values.get("messages", [])
                msg_end = len(msgs)
                self._turns.append({
                    "type": "turn",
                    "duration_s": time.monotonic() - turn_start,
                    "interrupted": current.is_cancelled,
                    "msg_start": msg_start,
                    "msg_end": msg_end,
                })
                write_session(self.thread_id, msgs, self._cumulative_input_tokens, self._turns)
                self._run_extraction(msgs)
            except Exception:
                pass  # persistence is best-effort; don't crash UI

            # remove queued message widgets and flush merged queue
            if self._queue:
                for _text, w in self._queue:
                    w.remove()
                merged = "\n".join(t for t, _ in self._queue)
                self._queue.clear()
                self._add_message(f"> {merged}", classes="message-user")
                self._streaming_worker = self._start_stream(merged)

    def action_interrupt(self) -> None:
        """Esc handler: double-tap clears input, single-tap only acts during streaming."""
        now = time.monotonic()
        inp = self.query_one("#input", ChatInput)
        if now - self._last_esc_time < _DOUBLE_ESC_WINDOW:
            inp.text = ""
            self._last_esc_time = 0.0
            return
        self._last_esc_time = now
        if self._streaming_worker is None or not self._streaming_worker.is_running:
            return
        self._streaming_worker.cancel()
        self._add_message("[已中断]", classes="message-queued")
        text = self.backfill_last()
        if text is not None:
            inp.text = text
            inp._cursor_to_end()

    def history_up(self, at_start: bool, current_text: str = "") -> str | None:
        """Navigate history backward. Returns text to load, or None for cursor movement."""
        if self._hist_index < 0:
            if not at_start:
                return None
            self._draft = current_text
            self._hist_index = len(self._input_history.items)
        else:
            self._edits[self._hist_index] = current_text
        if self._hist_index > 0:
            self._hist_index -= 1
        if self._input_history.items:
            return self._edits.get(self._hist_index, self._input_history.items[self._hist_index])
        return None

    def history_down(self, current_text: str = "") -> str | None:
        """Navigate history forward. Returns text, draft on exit, or None if not browsing."""
        if self._hist_index < 0:
            return None
        self._edits[self._hist_index] = current_text
        items = self._input_history.items
        if self._hist_index < len(items) - 1:
            self._hist_index += 1
            return self._edits.get(self._hist_index, items[self._hist_index])
        self._hist_index = -1
        return self._draft

    def backfill_last(self) -> str | None:
        """Load last sent input into the box and enter browsing mode."""
        if self._input_history.items:
            self._hist_index = len(self._input_history.items) - 1
            return self._input_history.items[-1]
        return None

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
        self._cumulative_input_tokens = 0
        self.agent = create_agent()
        self._turns = []
        self._queue.clear()
        self.query_one("#chat-view").remove_children()
        self._refresh_status_bar()
        self._memory_loader.reset()
        self._run_governance_check()

    def _do_reload_skills(self) -> None:
        """Rescan skill directories, refresh in-memory catalog."""
        metas = scan_skills()
        self._apply_skill_catalog(metas)
        self._add_message(f"已加载 {len(metas)} 个 skill")

    def _activate_skill(self, skill_name: str) -> None:
        """Inject a read_file tool call targeting the skill's skill.md.

        Agent processes the tool call, reads the file, and skill content enters
        history naturally — no synthetic ToolMessage needed.
        """
        skill_path = f".finagent/skills/{skill_name}/skill.md"
        msg = AIMessage(
            content=f"/{skill_name}",
            tool_calls=[{
                "name": "read_file",
                "args": {"file_path": skill_path},
                "id": f"activate-{skill_name}",
            }],
        )
        self.agent.update_state(
            values={"messages": [msg]},
            config={"configurable": {"thread_id": self.thread_id}},
        )
        self._add_message(f"✓ skill 已激活: {skill_name}", classes="message-tool")

    @work
    async def _run_extraction(self, messages: list) -> None:
        """Per-turn memory extraction. Async, non-blocking."""
        try:
            await extract_from_turn(messages)
        except Exception as e:
            self._set_status(f"记忆提取失败: {e}")
            self._refresh_status_bar()

    @work
    async def _run_governance_check(self) -> None:
        """Check if governance is needed, run if so."""
        if not check_governance_needed():
            return
        self._set_status("记忆治理中...")
        try:
            await run_governance()
        except Exception as e:
            self._add_message(f"治理失败: {e}", classes="message-error")
        finally:
            self._refresh_status_bar()

def main():
    # Textual replaces file descriptors (stdin/stdout/stderr). Python's default
    # 'spawn' start method needs to pass these fds to child processes, which fails
    # with "bad value(s) in fds_to_keep". 'fork' inherits fds without spawning.
    multiprocessing.set_start_method("fork", force=True)

    import argparse
    parser = argparse.ArgumentParser(description="FinAgent — A股财报点评助手")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复指定 session ID")
    args = parser.parse_args()

    app = FinAgentApp(resume_session_id=args.resume)
    app.run()

    # Print resume hint on exit
    if app.thread_id:
        print(f"\nResume this session with:\n"
              f"python -m finagent --resume {app.thread_id}")
