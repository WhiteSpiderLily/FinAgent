# FinAgent 记忆模块设计

## 概述

为 FinAgent 添加三层记忆系统，使 agent 具备跨会话持久记忆能力。

| 层 | 机制 | 存储 | 写入者 |
|---|---|---|---|
| 短期记忆 | 会话序列化 + 恢复 | `.finagent/sessions/<id>.jsonl` | 系统自动（每轮 stream 结束） |
| 长期记忆 | 手动维护的上下文文件 | `~/.finagent/finagent.md` + `./.finagent/finagent.md` | 用户手动 |
| 自动记忆 | LLM 提取 + 治理 | `./.finagent/memory/*.md` | 轮次提取（增量写）+ 定期治理（全量重写） |

## 目录结构

```
.finagent/
├── finagent.md                    # 项目级长期记忆（手动维护）
├── sessions/
│   └── <uuid>.jsonl               # 会话记录（每条消息一行 JSON）
└── memory/
    ├── memory.md                  # 摘要 + 索引（≤200 行 / 25KB）
    ├── preference.md              # 用户偏好
    ├── project.md                 # 项目规则
    ├── feedback.md                # 用户反馈
    ├── reference.md               # 参考信息
    └── .last_governance           # JSON: {timestamp, processed_sessions}

~/.finagent/
└── finagent.md                    # 用户级长期记忆（手动维护）
```

## 模块划分（方案 B）

三个新文件，各管一个关注点：

| 文件 | 职责 | 预估行数 |
|---|---|---|
| `finagent/session.py` | JSONL 序列化/反序列化 + session 生命周期 | ~100 行 |
| `finagent/memory.py` | 文件加载 + mtime 校验 + 条件注入 | ~80 行 |
| `finagent/governance.py` | 轮次提取 + 定期治理 | ~150 行 |

现有文件改动：`tui.py`、`agent.py`、`__main__.py`。

---

## 1. `finagent/session.py`

### 职责

- JSONL 序列化/反序列化 LangGraph 消息
- 原子写入（temp + rename）
- `--resume` 支持的加载入口

### 接口

```python
"""Session persistence: JSONL serialization for LangGraph messages."""

from pathlib import Path
import json
import tempfile
from datetime import datetime

# ponytail: CWD-relative. If --resume run from different dir, session not found.
# Acceptable for TUI app always run from project root. Upgrade to project-root
# anchoring if multi-dir usage becomes real.
SESSIONS_DIR = Path(".finagent/sessions")


def session_path(session_id: str) -> Path:
    """Return .finagent/sessions/<id>.jsonl path."""
    return SESSIONS_DIR / f"{session_id}.jsonl"


def _atomic_write(path: Path, content: str) -> None:
    """Write to temp file then rename. Atomic on same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def serialize_message(msg) -> dict:
    """Convert LangGraph message to JSON dict.

    Fields: type, content, id, tool_calls, tool_call_id, name, usage_metadata, ts.

    `id` is required: update_state's add_messages reducer deduplicates by id.
    Omitting it causes duplicate messages on re-seed.
    """
    # type HumanMessage/AIMessage/ToolMessage → "human"/"ai"/"tool"
    ...


def deserialize_message(data: dict):
    """Reverse of serialize_message. JSON dict → LangGraph message."""
    ...


def write_session(session_id: str, messages: list, cumulative_tokens: int) -> None:
    """Overwrite .finagent/sessions/<id>.jsonl with full message list.

    Called in _start_stream finally block. Snapshots complete conversation.
    Appends a final meta line with cumulative_tokens.
    """
    ...


def load_session(session_id: str) -> tuple[list, int]:
    """Read JSONL, return (deserialized messages, cumulative_tokens).

    Skips malformed lines (corruption tolerance).
    Returns ([], 0) if file doesn't exist.
    """
    ...


def count_sessions() -> int:
    """Count .jsonl files in sessions dir. Used by governance trigger."""
    ...
```

### JSONL 行格式

每行一个 JSON 对象：

```jsonl
{"type": "human", "content": "分析000001 2024Q3", "ts": "2026-08-10T22:00:00"}
{"type": "ai", "content": "营收同比...", "tool_calls": [{"name": "get_financials", "args": {...}, "id": "call_xxx"}], "usage_metadata": {"input_tokens": 1234}, "ts": "..."}
{"type": "tool", "content": "...", "tool_call_id": "call_xxx", "name": "get_financials", "ts": "..."}
{"type": "meta", "cumulative_tokens": 5678}
```

末尾 meta 行存储累计 token 计数，用于 resume 时恢复 `_cumulative_input_tokens`。

### 序列化细节

LangGraph 消息类型映射：

| LangGraph 类 | JSON `"type"` | 特殊字段 |
|---|---|---|
| `HumanMessage` | `"human"` | `content`, `id` |
| `AIMessage` | `"ai"` | `content`, `id`, `tool_calls`, `usage_metadata` |
| `ToolMessage` | `"tool"` | `content`, `id`, `tool_call_id`, `name` |

反序列化时根据 `"type"` 构造对应类的实例。

---

## 2. `finagent/memory.py`

### 职责

- 加载 memory 文件（长期记忆 + 自动记忆摘要）
- mtime 校验：只返回变更内容
- 提供注入用文本

### 接口

```python
"""Memory file loading with mtime-based conditional injection."""

from pathlib import Path

# 统一处理的文件列表（顺序决定注入顺序）
MEMORY_FILES = [
    ("用户级长期记忆", Path.home() / ".finagent" / "finagent.md"),
    ("项目级长期记忆", Path(".finagent") / "finagent.md"),
    ("自动记忆摘要", Path(".finagent") / "memory" / "memory.md"),
]


class MemoryLoader:
    """Tracks file mtimes, returns changed content for injection."""

    def __init__(self):
        self._last_mtimes: dict[Path, float | None] = {}
        for _label, path in MEMORY_FILES:
            self._last_mtimes[path] = None  # None = never loaded

    def get_injectable(self) -> str | None:
        """Return memory content if any file changed since last check.

        - First call: all existing files' content returned
        - Subsequent calls: only changed files' content returned
        - No changes: returns None
        - Non-existent files: skipped (no error)
        """
        ...

    def reset(self) -> None:
        """Clear mtime tracking. Called on /clear."""
        self._last_mtimes = {p: None for p in self._last_mtimes}
```

### 注入格式

当 `get_injectable()` 返回非 None 时，拼接为：

```
<system-reminder>
## 用户级长期记忆
{~/.finagent/finagent.md 内容}

## 项目级长期记忆
{./.finagent/finagent.md 内容}

## 自动记忆摘要
（以下为最新记忆，如有与上文冲突，以此为准）
{./.finagent/memory/memory.md 内容}
</system-reminder>
```

### 缓存行为

- 会话首条消息：memory 全量注入 → 进入缓存前缀
- 后续消息：mtime 不变 → `get_injectable()` 返回 None → 不注入 → 省 token
- `/clear` 后：`reset()` 重置 tracker → 首条消息重新注入
- 用户 mid-session 改 `.md` → mtime 变 → 下条消息重注入该文件内容
- 轮次提取写入 memory.md → mtime 变 → 下条消息自动重注入（**无需手动 reset**）
- mid-session 重注入时，旧内容仍在历史消息中。新 `<system-reminder>` 标注
  "（以下为最新记忆，如有与上文冲突，以此为准）"。旧内容在缓存前缀中不额外计费
- ponytail: mtime 粒度依赖文件系统。APFS 纳秒级，本地无问题。
  若部署到 ext4/网络挂载（秒级），同秒内快速编辑+发送可能漏检。

---

## 3. `finagent/governance.py`

### 职责

- **轮次提取**：每轮结束后，LLM 分析本轮对话，增量写入 detail 文档 + memory.md
- **定期治理**：读 memory/ 目录现有文件，LLM 全量重写（去重、消冲突、删过期）

两个入口通过模块级 `asyncio.Lock` 互斥执行。`@work(exclusive=True)` 会**抢占**（取消同组 worker），不保证顺序，因此不能用于写互斥。改用 `asyncio.Lock` 确保提取和治理不会同时写 memory/ 文件。

### 接口

```python
"""Memory governance: per-turn extraction + periodic maintenance."""

import asyncio

# 模块级锁，确保提取和治理不会同时写 memory/ 文件
_memory_lock = asyncio.Lock()

# 注意：prompt 模板中含字面 JSON 花括号，不能用 str.format()。
# 用 .replace("{messages}", text) / .replace("{current_memory}", text) 填充。
EXTRACT_PROMPT = """分析以下对话轮次，提取适合长期记忆的内容。

只提取明确、持久的信息。不确定的不提取。
分类写入：
- preference: 用户明确表达的偏好（格式、风格、工作方式）
- project: 项目规则、约束、技术决策
- feedback: 用户对 agent 行为的纠正/指导
- reference: 外部信息来源（URL、文档路径、工具用法）

输出 JSON，每类一个列表。无内容则空列表。
{"preference": [...], "project": [...], "feedback": [...], "reference": [...]}

对话：
{messages}
"""

GOVERNANCE_PROMPT = """你是记忆维护助手。以下是当前记忆文件内容。

任务：合并、去重、消解冲突、删除过期内容，输出整洁版本。

规则：
1. 重复内容合并为一条
2. 冲突内容保留最新，标注旧值已废弃
3. 明确过期的删除
4. 重新生成 memory.md 摘要 + 索引
5. memory.md 不超过 200 行 / 25KB。超限时优先压缩低价值条目
6. detail 文档无大小限制（不注入上下文，按需读取）
7. 用 === FILE: <name> === 分隔各文件输出

当前记忆：
{current_memory}
"""

MEMORY_MD_MAX_LINES = 200
MEMORY_MD_MAX_BYTES = 25_600  # 25KB


async def extract_from_turn(messages: list) -> None:
    """轮次提取。LLM 分析本轮对话 → 分类提取 → 追加写入 detail + memory.md。

    互斥：通过 _memory_lock 确保与 run_governance 不并发。

    流程：
    1. 只取最近一轮 user message + AI reply（不含完整 tool call 链），
       减少 API 成本。跳过短于阈值（<50 字符）的轮次。
    2. prompt 模板用 .replace("{messages}", text) 填充（不能用 .format()，
       因为模板含字面 JSON 花括号）
    3. 调 get_llm()，解析 JSON 输出
    4. 所有分类为空 → 直接返回，不写任何文件
    5. 每个非空分类 → 原子追加到 .finagent/memory/<cat>.md
    6. 更新 memory.md：追加摘要行。若行数 > MEMORY_MD_MAX_LINES 或
       字节数 > MEMORY_MD_MAX_BYTES，从头部裁剪最旧的摘要行
    """
    ...


async def run_governance() -> None:
    """定期维护。读 memory/ 目录现有文件，全量重写。

    互斥：通过 _memory_lock 确保与 extract_from_turn 不并发。

    不读 sessions。只整理 memory/ 下已有的增量写入。

    流程：
    1. 读 memory.md + 4 份 detail 文档现有内容
    2. prompt 模板用 .replace("{current_memory}", text) 填充
    3. 调 get_llm()，用 GOVERNANCE_PROMPT 分析
    4. 解析输出（=== FILE: <name> === 分隔）
    5. 将 5 个文件内容暂存到临时目录，全部就绪后原子 rename 覆盖
       （保证文件集合一致性：要么全更新，要么全不更新）
    6. 更新 .last_governance（timestamp + processed_session_count）
    """
    ...


def check_governance_needed() -> bool:
    """检查是否需要治理：now - last > 24h 且 new_sessions >= 5。

    session 计数 = count_sessions() - .last_governance.processed_sessions
    """
    ...
```

### .last_governance 格式

```json
{"timestamp": "2026-08-10T22:00:00", "processed_sessions": 12}
```

---

## 4. 现有文件改动

### `finagent/agent.py`

新增支持从历史消息恢复 checkpoint：

```python
def create_agent_with_history(thread_id: str, messages: list):
    """Build agent, pre-seed checkpointer with message history.

    Uses update_state to write messages into checkpoint without
    triggering graph execution.
    """
    agent = create_agent()
    agent.update_state(
        {"configurable": {"thread_id": thread_id}},
        {"messages": messages}
    )
    return agent
```

`reset_checkpoint()` 不变 — 仍用于 `/clear` 新建空 checkpoint。

### `finagent/__main__.py`

```python
import argparse
import multiprocessing

def main():
    # 必须在 app.run() 前：Textual 重写 stdin/stdout/stderr，
    # spawn 模式下会 "bad value(s) in fds_to_keep"
    multiprocessing.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复指定 session")
    args = parser.parse_args()

    app = FinAgentApp(resume_session_id=args.resume)
    app.run()

    # 退出后打印 resume 提示
    if app.thread_id:
        print(f"\nResume this session with:\n"
              f"python -m finagent --resume {app.thread_id}")
```

### `finagent/tui.py`

#### `__init__` 改动

```python
def __init__(self, resume_session_id: str | None = None):
    super().__init__()
    load_env()
    self._memory_loader = MemoryLoader()
    if resume_session_id:
        messages, tokens = load_session(resume_session_id)
        self.thread_id = resume_session_id
        self._cumulative_input_tokens = tokens
        self.agent = create_agent_with_history(self.thread_id, messages)
    else:
        self.thread_id = str(uuid.uuid4())
        self.agent = create_agent()
        self._cumulative_input_tokens = 0
    self._queue: list[tuple[str, Static]] = []
    self._streaming_worker = None
    metas = scan_skills()
    self._skill_catalog_names = frozenset(metas.keys())
    self._skill_catalog = render_catalog(metas)
```

#### `_build_user_message` 改动

```python
def _build_user_message(self, user_input: str) -> str:
    parts = [user_input]

    # memory 注入（mtime 条件）
    memory_content = self._memory_loader.get_injectable()
    if memory_content:
        parts.append(f"<system-reminder>\n{memory_content}\n</system-reminder>")

    # skill catalog（每轮注入）
    parts.append(
        f"<system-reminder>\n"
        f"可用 skills(用 load_skill 工具加载,或用户输入 /<name>):\n"
        f"{self._skill_catalog}\n"
        f"</system-reminder>"
    )

    return "\n\n".join(parts)
```

#### `_start_stream` finally 块改动

```python
finally:
    current = get_current_worker()
    if self._streaming_worker is not current:
        return
    self._refresh_status_bar()
    self._streaming_worker = None

    # 写 session JSONL（同步）
    state = await self.agent.aget_state(
        config={"configurable": {"thread_id": self.thread_id}}
    )
    messages = state.values.get("messages", [])
    write_session(self.thread_id, messages, self._cumulative_input_tokens)

    # 异步轮次提取（后台 worker）
    self._run_extraction(messages)

    # flush queue (现有逻辑不变)
    if self._queue:
        ...
```

#### `_do_clear` 改动

```python
def _do_clear(self) -> None:
    # ... 现有清理逻辑不变 ...
    self._memory_loader.reset()
    self._run_governance_check()
```

#### `on_mount` 改动

```python
def on_mount(self) -> None:
    # ... 现有逻辑不变 ...
    self._run_governance_check()
```

#### 新增异步 worker 方法

```python
@work
async def _run_extraction(self, messages: list) -> None:
    """轮次结束后异步提取记忆。静默执行，失败不吞。"""
    try:
        await extract_from_turn(messages)
    except Exception as e:
        # 不静默吞：在 status bar 短暂提示，不弹消息不打扰
        self._set_status(f"记忆提取失败: {e}")
        self._refresh_status_bar()

@work
async def _run_governance_check(self) -> None:
    """启动/clear 时检查，满足条件则异步执行治理。"""
    if not check_governance_needed():
        return
    self._set_status("记忆治理中...")
    try:
        await run_governance()
    except Exception as e:
        self._add_message(f"治理失败: {e}", classes="message-error")
    finally:
        self._refresh_status_bar()
```

注意：`_run_extraction` 和 `_run_governance_check` 不使用 Textual `@work(exclusive=True, group=...)`
做互斥 — Textual 的 exclusive 是**抢占式取消**同组 worker，不是排队等待。
互斥由 `governance.py` 内的模块级 `_memory_lock = asyncio.Lock()` 保证。
两个异步函数在各自入口 `async with _memory_lock:` 获取锁后才进行文件读写。

---

## 数据流总览

```
用户消息 → _build_user_message
  → MemoryLoader.get_injectable() (mtime 检查)
    → 有变更 → 拼接 <system-reminder> (memory 内容)
    → 无变更 → 跳过
  → 拼接 <system-reminder> (skill catalog, 每轮)
→ agent.astream → DeepSeek API
→ finally:
  → aget_state → 获取完整 messages
  → write_session() (JSONL 原子覆盖写)
  → _run_extraction() (异步, group=memory_writer)
    → extract_from_turn()
      → LLM 分析本轮 → 分类提取
      → 原子追加 detail 文档 + memory.md
      → 无内容则不写

启动 / /clear:
  → MemoryLoader.reset() (仅 /clear)
  → _run_governance_check() (异步, group=memory_writer)
    → check_governance_needed(): 24h + 5 sessions?
      → 是 → run_governance()
        → 读 memory/ 现有文件 → LLM 全量重写 → 原子覆盖
        → 更新 .last_governance
      → 否 → 返回

CLI --resume <id>:
  → load_session() → (messages, tokens)
  → create_agent_with_history(thread_id, messages)
  → 恢复 _cumulative_input_tokens
```

## 错误处理

| 场景 | 处理 |
|---|---|
| 轮次提取 LLM 失败 | status bar 短暂提示 `记忆提取失败: <error>`，下次自然重试 |
| 治理 LLM 失败 | TUI 显示 `治理失败: <error>`，下次满足条件重试 |
| JSONL 损坏行 | `load_session` 跳过该行，不整体失败 |
| memory/ 文件不存在 | `MemoryLoader` 跳过，不注入该文件 |
| `--resume` session 不存在 | `load_session` 返回空列表，正常启动新会话 |

## 约束

- memory.md ≤ 200 行 / 25KB（治理时保证，提取追加时也强制裁剪）
- detail 文档无大小限制
- session JSONL 无限保留
- 所有文件写入使用原子写（temp + rename）
- 治理覆盖写 5 个文件：全部暂存到临时目录，就绪后统一 rename（保证集合一致性）
- 提取和治理通过 `asyncio.Lock` 互斥（不用 Textual `@work(exclusive=True)`，那是抢占不是排队）
- 提取只传最近一轮 user + AI 文本（不含完整 tool call 链），跳过 <50 字符的轮次

## 不包含

- 不造 `read_memory` 工具 — agent 用现有 `read_file` 读 detail 文档
- 不热加载 system prompt — memory 通过 `<system-reminder>` 注入
- 治理不分析 sessions — 只整理 memory/ 已有内容
- 不预创建空模板文件 — 首次治理/提取后才生成
