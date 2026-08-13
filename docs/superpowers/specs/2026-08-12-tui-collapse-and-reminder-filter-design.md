# TUI 会话恢复过滤 + ReAct 中间过程折叠

## 背景

FinAgent TUI（`finagent/tui.py`）基于 Textual 8.2.8。当前存在两个 UX 问题：

1. **恢复会话时显示 system-reminder**：`_build_user_message()` 将 `<system-reminder>` 块（记忆内容 + skill 目录）注入 `HumanMessage.content`。这些块序列化进 session JSONL 文件，恢复时 `on_mount()` 原样显示给用户。
2. **ReAct 中间过程无法折叠**：一个轮次中，工具调用（`🔧 name ⏳ → ✓`）和过渡文本全部平铺显示，不区分中间推理过程和最终回答。用户期望类似 Claude Code 的体验——中间过程实时可见，轮次结束后自动折叠为一行摘要。

## 设计决策

通过 12 轮 grilling 确定以下决策：

| # | 决策 |
|---|------|
| Q1 | 显示时剥 `<system-reminder>` 块 + 跳过空 content 的 AIMessage |
| Q2 | 折叠所有中间内容（工具调用 + 过渡文本），只留最终回答 |
| Q3 | 摘要行仅时长：`思考了 12s` |
| Q4 | 无工具调用 → 不折叠 |
| Q5 | 恢复的轮次也折叠，体验与实时一致 |
| Q6 | 改 session 序列化格式，存每轮起止时间 |
| Q7 | 展开后工具调用格式不变：`🔧 name ✓` |
| Q8 | 中断也折叠：`思考了 Xs [已中断]` |
| Q9 | 独立轮次元数据行（JSONL 中单独的 `type: "turn"` 行） |
| Q10 | 旧 session 不兼容 |
| Q11 | 旧 session → `load_session` 返回空 turns → 平铺降级显示 |
| Q12 | 无最终回答的轮次 → 整轮进折叠区，外面无可见内容 |

## 改动范围

仅两个文件：

| 文件 | 改什么 |
|------|--------|
| `session.py` | `write_session` 加 `turns` 参数；`load_session` 返回 `(messages, tokens, turns)` |
| `tui.py` | `_start_stream` 计时 + 折叠；`on_mount` 恢复分组 + 过滤；`__init__`/`_do_clear` 维护 `self._turns` |

不改：`agent.py`、`prompts.py`、`tools.py`、`governance.py`、`memory.py`、`_build_user_message()`（reminder 仍需注入给 LLM）。

## 详细设计

### 1. Session 格式扩展

#### 1.1 新增 turn 元数据行

JSONL 文件中，turn 元数据行集中在 meta 行之前：

```
{"type":"human","content":"...","ts":"..."}
{"type":"ai","content":"...","ts":"...","tool_calls":[...]}
{"type":"tool","content":"...","ts":"...","tool_call_id":"...","name":"..."}
{"type":"ai","content":"...","ts":"...","usage_metadata":{...}}
{"type":"turn","duration_s":12.3,"interrupted":false,"msg_start":0,"msg_end":4}
{"type":"meta","cumulative_tokens":12345}
```

字段说明：
- `msg_start` / `msg_end`：消息列表中的索引范围（左闭右开），用于恢复时分组
- `duration_s`：float，`_start_stream` 的 `time.monotonic()` 起止差值
- `interrupted`：bool，worker 被 cancel 则 true

#### 1.2 `write_session` 签名变更

```python
def write_session(
    session_id: str,
    messages: list,
    cumulative_tokens: int,
    turns: list[dict] | None = None,
) -> None:
```

写文件时，消息行之后、meta 行之前，写入所有 turn 行。

#### 1.3 `load_session` 返回值变更

```python
def load_session(session_id: str) -> tuple[list, int, list[dict]]:
    """返回 (messages, cumulative_tokens, turns)。"""
```

解析循环需三个分支（**不能**走 `deserialize_message`）：

```python
if data.get("type") == "meta":
    cumulative_tokens = data.get("cumulative_tokens", 0)
elif data.get("type") == "turn":
    turns.append(data)
else:
    messages.append(deserialize_message(data))
```

**关键**：`type == "turn"` 分支必须在 `deserialize_message` 调用之前拦截。否则 turn 行触发 `ValueError`（未知类型），被 `except (KeyError, ValueError)` 静默跳过，turns 永远为空。`deserialize_message` 不改——`turn` 是 session 元数据，非 LangGraph 消息类型。

旧文件无 turn 行 → 返回空列表。

#### 1.4 旧 session 处理

`load_session` 遇到无 turn 行的旧文件，返回空 turns 列表。`on_mount` 检测到空 turns 时平铺显示（Feature 1 的 reminder 过滤仍生效）。这是 3 行 null check，不是兼容 shim。

缺失文件早返回路径也需更新：`return [], 0` → `return [], 0, []`。

#### 1.5 受影响的调用点清单

`load_session` 返回值从 2 元组变 3 元组，以下调用点需改：

| 文件 | 行 | 当前 | 改为 |
|------|-----|------|------|
| `finagent/tui.py` | 89 | `messages, tokens = load_session(...)` | `messages, tokens, turns = load_session(...)` |
| `tests/test_session.py` | 98 | 2 元组解包 | 3 元组解包 |
| `tests/test_session.py` | 112 | 2 元组解包 | 3 元组解包 |
| `tests/test_session.py` | 127 | 2 元组解包 | 3 元组解包 |
| `tests/test_tui.py` | 668 | mock `return_value=([], 0)` | `return_value=([], 0, [])` |

`write_session` 的 `turns` 参数默认 `None`，现有测试调用点（`test_session.py:97, 137`）不传 turns 仍兼容。

#### 1.6 索引偏移风险（W3）

`load_session` 会跳过 JSON 解码失败和反序列化失败的行。如果某条消息行被跳过，`msg_start/msg_end` 索引会与实际加载的消息列表错位，导致 turn 分组显示错误。

**决策**：接受降级。格式错误行极少见，发生时 turn 分组可能错位但不会崩溃。不按消息 ID 分组——增加复杂度，收益不足以覆盖成本。

### 2. Feature 1：恢复时过滤 system-reminder

`on_mount()` 中对每条恢复的消息：

1. **HumanMessage**：正则剥 `<system-reminder>...</system-reminder>` 块，strip 后若为空则跳过
2. **AIMessage 空 content + 有 tool_calls**：跳过，不渲染空行
3. **其余**：正常显示

正则：`re.sub(r"<system-reminder>.*?</system-reminder>", "", content, flags=re.DOTALL)`

### 3. Feature 2：实时折叠

#### 3.1 计时

`_start_stream` 方法改动：

- 入口：`turn_start = time.monotonic()`
- 调用 `astream` 前：`msg_start = len(self.agent.get_state(...).values.get("messages", []))`
- finally 块：`duration_s = time.monotonic() - turn_start`
- finally 块：`msg_end = len(state.values.get("messages", []))`（已在现有代码中获取 state）
- finally 块：`interrupted = get_current_worker().is_cancelled`

#### 3.2 Widget 追踪

新增 `turn_widgets: list[Static]`，在 `_start_stream` 作用域内。每次 `_add_message` 创建 widget 时，同步 append 到 `turn_widgets`。

需覆盖的创建路径（全部经过 `_add_message`）：
- 行 243：初始 reply_widget
- 行 218-220：tool widget（`_handle_update_msg` 内，需传入 `turn_widgets` 或返回 widget 引用）
- 行 269：tool result 后的新 reply_widget

#### 3.3 流结束折叠逻辑

finally 块内，session 持久化之前执行：

```
本轮是否有工具调用？（检查 turn_widgets 中是否有 message-tool class 的 widget）
├── 否（Q4）→ 不折叠，widget 保持原样
└── 是
    ├── 最终回答 = turn_widgets 中最后一个有非空 content 的非 tool widget
    ├── 最终回答存在？
    │   ├── 是 →
    │   │   1. 其余 widget 从 DOM 移除（await widget.remove()）
    │   │   2. 创建 Collapsible(*new_statics, title=f"思考了 {duration_s:.0f}s", collapsed=True)
    │   │      new_statics 用新 Static 复制旧 widget 内容
    │   │   3. await chat_view.mount(collapsible, before=final_widget)
    │   └── 否（Q12）→
    │       全部 widget 移除，创建 Collapsible 包含全部内容，mount 到原位置
    └── interrupted=true → title 追加 " [已中断]"
```

#### 3.4 运行中行为

流式过程中所有 widget 正常显示（不折叠）。用户实时看到中间过程。折叠只在 `astream` 结束后的 finally 块中发生。

#### 3.5 Collapsible 内容

折叠区内每个 widget 用新 `Static` 复制原 widget 的渲染内容：
- 过渡文本：原样
- 工具调用：`🔧 name ✓`（Q7，格式不变）

### 4. Feature 2：恢复时折叠

`on_mount()` 恢复逻辑重构——从逐条显示改为按 turn 分组。

#### 4.1 分组逻辑

```
有 turns 数据？
├── 否（旧 session）→ 平铺显示（Feature 1 的过滤仍生效）
└── 是 → 按 msg_start:msg_end 切分消息列表
    每组：
    ├── 剥 reminder（Feature 1）
    ├── 跳过空 AIMessage
    ├── HumanMessage 显示为 "> 原始文本"（已剥 reminder）
    ├── 组内有工具调用？（组内任一 AIMessage 有 tool_calls）
    │   ├── 否 → 直接显示最终 AIMessage content
    │   └── 是 →
    │       ├── 最终回答 = 组内最后一条 content 非空且无 tool_calls 的 AIMessage
    │       ├── 中间内容（工具行 + 过渡文本）创建为 Static widgets 包进 Collapsible
    │       ├── title = f"思考了 {duration_s:.0f}s"
    │       ├── interrupted=true → title 追加 " [已中断]"
    │       ├── 有最终回答 → Collapsible mount 后 mount 最终回答
    │       └── 无最终回答（Q12）→ 只 mount Collapsible
```

#### 4.2 最终回答判定规则

实时轮次和恢复轮次统一规则：

> 最后一条 content 非空且无 tool_calls 的 AIMessage。

如果 AIMessage 同时有 content 和 tool_calls，说明模型还没回答完，content 是过渡文本。

### 5. `self._turns` 生命周期

| 时机 | 操作 |
|------|------|
| `__init__`（新会话） | `self._turns = []` |
| `__init__`（恢复会话） | `self._turns = load_session(...) 的 turns` |
| `_start_stream` finally | `self._turns.append({...})` |
| `_do_clear` | `self._turns = []` |
| `write_session` 调用 | 传入 `self._turns` |

### 6. 验证结果

已通过 Textual 8.2.8 API 验证：

- `Collapsible(*children, title=..., collapsed=True)` 构造正常
- `collapsed=True` 时 `Contents` 的 CSS `display: none` 立即生效，无闪烁
- `await widget.remove()` 正常移除 widget
- `await container.mount(collapsible, before=ref_widget)` 正确插入到目标位置前面
- DOM 顺序验证正确：`Static(用户消息) → Collapsible(中间过程) → Static(最终回答)`

## 错误处理

| 场景 | 处理 |
|------|------|
| Collapsible mount 失败 | fallback 平铺（best-effort） |
| turn 元数据行解析失败 | 跳过该行，不崩溃 |
| 正则剥 reminder 失败 | 原样显示 |
| `self.agent.get_state()` 失败 | turn 不记录 msg_start/msg_end，跳过持久化 |

## 测试范围

| 模块 | 测试内容 |
|------|----------|
| `session.py` | `write_session` + `load_session` round-trip with turns；旧文件返回空 turns |
| `tui.py` | 恢复分组逻辑可抽函数单独测；reminder 正则过滤测 |
| 不测 | Textual Collapsible 本身（第三方库） |

## 不做的事

- 不改 `_build_user_message()`——reminder 仍注入给 LLM
- 不改 agent/tools/governance/memory
- 不做向后兼容迁移——旧 session 返回空 turns 自动降级
- Collapsible 展开后不显示工具参数/结果详情（Q7，以后按需加）
- 恢复的轮次不显示真实时长（时长从 turn 元数据读取，非真实计时）
