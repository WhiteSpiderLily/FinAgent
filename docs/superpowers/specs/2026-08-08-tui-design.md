# FinAgent TUI 设计方案

**日期:** 2026-08-08
**状态:** 已确认

## 背景与目标

当前 FinAgent 使用 `rich.Console` 实现的 CLI 输入循环（`cli.py`）。痛点：

- `agent.invoke()` 阻塞，用户等待整个回合才看到结果，无流式输出
- 聊天历史随输出滚动消失，输入框不固定
- 无状态栏、工具调用进度等视觉结构

目标：用 Textual 实现 TUI，提供流式输出、持久布局、视觉结构。完全替换旧 CLI。

## 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 框架 | Textual | 基于 rich，async-first，内置聊天/输入/状态栏 widget |
| 布局 | 单栏聊天 | Header + ChatLog + Input + StatusBar |
| 流式深度 | 工具调用进度行 + 最终回复逐字流式 | 能看到 agent 在干什么，不吵 |
| 旧 CLI | 完全替换 | 一个入口，零维护重复 |
| `/report` | 写文件 + 聊天区摘要 | 保持聊天区干净 |
| 输入并发 | 永不禁用，排队合并 | 参考 Claude Code / Codex |
| 中断 | Esc 取消当前回合 | 防卡死 |
| 滚动 | 粘性底部自动滚 | 跟随流式输出 |
| 输入框 | 单行，Enter 发送，Shift+Enter 换行 | chat 直觉 |
| 标题栏 | 静态标题，不解析公司 | 非核心，本次不做 |

## 架构

```
python -m finagent → tui.py (Textual App)

┌─────────────────────────────────────────┐
│ Header    — FinAgent · A股财报点评       │
├─────────────────────────────────────────┤
│ ChatLog   — 滚动消息区，粘性底部自动滚   │
│            · 用户消息                    │
│            · 🔧 工具名 ✓ (进度行)        │
│            · agent 最终回复 (流式追加)    │
│            · 排队消息 (灰显 · 排队中)     │
├─────────────────────────────────────────┤
│ Input     — 单行 · Enter发送 · 永不禁用  │
├─────────────────────────────────────────┤
│ StatusBar — 思考中/就绪 · thread id      │
└─────────────────────────────────────────┘
```

**一个文件 `finagent/tui.py` 替换 `cli.py`。** `__main__.py` 改调 `tui.main()`。`agent.py`、`report.py`、`config.py`、`prompts.py`、`tools.py` 不变。

## 组件

### ChatLog

Textual `RichLog`（`auto_scroll=True`）。内容经 `rich.markdown.Markdown` 渲染后写入。消息类型：

- **用户消息**：`> <内容>`
- **工具调用进度**：一行 `🔧 <工具显示名> ✓`（运行中 spinner / 完成打勾），参数和返回值隐藏
- **agent 最终回复**：Markdown 渲染，token 逐字流式追加
- **排队消息**：灰显 + "排队中"标记，回合结束发送后转正常颜色
- **错误**：红字显示

### Input

Textual `Input`，单行。Enter 触发 `on_submit`。处理期间不锁定。

### StatusBar

Textual `Static`，状态切换：就绪 / 思考中... / thread id。

### Header

Textual `Static`，静态标题 "FinAgent — A股财报点评"。

## 数据流 — 消息队列模型

```
用户输入 (任意时刻)
    │
    ├─ agent 空闲 → 立即发送，启动 astream worker
    │
    └─ agent 运行中 → 入队，聊天区灰显 "排队中"
                        │
                        回合结束 → 队列合并为一条消息（换行分隔）→ 发送
                                   灰显消息转正常

Esc → 取消当前 astream worker → 保留已生成内容至检查点 → 输入可用
```

关键规则：

- 输入框永不禁用
- agent 运行中用户输入的消息入队列，聊天区灰显
- 当前回合结束后，队列中的消息合并为一条用户消息（换行分隔），作为下一轮 `astream` 输入
- Esc 中断当前 `astream` worker，已生成内容保留进检查点，中断后队列中的消息照常刷新

## 流式渲染

```python
async for event in agent.astream(
    input, config={"configurable": {"thread_id": thread_id}},
    stream_mode=["messages", "updates"],
):
    if event 是 tool_call:      # 来自 updates stream
        chatlog.write("🔧 akshare_xxx ✓")
    elif event 是 token:         # 来自 messages stream
        追加到当前回复块         # 流式追加
```

- `stream_mode="updates"` 捕获工具调用事件
- `stream_mode="messages"` 捕获 LLM token 级输出
- agent `astream` 在 Textual worker 中运行，UI 保持响应

## 命令处理

命令在 Input 层拦截，不进 agent：

| 命令 | 行为 |
|------|------|
| `/report` | `generate_report()` → 写文件 → 聊天区显示报告标题（首个 Markdown 标题）+ 文件路径，不显示完整报告内容 |
| `/clear` | `reset_checkpoint()` + 新 `thread_id` + 清空 ChatLog |
| `/help` | 聊天区显示帮助文本 |
| `/quit` | 退出 app |

## 错误处理

- Agent 异常（含中断取消）→ 聊天区红字显示，输入保持可用
- `/report` 异常 → 聊天区红字显示
- 不 crash app，错误后状态恢复为就绪

## 测试

| 层 | 方法 |
|----|------|
| 逻辑层 | 纯函数：队列合并、命令解析 → pytest 单测 |
| UI 层 | Textual `Pilot` 冒烟测试：启动 app、模拟输入、断言消息渲染。不 mock 整个 LLM agent |

## 新增依赖

`textual>=0.40`（加入 `pyproject.toml` `dependencies`）

## 涉及文件

| 文件 | 变更 |
|------|------|
| `finagent/tui.py` | 新建 — Textual App + 全部交互逻辑 |
| `finagent/cli.py` | 删除 |
| `finagent/__main__.py` | 改调 `tui.main()` |
| `pyproject.toml` | 加 `textual>=0.40` |
| `tests/test_tui.py` | 新建 — Pilot 冒烟测试 + 逻辑层单测 |
