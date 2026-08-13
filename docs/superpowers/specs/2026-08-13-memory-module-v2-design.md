# Memory Module v2 Design

## 背景

当前记忆模块存在三个问题：

1. **重复记录过多** — `extract_from_turn` 盲写追加，不检查已有内容。LLM 每轮看到相同上下文，重复提取相同事实。`project.md` 中"报告结构固定六段+多空辩论"出现三次。
2. **memory.md 职责混乱** — 应为子文档要点摘要，实际退化为原始提取日志。32 行中约 25 行为重复或原始 dict 垃圾。
3. **格式异常写入** — LLM 返回 dict 而非 str，代码无类型校验，`str(dict)` 写入 markdown。

根因：提取路径缺乏去重与校验，且 memory.md 在提取阶段被独立追加，与 governance 重新生成职责冲突。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 提取时是否传已有记忆 | 全量注入（各子文档尾部 50 行） | LLM 需知道已有内容才能避免重复提取 |
| 校验范围 | 精确去重 + 格式校验 | 确定性可靠；语义去重交给 governance LLM |
| memory.md 写入权 | 提取不碰，governance 独占 | 单一写入源，根治漂移 |
| 子文档注入量 | 各取尾部 50 行 | 控制 token 成本 |
| 去重比较方式 | strip existing 行的 `- ` 前缀后精确匹配 | 对齐 LLM 输出格式 |
| cap 超限处理 | LLM 压缩重试 3 次 → 兜底截断 | 精准操作，只动 memory.md |

## 数据流

```
轮次结束
  │
  ▼
extract_from_turn
  │  1. 读取 4 个子文档各尾部 50 行 → 拼入 prompt
  │  2. LLM 提取（增量语义：已有记忆注入，仅输出新增）
  │  3. validate_and_dedup：类型检查 + 空值检查 + 精确去重
  │  4. 仅写入通过校验的新条目到 {category}.md
  │
  ╳  memory.md 不触碰

24h + 5 sessions
  │
  ▼
run_governance
  │  1. 读取全量子文档 → LLM 去重/消解冲突/重整
  │  2. LLM 重新生成 memory.md（从子文档派生摘要 + 索引）
  │  3. memory.md cap 执行（在 _memory_lock 外）：
  │     a. _within_cap 通过 → 直接用
  │     b. 超限 → LLM 压缩重试最多 3 次（含异常重试）
  │     c. 重试全部失败 → _truncate_preserve_header 兜底
  │  4. 获取 _memory_lock，stage + rename 全部 5 文件
```

## 组件设计

### validate_and_dedup

```python
def validate_and_dedup(items: list, existing: str) -> list[str]:
    """过滤无效条目 + 精确去重。

    items: LLM 返回的原始列表（可能含 dict/空串/重复/非 list）
    existing: 子文档当前全量内容
    返回: 通过校验且不重复的 str 列表
    """
```

校验规则（按顺序）：

0. **列表检查**：`isinstance(items, list)` → 非 list 返回空列表（防止字符串遍历为字符）
1. **类型检查**：`isinstance(item, str)` → 非 str 丢弃
2. **空值检查**：`item.strip()` 为空 → 丢弃
3. **精确去重**：
   - item 规范化 = `item.strip()`
   - existing 规范化 = 每行 `line.strip().removeprefix("- ").strip()`，构成 `set`
   - `removeprefix` 而非 `lstrip`——`lstrip("- ")` 是字符集操作，会过度去重 `-- foo` 等
   - `if normalized in existing_set: skip`

不做：语义去重、模糊匹配、标点归一化、子串匹配。

### EXTRACT_PROMPT 改造

新增 `{existing_memory}` 模板变量。prompt 明确指示：与已有记忆语义重复的不输出，没有值得记忆的内容全部输出空列表。

```python
EXTRACT_PROMPT = """分析以下对话轮次，提取适合长期记忆的内容。

以下是当前已有记忆（仅最近 50 行）：
{existing_memory}

只提取明确、持久的信息。不确定的不提取。
与已有记忆语义重复的、无关的，不需要输出。
没有值得记忆的内容就什么都不做，全部输出空列表。
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
```

`extract_from_turn` 中 existing_memory 构建：读取 4 个子文档，各取尾部 50 行，按 `### {category}` 分隔拼接。子文档内容读取一次，同时用于 prompt 注入和 dedup 比较，避免重复 IO。模板替换用 `.replace()`（模板含 JSON 花括号，不能用 `.format()`）。

注意：`.replace()` 模板替换有注入风险——用户消息含 `{existing_memory}` 或 `{messages}` 字面量时会被误替换。这是既有问题，本设计不引入新风险，但实现时先替换 `{existing_memory}` 再替换 `{messages}` 以缩小窗口。

### extract_from_turn 写入逻辑

- 加 `validate_and_dedup` 调用，校验 + 去重后才追加
- `new_items` 为空则跳过该 category
- 删除 `has_content` 标志和 `_append_memory_md` 调用——不再写 memory.md
- 删除 `_append_memory_md` 函数本身（唯一调用点在 118 行，删除安全）
- `_memory_lock` 并发安全不变

### _enforce_memory_md_cap 改造

变为 `async`，三级降级：

```python
COMPRESS_PROMPT = """压缩以下记忆摘要文件，控制在 {max_lines} 行 / {max_bytes} 字节以内。

保留索引结构和重要条目，删除低价值条目。
格式不变。

内容：
{content}
"""
```

1. **_within_cap 通过** → 直接返回
2. **LLM 压缩重试**：发 COMPRESS_PROMPT，最多重试 3 次（含网络/限速/异常重试）。每次重试检查 `_within_cap`，通过则返回
3. **兜底截断**：3 次重试全部失败（异常或仍超限）→ `_truncate_preserve_header`

**锁与异步**：cap 执行（含 LLM 调用）在 `_memory_lock` **外**完成。`run_governance` 先执行 cap，再获取锁进行 stage + rename。避免 LLM 响应期间阻塞 `extract_from_turn`。

辅助函数：

- `_within_cap(content) -> bool`：行数 ≤ 200 且字节 ≤ 25600
- `_truncate_preserve_header(content) -> str`：
  - 定位 header 边界：从开头到第一个空行（含）
  - 无空行 → 整个内容视为 body（header_end = 0）
  - header 本身超限 → 硬截断 header 到 cap 上限
  - body 取尾部（保留最新条目）
  - 字节超限时从 body 头部继续删

GOVERNANCE_PROMPT 第 6 条修正：
```
旧：detail 文档无大小限制（不注入上下文，按需读取）
新：detail 文档无大小限制（提取时各注入尾部 50 行，按需读取）
```

已知上限：子文档无大小限制，governance 间隔（24h + 5 sessions）期间增长依赖 governance LLM 压缩可靠性。`# ponytail: detail-doc growth depends on governance LLM discipline; revisit if any file exceeds ~1k lines`

## 改动文件

| 文件 | 改动 |
|------|------|
| `finagent/governance.py` | 改 EXTRACT_PROMPT、extract_from_turn、加 validate_and_dedup、删 _append_memory_md、改 _enforce_memory_md_cap（async + 重试3次 + 兜底）、加 _within_cap / _truncate_preserve_header / COMPRESS_PROMPT、改 GOVERNANCE_PROMPT 第 6 条、cap 移到锁外 |
| `tests/test_governance.py` | 新增 16 个测试，修改 2 个现有测试 |

## 测试计划

### 新增 validate_and_dedup 测试

| 测试 | 验证点 |
|------|--------|
| `test_validate_rejects_non_str` | dict/int item 被丢弃 |
| `test_validate_rejects_empty` | 空串/纯空格被丢弃 |
| `test_validate_rejects_non_list` | items 为字符串 → 返回空列表（不遍历字符） |
| `test_validate_dedup_exact` | 精确重复被丢弃 |
| `test_validate_dedup_strips_prefix` | existing 行 `- foo`，item `foo` → 去重命中；`-- foo` 不过度去重 |
| `test_validate_passes_new` | 新条目通过 |
| `test_validate_mixed` | 单次调用混合 pass + reject + dedup |

### 新增/修改 extract_from_turn 测试

| 测试 | 验证点 |
|------|--------|
| `test_extract_injects_existing_memory` | 提取 prompt 含已有记忆尾部 50 行 |
| `test_extract_no_memory_md_write` | 提取后 memory.md 不存在或不变 |
| `test_extract_dedup_at_write` | LLM 返回已有内容 → 子文档无追加 |
| `test_extract_from_turn_with_findings`（修改） | 去掉 memory.md 断言，加子文档去重验证 |

### 新增 cap 测试

| 测试 | 验证点 |
|------|--------|
| `test_within_cap_under` | 限制内返回 True |
| `test_within_cap_over` | 超限返回 False |
| `test_truncate_preserve_header` | 头部结构保留，body 截断 |
| `test_truncate_no_blank_line` | 无空行 → 整体视为 body |
| `test_truncate_header_over_cap` | header 本身超限 → 硬截断 |
| `test_enforce_cap_llm_compress` | mock LLM 压缩成功 |
| `test_enforce_cap_retry_then_success` | 前 2 次失败第 3 次成功 |
| `test_enforce_cap_fallback` | 3 次重试全失败 → 兜底截断 |
| `test_run_governance_caps_memory_md` | governance 端到端：超限 → 压缩/截断 → 原子写入 |
