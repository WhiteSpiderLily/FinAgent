"""Memory governance: per-turn extraction + periodic maintenance."""
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

from finagent.config import get_llm
from finagent.session import atomic_write, count_sessions

MEMORY_DIR = Path(".finagent/memory")

# Module-level lock ensures extraction and governance never write concurrently
_memory_lock = asyncio.Lock()

MEMORY_MD_MAX_LINES = 200
MEMORY_MD_MAX_BYTES = 25_600  # 25KB

KNOWN_FILES = ("memory.md", "preference.md", "project.md", "feedback.md", "reference.md")

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


def check_governance_needed() -> bool:
    """Check if governance should run: >24h since last AND >=5 new sessions."""
    lg_path = MEMORY_DIR / ".last_governance"
    if lg_path.exists():
        try:
            data = json.loads(lg_path.read_text(encoding="utf-8"))
            last_time = datetime.fromisoformat(data["timestamp"])
            processed = data.get("processed_sessions", 0)
        except (json.JSONDecodeError, KeyError, ValueError):
            return False
    else:
        last_time = datetime.min
        processed = 0

    hours_elapsed = (datetime.now() - last_time).total_seconds() / 3600
    if hours_elapsed < 24:
        return False

    new_sessions = count_sessions() - processed
    return new_sessions >= 5


async def extract_from_turn(messages: list) -> None:
    """Per-turn extraction. LLM analyzes the latest turn, appends to detail + memory.md.

    Uses .replace() for prompt templating (template contains literal JSON braces).
    Only passes the latest user msg + AI reply (not full tool call chain).
    Skips turns with <50 chars of text.
    """
    # Extract only the latest user turn + AI reply
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_user_idx = i
            break
    if last_user_idx is None:
        return

    turn_msgs = messages[last_user_idx:]
    lines = []
    for m in turn_msgs:
        if isinstance(m, HumanMessage) and m.content:
            lines.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"助手: {m.content}")
    turn_text = "\n".join(lines)

    if len(turn_text) < 50:
        return

    prompt = EXTRACT_PROMPT.replace("{messages}", turn_text)
    llm = get_llm()

    response = await llm.ainvoke(prompt)
    content = response.content
    # Extract JSON from response (may have markdown fences)
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        return
    try:
        findings = json.loads(content[start:end])
    except json.JSONDecodeError:
        return

    async with _memory_lock:
        has_content = False
        for category in ("preference", "project", "feedback", "reference"):
            items = findings.get(category, [])
            if not items:
                continue
            has_content = True
            detail_path = MEMORY_DIR / f"{category}.md"
            existing = detail_path.read_text(encoding="utf-8") if detail_path.exists() else ""
            new_section = "\n".join(f"- {item}" for item in items)
            atomic_write(detail_path, existing + new_section + "\n")

        if has_content:
            _append_memory_md(findings)


def _enforce_memory_md_cap(content: str) -> str:
    """Trim memory.md content to within line + byte caps (keeps most recent)."""
    lines = content.split("\n")
    if len(lines) > MEMORY_MD_MAX_LINES:
        lines = lines[-MEMORY_MD_MAX_LINES:]
        content = "\n".join(lines)
    while len(content.encode("utf-8")) > MEMORY_MD_MAX_BYTES and lines:
        lines.pop(0)
        content = "\n".join(lines)
    return content


def _append_memory_md(findings: dict) -> None:
    """Append summary lines to memory.md, enforcing size cap."""
    path = MEMORY_DIR / "memory.md"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    for category in ("preference", "project", "feedback", "reference"):
        items = findings.get(category, [])
        for item in items:
            content += f"- [{category}]({category}.md) — {item}\n"
    content = _enforce_memory_md_cap(content)
    atomic_write(path, content)


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


async def run_governance() -> None:
    """Periodic maintenance. Reads memory/ files, LLM rewrites them cleanly.

    Does NOT read sessions — only compacts existing memory/ content.
    Stages all 5 files in a temp dir, then per-file atomic rename.
    """
    # Read current memory files
    current_parts = []
    for name in KNOWN_FILES:
        path = MEMORY_DIR / name
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                current_parts.append(f"=== FILE: {name} ===\n{content}")

    current_memory = "\n\n".join(current_parts) if current_parts else "(空)"
    prompt = GOVERNANCE_PROMPT.replace("{current_memory}", current_memory)

    llm = get_llm()

    response = await llm.ainvoke(prompt)
    output = response.content

    # Parse output: split by === FILE: <name> === markers
    files = {}
    current_name = None
    current_lines = []
    for line in output.split("\n"):
        if line.startswith("=== FILE: ") and line.endswith(" ==="):
            if current_name:
                files[current_name] = "\n".join(current_lines).strip()
            current_name = line[len("=== FILE: "):-len(" ===")]
            current_lines = []
        else:
            current_lines.append(line)
    if current_name:
        files[current_name] = "\n".join(current_lines).strip()

    # Guard: if LLM returned prose without markers, don't wipe memory
    if not files:
        return

    async with _memory_lock:
        # Ensure all 5 files exist (missing sections default to empty)
        for name in KNOWN_FILES:
            if name not in files:
                files[name] = ""

        # Enforce memory.md cap before staging
        files["memory.md"] = _enforce_memory_md_cap(files["memory.md"])

        # Stage all files in temp subdir, then rename
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        staging = MEMORY_DIR / ".staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(exist_ok=True)
        for name in KNOWN_FILES:
            (staging / name).write_text(files[name] + "\n", encoding="utf-8")
        # Per-file atomic swap
        for name in KNOWN_FILES:
            (staging / name).rename(MEMORY_DIR / name)
        staging.rmdir()

        # Update .last_governance
        lg_path = MEMORY_DIR / ".last_governance"
        lg_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_sessions": count_sessions(),
        }
        atomic_write(lg_path, json.dumps(lg_data, ensure_ascii=False))
