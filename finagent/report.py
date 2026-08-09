"""Report synthesis chain — generates structured 财报点评 from conversation history."""
import re
from pathlib import Path
from typing import NamedTuple

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from finagent.config import get_llm
from finagent.prompts import REPORT_SYSTEM_PROMPT, REPORT_TEMPLATE

_current_report_path: Path | None = None


def get_current_report() -> Path | None:
    """返回当前 session 的报告文件路径，无报告时返回 None。"""
    return _current_report_path


def set_current_report(path: Path | None) -> None:
    """更新当前报告路径；传 None 清空状态（/clear 时使用）。"""
    global _current_report_path
    _current_report_path = path


class Section(NamedTuple):
    title: str  # 去掉 "## " 前缀的完整标题（含"一、"编号）
    body: str   # 段正文（已 rstrip 换行）


def _is_h1(line: str) -> bool:
    return line.startswith("# ") and not line.startswith("##")


def _is_h2(line: str) -> bool:
    return line.startswith("## ") and not line.startswith("###")


def parse_sections(markdown: str) -> tuple[str, list[Section]]:
    """把 markdown 按 H2 标题切段。H1 作为 title_line 独立返回，H3+ 归入上一个 H2 段的 body。"""
    title_line = ""
    sections: list[Section] = []
    cur_title: str | None = None
    cur_body: list[str] = []
    for line in markdown.splitlines():
        if _is_h1(line):
            if cur_title is not None:
                sections.append(Section(cur_title, "".join(cur_body).rstrip("\n")))
                cur_title, cur_body = None, []
            title_line = line
        elif _is_h2(line):
            if cur_title is not None:
                sections.append(Section(cur_title, "".join(cur_body).rstrip("\n")))
            cur_title = line[3:].strip()
            cur_body = []
        else:
            if cur_title is not None:
                cur_body.append(line + "\n")
    if cur_title is not None:
        sections.append(Section(cur_title, "".join(cur_body).rstrip("\n")))
    return title_line, sections


def serialize_sections(title_line: str, sections: list[Section]) -> str:
    """把 title_line + sections 拼回完整 markdown。段间用空行分隔。"""
    parts = [title_line] if title_line else []
    parts.extend(f"## {s.title}\n{s.body}" for s in sections)
    return "\n\n".join(parts)


def _messages_to_text(messages: list) -> str:
    """Flatten message list to readable transcript text."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "user")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        label = {"user": "分析师", "human": "分析师", "assistant": "助手", "ai": "助手", "tool": "工具结果"}.get(role, role)
        lines.append(f"[{label}]\n{content}")
    return "\n\n".join(lines)


def _write_report_file(stock_code: str, period: str, content: str) -> Path:
    """写入报告文件并 set_current_report。返回文件路径。"""
    reports_dir = Path.cwd() / "reports"
    reports_dir.mkdir(exist_ok=True)
    period_safe = re.sub(r"[^\w一-鿿]", "", period)
    filepath = reports_dir / f"{stock_code}_{period_safe}_点评.md"
    filepath.write_text(content, encoding="utf-8")
    set_current_report(filepath)
    return filepath


def _generate_report_core(stock_code: str, period: str, transcript: str) -> tuple[str, str]:
    """核心生成逻辑。显式 code/period + transcript → 写文件 + set_current_report → (filepath, content)。

    Raises ValueError on LLM error ([ERROR] substring in response content).
    """
    full_prompt = (
        f"{REPORT_SYSTEM_PROMPT}\n\n"
        f"模板结构:\n{REPORT_TEMPLATE}\n\n"
        f"对话历史:\n{transcript}\n\n"
        f"请按模板生成完整报告。"
    )
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=full_prompt)])
    content = response.content if hasattr(response, "content") else str(response)

    if "[ERROR]" in content:
        raise ValueError(content)

    filepath = _write_report_file(stock_code, period, content)
    return str(filepath), content


def generate_report(messages: list) -> tuple[str, str]:
    """Synthesize a 财报点评 report from conversation history.

    向后兼容入口：从 messages 推导 code/period 后调 `_generate_report_core`。

    Returns (filepath, content). Raises ValueError if LLM cannot determine company/period.
    """
    transcript = _messages_to_text(messages)
    # extract code + period for filename (fallback to 'report' if not found)
    code_match = re.search(r"\b(\d{6})\b", transcript)
    period_match = re.search(r"(\d{4}Q\d|\d{4}[一二三四半年度季报]+报|\d{4}-\d{2}-\d{2})", transcript)
    code = code_match.group(1) if code_match else "report"
    period = period_match.group(1) if period_match else "unknown"
    return _generate_report_core(code, period, transcript)


@tool
def read_report() -> str:
    """读取当前报告全文与段落结构。

    返回段落标题列表 + 全文 markdown，便于后续 update_section / delete_section 操作。
    """
    path = get_current_report()
    if path is None:
        return "当前无报告。请先用 generate_report 生成，或通过 /report 命令生成。"
    try:
        md = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"读取报告失败: {e}"
    _, sections = parse_sections(md)
    listing = "\n".join(f"{i + 1}. {s.title}" for i, s in enumerate(sections))
    return f"当前报告: {path}\n段落列表:\n{listing}\n\n--- 全文 ---\n{md}"


@tool
def update_section(title: str, content: str) -> str:
    """替换或追加报告段落。title 精确匹配现有段落标题（含"一、"编号）；未命中则追加新段到末尾。

    Args:
        title: 段落标题，如 "五、风险提示" 或新增段 "行业对比"
        content: 该段新内容（markdown 正文，不含 ## 标题行）
    """
    path = get_current_report()
    if path is None:
        return "当前无报告。请先用 generate_report 生成，或通过 /report 命令生成。"
    try:
        md = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"读取报告失败: {e}"
    title_line, sections = parse_sections(md)
    new_body = content.rstrip("\n")
    found = False
    updated: list[Section] = []
    for s in sections:
        if s.title == title:
            updated.append(Section(title, new_body))
            found = True
        else:
            updated.append(s)
    if not found:
        updated.append(Section(title, new_body))
    try:
        path.write_text(serialize_sections(title_line, updated), encoding="utf-8")
    except OSError as e:
        return f"写入报告失败: {e}"
    action = "替换" if found else "追加"
    listing = "\n".join(f"{i + 1}. {s.title}" for i, s in enumerate(updated))
    return f"已{action}段落: {title}\n当前段落:\n{listing}"


@tool
def delete_section(title: str) -> str:
    """删除报告段落。title 精确匹配；未命中时列出可用段落标题。

    Args:
        title: 段落标题，如 "五、风险提示"
    """
    path = get_current_report()
    if path is None:
        return "当前无报告。请先用 generate_report 生成，或通过 /report 命令生成。"
    try:
        md = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"读取报告失败: {e}"
    title_line, sections = parse_sections(md)
    filtered = [s for s in sections if s.title != title]
    if len(filtered) == len(sections):
        listing = "\n".join(f"{i + 1}. {s.title}" for i, s in enumerate(sections))
        return f"段落不存在: {title}\n可用段落:\n{listing}"
    try:
        path.write_text(serialize_sections(title_line, filtered), encoding="utf-8")
    except OSError as e:
        return f"写入报告失败: {e}"
    listing = "\n".join(f"{i + 1}. {s.title}" for i, s in enumerate(filtered))
    return f"已删除段落: {title}\n当前段落:\n{listing}"


@tool
def generate_report_tool(stock_code: str, period: str, content: str) -> str:
    """生成财报点评报告并保存到文件。agent 根据对话中的分析数据，按固定模板结构生成完整 markdown 报告内容，调用此工具写入 reports/ 目录。

    Args:
        stock_code: 6 位股票代码，如 002415
        period: 报告期，如 2024Q3 / 2024三季报 / 2024-09-30
        content: 完整报告 markdown 内容，按六段结构（# 标题 + ## 一、事件概述 / ## 二、财务分析 / ## 三、经营要点 / ## 四、影响评估 / ## 五、风险提示 / ## 六、免责声明）
    """
    try:
        filepath = _write_report_file(stock_code, period, content)
    except OSError as e:
        return f"写入报告失败: {e}"
    return f"报告已保存\n路径: {filepath}"
