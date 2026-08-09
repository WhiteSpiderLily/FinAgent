from pathlib import Path

import pytest

import finagent.report as report_mod
from finagent.report import parse_sections, serialize_sections, Section


@pytest.fixture(autouse=True)
def _reset_current_report_path():
    """Reset module-level report path before each test for isolation."""
    old = report_mod._current_report_path
    report_mod._current_report_path = None
    yield
    report_mod._current_report_path = old


def test_parse_empty_string():
    title, sections = parse_sections("")
    assert title == ""
    assert sections == []


def test_parse_only_h1_no_sections():
    title, sections = parse_sections("# 标题\n")
    assert title == "# 标题"
    assert sections == []


def test_parse_six_section_template():
    md = (
        "# 海康威视(002415) 2024Q3财报点评\n"
        "\n"
        "## 一、事件概述\n"
        "报告期、披露日期、核心财务数据摘要。\n"
        "\n"
        "## 二、财务分析\n"
        "营收/归母净利 同比环比。\n"
        "\n"
        "## 三、经营要点\n"
        "业务分部变化。\n"
    )
    title, sections = parse_sections(md)
    assert title == "# 海康威视(002415) 2024Q3财报点评"
    assert len(sections) == 3
    assert sections[0].title == "一、事件概述"
    assert sections[0].body == "报告期、披露日期、核心财务数据摘要。"
    assert sections[1].title == "二、财务分析"
    assert sections[2].title == "三、经营要点"


def test_parse_h3_nested_into_parent_body():
    md = (
        "## 一、A\n"
        "正文。\n"
        "\n"
        "### 子小节\n"
        "子内容。\n"
    )
    _, sections = parse_sections(md)
    assert len(sections) == 1
    assert "### 子小节" in sections[0].body
    assert "子内容" in sections[0].body


def test_serialize_roundtrip_preserves_structure():
    md = (
        "# Title\n"
        "\n"
        "## 一、A\n"
        "content a\n"
        "\n"
        "## 二、B\n"
        "content b"
    )
    title, sections = parse_sections(md)
    assert serialize_sections(title, sections) == md


def test_serialize_empty_sections():
    assert serialize_sections("# Only Title", []) == "# Only Title"


def test_current_report_default_none():
    # 模块级变量默认值
    assert report_mod._current_report_path is None


def test_set_and_get_current_report(monkeypatch):
    p = Path("/tmp/fake_report.md")
    monkeypatch.setattr(report_mod, "_current_report_path", p)
    assert report_mod.get_current_report() == p


def test_set_current_report_updates_state(monkeypatch):
    monkeypatch.setattr(report_mod, "_current_report_path", None)
    p = Path("/tmp/another.md")
    report_mod.set_current_report(p)
    assert report_mod.get_current_report() == p


def test_set_current_report_accepts_none(monkeypatch):
    """I1: set_current_report(None) 必须清空路径（/clear 后调用）。"""
    monkeypatch.setattr(report_mod, "_current_report_path", Path("/tmp/x.md"))
    report_mod.set_current_report(None)
    assert report_mod.get_current_report() is None


def test_generate_report_core_writes_file_and_sets_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class FakeResponse:
        content = "# 标题\n\n## 一、事件概述\n内容。"

    class FakeLLM:
        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr("finagent.report.get_llm", lambda: FakeLLM())

    from finagent.report import _generate_report_core, get_current_report
    filepath, content = _generate_report_core("002415", "2024Q3", "fake transcript")

    assert "002415" in filepath
    assert "事件概述" in content
    assert (tmp_path / "reports").exists()
    assert get_current_report() is not None
    assert "002415" in str(get_current_report())


def test_generate_report_core_raises_on_llm_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class FakeResponse:
        content = "[ERROR] LLM 无法生成报告"

    class FakeLLM:
        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr("finagent.report.get_llm", lambda: FakeLLM())

    from finagent.report import _generate_report_core
    import pytest
    with pytest.raises(ValueError, match="\\[ERROR\\]"):
        _generate_report_core("002415", "2024Q3", "transcript")


from finagent.report import read_report


def test_read_report_no_current_report(monkeypatch):
    monkeypatch.setattr("finagent.report._current_report_path", None)
    result = read_report.invoke({})
    assert "当前无报告" in result


def test_read_report_returns_listing_and_full_content(monkeypatch, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text(
        "# 标题\n\n## 一、事件概述\n内容A\n\n## 二、财务分析\n内容B",
        encoding="utf-8",
    )
    monkeypatch.setattr("finagent.report._current_report_path", md_file)
    result = read_report.invoke({})
    assert str(md_file) in result
    assert "一、事件概述" in result
    assert "二、财务分析" in result
    assert "内容A" in result  # 全文部分
    assert "段落列表" in result


def test_read_report_io_error_returns_friendly_string(monkeypatch, tmp_path):
    missing = tmp_path / "ghost.md"
    monkeypatch.setattr("finagent.report._current_report_path", missing)
    result = read_report.invoke({})
    assert "读取报告失败" in result


from finagent.report import update_section


def test_update_section_replaces_existing(monkeypatch, tmp_path):
    md_file = tmp_path / "r.md"
    md_file.write_text("# T\n\n## 一、A\nold\n\n## 二、B\nkeep", encoding="utf-8")
    monkeypatch.setattr("finagent.report._current_report_path", md_file)

    result = update_section.invoke({"title": "一、A", "content": "new content"})

    assert "替换" in result
    assert "一、A" in result
    on_disk = md_file.read_text(encoding="utf-8")
    assert "new content" in on_disk
    assert "old" not in on_disk
    assert "keep" in on_disk  # 其他段不动


def test_update_section_appends_when_title_missing(monkeypatch, tmp_path):
    md_file = tmp_path / "r.md"
    md_file.write_text("# T\n\n## 一、A\ncontent", encoding="utf-8")
    monkeypatch.setattr("finagent.report._current_report_path", md_file)

    result = update_section.invoke({"title": "行业对比", "content": "新增段。"})

    assert "追加" in result
    on_disk = md_file.read_text(encoding="utf-8")
    assert "## 行业对比" in on_disk
    assert "新增段" in on_disk
    assert "一、A" in on_disk


def test_update_section_no_current_report(monkeypatch):
    monkeypatch.setattr("finagent.report._current_report_path", None)
    result = update_section.invoke({"title": "X", "content": "Y"})
    assert "当前无报告" in result


from finagent.report import delete_section


def test_delete_section_removes_existing(monkeypatch, tmp_path):
    md_file = tmp_path / "r.md"
    md_file.write_text("# T\n\n## 一、A\ncontent\n\n## 二、B\nkeep", encoding="utf-8")
    monkeypatch.setattr("finagent.report._current_report_path", md_file)

    result = delete_section.invoke({"title": "一、A"})

    assert "删除" in result
    on_disk = md_file.read_text(encoding="utf-8")
    assert "一、A" not in on_disk
    assert "content" not in on_disk
    assert "二、B" in on_disk


def test_delete_section_missing_title_lists_available(monkeypatch, tmp_path):
    md_file = tmp_path / "r.md"
    md_file.write_text("# T\n\n## 一、A\nx\n\n## 二、B\ny", encoding="utf-8")
    monkeypatch.setattr("finagent.report._current_report_path", md_file)

    result = delete_section.invoke({"title": "三、C"})

    assert "不存在" in result or "未找到" in result
    assert "一、A" in result  # 列出可用段
    assert "二、B" in result
    on_disk = md_file.read_text(encoding="utf-8")
    assert "一、A" in on_disk  # 文件未改


def test_delete_section_no_current_report(monkeypatch):
    monkeypatch.setattr("finagent.report._current_report_path", None)
    result = delete_section.invoke({"title": "X"})
    assert "当前无报告" in result


def test_generate_report_tool_writes_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from finagent.report import generate_report_tool, get_current_report

    content = "# 海康威视(002415) 2024Q3财报点评\n\n## 一、事件概述\n内容。"
    result = generate_report_tool.invoke({
        "stock_code": "002415",
        "period": "2024Q3",
        "content": content,
    })

    assert "报告已保存" in result
    assert "002415" in result
    assert get_current_report() is not None
    written = get_current_report().read_text(encoding="utf-8")
    assert written == content


def test_generate_report_tool_io_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from finagent.report import generate_report_tool

    from pathlib import Path
    def raise_oserror(self, *a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "write_text", raise_oserror)

    result = generate_report_tool.invoke({
        "stock_code": "002415",
        "period": "2024Q3",
        "content": "fake",
    })
    assert "写入报告失败" in result


def test_tools_list_includes_report_tools():
    from finagent.tools import tools
    names = {t.name for t in tools}
    assert "generate_report_tool" in names
    assert "read_report" in names
    assert "update_section" in names
    assert "delete_section" in names


def test_research_prompt_mentions_report_tools():
    from finagent.prompts import RESEARCH_SYSTEM_PROMPT
    assert "generate_report" in RESEARCH_SYSTEM_PROMPT
    assert "update_section" in RESEARCH_SYSTEM_PROMPT
