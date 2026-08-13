"""tests/test_prompts.py"""
from finagent.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    SKILL_RECIPES,
)


def test_research_prompt_mentions_constraints():
    assert "不出评级" in RESEARCH_SYSTEM_PROMPT or "无评级" in RESEARCH_SYSTEM_PROMPT
    assert "不编造" in RESEARCH_SYSTEM_PROMPT
    assert "A股" in RESEARCH_SYSTEM_PROMPT


def test_skill_recipes_contains_tools():
    for tool in [
        "get_company_info",
        "get_financials",
        "get_valuation",
        "get_industry_ranking",
        "get_research_reports",
        "get_holder_change",
        "get_dividend_history",
        "get_fund_flow",
    ]:
        assert tool in SKILL_RECIPES


def test_system_prompt_includes_recipes():
    assert SKILL_RECIPES in RESEARCH_SYSTEM_PROMPT


def test_report_constants_removed():
    import pytest
    with pytest.raises(ImportError):
        from finagent.prompts import REPORT_SYSTEM_PROMPT  # noqa: F401
    with pytest.raises(ImportError):
        from finagent.prompts import REPORT_TEMPLATE  # noqa: F401


def test_skill_recipes_no_report_subsection():
    from finagent.prompts import SKILL_RECIPES
    assert "报告生成与编辑" not in SKILL_RECIPES
    assert "generate_report_tool" not in SKILL_RECIPES
    assert "update_section" not in SKILL_RECIPES


def test_research_system_prompt_has_file_operation_guidance():
    from finagent.prompts import RESEARCH_SYSTEM_PROMPT
    assert "read_file" in RESEARCH_SYSTEM_PROMPT
    assert ".finagent/skills/" in RESEARCH_SYSTEM_PROMPT
    assert ".finagent/reports/" in RESEARCH_SYSTEM_PROMPT
