"""tests/test_prompts.py"""
from finagent.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
    REPORT_TEMPLATE,
    SKILL_RECIPES,
)


def test_research_prompt_mentions_constraints():
    assert "不出评级" in RESEARCH_SYSTEM_PROMPT or "无评级" in RESEARCH_SYSTEM_PROMPT
    assert "不编造" in RESEARCH_SYSTEM_PROMPT
    assert "A股" in RESEARCH_SYSTEM_PROMPT


def test_report_prompt_has_all_sections():
    for section in ["事件概述", "财务分析", "经营要点", "影响评估", "风险提示", "免责声明"]:
        assert section in REPORT_TEMPLATE


def test_report_prompt_no_rating():
    assert "评级" in REPORT_SYSTEM_PROMPT
    assert "目标价" in REPORT_SYSTEM_PROMPT


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
