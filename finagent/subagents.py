"""Subagent definitions for Deep Agents — bull/bear debate specs."""
from deepagents.middleware.filesystem import FilesystemMiddleware

# ── Global tool blacklist (subagents forbidden) ─────────────────────

GLOBAL_TOOL_BLACKLIST = frozenset({
    "generate_report_tool",
    "update_section",
    "delete_section",
    "read_sandbox_file",
})


# ── Restricted filesystem middleware (security: no execute/write) ───

FILESYSTEM_MIDDLEWARE = FilesystemMiddleware(tools=["read_file"])


# ── Bull / Bear system prompts ──────────────────────────────────────

BULL_SYSTEM_PROMPT = """你是一名 A股多方分析师。从给定数据和工具结果中找出支持看多的理由。

规则：
1. 聚焦正面信号：营收增长、利润扩张、估值偏低、资金流入、行业景气、筹码集中。
2. 只使用工具返回的数据，绝不编造数字。
3. 不做买卖评级，不给目标价。
4. 必须先调用工具获取数据，再给出分析。
5. 输出 3-5 个核心看多逻辑，每个附数据支撑。
"""

BEAR_SYSTEM_PROMPT = """你是一名 A股空方分析师。从给定数据和工具结果中找出支持看空的理由。

规则：
1. 聚焦负面信号：利润下滑、毛利率收窄、估值偏高、债务风险、资金流出、竞争恶化。
2. 只使用工具返回的数据，绝不编造数字。
3. 不做买卖评级，不给目标价。
4. 必须先调用工具获取数据，再给出分析。
5. 输出 3-5 个核心看空逻辑，每个附数据支撑。
"""


# ── Tool resolution ──────────────────────────────────────────────────

def resolve_subagent_tools() -> list:
    """Return financial tools minus blacklist."""
    from finagent.tools import tools as all_tools
    return [t for t in all_tools if t.name not in GLOBAL_TOOL_BLACKLIST]


# ── Deep Agents subagent specs ───────────────────────────────────────

def build_subagent_specs() -> list:
    """Build Deep Agents subagent dicts. Called at agent creation, not import."""
    # ponytail: shared list — deepagents reads but does not mutate tool lists
    tools = resolve_subagent_tools()
    return [
        {
            "name": "bull",
            "description": "多方分析师，从财务/估值/资金/行业角度给出看多理由",
            "system_prompt": BULL_SYSTEM_PROMPT,
            "tools": tools,
            "middleware": [FILESYSTEM_MIDDLEWARE],
        },
        {
            "name": "bear",
            "description": "空方分析师，从财务/估值/资金/行业角度给出看空理由",
            "system_prompt": BEAR_SYSTEM_PROMPT,
            "tools": tools,
            "middleware": [FILESYSTEM_MIDDLEWARE],
        },
    ]
