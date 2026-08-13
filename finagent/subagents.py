"""Subagent definitions for Deep Agents — bull/bear debate specs."""
from deepagents import FilesystemPermission
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

# ponytail: tools list lives in tools.py; alias avoids a second source of truth.
from finagent.tools import tools as DATA_TOOLS

# ponytail: root_dir="." so agent paths like ".finagent/skills/..." resolve to
# <project>/.finagent/skills/... without double-prefix. virtual_mode still
# blocks ".." traversal; permission rules enforce the .finagent/ boundary.
BACKEND = FilesystemBackend(root_dir=".", virtual_mode=True)

MAIN_AGENT_PERMISSIONS = [
    FilesystemPermission(
        operations=["read"],
        paths=["/.finagent", "/.finagent/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["write"],
        paths=["/.finagent/reports/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**", "/.finagent/**"],
        mode="deny",
    ),
]

SUBAGENT_PERMISSIONS = [
    FilesystemPermission(
        operations=["read"],
        paths=["/.finagent", "/.finagent/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**", "/.finagent/**"],
        mode="deny",
    ),
]

# ponytail: custom middleware replaces default filesystem middleware on subagents.
# The spec-level `permissions=` key is intentionally omitted — enforcement comes
# entirely from `_permissions=` here, since custom `middleware=` replaces defaults.
SUBAGENT_MIDDLEWARE = FilesystemMiddleware(
    backend=BACKEND,
    _permissions=SUBAGENT_PERMISSIONS,
    tools=["read_file"],
)

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


def build_subagent_specs() -> list:
    """Build Deep Agents subagent dicts. Called at agent creation, not import."""
    shared = {
        "tools": DATA_TOOLS,
        "middleware": [SUBAGENT_MIDDLEWARE],
    }
    return [
        {
            "name": "bull",
            "description": "多方分析师，从财务/估值/资金/行业角度给出看多理由",
            "system_prompt": BULL_SYSTEM_PROMPT,
            **shared,
        },
        {
            "name": "bear",
            "description": "空方分析师，从财务/估值/资金/行业角度给出看空理由",
            "system_prompt": BEAR_SYSTEM_PROMPT,
            **shared,
        },
    ]
