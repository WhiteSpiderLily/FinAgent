"""ReAct agent construction using LangGraph."""
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent as create_langchain_agent

from finagent.config import get_llm
from finagent.prompts import RESEARCH_SYSTEM_PROMPT
from finagent.tools import tools

# shared in-memory checkpointer — one per process, cleared on exit
_checkpointer = MemorySaver()


def create_agent():
    """Build a ReAct agent bound to a conversation thread."""
    llm = get_llm()
    agent = create_langchain_agent(
        model=llm,
        tools=tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )
    return agent


def reset_checkpoint() -> None:
    """重置 checkpointer，使 /clear 后新对话从干净状态开始。"""
    global _checkpointer
    _checkpointer = MemorySaver()
