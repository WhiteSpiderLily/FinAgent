"""Test agent history pre-seeding via update_state."""
from langchain_core.messages import HumanMessage, AIMessage

from finagent.agent import create_agent_with_history


def test_create_agent_with_history_seeds_checkpoint():
    """update_state injects messages into checkpointer without executing graph."""
    messages = [
        HumanMessage(content="你好", id="m1"),
        AIMessage(content="你好！", id="m2"),
    ]
    agent = create_agent_with_history("test-thread", messages)

    # Verify messages are in the checkpoint
    state = agent.get_state(config={"configurable": {"thread_id": "test-thread"}})
    checkpoint_msgs = state.values.get("messages", [])
    assert len(checkpoint_msgs) >= 2
    assert checkpoint_msgs[0].content == "你好"
    assert checkpoint_msgs[1].content == "你好！"
