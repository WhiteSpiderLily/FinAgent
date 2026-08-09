"""tests/test_agent.py"""
from finagent.agent import reset_checkpoint


def test_reset_checkpoint_does_not_error():
    reset_checkpoint()  # should not raise
