"""Tests for finagent.config constants."""
from finagent import config


def test_model_name_is_string():
    assert isinstance(config.MODEL_NAME, str)
    assert config.MODEL_NAME  # non-empty


def test_context_window_positive_int():
    assert isinstance(config.CONTEXT_WINDOW_TOKENS, int)
    assert config.CONTEXT_WINDOW_TOKENS > 0
