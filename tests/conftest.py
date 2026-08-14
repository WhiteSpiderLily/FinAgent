"""Shared test fixtures."""
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# CI has no .env — set dummy key so get_llm() can instantiate ChatDeepSeek.
# API is never called in tests; this only allows model object construction.
os.environ.setdefault("DEEPSEEK_API_KEY", "dummy-key-for-tests")


@pytest.fixture(autouse=True)
def _mock_tui_persistence(tmp_path, monkeypatch):
    """Auto-mock persistence layer in TUI tests.

    Patches at the tui.py import level so direct module tests are unaffected.
    """
    monkeypatch.setattr("finagent.tui.INPUT_HISTORY_PATH", tmp_path / "ih.json")
    monkeypatch.setattr("finagent.tui.COMMAND_FREQ_PATH", tmp_path / "freq.json")
    with patch("finagent.tui.write_session"), \
         patch("finagent.tui.extract_from_turn", new_callable=AsyncMock), \
         patch("finagent.tui.run_governance", new_callable=AsyncMock), \
         patch("finagent.tui.check_governance_needed", return_value=False):
        yield
