"""Command/skill usage frequency persisted to JSON."""
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class CommandFreq:
    """Usage frequency counter persisted under command_freq key in JSON file."""

    def __init__(self, path: Path):
        self._path = path
        self._freq: dict[str, int] = {}
        self.load()

    def load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                raw = data.get("command_freq", {}) if isinstance(data, dict) else {}
                self._freq = {
                    k: v for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)
                }
        except (json.JSONDecodeError, OSError) as e:
            log.warning("command_freq load failed: %s", e)
            self._freq = {}

    def get(self, name: str) -> int:
        return self._freq.get(name, 0)

    def increment(self, name: str) -> None:
        self._freq[name] = self._freq.get(name, 0) + 1

    def save(self) -> None:
        try:
            if self._path.exists():
                existing = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            else:
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
        existing["command_freq"] = self._freq
        self._path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
