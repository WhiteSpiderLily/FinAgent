"""Input history with file persistence, consecutive dedup, and FIFO cap."""
import json
from collections import deque
from pathlib import Path

_MAX_HISTORY = 100


class InputHistory:
    def __init__(self, path: Path):
        self._path = path
        self._items: deque[str] = deque(maxlen=_MAX_HISTORY)
        self.load()

    @property
    def items(self) -> list[str]:
        return list(self._items)

    def append(self, text: str) -> None:
        if self._items and self._items[-1] == text:
            return
        self._items.append(text)

    def load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._items = deque(data, maxlen=_MAX_HISTORY)
        except (json.JSONDecodeError, OSError):
            self._items = deque(maxlen=_MAX_HISTORY)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(list(self._items), ensure_ascii=False), encoding="utf-8"
        )
