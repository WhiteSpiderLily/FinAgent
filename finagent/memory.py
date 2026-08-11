"""Memory file loading with mtime-based conditional injection."""
from pathlib import Path

# Default file list: (label, path) pairs. Order determines injection order.
DEFAULT_MEMORY_FILES = [
    ("用户级长期记忆", Path.home() / ".finagent" / "finagent.md"),
    ("项目级长期记忆", Path(".finagent") / "finagent.md"),
    ("自动记忆摘要", Path(".finagent") / "memory" / "memory.md"),
]


class MemoryLoader:
    """Tracks file mtimes, returns changed content for injection.

    ponytail: mtime resolution depends on filesystem. APFS = nanosecond
    (fine locally). ext4/network mounts may be second-coarse — rapid
    edit+send within same second could miss a change.
    """

    def __init__(self, files: list[tuple[str, Path]] | None = None):
        self._files = files if files is not None else DEFAULT_MEMORY_FILES
        self._last_mtimes: dict[Path, float | None] = {}
        for _label, path in self._files:
            self._last_mtimes[path] = None

    def get_injectable(self) -> str | None:
        """Return memory content if any tracked file changed since last check.

        - First call: all existing files' content returned
        - Subsequent calls: only changed files' content returned
        - No changes: returns None
        - Non-existent files: skipped
        """
        changed_sections = []
        for label, path in self._files:
            if not path.exists():
                self._last_mtimes[path] = None
                continue
            current_mtime = path.stat().st_mtime
            last_mtime = self._last_mtimes.get(path)
            if last_mtime is not None and current_mtime == last_mtime:
                continue
            self._last_mtimes[path] = current_mtime
            content = path.read_text(encoding="utf-8").strip()
            if content:
                changed_sections.append(f"## {label}\n{content}")
        if not changed_sections:
            return None
        return "\n\n".join(changed_sections)

    def reset(self) -> None:
        """Clear mtime tracking so next call re-reads all files."""
        self._last_mtimes = {p: None for p in self._last_mtimes}
