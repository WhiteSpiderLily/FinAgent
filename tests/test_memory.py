"""Tests for MemoryLoader mtime-based injection."""
from pathlib import Path

from finagent.memory import MemoryLoader


def test_get_injectable_first_call_returns_all(tmp_path):
    """First call returns content from all existing files."""
    user_file = tmp_path / "user.md"
    user_file.write_text("# 用户偏好\n喜欢简洁报告", encoding="utf-8")
    proj_file = tmp_path / "proj.md"
    proj_file.write_text("# 项目规则\nA股only", encoding="utf-8")
    mem_file = tmp_path / "memory.md"
    mem_file.write_text("# 摘要\n- 偏好: 简洁", encoding="utf-8")

    loader = MemoryLoader(files=[
        ("用户级", user_file),
        ("项目级", proj_file),
        ("自动记忆", mem_file),
    ])
    result = loader.get_injectable()
    assert result is not None
    assert "用户偏好" in result
    assert "项目规则" in result
    assert "摘要" in result


def test_get_injectable_no_changes_returns_none(tmp_path):
    """Second call with no file changes returns None."""
    f = tmp_path / "f.md"
    f.write_text("content", encoding="utf-8")
    loader = MemoryLoader(files=[("test", f)])
    first = loader.get_injectable()
    assert first is not None
    second = loader.get_injectable()
    assert second is None


def test_get_injectable_file_changed_returns_new_content(tmp_path):
    """When a file changes, get_injectable returns its new content."""
    f = tmp_path / "f.md"
    f.write_text("old", encoding="utf-8")
    loader = MemoryLoader(files=[("test", f)])
    loader.get_injectable()  # mark as seen
    f.write_text("new content", encoding="utf-8")
    result = loader.get_injectable()
    assert result is not None
    assert "new content" in result


def test_get_injectable_nonexistent_file_skipped(tmp_path):
    """Non-existent files are silently skipped."""
    missing = tmp_path / "nope.md"
    loader = MemoryLoader(files=[("missing", missing)])
    result = loader.get_injectable()
    assert result is None


def test_reset_clears_tracking(tmp_path):
    """After reset, next call returns content again."""
    f = tmp_path / "f.md"
    f.write_text("content", encoding="utf-8")
    loader = MemoryLoader(files=[("test", f)])
    loader.get_injectable()  # mark seen
    assert loader.get_injectable() is None  # no change
    loader.reset()
    result = loader.get_injectable()
    assert result is not None
    assert "content" in result
