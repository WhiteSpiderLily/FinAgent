"""Tests for finagent.skills."""
import pytest
from pathlib import Path

from finagent import skills


def test_get_finagent_roots_order(tmp_path, monkeypatch):
    """Project-local root precedes user home root."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    roots = skills.get_finagent_roots()
    assert roots[0] == tmp_path / ".finagent"
    assert roots[1] == fake_home / ".finagent"


def test_get_skill_roots_under_finagent(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    roots = skills.get_skill_roots()
    assert roots[0] == tmp_path / ".finagent" / "skills"
    assert roots[1] == fake_home / ".finagent" / "skills"


SKILL_MD_VALID = """\
---
name: news-radar
description: 资讯雷达 - 扫描A股实时新闻
---

# 资讯雷达
"""


def _make_skill(root: Path, name: str, body: str = SKILL_MD_VALID) -> Path:
    """Helper: write <root>/skills/<name>/skill.md with body."""
    d = root / "skills" / name
    d.mkdir(parents=True)
    (d / "skill.md").write_text(body, encoding="utf-8")
    return d / "skill.md"


def test_scan_skills_finds_single(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _make_skill(tmp_path / ".finagent", "news-radar")

    metas = skills.scan_skills()
    assert "news-radar" in metas
    assert metas["news-radar"].name == "news-radar"
    assert "资讯雷达" in metas["news-radar"].description


def test_scan_skills_project_overrides_global(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _make_skill(fake_home / ".finagent", "dup", "---\nname: dup\ndescription: global\n---\n")
    _make_skill(tmp_path / ".finagent", "dup", "---\nname: dup\ndescription: project\n---\n")

    metas = skills.scan_skills()
    assert metas["dup"].description == "project"


def test_scan_skills_skips_malformed_no_frontmatter(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    _make_skill(tmp_path / ".finagent", "bad", "no frontmatter here")

    metas = skills.scan_skills()
    assert "bad" not in metas


def test_scan_skills_skips_missing_name(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    _make_skill(tmp_path / ".finagent", "bad", "---\ndescription: no name\n---\n")

    metas = skills.scan_skills()
    assert "bad" not in metas


def test_scan_skills_skips_invalid_name(tmp_path, monkeypatch):
    """Names containing / are rejected (would break slash-command parsing)."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    # The dir name has no slash (filesystem won't allow), but frontmatter name could
    _make_skill(tmp_path / ".finagent", "ok", "---\nname: has/slash\ndescription: d\n---\n")

    metas = skills.scan_skills()
    assert "has/slash" not in metas


def test_scan_skills_skips_reserved_names(tmp_path, monkeypatch):
    """Reserved command names cannot be skills (case-insensitive)."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    _make_skill(tmp_path / ".finagent", "rlow", "---\nname: report\ndescription: x\n---\n")
    # Uppercase variant must also be rejected (would collide with /Report routing).
    _make_skill(tmp_path / ".finagent", "rupp", "---\nname: Report\ndescription: y\n---\n")

    metas = skills.scan_skills()
    assert "report" not in metas
    assert "Report" not in metas


def test_render_catalog_includes_name_and_desc():
    metas = {
        "news-radar": skills.SkillMeta(
            name="news-radar",
            description="资讯雷达",
            path=Path("/x/skills/news-radar/skill.md"),
        ),
        "report-edit": skills.SkillMeta(
            name="report-edit",
            description="报告编辑",
            path=Path("/x/skills/report-edit/skill.md"),
        ),
    }
    out = skills.render_catalog(metas)
    assert "news-radar" in out
    assert "资讯雷达" in out
    assert "report-edit" in out
    assert "报告编辑" in out


def test_render_catalog_empty_dict_returns_empty_hint():
    out = skills.render_catalog({})
    assert out.strip() == "(无可用 skill)"


def test_read_skill_md_returns_full_text(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    md_path = _make_skill(tmp_path / ".finagent", "news-radar", SKILL_MD_VALID)

    out = skills.read_skill_md("news-radar")
    assert "资讯雷达" in out
    assert "# 资讯雷达" in out


def test_read_skill_md_project_overrides_global(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _make_skill(fake_home / ".finagent", "dup", "---\nname: dup\ndescription: g\n---\nbody global")
    _make_skill(tmp_path / ".finagent", "dup", "---\nname: dup\ndescription: p\n---\nbody project")

    out = skills.read_skill_md("dup")
    assert "body project" in out


def test_read_skill_md_not_found_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    with pytest.raises(FileNotFoundError):
        skills.read_skill_md("nope")


def test_read_skill_md_rejects_traversal_name(tmp_path, monkeypatch):
    """Names containing path separators must be rejected — sandbox bypass."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    # Place a target file that the unguarded path would reach
    escape_dir = tmp_path / "escape"
    escape_dir.mkdir()
    (escape_dir / "skill.md").write_text("secret", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        skills.read_skill_md("../escape")
