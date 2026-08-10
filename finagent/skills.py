"""Skill discovery and loading.

Skills live under <root>/.finagent/skills/<name>/skill.md where <root> is
either the project working directory or the user home directory. Project
skills override user skills on name conflict.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


def get_finagent_roots() -> list[Path]:
    """Return [cwd/.finagent, home/.finagent]. Project root first."""
    return [Path.cwd() / ".finagent", Path.home() / ".finagent"]


def get_skill_roots() -> list[Path]:
    """Return [cwd/.finagent/skills, home/.finagent/skills]. Project first."""
    return [root / "skills" for root in get_finagent_roots()]


# Names reserved for built-in slash commands; cannot be skill names.
RESERVED_NAMES = frozenset({"report", "clear", "help", "quit", "reload_skills"})

# Legal skill name: letters, digits, underscore, hyphen only.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class SkillMeta:
    """Metadata for one discoverable skill."""
    name: str
    description: str
    path: Path  # absolute path to skill.md


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str] | None:
    """Parse '---\\nkey: value\\n---\\nbody' frontmatter. Return (fields, body) or None."""
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None
    fields: dict[str, str] = {}
    for line in lines[1:end_idx]:
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fields[k.strip()] = v.strip()
    body = "\n".join(lines[end_idx + 1:])
    return fields, body


def scan_skills() -> dict[str, SkillMeta]:
    """Scan all skill roots, merge with project-overrides-global semantics.

    Skips skills with: missing frontmatter, missing name/description,
    illegal name chars, or reserved names.
    """
    metas: dict[str, SkillMeta] = {}
    # Iterate global->project so project overrides on name collision
    for root in reversed(get_skill_roots()):
        if not root.is_dir():
            continue
        for skill_md in root.glob("*/skill.md"):
            try:
                text = skill_md.read_text(encoding="utf-8")
            except OSError:
                log.warning("skipping skill at %s: read failed", skill_md)
                continue
            parsed = _parse_frontmatter(text)
            if parsed is None:
                log.warning("skipping skill at %s: missing frontmatter", skill_md)
                continue
            fields, _body = parsed
            name = fields.get("name", "")
            desc = fields.get("description", "")
            if not name or not desc:
                log.warning("skipping skill at %s: missing name or description", skill_md)
                continue
            if not _NAME_RE.match(name):
                log.warning("skipping skill at %s: invalid name %r", skill_md, name)
                continue
            if name.lower() in RESERVED_NAMES:
                log.warning("skipping skill at %s: reserved name %r", skill_md, name)
                continue
            metas[name] = SkillMeta(name=name, description=desc, path=skill_md.resolve())
    return metas


def render_catalog(metas: dict[str, SkillMeta]) -> str:
    """Render catalog as a system-reminder text block listing each skill."""
    if not metas:
        return "(无可用 skill)"
    lines = [f"- {m.name}: {m.description}" for m in metas.values()]
    return "\n".join(lines)


def read_skill_md(name: str) -> str:
    """Read skill.md full text by name. Project root wins on conflict.

    Raises FileNotFoundError if no skill with this name exists in any root.
    """
    if not _NAME_RE.match(name):
        raise FileNotFoundError(f"skill not found: {name}")
    for root in get_skill_roots():
        candidate = root / name / "skill.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"skill not found: {name}")
