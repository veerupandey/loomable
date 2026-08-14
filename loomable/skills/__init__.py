"""Bundled loomable skills (progressive disclosure via SKILL.md).

Resolve skill names or paths for ``Agent(skills=...)`` /
``create_deep_agent(skills=...)``::

    from loomable.skills import resolve_skills, bundled_skills_root

    paths = resolve_skills(["research"])  # package skill
    paths = resolve_skills([Path("./my_skills")])  # catalog or skill dir
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

__all__ = [
    "bundled_skills_root",
    "list_bundled_skills",
    "resolve_skills",
]


def bundled_skills_root() -> Path:
    """Return the directory that contains packaged skills (e.g. ``research/``)."""
    return Path(__file__).resolve().parent


def list_bundled_skills() -> list[str]:
    """Names of skills shipped inside the loomable package."""
    root = bundled_skills_root()
    names: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            names.append(child.name)
    return names


def resolve_skills(skills: Sequence[str | Path] | None) -> list[Path]:
    """Normalize skill refs to directories the SkillLoader can discover.

    Accepts:
    - bundled names (``\"research\"``)
    - a skill directory containing ``SKILL.md``
    - a catalog directory containing skill subfolders
    - filesystem paths as strings
    """
    if not skills:
        return []
    bundled = bundled_skills_root()
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in skills:
        if isinstance(raw, Path):
            path = raw
        else:
            text = str(raw).strip()
            if not text:
                continue
            candidate = bundled / text
            if candidate.is_dir() and (candidate / "SKILL.md").is_file():
                path = candidate
            else:
                path = Path(text)
        path = path.expanduser()
        try:
            path = path.resolve()
        except OSError:
            pass
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out
