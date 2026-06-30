"""Tests for nanobot.agent.skills.SkillsLoader extra_skills_dirs parameter.

spec: change meeting-content-identification (decision 4a) — project-side skill dirs
discovered alongside workspace_skills and builtin_skills, with the same name-skip
precedence (earlier sources shadow later).
"""

from __future__ import annotations

import json
from pathlib import Path

from nanobot.agent.skills import SkillsLoader


def _write_skill(base: Path, name: str, *, body: str = "# Skill\n") -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "\n".join(["---", f"name: {name}", "---", "", body]),
        encoding="utf-8",
    )
    return path


def test_extra_skills_dir_scanned(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    extra = tmp_path / "project_skills"
    extra.mkdir()
    path = _write_skill(extra, "meeting-suggest", body="# MS")

    loader = SkillsLoader(
        workspace, builtin_skills_dir=tmp_path / "builtin", extra_skills_dirs=[extra]
    )
    entries = loader.list_skills(filter_unavailable=False)
    assert entries == [{"name": "meeting-suggest", "path": str(path), "source": "extra"}]


def test_extra_skills_merges_with_workspace_and_builtin(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    extra = tmp_path / "project_skills"
    extra.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    _write_skill(ws_skills, "ws_only", body="# W")
    _write_skill(extra, "extra_only", body="# E")
    _write_skill(builtin, "bi_only", body="# B")

    loader = SkillsLoader(
        workspace, builtin_skills_dir=builtin, extra_skills_dirs=[extra]
    )
    entries = sorted(loader.list_skills(filter_unavailable=False), key=lambda e: e["name"])
    assert [e["name"] for e in entries] == ["bi_only", "extra_only", "ws_only"]
    assert {e["source"] for e in entries} == {"workspace", "extra", "builtin"}


def test_workspace_shadows_extra_same_name(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    extra = tmp_path / "project_skills"
    extra.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    ws_path = _write_skill(ws_skills, "dup", body="# WS wins")
    _write_skill(extra, "dup", body="# Extra")

    loader = SkillsLoader(
        workspace, builtin_skills_dir=builtin, extra_skills_dirs=[extra]
    )
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["path"] == str(ws_path)
    assert entries[0]["source"] == "workspace"


def test_extra_shadows_builtin_same_name(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    extra = tmp_path / "project_skills"
    extra.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    extra_path = _write_skill(extra, "dup", body="# Extra wins")
    _write_skill(builtin, "dup", body="# Builtin")

    loader = SkillsLoader(
        workspace, builtin_skills_dir=builtin, extra_skills_dirs=[extra]
    )
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["path"] == str(extra_path)
    assert entries[0]["source"] == "extra"


def test_extra_skills_dir_missing_is_silent(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(
        workspace,
        builtin_skills_dir=builtin,
        extra_skills_dirs=[tmp_path / "does_not_exist"],
    )
    assert loader.list_skills(filter_unavailable=False) == []


def test_extra_skills_dir_none_equivalent_to_empty(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader_default = SkillsLoader(workspace, builtin_skills_dir=builtin)
    loader_none = SkillsLoader(
        workspace, builtin_skills_dir=builtin, extra_skills_dirs=None
    )
    assert loader_default.list_skills(filter_unavailable=False) == \
        loader_none.list_skills(filter_unavailable=False)


def test_load_skill_search_order_workspace_then_extra_then_builtin(tmp_path: Path) -> None:
    """load_skill should mirror list_skills precedence: workspace → extra → builtin."""
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    extra = tmp_path / "project_skills"
    extra.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    _write_skill(ws_skills, "in_ws", body="# WS body")
    _write_skill(extra, "in_extra", body="# Extra body")
    _write_skill(builtin, "in_builtin", body="# Builtin body")

    loader = SkillsLoader(
        workspace, builtin_skills_dir=builtin, extra_skills_dirs=[extra]
    )
    assert "WS body" in loader.load_skill("in_ws")
    assert "Extra body" in loader.load_skill("in_extra")
    assert "Builtin body" in loader.load_skill("in_builtin")
    # workspace wins over extra for same name
    _write_skill(ws_skills, "dup", body="# WS body")
    _write_skill(extra, "dup", body="# Extra body")
    assert "WS body" in loader.load_skill("dup")
