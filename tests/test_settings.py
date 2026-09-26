"""The FIGLIB_* settings: environment over profile over default, and nothing set at import.

Before settings.py, four modules ran `os.environ.setdefault` when imported, so a result
depended on which of them had been imported first (NOTES.md, 2026-09-25).
"""

from __future__ import annotations

import importlib
import os

import pytest

from src.figlib import settings as S


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FIGLIB_"):
            monkeypatch.delenv(k)
    monkeypatch.setattr(S, "_module_default", None)


def test_defaults_with_nothing_set():
    assert S.get("FIGLIB_CORPUS") == "core"
    assert S.get("FIGLIB_LENS") == "rectilinear"
    assert not S.flag("FIGLIB_POSE_LEDGER")
    assert S.get("FIGLIB_DETS") is None
    r = S.resolved()
    assert r["profile"] is None
    assert set(r["values"]) == set(S.SETTINGS)
    assert set(r["source"].values()) == {"default"}


def test_profile_then_environment(monkeypatch):
    monkeypatch.setenv("FIGLIB_PROFILE", "calibrated")
    assert S.get("FIGLIB_LENS") == "fisheye" and S.flag("FIGLIB_POSE_LEDGER")
    monkeypatch.setenv("FIGLIB_LENS", "rectilinear")
    r = S.resolved()
    assert r["values"]["FIGLIB_LENS"] == "rectilinear" and r["source"]["FIGLIB_LENS"] == "env"
    assert r["source"]["FIGLIB_POSE_LEDGER"] == "profile"
    assert r["profile"]["name"] == "calibrated" and r["profile"]["selected_by"] == "env"
    assert len(r["profile"]["sha256"]) == 64


def test_module_default_yields_to_named_profile(monkeypatch):
    S.default_profile("calibrated")
    assert S.get("FIGLIB_LENS") == "fisheye"
    assert S.resolved()["profile"]["selected_by"] == "module default"
    monkeypatch.setenv("FIGLIB_PROFILE", "recent")
    assert S.get("FIGLIB_POSE_LEDGER_PATH") == "out/recent/pose_ledger.json"


def test_bad_profiles_fail_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("FIGLIB_PROFILE", "no-such-profile")
    with pytest.raises(ValueError, match="no-such-profile"):
        S.get("FIGLIB_LENS")
    (tmp_path / "typo.toml").write_text('FIGLIB_LENZ = "fisheye"\n')
    monkeypatch.setattr(S, "CONFIGS", tmp_path)
    monkeypatch.setenv("FIGLIB_PROFILE", "typo")
    with pytest.raises(ValueError, match="FIGLIB_LENZ"):
        S.get("FIGLIB_LENS")
    with pytest.raises(KeyError):
        S.get("FIGLIB_LENZ")


def test_every_profile_parses():
    for p in S.CONFIGS.glob("*.toml"):
        assert S.profile_values(p.stem)


@pytest.mark.parametrize("name", ["coverage", "bias", "fig_bearing", "terrain_range"])
def test_importing_sets_nothing(name):
    importlib.reload(importlib.import_module(f"src.figlib.{name}"))
    assert not [k for k in os.environ if k.startswith("FIGLIB_")]
    assert S.get("FIGLIB_LENS") == "rectilinear"


def test_refinement_is_on_unless_set_to_zero(monkeypatch):
    from src.figlib.geolocate import refine_step
    assert refine_step() == 0.01
    monkeypatch.setenv("FIGLIB_REFINE_KM", "0")
    assert refine_step() is None
    monkeypatch.setenv("FIGLIB_REFINE_KM", "0.05")
    assert refine_step() == 0.05
