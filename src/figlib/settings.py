"""Every FIGLIB_* switch in one place: its default, and where its value came from.

A value comes from, in order:

1. the environment (`FIGLIB_LENS=fisheye python -m ...`);
2. a named profile, a committed file `configs/<name>.toml` selected with `FIGLIB_PROFILE`;
3. a module's own default profile, named by that module's `main()` (`default_profile`);
4. the default below.

Values are read when they are asked for, never written back into `os.environ`, so importing
a module cannot change what another one sees. The run log (`provenance.record`) stores the
whole resolved set, defaults included, with the profile's name and hash.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

# name -> (default, what it does). None means unset.
SETTINGS: dict[str, tuple[str | None, str]] = {
    "FIGLIB_CORPUS": ("core", "which archives and outputs: core, extra, all, recent"),
    "FIGLIB_TIER": (None, "restrict scoring to contamination tiers, comma-separated"),
    "FIGLIB_DETS": (None, "alternative detection directory (default: the corpus's own)"),
    "FIGLIB_CAMS": (None, "alternative camera table (default: data/meta/cams.json)"),
    "FIGLIB_LENS": ("rectilinear", "rectilinear, or fisheye for the star-measured lens"),
    "FIGLIB_POSE_LEDGER": ("0", "1: per-camera star azimuths from the pose ledger"),
    "FIGLIB_POSE_LEDGER_PATH": (None, "alternative pose ledger (default: data/meta/pose_ledger.json)"),
    "FIGLIB_POSE_FULL": ("0", "1: bearings through the whole solved camera (lens, pitch, roll)"),
    "FIGLIB_REFINE_KM": ("0.01", "re-find the likelihood peak at this step off the grid; 0 for grid nodes"),
    "FIGLIB_VARIANTS": (None, "bearing variants to score, comma-separated (default: the box variants)"),
    "FIGLIB_WIND_GATE": ("1", "0: mask fits may lean against the wind"),
    "FIGLIB_BAND_PX": ("100", "terrain_range: allowed rows above the terrain line"),
    "FIGLIB_ALLOW_UNPINNED_MODEL": ("0", "1: run detection on weights other than the pinned ones"),
}

_module_default: str | None = None


def default_profile(name: str) -> None:
    """The profile a module's `main()` runs under when FIGLIB_PROFILE is not set.

    Call it from `main()`, never at import: it applies to the whole process."""
    global _module_default
    _module_default = name


def _profile() -> tuple[str, str, dict[str, str]] | None:
    """(name, source, values) of the active profile, or None."""
    name, source = os.environ.get("FIGLIB_PROFILE") or None, "env"
    if name is None and _module_default:
        name, source = _module_default, "module default"
    if name is None:
        return None
    return name, source, profile_values(name)


def profile_path(name: str) -> Path:
    p = CONFIGS / f"{name}.toml"
    if not p.exists():
        have = sorted(q.stem for q in CONFIGS.glob("*.toml"))
        raise ValueError(f"FIGLIB_PROFILE={name!r}: no {p}; profiles are {have}")
    return p


def profile_values(name: str) -> dict[str, str]:
    raw = tomllib.loads(profile_path(name).read_text())
    bad = sorted(set(raw) - set(SETTINGS))
    if bad:
        raise ValueError(f"configs/{name}.toml sets unknown {bad}; known are {sorted(SETTINGS)}")
    return {k: ("1" if v is True else "0" if v is False else str(v)) for k, v in raw.items()}


def get(name: str) -> str | None:
    """The value of one FIGLIB_* setting: environment, then profile, then default."""
    if name not in SETTINGS:
        raise KeyError(f"{name} is not a setting; add it to settings.SETTINGS")
    v = os.environ.get(name)
    if v:
        return v
    prof = _profile()
    if prof and name in prof[2]:
        return prof[2][name]
    return SETTINGS[name][0]


def flag(name: str) -> bool:
    return get(name) == "1"


def resolved() -> dict:
    """Every setting's value and its source, and the profile: what the run log stores."""
    prof = _profile()
    values, source = {}, {}
    for name, (default, _) in SETTINGS.items():
        values[name] = get(name)
        source[name] = ("env" if os.environ.get(name) else
                        "profile" if prof and name in prof[2] else "default")
    out = {"values": values, "source": source, "profile": None}
    if prof:
        p = profile_path(prof[0])
        out["profile"] = {"name": prof[0], "selected_by": prof[1], "file": str(p.relative_to(ROOT)),
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def main() -> None:
    r = resolved()
    if r["profile"]:
        print(f"profile {r['profile']['name']} ({r['profile']['file']}, {r['profile']['sha256'][:12]})")
    for name, (default, doc) in SETTINGS.items():
        print(f"  {name:28s} {str(r['values'][name]):12s} {r['source'][name]:8s} {doc}")


if __name__ == "__main__":
    main()
