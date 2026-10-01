"""Per-camera, per-date pose corrections, and the rule for when one applies.

A published HPWREN azimuth is a nameplate number, and cameras get re-aimed: om-s-mobo-c
solved at -10.5 deg off in October 2019 (its color and monochrome units agree) and at
-0.4 deg in June 2024. So a correction is a measurement of one camera on one date, and
applying it to a detection on another date is a claim that the camera didn't move in
between. This module makes that claim explicit, and conservative.

The rules and the file format are star_calibration.ledger's; this module adds the switches
that decide whether geolocation uses them, and `corrected_cam`, which applies them to a
camera. Entries come from star-track solves (stars/solve.py) and live in
data/meta/pose_ledger.json, one per solve:
    {"camera", "epoch", "frame_w", "d_az", "d_pitch", "d_roll", "k_ratio", "k1", "cx", "cy",
     "n_stars", "median_px", "source", "sky_model", "solver"}

`cx`, `cy` are the optical centre the solve assumed, in pixels right and down from the frame's
middle (star_calibration.intrinsics); a pose is only right together with its centre. An entry
without them means (0, 0).

`sky_model` is star_calibration.catalog.model_id() at solve time: the catalog and which corrections
(proper motion, precession, refraction) were applied. Poses solved under different models
differ by up to ~0.3 deg for reasons that have nothing to do with the camera, so `load`
refuses a ledger that mixes them. Entries from before the field existed read as
"legacy: J2000, uncorrected".

Lookup for (camera, epoch), first rule that fires:
  1. same-night: solves within SAME_DAYS -> their median d_az;
  2. bracketed: the nearest solve on each side agree within AGREE_DEG -> their mean (the
     camera held still across the gap as far as two measurements can say). If they
     disagree, it moved somewhere in between -> no correction;
  3. one-sided: the nearest solve, if within MAX_DAYS -> it; otherwise no correction.
A solve never carries across a change of frame format: a 2048x1536 unit replaced by a
3072x2048 one under the same camera name is a new installation, whatever cams.json says.
`corrected_cam` applies d_az and the centre the solve assumed (geom.offset_bearing_deg reads
a column relative to it); pitch, roll and lens reach a bearing only with FIGLIB_POSE_FULL.

Opt in with FIGLIB_POSE_LEDGER=1 (FIGLIB_POSE_LEDGER_PATH to point at another file).
"""

from __future__ import annotations

import json
from pathlib import Path

from star_calibration import ledger as L
from star_calibration.ledger import (AGREE_DEG, DAY_S, LEGACY_MODEL, LENS_KEYS,  # noqa: F401
                                     MAX_DAYS, SAME_DAYS, lookup)

from . import settings

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "meta" / "pose_ledger.json"


def enabled() -> bool:
    return settings.flag("FIGLIB_POSE_LEDGER")


def full_enabled() -> bool:
    """FIGLIB_POSE_FULL=1: bearings go through the whole solved camera -- its own lens, pitch
    and roll, read at the box's foot -- instead of the shared lens along the middle row."""
    return settings.flag("FIGLIB_POSE_FULL")


def path() -> Path:
    return Path(settings.get("FIGLIB_POSE_LEDGER_PATH") or LEDGER)


def load(p: Path | None = None) -> list[dict]:
    """The ledger's entries (FIGLIB_POSE_LEDGER_PATH, or data/meta/pose_ledger.json)."""
    return L.load(p or path())


def corrected_cam(camera: str, cam: dict, epoch: float,
                  entries: list[dict] | None = None) -> tuple[dict, dict | None]:
    """`cam` with the ledger's d_az folded into its azimuth and the solves' optical centre
    (`cx`, `cy`) attached, and the lookup that did it. The d_az is measured from that centre:
    read from the frame's middle instead, a 30 px offset is ~1 deg of bearing.

    Unchanged (and None) when the ledger is off or no rule applies, so callers can pass
    every camera through without special cases.
    """
    if not enabled():
        return cam, None
    hit = lookup(load() if entries is None else entries, camera, epoch, cam.get("frame_w"))
    if hit is None:
        return cam, None
    out = {**cam, "az": cam["az"] + hit["d_az"], "cx": hit.get("cx", 0.0), "cy": hit.get("cy", 0.0)}
    if full_enabled() and all(k in hit for k in LENS_KEYS):
        # the rest of the solved camera, for geom.offset_bearing_deg to read a pixel through
        out["solved"] = {k: hit[k] for k in LENS_KEYS}
    return out, hit


def solved_pose(entry: dict) -> dict:
    """One ledger entry's solved camera: pose, lens and the centre it was solved through."""
    return {k: entry.get(k, 0.0) for k in ("d_az", *LENS_KEYS)}


def centred(cam: dict, pose: dict) -> dict:
    """`cam` with `pose`'s optical centre, for star_calibration.fisheye to project through."""
    return {**cam, "cx": pose.get("cx", 0.0), "cy": pose.get("cy", 0.0)}


def build(solve_summary: list[dict], t0_by_seq: dict[str, float],
          dest: Path = LEDGER) -> list[dict]:
    """One ledger entry per solved star-track sequence (star_calibration.ledger.build)."""
    return L.build(solve_summary, t0_by_seq, dest)


if __name__ == "__main__":
    from star_calibration.hpwren import nights
    sky = ROOT / "out" / "sky" / "data"
    t0 = {s["seq"]: s["t0"] for s in json.loads((ROOT / "data/meta/all/sequences.json").read_text())}
    t0.update({k: v["t0"] for k, v in nights.sequences().items()})
    rows = build(json.loads((sky / "star_tracks" / "solve_summary.json").read_text()), t0)
    print(f"wrote {LEDGER} ({len(rows)} solves, {len({r['camera'] for r in rows})} cameras)")
