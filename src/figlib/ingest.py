"""Build a manifest of FIgLib sequences from the downloaded archives.

FIgLib frames are named `<epoch>_<signed offset>.jpg`, where the offset is seconds
from the moment the plume became visible *to that camera*. Within a sequence
`epoch - offset` is constant, so each sequence carries its own t0.

Cameras watching the same fire do NOT agree on t0 -- spreads of a few minutes are
normal, since each view was annotated separately and a plume clears one horizon
before another. Absolute epoch is therefore the only clock that survives across
cameras, and everything downstream keys off it.
"""

from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path

from . import corpus as C

ROOT = Path(__file__).resolve().parents[2]
TGZ_DIR = ROOT / "data" / "tgz"
META_DIR = ROOT / "data" / "meta"      # camera table; outputs follow FIGLIB_CORPUS

FRAME_RE = re.compile(r"(?P<epoch>\d{9,11})_(?P<offset>[+-]\d+)\.jpg$")
# Fallback split when a camera id is absent from cams.json (retired hardware).
SEQ_RE = re.compile(r"^(?P<event>\d{8}_.+?)_(?P<camera>(?:[a-z0-9]+-)+[a-z0-9]+)$")


@dataclass
class Sequence:
    seq: str
    event: str
    camera: str
    n_frames: int
    t0: int                 # epoch at which the plume became visible to this camera
    first_epoch: int
    last_epoch: int
    min_offset: int
    max_offset: int
    n_pre: int              # frames before plume appearance -- the hard negatives
    n_post: int
    median_dt: float | None  # nominal frame spacing, seconds
    has_pose: bool
    site: str | None = None
    lat: float | None = None
    lon: float | None = None
    elev: float | None = None
    az: float | None = None
    fov: float | None = None
    imager: str | None = None


def load_cams() -> dict:
    return json.loads((META_DIR / "cams.json").read_text())


def split_seq_name(name: str, cams: dict) -> tuple[str, str]:
    """Split `20190924_FIRE_bl-s-mobo-c` into event and camera id.

    Prefer matching a known camera id as a suffix -- event names themselves contain
    hyphens and underscores (`20201202_WillowFire-nightime-near-CDF-HQ`), so a purely
    positional split is not safe.
    """
    for cam in cams:
        if name.endswith("_" + cam):
            return name[: -(len(cam) + 1)], cam
    # Two FIgLib archives join event and camera with a hyphen instead
    # (`20190814_FIRE-pi-s-mobo-c`). Tried only after every underscore split fails, so
    # no name that parsed before parses differently now.
    for cam in sorted(cams, key=len, reverse=True):
        if name.endswith("-" + cam) and "_" in name[: -(len(cam) + 1)]:
            return name[: -(len(cam) + 1)], cam
    m = SEQ_RE.match(name)
    if m:
        return m.group("event"), m.group("camera")
    raise ValueError(f"cannot split sequence name: {name}")


def read_archive(path: Path, cams: dict) -> list[Sequence]:
    """Parse one archive into one Sequence per distinct t0.

    A handful of archives bundle two annotation passes over the same camera and
    fire (`20250823_Clubfire_starr-n-mobo-c` carries two t0 six minutes apart).
    Splitting rather than dropping keeps the frames; the two halves cluster back
    together into a single fire downstream.
    """
    seq = path.name[: -len(".tgz")]
    event, camera = split_seq_name(seq, cams)

    frames: list[tuple[int, int]] = []
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            m = FRAME_RE.search(member.name)
            if m:
                frames.append((int(m.group("epoch")), int(m.group("offset"))))
    if not frames:
        raise ValueError(f"no frames parsed in {path.name}")

    groups: dict[int, list[tuple[int, int]]] = {}
    for epoch, offset in frames:
        groups.setdefault(epoch - offset, []).append((epoch, offset))

    cam = cams.get(camera)
    out: list[Sequence] = []
    for i, (t0, group) in enumerate(sorted(groups.items())):
        group.sort()
        epochs = [e for e, _ in group]
        offsets = [o for _, o in group]
        diffs = sorted(b - a for a, b in zip(epochs, epochs[1:]))
        out.append(Sequence(
            seq=seq if len(groups) == 1 else f"{seq}#{i + 1}",
            event=event, camera=camera, n_frames=len(group), t0=t0,
            first_epoch=epochs[0], last_epoch=epochs[-1],
            min_offset=min(offsets), max_offset=max(offsets),
            n_pre=sum(1 for o in offsets if o < 0),
            n_post=sum(1 for o in offsets if o >= 0),
            median_dt=float(diffs[len(diffs) // 2]) if diffs else None,
            has_pose=cam is not None,
            site=cam and cam["site"], lat=cam and cam["lat"], lon=cam and cam["lon"],
            elev=cam and cam["elev"], az=cam and cam["az"], fov=cam and cam["fov"],
            imager=cam and cam["imager"],
        ))
    return out


def build(tgz_dir: Path | None = None, errors: list[str] | None = None) -> list[Sequence]:
    """Parse every archive in `tgz_dir`, or in the current corpus when it is omitted."""
    cams = load_cams()
    out = []
    errors = [] if errors is None else errors
    paths = sorted(tgz_dir.glob("*.tgz")) if tgz_dir is not None else C.tgz_paths()
    for path in paths:
        try:
            out.extend(read_archive(path, cams))
        except (ValueError, tarfile.TarError) as exc:
            errors.append(f"{path.name}: {exc}")
    for e in errors:
        print(f"  SKIP {e}")
    return out


def main() -> None:
    from . import provenance as P
    started = P.utc_now()
    skipped: list[str] = []
    seqs = build(errors=skipped)
    dest = C.current().meta / "sequences.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps([asdict(s) for s in seqs], indent=1) + "\n")

    events = {s.event for s in seqs}
    posed = [s for s in seqs if s.has_pose]
    multi = {e for e in events
             if len({s.site for s in posed if s.event == e and s.site}) >= 2}
    print(f"\n{len(seqs)} sequences, {len(events)} events -> {dest.relative_to(ROOT)}")
    print(f"pose resolved: {len(posed)}/{len(seqs)} sequences")
    print(f"events with >=2 distinct posed sites: {len(multi)}")
    print(f"frames: {sum(s.n_frames for s in seqs)} "
          f"({sum(s.n_pre for s in seqs)} pre-ignition negatives)")

    spreads = []
    for e in events:
        t0s = [s.t0 for s in seqs if s.event == e]
        if len(t0s) > 1:
            spreads.append(max(t0s) - min(t0s))
    if spreads:
        spreads.sort()
        print(f"per-camera t0 disagreement across {len(spreads)} multi-cam events: "
              f"median {spreads[len(spreads)//2]}s, max {spreads[-1]}s")
    P.record("ingest", [dest], params={"skipped": skipped}, started=started)


if __name__ == "__main__":
    main()
