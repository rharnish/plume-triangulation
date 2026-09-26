"""Which archives a run reads, where its results go, and what the detector may have seen.

The published numbers come from the 189 archives `data/fetch.sh` pulls. The remaining
FIgLib archives are fetched separately, and their mere presence on disk must never change
a published number. So every pipeline stage resolves its paths here, from one variable:

| `FIGLIB_CORPUS` | archives | metadata | detections | results |
|---|---|---|---|---|
| `core` (default) | `data/tgz` | `data/meta` | `out/yolo` | `out` |
| `extra` | `out/figlib_extra/tgz` | `data/meta/extra` | `out/extra/yolo` | `out/extra` |
| `all` | both of the above | `data/meta/all` | `out/all/yolo` | `out/all` |
| `recent` | none: CDN frames, see recent.py | `data/meta/recent` | `out/recent/yolo` | `out/recent` |

Unset means `core`, so every existing command reads and writes exactly what it did
before. `out/all/yolo` holds links to the other two detection directories rather than a
second detection pass (`python -m src.figlib.corpus link`). The camera table and the
WFIGS and wind caches are shared across corpora and stay in `data/meta`.

`FIGLIB_TIER` restricts scoring to sequences by what the detector could have trained on;
see `contamination`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import settings

ROOT = Path(__file__).resolve().parents[2]
SHARED_META = ROOT / "data" / "meta"

# pyronear's FIgLib training source, `data/raw/FIGLIB_ANNOTATED_RESIZED.dvc` in
# earthtoolsmaker/pyro-dataset, has exactly one commit, on this date, never updated. A
# fire after it cannot be in that snapshot; a fire on or before it may be.
FIGLIB_SNAPSHOT = "20250414"
# `yolo11s_rapid-raccoon_v8.1.0` was created on Hugging Face on this date. Nothing after
# it can have been trained on, from any source.
MODEL_RELEASE = "20260525"

TIERS = ("possibly_seen", "likely_unseen", "unseen")


@dataclass(frozen=True)
class Corpus:
    name: str
    tgz_dirs: tuple[Path, ...]
    meta: Path
    dets: Path
    out: Path


CORPORA = {
    "core": Corpus("core", (ROOT / "data" / "tgz",), SHARED_META,
                   ROOT / "out" / "yolo", ROOT / "out"),
    "extra": Corpus("extra", (ROOT / "out" / "figlib_extra" / "tgz",),
                    SHARED_META / "extra", ROOT / "out" / "extra" / "yolo",
                    ROOT / "out" / "extra"),
    "all": Corpus("all", (ROOT / "data" / "tgz", ROOT / "out" / "figlib_extra" / "tgz"),
                  SHARED_META / "all", ROOT / "out" / "all" / "yolo", ROOT / "out" / "all"),
    # Extra cameras for recent fires, fetched from the CDN as frames rather than archives
    # (recent.py). Scored against the `all` corpus's fires and ground truth.
    "recent": Corpus("recent", (), SHARED_META / "recent", ROOT / "out" / "recent" / "yolo",
                     ROOT / "out" / "recent"),
}


def current() -> Corpus:
    name = settings.get("FIGLIB_CORPUS")
    if name not in CORPORA:
        raise ValueError(f"FIGLIB_CORPUS={name!r}; expected one of {sorted(CORPORA)}")
    return CORPORA[name]


def tgz_paths(corpus: Corpus | None = None) -> list[Path]:
    """Every archive in the corpus, by name. A name in two directories is an error, not a
    silent preference for one copy."""
    corpus = corpus or current()
    seen: dict[str, Path] = {}
    for d in corpus.tgz_dirs:
        for p in sorted(d.glob("*.tgz")):
            if p.name in seen:
                raise ValueError(f"{p.name} is in both {seen[p.name].parent} and {d}")
            seen[p.name] = p
    return [seen[k] for k in sorted(seen)]


def contamination(name: str) -> str:
    """What pyronear v8.1.0 could have trained on, from the fire date in a sequence or
    fire name (`20240701_Kitchenfire...`).

    * `possibly_seen` -- on or before the FIgLib snapshot. Not proven to be in training:
      pyro-dataset's `filter_data_figlib_smoke` drops FIgLib background frames, so these
      sequences' negatives may be unseen even where their smoke frames were not.
    * `likely_unseen` -- after the snapshot, before the model existed. Unseen unless
      pyronear added FIgLib from another route, which nothing found suggests.
    * `unseen` -- after the model was released.
    """
    date = name[:8]
    if not date.isdigit():
        raise ValueError(f"no leading YYYYMMDD in {name!r}")
    if date <= FIGLIB_SNAPSHOT:
        return "possibly_seen"
    if date <= MODEL_RELEASE:
        return "likely_unseen"
    return "unseen"


def tier_filter() -> tuple[str, ...] | None:
    """Tiers named in FIGLIB_TIER, or None to score everything."""
    raw = settings.get("FIGLIB_TIER")
    if not raw:
        return None
    tiers = tuple(t.strip() for t in raw.split(",") if t.strip())
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise ValueError(f"FIGLIB_TIER has {bad}; expected from {TIERS}")
    return tiers


def in_tier(name: str, tiers: tuple[str, ...] | None) -> bool:
    return tiers is None or contamination(name) in tiers


def tier_suffix(tiers: tuple[str, ...] | None) -> str:
    """File-name suffix for a tier-restricted result, empty when unrestricted."""
    return "" if tiers is None else "_" + "+".join(tiers)


def link_all() -> int:
    """Populate `out/all/yolo` with relative links to the core and extra detections."""
    dest = CORPORA["all"].dets
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in (CORPORA["core"].dets, CORPORA["extra"].dets):
        for p in sorted(src.glob("*.json")):
            link = dest / p.name
            target = Path(os.path.relpath(p, dest))
            if link.is_symlink() and Path(os.readlink(link)) == target:
                continue
            if link.exists() or link.is_symlink():
                raise ValueError(f"{link} already exists and does not point at {p}")
            link.symlink_to(target)
            n += 1
    return n


def main(argv: list[str]) -> None:
    if argv[:1] == ["link"]:
        print(f"linked {link_all()} detection files into {CORPORA['all'].dets}")
    elif argv[:1] == ["stems"]:
        for p in tgz_paths():
            print(p.name[:-4])
    else:
        c = current()
        print(f"corpus {c.name}: {len(tgz_paths(c))} archives in "
              f"{', '.join(str(d.relative_to(ROOT)) for d in c.tgz_dirs)}")
        print(f"  metadata {c.meta.relative_to(ROOT)}  detections "
              f"{c.dets.relative_to(ROOT)}  results {c.out.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
