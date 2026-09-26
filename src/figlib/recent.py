"""Extra cameras for recent fires, fetched from HPWREN's public CDN while it still has them.

A FIgLib archive holds the cameras someone annotated, not every camera that saw the fire.
For a fire inside the CDN's public window (~90 days; older frames sit in Glacier Deep
Archive and need an HPWREN staff restore) the rest can still be fetched. That turns a
single-site fire into a triangulable one and adds crossing angles where the geometry was
weak. This module fetches those frames, runs the same detector over them, and scores each
fire from its FIgLib cameras alone, with every extra camera added, and from every subset
of sites in between.

`candidates` chooses the cameras: fixed color Mobotix units not in the archive, within
MAX_KM of the official point, with it inside the field of view, where a plume rising
MAX_RISE_M or less clears the terrain, and whose frames the CDN still serves. The frames are
named like FIgLib's, `<epoch>_<signed offset>.jpg`. The offset is taken from the fire's
FIgLib t0, so a frame here and a frame in the archive with the same offset were taken at
the same moment. The window is FIgLib's, +-40 min.

Corpus `recent` (corpus.py):
    data/hpwren_recent/<fire_id>/<cam>/*.jpg        frames (gitignored)
    data/meta/recent/manifest.json                  source URL, SHA-256, Last-Modified per frame
    out/recent/yolo/<fire_id>_<cam>.json            detections, in detect_yolo's shape
    out/recent/geolocation{_fisheye}{_ledger}.json  scores

Fires, sequences and ground truth come from the `all` corpus, so no FIgLib result moves.
Only the box-center bearings are scored (`center`, `early`), which need no wind lookup.

Data credit: HPWREN, https://www.hpwren.ucsd.edu/

    python -m src.figlib.recent candidates 20260715_ThornFire ...   # -> out/recent/candidates/
    python -m src.figlib.recent fetch 20260629_JunctionFire bi-e-mobo-c mpo-s-mobo-c ...
    python -m src.figlib.recent detect
    python -m src.figlib.recent score     # FIGLIB_LENS and FIGLIB_POSE_LEDGER as in geolocate
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import corpus as C

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "data" / "hpwren_recent"
CORPUS = C.CORPORA["recent"]
MANIFEST = CORPUS.meta / "manifest.json"
CDN = "https://cdn.hpwren.ucsd.edu"
LA = ZoneInfo("America/Los_Angeles")   # Q blocks are local time: Q1 = 00:00-02:59
HALF_S = 2400
VARIANTS = (("center", "best", (0, 2400)), ("early", "earliest", (0, 900)))
MAX_KM = 45.0          # candidates: farther than this, a young plume is a few pixels
MAX_RISE_M = 200.0     # candidates: how far a plume may have to rise to clear the terrain
EDGE_DEG = 1.0         # candidates: a fire in the frame's outermost degree is half cut off


def _get(url: str, tries: int = 3) -> tuple[bytes, str | None]:
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read(), r.headers.get("Last-Modified")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):     # 403: gone to Glacier; 404: camera offline
                raise
            err = exc
        except Exception as exc:
            err = exc
        time.sleep(2 * (k + 1))
    raise err


def _t0(fire_id: str) -> int:
    fires = json.loads((C.CORPORA["all"].meta / "fires.json").read_text())
    return next(f["t0_median"] for f in fires if f["fire_id"] == fire_id)


def _blocks(t0: int) -> list[tuple[str, int]]:
    """The (YYYYMMDD, Q) blocks the +-40 min window touches."""
    out = []
    for t in (t0 - HALF_S, t0, t0 + HALF_S):
        lt = datetime.fromtimestamp(t, UTC).astimezone(LA)
        b = (lt.strftime("%Y%m%d"), lt.hour // 3 + 1)
        if b not in out:
            out.append(b)
    return out


def fetch(fire_id: str, cam: str, prior: dict | None = None) -> dict:
    """Every frame of `cam` within +-40 min of the fire's t0, skipping files on disk.

    `prior` is this camera's earlier manifest entry, whose Last-Modified dates are kept for
    frames already on disk.
    """
    t0 = _t0(fire_id)
    known = {f["file"]: f.get("last_modified") for f in (prior or {}).get("frames", [])}
    rec = {"fire": fire_id, "cam": cam, "t0": t0, "frames": [], "missing": [], "errors": []}
    wanted = set()
    for day, q in _blocks(t0):
        try:
            body, _ = _get(f"{CDN}/hpwren-cameras/{cam}/{day[:4]}/{day}/{day}_{cam}_Q{q}.txt")
        except Exception as exc:
            rec["errors"].append(f"list {day} Q{q}: {exc}")
            continue
        for name in body.decode().split():
            if name.endswith(".jpg") and abs(int(name[:-4]) - t0) <= HALF_S:
                wanted.add((day, q, int(name[:-4])))
    d = FRAMES / fire_id / cam
    d.mkdir(parents=True, exist_ok=True)
    for day, q, ep in sorted(wanted, key=lambda w: w[2]):
        p = d / f"{ep}_{ep - t0:+06d}.jpg"
        url = f"{CDN}/MTA/{cam}/large/{day}/Q{q}/{ep}.jpg"
        lm = known.get(p.name)
        if p.exists() and p.stat().st_size > 0:
            data = p.read_bytes()
        else:
            try:
                data, lm = _get(url)
            except Exception as exc:
                rec["missing"].append([ep, str(exc)[:80]])
                continue
            p.write_bytes(data)
        rec["frames"].append({"file": p.name, "url": url, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest(), "last_modified": lm})
    rec["fetched_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    return rec


def _archive_cameras(fire: dict) -> set[str]:
    return {s.split("#")[0].split("_", 2)[2] for s in fire["sequences"]}


def select(fire: dict, truth: dict, cams: dict, d_az: dict, rise_m) -> list[dict]:
    """Candidate cameras for one fire, before asking the CDN.

    Every fixed color Mobotix (`-mobo-c`) not already in the fire's archive, within MAX_KM of
    the official point and with it inside the frame (published azimuth plus the star `d_az`
    where one exists; a 90 deg unit's frame is the star-measured fisheye's, ~+-53 deg). `rise_m(cam)` gives how far a plume must rise above the ignition
    to clear the terrain; it is only asked of cameras that pass the cheap tests. Returns them
    all with `ok` set where the rise is within MAX_RISE_M, nearest first.
    """
    from .geom import angdiff_deg, bearing_deg, haversine_km
    have = _archive_cameras(fire)
    fish = fisheye_half_deg()
    out = []
    for name, cam in cams.items():
        if not name.endswith("-mobo-c") or name in have or cam.get("fov") is None:
            continue
        km = haversine_km(cam["lat"], cam["lon"], truth["lat"], truth["lon"])
        if km > MAX_KM:
            continue
        off = angdiff_deg(bearing_deg(cam["lat"], cam["lon"], truth["lat"], truth["lon"]),
                          cam["az"] + d_az.get(name, 0.0))
        if abs(off) > (fish if cam["fov"] == 90 else cam["fov"] / 2.0) - EDGE_DEG:
            continue
        rise = rise_m(cam)
        out.append({"camera": name, "km": round(km, 1), "off_axis_deg": round(off, 1),
                    "d_az": d_az.get(name), "rise_m": round(rise), "ok": rise <= MAX_RISE_M})
    return sorted(out, key=lambda r: r["km"])


def fisheye_half_deg() -> float:
    """Half the horizontal field of a 3072 px 90 deg unit under the star-measured lens: the
    off-axis angle that `geom.bearing_x_frac` maps to the frame edge."""
    import math
    from .geom import FISHEYE_K1, FISHEYE_K_RATIO
    lo, hi = 0.0, math.radians(90.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        x = 0.5 + mid * (1 + FISHEYE_K1 * mid * mid) * FISHEYE_K_RATIO / math.radians(90.0)
        lo, hi = (mid, hi) if x < 1.0 else (lo, mid)
    return math.degrees(lo)


def _rise_m(dem, truth: dict):
    """How far above the ground at the official point a plume must be to clear the terrain."""
    import math
    from . import terrain as T

    def rise(cam: dict) -> float:
        g = T.sightline(dem, cam, truth["lat"], truth["lon"])
        return max(0.0, (math.tan(math.radians(g["occ_el"])) - math.tan(math.radians(g["el"])))
                   * g["km"] * 1000.0)
    return rise


def _star_d_az(epoch: float) -> dict[str, float]:
    """Each camera's d_az from its nearest star solve in time, recent ledger first."""
    from . import pose_ledger
    p = CORPUS.out / "pose_ledger.json"
    entries = pose_ledger.load(p if p.exists() else None)
    best: dict[str, dict] = {}
    for e in entries:
        c = e["camera"]
        if c not in best or abs(e["epoch"] - epoch) < abs(best[c]["epoch"] - epoch):
            best[c] = e
    return {c: e["d_az"] for c, e in best.items()}


def cdn_status(cam: str, t0: int) -> dict:
    """Does the CDN still serve this camera's frames around t0? `live`, `403` (gone to
    Glacier), `404` (offline or no such listing), or an error. Fetches one frame to be sure:
    a listing can outlive the frames it names."""
    frames = []
    for day, q in _blocks(t0):
        try:
            body, _ = _get(f"{CDN}/hpwren-cameras/{cam}/{day[:4]}/{day}/{day}_{cam}_Q{q}.txt")
        except urllib.error.HTTPError as exc:
            return {"status": str(exc.code), "n_listed": 0}
        except Exception as exc:
            return {"status": f"error: {str(exc)[:60]}", "n_listed": 0}
        frames += [(day, q, int(n[:-4])) for n in body.decode().split()
                   if n.endswith(".jpg") and abs(int(n[:-4]) - t0) <= HALF_S]
    if not frames:
        return {"status": "no frames listed", "n_listed": 0}
    day, q, ep = min(frames, key=lambda f: abs(f[2] - t0))
    try:
        _get(f"{CDN}/MTA/{cam}/large/{day}/Q{q}/{ep}.jpg", tries=2)
    except urllib.error.HTTPError as exc:
        return {"status": str(exc.code), "n_listed": len(frames)}
    except Exception as exc:
        return {"status": f"error: {str(exc)[:60]}", "n_listed": len(frames)}
    return {"status": "live", "n_listed": len(frames)}


def candidates(fire_id: str) -> Path:
    from .geom import load_cams
    from .terrain import Dem
    meta = C.CORPORA["all"].meta
    fire = next(f for f in json.loads((meta / "fires.json").read_text()) if f["fire_id"] == fire_id)
    truth = next(r for r in json.loads((meta / "resolved.json").read_text())
                 if r["fire_id"] == fire_id)["truth"]
    t0 = fire["t0_median"]
    rows = select(fire, truth, load_cams(), _star_d_az(t0), _rise_m(Dem(), truth))
    ok = [r for r in rows if r["ok"]]
    with ThreadPoolExecutor(4) as pool:
        for r, st in zip(ok, pool.map(lambda r: cdn_status(r["camera"], t0), ok)):
            r.update(st)
    print(f"{fire_id}  archive: {', '.join(sorted(_archive_cameras(fire)))}")
    for r in rows:
        print(f"  {r['camera']:18s} {r['km']:5.1f} km  off-axis {r['off_axis_deg']:+6.1f}  "
              f"rise {r['rise_m']:5d} m  {r.get('status', 'terrain') if r['ok'] else 'terrain'}")
    live = [r["camera"] for r in ok if r.get("status") == "live"]
    print(f"  live: {' '.join(live) or '-'}")
    dest = CORPUS.out / "candidates" / f"{fire_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"fire_id": fire_id, "t0": t0, "truth": truth,
                                "max_km": MAX_KM, "max_rise_m": MAX_RISE_M, "edge_deg": EDGE_DEG,
                                "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                "cameras": rows, "live": live}, indent=1) + "\n")
    return dest


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _dirs() -> list[Path]:
    return sorted(d for d in FRAMES.glob("*/*") if d.is_dir() and any(d.glob("*.jpg")))


def detect() -> tuple[list[Path], dict]:
    """detect_yolo over every fetched camera, one detection file per fire and camera."""
    import cv2
    import numpy as np
    from dataclasses import asdict
    from . import detect_yolo as Y
    from .provenance import check_model

    CORPUS.dets.mkdir(parents=True, exist_ok=True)
    model = check_model(Y.MODEL)
    sess = Y.make_session()
    written = []
    for d in _dirs():
        dest = CORPUS.dets / f"{d.parent.name}_{d.name}.json"
        if dest.exists():
            continue
        recs = []
        for p in sorted(d.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[0])):
            img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            ep, off = (int(x) for x in p.stem.split("_"))
            recs.append(Y.FrameDets(ep, off, [asdict(x) for x in Y.detect(sess, img)]))
        dest.write_text(json.dumps([asdict(r) for r in recs]) + "\n")
        written.append(dest)
        print(f"{dest.stem}: {len(recs)} frames, {sum(len(r.dets) for r in recs)} dets", flush=True)
    return written, model


def _recent_sequences(cams: dict) -> dict[str, dict]:
    from PIL import Image
    out = {}
    for d in _dirs():
        seq = f"{d.parent.name}_{d.name}"
        if not (CORPUS.dets / f"{seq}.json").exists():
            continue
        c = cams.get(d.name) or {}
        with Image.open(next(d.glob("*.jpg"))) as im:
            w, h = im.size
        out[seq] = {"seq": seq, "event": d.parent.name, "camera": d.name,
                    "has_pose": bool(c) and c.get("az") is not None,
                    "dets_dir": str(CORPUS.dets), "frame_w": w, "frame_h": h}
    return out


def _site(camera: str) -> str:
    return camera.split("-")[0]


def _solve(bs, truth) -> dict:
    import numpy as np
    from .geolocate import credible_area_km2, solve
    from .geom import haversine_km
    center = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
    lats, lons, ll, la, lo = solve(bs, center)
    return {"est_lat": round(la, 5), "est_lon": round(lo, 5),
            "error_km": round(haversine_km(la, lo, truth["lat"], truth["lon"]), 2),
            "area95_km2": round(credible_area_km2(lats, lons, ll), 1)}


def score() -> Path:
    from . import pose_ledger
    from . import provenance as P
    from .geolocate import bearings_for_fire
    from .geom import load_cams

    started = P.utc_now()
    meta = C.CORPORA["all"].meta
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((meta / "sequences.json").read_text())}
    for s in seqs.values():
        s["dets_dir"] = str(C.CORPORA["all"].dets)
    fires = {f["fire_id"]: f for f in json.loads((meta / "fires.json").read_text())}
    resolved = {r["fire_id"]: r for r in json.loads((meta / "resolved.json").read_text())}
    recent = _recent_sequences(cams)

    rows = []
    for fid in sorted({s["event"] for s in recent.values()}):
        r, fire = resolved[fid], fires[fid]
        truth = r["truth"]
        extra = sorted(s for s in recent if recent[s]["event"] == fid)
        row = {"fire_id": fid, "tier": r["tier"], "truth_name": truth["name"],
               "truth_lat": truth["lat"], "truth_lon": truth["lon"],
               "figlib_cameras": fire["sequences"], "extra_cameras": [recent[s]["camera"] for s in extra]}
        for tag, pick, win in VARIANTS:
            kw = dict(use_wind=False, pick=pick, window_s=win)
            fb = bearings_for_fire(fire, seqs, cams, **kw)
            eb = bearings_for_fire({"sequences": extra}, recent, cams, **kw)
            out = {"bearings": [{"camera": b.camera, "deg": round(b.bearing_deg, 2), "conf": b.conf,
                                 "x": b.x_frac, "offset_s": b.epoch - fire["t0_median"],
                                 "extra": is_extra, "pose": b.pose}
                                for bs, is_extra in ((fb, False), (eb, True)) for b in bs]}
            for name, bs in (("figlib", fb), ("all", fb + eb)):
                sites = sorted({_site(b.camera) for b in bs})
                out[name] = ({"n_sites": len(sites), "sites": sites, **_solve(bs, truth)}
                             if len(sites) >= 2 else {"n_sites": len(sites), "status": "under 2 sites"})
            # Every subset of the sites that produced a bearing: how error falls as sites
            # are added, independent of which sites FIgLib happened to annotate.
            by_site: dict[str, list] = {}
            for b in fb + eb:
                by_site.setdefault(_site(b.camera), []).append(b)
            subsets = []
            for k in range(2, len(by_site) + 1):
                for combo in itertools.combinations(sorted(by_site), k):
                    s = _solve([b for c in combo for b in by_site[c]], truth)
                    subsets.append({"sites": list(combo), "error_km": s["error_km"],
                                    "area95_km2": s["area95_km2"]})
            out["subsets"] = subsets
            row[tag] = out
        rows.append(row)

    variant = (("_fisheye" if os.environ.get("FIGLIB_LENS") == "fisheye" else "")
               + ("_ledger" if pose_ledger.enabled() else "")
               + ("_full" if pose_ledger.enabled() and pose_ledger.full_enabled() else ""))
    dest = CORPUS.out / f"geolocation{variant}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=1) + "\n")
    P.record("recent-score", [dest], started=started,
             params={"variants": [v[0] for v in VARIANTS], "use_wind": False,
                     "lens": os.environ.get("FIGLIB_LENS", "rectilinear"),
                     "pose_ledger": str(pose_ledger.path()) if pose_ledger.enabled() else None,
                     "pose_full": pose_ledger.enabled() and pose_ledger.full_enabled()},
             extra_inputs=[meta / "sequences.json", meta / "fires.json", meta / "resolved.json",
                           C.CORPORA["all"].dets, CORPUS.dets, MANIFEST]
             + ([pose_ledger.path()] if pose_ledger.enabled() else []))
    report(rows)
    return dest


def report(rows: list[dict]) -> None:
    for tag, _, _ in VARIANTS:
        print(f"\n[{tag} bearing]")
        for r in rows:
            x = r[tag]
            cell = lambda s: (f"{s['error_km']:6.2f} km  {s['area95_km2']:6.1f} km2  {s['n_sites']} sites"
                              if "error_km" in s else f"  --  ({s['status']})")
            print(f"  {r['fire_id']} ({r['tier']})")
            print(f"    FIgLib cameras only   {cell(x['figlib'])}")
            print(f"    + extra cameras       {cell(x['all'])}")
            by_k: dict[int, list] = {}
            for s in x["subsets"]:
                by_k.setdefault(len(s["sites"]), []).append(s)
            for k, ss in sorted(by_k.items()):
                e = sorted(s["error_km"] for s in ss)
                a = sorted(s["area95_km2"] for s in ss)
                print(f"    {k} sites, {len(ss):3d} subsets: median {statistics.median(e):6.2f} km  "
                      f"best {e[0]:5.2f}  worst {e[-1]:6.2f}  <=2 km {sum(v <= 2 for v in e):3d}  "
                      f"median area {statistics.median(a):6.1f} km2")
        for r in rows:
            print(f"\n  {r['fire_id']} bearings ({tag}):")
            for b in r[tag]["bearings"]:
                pose = f"  d_az {b['pose']['d_az']:+.2f} ({b['pose']['rule']})" if b["pose"] else ""
                print(f"    {'+' if b['extra'] else ' '} {b['camera']:18s} {b['deg']:7.2f} deg  "
                      f"conf {b['conf']:.2f}  x {b['x']:.3f}  at {b['offset_s']:+5d} s{pose}")


def main(argv: list[str]) -> None:
    from . import provenance as P
    # The run log goes to the `recent` corpus's metadata, whatever the shell has set.
    os.environ["FIGLIB_CORPUS"] = "recent"
    cmd, args = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "candidates":
        for fire_id in args:
            started = P.utc_now()
            dest = candidates(fire_id)
            P.record("recent-candidates", [dest], started=started,
                     params={"fire": fire_id, "max_km": MAX_KM, "max_rise_m": MAX_RISE_M,
                             "edge_deg": EDGE_DEG})
    elif cmd == "fetch":
        fire_id, cam_ids = args[0], args[1:]
        started = P.utc_now()
        man = _manifest()
        with ThreadPoolExecutor(4) as pool:
            for rec in pool.map(lambda c: fetch(fire_id, c, man.get(f"{fire_id}/{c}")), cam_ids):
                man[f"{rec['fire']}/{rec['cam']}"] = rec
                print(f"{rec['fire']} {rec['cam']:18s} {len(rec['frames'])} frames, "
                      f"{len(rec['missing'])} missing {rec['errors'][:1]}", flush=True)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        P.record("recent-fetch", [MANIFEST], started=started, params={"fire": fire_id, "cameras": cam_ids})
    elif cmd == "detect":
        started = P.utc_now()
        written, model = detect()
        P.record("recent-detect", written, started=started, model=model)
    elif cmd == "score":
        score()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
