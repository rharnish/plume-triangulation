"""Surveyed landmarks as an independent check on the star-solved poses.

Every pose in the ledger comes from stars, and so does the evidence that the star solver's
missing precession biases those poses by about -0.28 deg in azimuth (NOTES.md, 2026-09-24).
This checks both against objects whose positions are known without looking at the sky:

* **FCC Antenna Structure Registration** towers -- surveyed coordinates and heights for every
  registered structure, many with red obstruction lights that show up at night in the very
  frames the stars were solved from, so a day/night change of pose cannot confound it;
* **other HPWREN sites** (cams.json), lit or visible from their neighbours;
* water bodies (OpenStreetMap, (c) OpenStreetMap contributors, ODbL), for a daytime look at
  bm-s-mobo-c's reservoir.

Geometry is exact: WGS84 Earth-centred coordinates rotated into the camera's local
east-north-up frame, which is the frame the star alt/az live in. A flat-earth or spherical
bearing is not good enough here -- over tens of kilometres the spherical azimuth can be off
by a sizeable fraction of the 0.28 deg under test. NAD83 (FCC) and WGS84 differ by ~1 m;
heights are orthometric on both ends, and the geoid (~ -35 m here) varies by a few metres
across 80 km, which moves an elevation angle by arcseconds.

The test is paired: each night block is star-solved twice with the same solver
(`solve.solve_wide`), once under the current sky model and once with precession and
refraction (catalog.set_model), and every matched landmark's azimuth residual is taken
under both. If precession is real, the corrected poses centre the landmarks on zero.
Azimuth barely depends on terrestrial refraction or curvature, so it is the clean axis;
elevation residuals are reported but carry the refraction coefficient's uncertainty.

    python -m src.figlib.landmarks build             # data/meta/landmarks.json
    python -m src.figlib.landmarks lights <seq>...   # static lights per night block (cached)
    python -m src.figlib.landmarks solve             # both-model solves for every block
    python -m src.figlib.landmarks check             # match + residuals -> summary
"""
from __future__ import annotations

import json
import math
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
LANDMARKS = META / "landmarks.json"
EXTERNAL = ROOT / "data" / "external"
FCC_URL = "https://data.fcc.gov/download/pub/uls/complete/r_tower.zip"
OUT = ROOT / "out" / "sky" / "data" / "landmarks"
RANGE_KM = 80.0

from .geom import A_WGS, E2_WGS as E2   # one WGS84 definition, shared with geom.bearing_deg
K_TERRESTRIAL = 0.13        # optical refraction coefficient; only elevation feels it


# ---- geometry ---------------------------------------------------------------------------------

def ecef(lat_deg, lon_deg, h_m):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    n = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h_m) * np.cos(la) * np.cos(lo),
                     (n + h_m) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h_m) * np.sin(la)], axis=-1)


def enu_basis(lat_deg, lon_deg):
    """Rows: east, north, up unit vectors at a geodetic point, in ECEF."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def direction(cam_lat, cam_lon, cam_h, lat, lon, h, refract: bool = True):
    """(az_deg, el_deg, dist_m) from a camera to a point, exact on the ellipsoid. `refract`
    adds the standard terrestrial refraction lift k*d/(2R) to the elevation."""
    d = ecef(lat, lon, h) - ecef(cam_lat, cam_lon, cam_h)
    e, n, u = enu_basis(cam_lat, cam_lon) @ np.atleast_2d(d).T
    dist = np.sqrt(e ** 2 + n ** 2 + u ** 2)
    az = np.degrees(np.arctan2(e, n)) % 360
    el = np.degrees(np.arcsin(u / dist))
    if refract:
        el = el + np.degrees(K_TERRESTRIAL * dist / (2 * 6_371_000.0))
    return az.squeeze()[()], el.squeeze()[()], dist.squeeze()[()]


def dms(deg, minutes, sec, hemi) -> float:
    v = int(deg) + int(minutes) / 60 + float(sec or 0) / 3600
    return -v if hemi in ("S", "W") else v


# ---- landmark catalogue ---------------------------------------------------------------------------

def _sites() -> dict:
    cams = json.loads((META / "cams.json").read_text())
    out = {}
    for name, c in cams.items():
        s = c.get("site")
        if s and s not in out:
            out[s] = {"lat": c["lat"], "lon": c["lon"], "elev": c.get("elev"), "agl": c.get("agl")}
    return out


def _near_any(lat, lon, sites, km=RANGE_KM) -> bool:
    for s in sites.values():
        if math.hypot((lat - s["lat"]) * 111.0, (lon - s["lon"]) * 111.0 * math.cos(math.radians(lat))) <= km:
            return True
    return False


def fcc_towers(sites: dict) -> list[dict]:
    """Constructed, standing FCC-registered structures within RANGE_KM of any HPWREN site.

    CO.dat: `CO|REG|file|reg_no|usi|coord_type|lat d|m|s|N/S|total|lon d|m|s|E/W|total`,
    NAD83. RA.dat (by reg_no): status [8], dismantled date [13], ground elevation [29],
    overall height above ground [30] and AMSL [31] (metres), structure type [32], FAA lighting
    specification [36], painting-and-lighting chapters of AC 70/7460-1 [37], marking/lighting
    code [38] ('1' = none). A structure counts as lit when [37] names chapters, since
    obstruction lighting is what those chapters specify."""
    zp = EXTERNAL / "r_tower.zip"
    if not zp.exists():
        import urllib.request
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FCC_URL, zp)
    z = zipfile.ZipFile(zp)
    co = {}
    for line in z.read("CO.dat").decode("latin-1").splitlines():
        f = line.split("|")
        if len(f) < 16 or f[5] != "T":
            continue
        try:
            lat, lon = dms(f[6], f[7], f[8], f[9]), dms(f[11], f[12], f[13], f[14])
        except ValueError:
            continue
        if _near_any(lat, lon, sites):
            co[f[3]] = (lat, lon)
    out = []
    for line in z.read("RA.dat").decode("latin-1").splitlines():
        f = line.split("|")
        if len(f) < 39 or f[3] not in co or f[8] != "C" or f[13]:
            continue
        try:
            ground, agl, amsl = float(f[29]), float(f[30]), float(f[31])
        except ValueError:
            continue
        pl = f[37].strip()
        lit = bool(pl) and pl.lower() != "none"
        lat, lon = co[f[3]]
        out.append({"id": f"asr:{f[3]}", "source": "fcc_asr", "lat": lat, "lon": lon,
                    "ground_m": ground, "agl_m": agl, "top_amsl_m": amsl, "type": f[32],
                    "lit": lit, "lighting": pl, "city": f[24]})
    return out


def water_osm(bbox=(33.08, -116.83, 33.14, -116.74)) -> list[dict]:
    """Named water polygons in a bounding box, from OpenStreetMap via Overpass."""
    import urllib.parse
    import urllib.request
    q = (f'[out:json][timeout:60];(way["natural"="water"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});'
         f'relation["natural"="water"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););out geom;')
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "plume-triangulation landmark check"})
    js = json.loads(urllib.request.urlopen(req, timeout=90).read())
    out = []
    for el in js["elements"]:
        rings = []
        if el["type"] == "way" and "geometry" in el:
            rings = [el["geometry"]]
        elif el["type"] == "relation":
            rings = [m["geometry"] for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
        for ring in rings:
            pts = [(p["lat"], p["lon"]) for p in ring]
            if len(pts) >= 20:
                out.append({"id": f"osm:{el['type']}/{el['id']}", "source": "osm_water",
                            "name": el.get("tags", {}).get("name"), "outline": pts})
    return out


def build() -> dict:
    sites = _sites()
    towers = fcc_towers(sites)
    hp = [{"id": f"hpwren:{k}", "source": "hpwren", "lat": v["lat"], "lon": v["lon"],
           "ground_m": v["elev"], "agl_m": v["agl"] or 0.0,
           "top_amsl_m": (v["elev"] or 0.0) + (v["agl"] or 0.0)} for k, v in sites.items()]
    try:
        water = water_osm()
    except Exception as exc:          # Overpass is a courtesy service; the towers don't need it
        print(f"  water: {exc}")
        water = []
    doc = {"pulled": {"fcc_asr": date.today().isoformat(), "osm_water": date.today().isoformat(),
                      "hpwren": "data/meta/cams.json"},
           "notes": "FCC ASR (public domain) constructed structures within "
                    f"{RANGE_KM:.0f} km of an HPWREN site; NAD83, heights in metres.",
           "attribution": {"osm_water": "Water outlines (c) OpenStreetMap contributors, available "
                                        "under the Open Database License (ODbL): openstreetmap.org/copyright"},
           "towers": towers, "hpwren": hp, "water": water}
    LANDMARKS.write_text(json.dumps(doc, indent=0) + "\n")
    print(f"{len(towers)} towers ({sum(t['lit'] for t in towers)} lit), {len(hp)} HPWREN sites, "
          f"{len(water)} water outlines -> {LANDMARKS}")
    return doc


def load() -> dict:
    return json.loads(LANDMARKS.read_text())


# ---- line of sight ----------------------------------------------------------------------------

def visible(dem, cam: dict, lat: float, lon: float, h_target: float, margin_deg: float = 0.02) -> bool:
    """Whether a point at height `h_target` clears the terrain between it and the camera
    (`terrain.sightline`: Copernicus DSM, 4/3 earth, the target's own last 150 m ignored)."""
    from . import terrain as T
    g = T.sightline(dem, cam, lat, lon, h_target)
    if g["km"] < 0.3:
        return False
    return bool(g["el"] >= g["occ_el"] - margin_deg)


# ---- static lights ----------------------------------------------------------------------------

def _frames(seq: str):
    from .stars import nights, solve as S
    if seq.startswith("hpwren_"):
        return nights.read_frames(seq)
    from . import corpus as C
    from .detect_yolo import read_frames
    arch = {p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}
    return sorted(read_frames(arch[seq.split("#")[0]]), key=lambda f: f[1])


def static_lights(detections: list[np.ndarray], n_frames: int, radius_px: float = 1.5,
                  min_frac: float = 0.05, max_sd_px: float = 1.0) -> list[dict]:
    """Points that hold still across a block's frames.

    `detections[i]` is an (m, k) array for frame i with columns x, y, amp, then any extra
    per-detection columns (colour). Detections are linked into clusters when within
    radius_px; a cluster is a light if it appears in at least min_frac of the frames (a
    flashing beacon is caught in only some), holds a position sd <= max_sd_px, and never
    appears twice in one frame (which would be two lights, or a star crossing)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    rows = [np.c_[np.full(len(d), i), d] for i, d in enumerate(detections) if len(d)]
    if not rows:
        return []
    X = np.concatenate(rows)
    pairs = cKDTree(X[:, 1:3]).query_pairs(radius_px, output_type="ndarray")
    n = len(X)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    out = []
    order = np.argsort(lab)
    splits = np.split(order, np.flatnonzero(np.diff(lab[order])) + 1)
    need = max(3, int(math.ceil(min_frac * n_frames)))
    for idx in splits:
        fr = X[idx, 0]
        nf = len(np.unique(fr))
        if nf < need or nf != len(fr):
            continue
        sd = float(np.hypot(X[idx, 1].std(), X[idx, 2].std()))
        if sd > max_sd_px:
            continue
        rec = {"x": float(np.median(X[idx, 1])), "y": float(np.median(X[idx, 2])), "sd_px": sd,
               "duty": nf / n_frames, "amp": float(np.median(X[idx, 3]))}
        if X.shape[1] > 4:
            rec["bgr"] = [float(v) for v in np.median(X[idx, 4:7], axis=0)]
        out.append(rec)
    return out


def detect_all(img: np.ndarray, thresh: int = 25, area=(1, 60)) -> np.ndarray:
    """Point sources over the whole frame (tracks.detect_points watches only the sky band),
    with the mean BGR of each one's 3x3 core: x, y, amp, b, g, r."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resid = cv2.subtract(g, cv2.medianBlur(g, 21))
    n, lab, st, cen = cv2.connectedComponentsWithStats((resid > thresh).astype(np.uint8), 8)
    keep = (st[1:, cv2.CC_STAT_AREA] >= area[0]) & (st[1:, cv2.CC_STAT_AREA] <= area[1])
    ids = np.flatnonzero(keep) + 1
    if not len(ids):
        return np.zeros((0, 6))
    c = cen[ids]
    xi = np.clip(np.round(c[:, 0]).astype(int), 1, g.shape[1] - 2)
    yi = np.clip(np.round(c[:, 1]).astype(int), 1, g.shape[0] - 2)
    amp = np.array([resid[lab == i].max() for i in ids]) if len(ids) < 400 else resid[yi, xi]
    bgr = np.mean([img[yi + dy, xi + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    return np.c_[c, amp, bgr]


def lights(seq: str, el_max: float = -8.0) -> dict:
    path = OUT / f"static_lights_{seq}.json"
    if path.exists():
        return json.loads(path.read_text())
    import cv2
    from .stars import solve as S
    from .stars.sun import sun as sun_altaz
    s = S.SEQS[seq]
    dets, n = [], 0
    for epoch, off, blob in _frames(seq):
        if sun_altaz(epoch, s["lat"], s["lon"])[0] > el_max:
            continue
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        dets.append(detect_all(img))
        n += 1
    rec = {"seq": seq, "n_frames": n, "W": W, "H": H, "lights": static_lights(dets, n)}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec) + "\n")
    return rec


# ---- both-model solves --------------------------------------------------------------------------

CUR = dict(proper_motion=False, precession=False, refraction=False)     # the pre-2026-09 solver
PR = dict(proper_motion=False, precession=True, refraction=True)        # as run on 2026-09-25


def _solve_both(seq: str) -> dict:
    from .stars import catalog as SG, solve as S
    out = {}
    for tag, model in (("cur", CUR), ("pr", PR)):
        p = OUT / f"solve_{tag}_{seq}.json"
        if p.exists():
            out[tag] = json.loads(p.read_text())
            continue
        try:
            with SG.using(**model):
                r = S.solve_wide(seq)
        except Exception as exc:
            r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        r["sky_model"] = model
        OUT.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1, default=float) + "\n")
        out[tag] = r
    return out


def blocks() -> list[str]:
    """Every night block behind a ledger entry, plus the whole-night blocks."""
    from .stars import solve as S
    ledger = json.loads((META / "pose_ledger.json").read_text())
    seqs = sorted({e["source"].split(":", 1)[1] for e in ledger if e["source"].startswith("star:")})
    seqs += sorted(s for s in S.SEQS if "_N_" in s)
    return [s for s in seqs if s in S.SEQS]


# ---- matching and residuals -------------------------------------------------------------------

def observed_dir(cam: dict, pose: dict, x: float, y: float, W: int, H: int):
    """Pixel -> (az, el) in the world through a solved pose and lens: pole.pixel_to_cam's
    lens inversion, then the pose's camera axes."""
    from .stars import pole as POLE
    from .stars.fisheye import initial_k
    d = POLE.pixel_to_cam(np.array([x]), np.array([y]), W, H, pose["k_ratio"] * initial_k(cam, W), pose["k1"])[0]
    M = POLE.axes_from_pose(cam, pose["d_az"], pose["d_pitch"], pose["d_roll"])
    w = M.T @ d
    return math.degrees(math.atan2(w[0], w[1])) % 360, math.degrees(math.asin(np.clip(w[2], -1, 1)))


def in_front(cam, pose, az, el, max_off_deg: float = 80.0) -> bool:
    """Within max_off_deg of the boresight. The lens model r = k*theta*(1 + k1*theta^2)
    folds back past ~110 deg, so a direction behind the camera can project into the frame."""
    from .stars import pole as POLE
    b = POLE.axes_from_pose(cam, pose["d_az"], pose["d_pitch"], pose["d_roll"])[2]
    a, e = math.radians(az), math.radians(el)
    v = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    return float(v @ b) > math.cos(math.radians(max_off_deg))


def predict_px(cam, pose, az, el, W, H):
    from .stars.fisheye import initial_k, project_fisheye
    x, y = project_fisheye(cam, np.atleast_1d(az), np.atleast_1d(el), W, H, pose["d_az"], pose["d_pitch"],
                           pose["d_roll"], pose["k_ratio"] * initial_k(cam, W), pose["k1"])
    return float(x[0] * W), float(y[0] * H)


def match(pred, lights_xy, gate_px=25.0, isolation_px=30.0):
    """Index of the light that is this landmark's, or None: the one light within gate_px of
    the prediction, and only if no other light sits within isolation_px of it.

    Isolation is what keeps the test honest. In a field of city lights, "the nearest light"
    is chosen *by* the prediction, so it drifts toward whichever pose made the prediction and
    that pose's residuals shrink for no physical reason. An isolated light is the same light
    under either pose, so both are scored on equal terms."""
    if not len(lights_xy):
        return None
    d = np.hypot(lights_xy[:, 0] - pred[0], lights_xy[:, 1] - pred[1])
    near = np.flatnonzero(d <= gate_px)
    if len(near) != 1:
        return None
    j = int(near[0])
    dj = np.hypot(lights_xy[:, 0] - lights_xy[j, 0], lights_xy[:, 1] - lights_xy[j, 1])
    return j if np.sum(dj <= isolation_px) == 1 else None


def check_block(seq: str, lm: dict, dem) -> list[dict]:
    from .stars import solve as S
    s = S.SEQS[seq]
    cam = S.CAMS[s["camera"]]
    sol = _solve_both(seq)
    if any(sol[t]["status"] != "solved" for t in ("cur", "pr")):
        return []
    L = lights(seq)
    if not L["lights"]:
        return []
    W, H = L["W"], L["H"]
    xy = np.array([[l["x"], l["y"]] for l in L["lights"]])
    bgr = np.array([l.get("bgr", [0, 0, 0]) for l in L["lights"]])
    red = bgr[:, 2] / (0.5 * (bgr[:, 0] + bgr[:, 1]) + 1.0)
    h_cam = cam["elev"] + (cam.get("agl") or 0.0)
    rows = []
    cands = [t for t in lm["towers"] if t["lit"]] + [h for h in lm["hpwren"]
                                                        if abs(h["lat"] - cam["lat"]) + abs(h["lon"] - cam["lon"]) > 1e-4]
    for t in cands:
        az, el, dist = direction(cam["lat"], cam["lon"], h_cam, t["lat"], t["lon"], t["top_amsl_m"])
        if dist > RANGE_KM * 1000 or dist < 300:
            continue
        rec = {"seq": seq, "camera": s["camera"], "id": t["id"], "source": t["source"], "dist_km": dist / 1000,
               "az": float(az), "el": float(el)}
        ok_both = True
        for tag in ("cur", "pr"):
            pose = sol[tag]["pose"]
            if not in_front(cam, pose, az, el):
                ok_both = False
                break
            px, py = predict_px(cam, pose, az, el, W, H)
            if not (0 <= px < W and 70 <= py < H):
                ok_both = False
                break
            rec[f"pred_{tag}"] = [px, py]
        if not ok_both or not visible(dem, cam, t["lat"], t["lon"], t["top_amsl_m"]):
            continue
        # one match, predicted halfway between the two poses, so neither is favoured
        j = match(0.5 * (np.array(rec["pred_cur"]) + np.array(rec["pred_pr"])), xy)
        if j is None:
            continue
        rec.update(light=L["lights"][j], redness=float(red[j]))
        for tag in ("cur", "pr"):
            oaz, oel = observed_dir(cam, sol[tag]["pose"], xy[j, 0], xy[j, 1], W, H)
            rec[f"daz_{tag}"] = ((oaz - az + 180) % 360 - 180) * math.cos(math.radians(el))
            rec[f"del_{tag}"] = oel - el
        rows.append(rec)
    return rows


def check() -> dict:
    from . import terrain as T
    from .stars import solve as S
    lm = load()
    dem = T.Dem()
    seqs = blocks()
    # cameras one at a time keeps the DEM window cache warm
    seqs.sort(key=lambda q: S.SEQS[q]["camera"])
    rows = []
    for q in seqs:
        t = time.time()
        r = check_block(q, lm, dem)
        rows += r
        print(f"  {q}: {len(r)} matched  ({time.time() - t:.0f} s)", flush=True)
    out = {"rows": rows, "n_blocks": len(seqs)}
    (OUT / "check.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def _prepare(seq: str):
    lights(seq)
    _solve_both(seq)
    return seq


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build()
    elif cmd == "lights":
        for q in sys.argv[2:]:
            L = lights(q)
            print(q, L["n_frames"], "frames,", len(L["lights"]), "static lights")
    elif cmd == "prepare":
        from multiprocessing import Pool
        qs = blocks()
        with Pool(4) as pool:
            for i, q in enumerate(pool.imap_unordered(_prepare, qs), 1):
                print(f"  [{i}/{len(qs)}] {q}", flush=True)
    elif cmd == "check":
        out = check()
        print(f"{len(out['rows'])} matches over {out['n_blocks']} blocks")


# ---- figures ----------------------------------------------------------------------------------

def figure_block(seq: str, rows: list[dict], dest: Path) -> Path:
    """The horizon band of one night block -- a max-projection over its dark frames, so every
    flashing beacon shows -- with each matched landmark's predicted position under the current
    pose (circle) and the precession + refraction pose (square), and the light it matched."""
    import cv2
    from .stars import solve as S
    from .stars.sun import sun as sun_altaz
    s = S.SEQS[seq]
    stack = None
    for epoch, off, blob in _frames(seq):
        if sun_altaz(epoch, s["lat"], s["lon"])[0] > -8:
            continue
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        stack = img if stack is None else np.maximum(stack, img)
    H, W = stack.shape[:2]
    ys = [r["light"]["y"] for r in rows] or [H / 2]
    y0 = int(max(0, min(ys) - 180)); y1 = int(min(H, max(ys) + 180))
    band = cv2.convertScaleAbs(stack[y0:y1], alpha=1.6, beta=0)
    CUR, PR, LIGHT = (0, 200, 255), (255, 0, 255), (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in rows:
        lx, ly = r["light"]["x"], r["light"]["y"] - y0
        cx, cy = r["pred_cur"][0], r["pred_cur"][1] - y0
        px, py = r["pred_pr"][0], r["pred_pr"][1] - y0
        cv2.circle(band, (int(cx), int(cy)), 9, CUR, 1, cv2.LINE_AA)
        cv2.rectangle(band, (int(px) - 8, int(py) - 8), (int(px) + 8, int(py) + 8), PR, 1, cv2.LINE_AA)
        cv2.drawMarker(band, (int(lx), int(ly)), LIGHT, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        lab = f"{r['id'].split(':')[1]} {r['dist_km']:.0f}km  cur {r['daz_cur']:+.2f}  p+r {r['daz_pr']:+.2f} deg"
        cv2.putText(band, lab, (int(lx) + 12, int(ly) - 14), font, 0.5, LIGHT, 1, cv2.LINE_AA)
    head = np.zeros((70, W, 3), np.uint8)
    cv2.putText(head, f"{seq}: landmark lights (max over the night block)", (12, 26), font, 0.8, LIGHT, 1, cv2.LINE_AA)
    cv2.putText(head, "+ observed light   o predicted, current pose   [] predicted, precession + refraction pose   "
                "(az residual x cos el)", (12, 56), font, 0.6, LIGHT, 1, cv2.LINE_AA)
    out = np.vstack([head, band])
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), out)
    return dest


# ---- candidates for a hand-verified list ----------------------------------------------------

def _maxproj(seq: str):
    import cv2
    from .stars import solve as S
    from .stars.sun import sun as sun_altaz
    s = S.SEQS[seq]
    stack = None
    for epoch, off, blob in _frames(seq):
        if sun_altaz(epoch, s["lat"], s["lon"])[0] > -8:
            continue
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        stack = img if stack is None else np.maximum(stack, img)
    return stack


def candidates(seqs: list[str], pose_tag: str = "pr", near_px: float = 40.0) -> list[dict]:
    """Every lit tower / HPWREN site that is in frame, in line of sight, and has at least one
    static light within near_px of where the pose puts it -- the pool a person (or a careful
    look at the crop) confirms or rejects one by one."""
    from . import terrain as T
    from .stars import solve as S
    lm, dem = load(), T.Dem()
    out = []
    for seq in seqs:
        s = S.SEQS[seq]
        cam = S.CAMS[s["camera"]]
        sol = _solve_both(seq)[pose_tag]
        if sol["status"] != "solved":
            continue
        L = lights(seq)
        if not L["lights"]:
            continue
        W, H = L["W"], L["H"]
        xy = np.array([[l["x"], l["y"]] for l in L["lights"]])
        h_cam = cam["elev"] + (cam.get("agl") or 0.0)
        for t in [t for t in lm["towers"] if t["lit"]] + lm["hpwren"]:
            az, el, dist = direction(cam["lat"], cam["lon"], h_cam, t["lat"], t["lon"], t["top_amsl_m"])
            if not (300 < dist < RANGE_KM * 1000) or not in_front(cam, sol["pose"], az, el):
                continue
            px, py = predict_px(cam, sol["pose"], az, el, W, H)
            if not (0 <= px < W and 70 <= py < H):
                continue
            d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
            if d.min() > near_px or not visible(dem, cam, t["lat"], t["lon"], t["top_amsl_m"]):
                continue
            out.append({"seq": seq, "camera": s["camera"], "id": t["id"], "source": t["source"],
                        "type": t.get("type"), "dist_km": float(dist / 1000), "az": float(az), "el": float(el),
                        "pred": [px, py], "lights": [L["lights"][i] for i in np.flatnonzero(d <= near_px)]})
    return out


def contact_sheet(cands: list[dict], dest: Path, per_row: int = 4, half: int = 60, zoom: int = 3):
    """One tile per candidate: the max-projection around the prediction, zoomed, with the
    prediction (cyan circle, 0.28 deg radius for scale) and each nearby static light numbered."""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    tiles, cache = [], {}
    for k, c in enumerate(cands):
        if c["seq"] not in cache:
            cache.clear()
            cache[c["seq"]] = _maxproj(c["seq"])
        img = cache[c["seq"]]
        H, W = img.shape[:2]
        px, py = c["pred"]
        x0, y0 = int(px) - half, int(py) - half
        crop = np.zeros((2 * half, 2 * half, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * half, W), min(y0 + 2 * half, H)
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        t = cv2.resize(cv2.convertScaleAbs(crop, alpha=1.8), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        P = lambda x, y: (int((x - x0) * zoom), int((y - y0) * zoom))
        cv2.circle(t, P(px, py), 7 * zoom, (255, 220, 80), 1, cv2.LINE_AA)
        for i, l in enumerate(c["lights"]):
            q = P(l["x"], l["y"])
            cv2.circle(t, q, 3 * zoom, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(t, str(i), (q[0] + 10, q[1] - 6), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        bar = np.zeros((40, t.shape[1], 3), np.uint8)
        cv2.putText(bar, f"#{k} {c['camera']} {c['id']} {c['type'] or ''}", (4, 15), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"{c['dist_km']:.0f} km  az {c['az']:.1f}  {c['seq'][-24:]}", (4, 33), font, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows = [np.hstack(tiles[i:i + per_row] + [np.zeros((th, tw, 3), np.uint8)] * (per_row - len(tiles[i:i + per_row])))
            for i in range(0, len(tiles), per_row)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.vstack(rows))
    return dest
