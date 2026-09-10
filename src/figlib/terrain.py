"""Synthesise the horizon a camera should see, from terrain.

HPWREN publishes each camera's position, height, azimuth and field of view, but nothing
that says whether those numbers are right. Terrain does. Marching a ray out from the
camera along every azimuth in its field of view and taking the highest elevation angle
reached gives the skyline the camera *ought* to record, and that predicted skyline can
be laid over the frame it actually recorded.

What the comparison buys:

* **Pose check.** A horizon that sits too high or low is a pitch error; one shifted left
  or right is an azimuth error; one that tilts is roll.
* **Distortion check.** A predicted skyline that matches at frame centre but diverges
  toward the edges is barrel distortion, and that is the error which matters most here,
  because it corrupts the pixel-to-bearing conversion exactly where detections often sit.
* **Range.** The march already knows how far along each ray the skyline was reached, so
  a detection on the horizon carries an implied distance -- the beginning of geolocating
  from a single camera rather than needing two.

Earth curvature and standard atmospheric refraction are both applied, via the usual
four-thirds effective Earth radius. Over 60 km the drop is roughly 240 m, which is the
difference between a ridge being visible and being hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEM_DIR = ROOT / "data" / "dem"

EARTH_R_M = 6_371_000.0
K_REFRACTION = 4.0 / 3.0          # effective radius multiplier, standard atmosphere
R_EFF = EARTH_R_M * K_REFRACTION


@dataclass
class HorizonProfile:
    az_deg: np.ndarray        # azimuths sampled, degrees clockwise from north
    elev_deg: np.ndarray      # apparent elevation angle of the skyline, degrees
    range_km: np.ndarray      # distance at which the skyline is reached
    peak_m: np.ndarray        # terrain height there


class Dem:
    """Lazily mosaics the Copernicus GLO-30 tiles overlapping a requested area."""

    def __init__(self, dem_dir: Path = DEM_DIR):
        self.dir = dem_dir
        self._cache: dict[tuple, tuple[np.ndarray, object]] = {}

    def window(self, lat0: float, lon0: float, pad_deg: float):
        import rasterio
        from rasterio.merge import merge
        key = (round(lat0, 2), round(lon0, 2), round(pad_deg, 2))
        if key in self._cache:
            return self._cache[key]

        bounds = (lon0 - pad_deg, lat0 - pad_deg, lon0 + pad_deg, lat0 + pad_deg)
        srcs = []
        for la in range(math.floor(bounds[1]), math.floor(bounds[3]) + 1):
            for lo in range(math.floor(bounds[0]), math.floor(bounds[2]) + 1):
                ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
                f = self.dir / (f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_"
                                f"{ew}{abs(lo):03d}_00_DEM.tif")
                if f.exists():
                    srcs.append(rasterio.open(f))
        if not srcs:
            raise FileNotFoundError(f"no DEM tiles cover {lat0},{lon0}")
        arr, transform = merge(srcs, bounds=bounds)
        for s in srcs:
            s.close()
        band = arr[0].astype(np.float32)
        band[band < -1000] = 0.0                     # sea / nodata
        self._cache.clear()                          # one camera at a time; keep memory flat
        self._cache[key] = (band, transform)
        return band, transform

    @staticmethod
    def sample(band: np.ndarray, transform, lats: np.ndarray, lons: np.ndarray):
        inv = ~transform
        cols, rows = inv * (lons, lats)
        r = np.clip(rows, 0, band.shape[0] - 1.001)
        c = np.clip(cols, 0, band.shape[1] - 1.001)
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        v = (band[r0, c0] * (1 - fr) * (1 - fc) + band[r0 + 1, c0] * fr * (1 - fc)
             + band[r0, c0 + 1] * (1 - fr) * fc + band[r0 + 1, c0 + 1] * fr * fc)
        return v


def horizon(cam: dict, dem: Dem, half_fov_pad: float = 8.0,
            step_deg: float = 0.2, max_km: float = 80.0,
            step_m: float = 60.0) -> HorizonProfile:
    """March rays across the camera's field of view and record the skyline."""
    half = cam["fov"] / 2.0 + half_fov_pad
    az = np.arange(cam["az"] - half, cam["az"] + half + 1e-9, step_deg)
    d_m = np.arange(step_m, max_km * 1000.0, step_m)

    lat0, lon0 = cam["lat"], cam["lon"]
    h_cam = cam["elev"] + (cam.get("agl") or 0.0)

    band, transform = dem.window(lat0, lon0, max_km / 111.0 + 0.05)

    # Local flat approximation: over 80 km the along-ray geodesic error is well under a
    # DEM pixel, and curvature is handled separately in the elevation term.
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    A = np.radians(az)[:, None]
    D = d_m[None, :]
    lats = lat0 + (D * np.cos(A)) / m_per_deg_lat
    lons = lon0 + (D * np.sin(A)) / m_per_deg_lon

    h = Dem.sample(band, transform, lats.ravel(), lons.ravel()).reshape(lats.shape)

    drop = D ** 2 / (2.0 * R_EFF)                    # curvature + refraction
    ang = np.degrees(np.arctan2(h - h_cam - drop, D))

    i = np.argmax(ang, axis=1)
    j = np.arange(ang.shape[0])
    return HorizonProfile(az_deg=az, elev_deg=ang[j, i],
                          range_km=d_m[i] / 1000.0, peak_m=h[j, i])


def vfov_deg(cam: dict, width: int, height: int) -> float:
    """Vertical field of view implied by the horizontal one under a rectilinear lens."""
    half_h = math.radians(cam["fov"] / 2.0)
    return 2.0 * math.degrees(math.atan(math.tan(half_h) * height / width))


def project(cam: dict, az_deg: np.ndarray, elev_deg: np.ndarray,
            width: int, height: int, pitch_deg: float = 0.0,
            roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Rectilinear projection of (azimuth, elevation) into fractional image coords."""
    half_h = math.radians(cam["fov"] / 2.0)
    half_v = math.radians(vfov_deg(cam, width, height) / 2.0)

    d_az = np.radians((az_deg - (cam["az"] + cam.get("yaw", 0.0)) + 180) % 360 - 180)
    d_el = np.radians(elev_deg - pitch_deg - (cam.get("pitch") or 0.0))

    x = 0.5 + np.tan(d_az) / (2 * np.tan(half_h))
    y = 0.5 - np.tan(d_el) / (2 * np.tan(half_v)) / np.cos(d_az)

    r = math.radians(roll_deg + (cam.get("roll") or 0.0))
    if r:
        xc, yc = x - 0.5, (y - 0.5) * height / width
        x = 0.5 + (xc * math.cos(r) - yc * math.sin(r))
        y = 0.5 + (xc * math.sin(r) + yc * math.cos(r)) * width / height
    return x, y


def prominent_peaks(prof: "HorizonProfile", min_prominence_deg: float = 0.25,
                    max_peaks: int = 12) -> list[int]:
    """Indices of summits worth calling summits, by topographic prominence.

    Taking every local maximum of the elevation profile returns twenty-odd "peaks" on a
    single ridge, most of them a tenth of a degree of noise on a smooth crest -- crowded,
    unmatchable, and useless for pinning azimuth. Prominence is the standard fix and the
    right one here: how far a summit stands above the highest saddle connecting it to
    anything taller. It keeps the handful of features a person would actually point at.
    """
    e = np.asarray(prof.elev_deg, dtype=float)
    n = len(e)
    cand = [i for i in range(1, n - 1) if e[i] >= e[i - 1] and e[i] > e[i + 1]]
    out = []
    for i in cand:
        key = e[i]
        left = i
        while left > 0 and e[left - 1] <= key:
            left -= 1
        right = i
        while right < n - 1 and e[right + 1] <= key:
            right += 1
        # Saddle on each side: the lowest point before reaching higher ground. Where the
        # profile never rises again, the frame edge bounds it.
        lo_l = e[left:i + 1].min() if left < i else e[i]
        lo_r = e[i:right + 1].min() if right > i else e[i]
        prom = key - max(lo_l, lo_r)
        if prom >= min_prominence_deg:
            out.append((prom, i))
    out.sort(reverse=True)
    return sorted(i for _p, i in out[:max_peaks])


@dataclass
class RidgeField:
    """Every terrain silhouette a camera can see, not just the outermost one.

    A frame of mountains is not one curve. It is a stack of nested crests at different
    ranges, and the eye reads depth from them directly. `horizon()` collapses that stack
    to its top edge and discards the rest, which is most of the geometric information in
    the picture -- and the part that carries range.
    """
    az_deg: np.ndarray        # azimuth of the silhouette point
    elev_deg: np.ndarray      # apparent elevation angle
    range_km: np.ndarray      # distance to the crest
    peak_m: np.ndarray        # terrain height there
    layer: np.ndarray         # chain id: points sharing one continuous ridge


def ridges(cam: dict, dem: Dem, half_fov_pad: float = 8.0, step_deg: float = 0.1,
           max_km: float = 80.0, step_m: float = 60.0, min_step_deg: float = 0.6,
           link_tol: float = 0.25, link_d_elev: float = 0.35,
           min_km: float = 1.5) -> RidgeField:
    """All visible ridge crests, near to far, grouped into continuous ridgelines.

    March each ray as `horizon()` does, but keep the whole elevation-angle profile rather
    than its argmax. A terrain point is *visible* exactly when its angle exceeds every
    closer point's -- that is, when it sets a new running maximum. So the running-max
    staircase along a ray has one step per silhouette the eye can see, and the top step
    is the horizon. Reading off the steps costs nothing beyond the march already done.

    `min_step_deg` is what separates a genuinely new ridge standing clear behind the one
    in front from a metre of noise on a single slope. Below about 0.3 deg the count runs
    to twenty-odd per ray; at 0.6 it settles to the three-to-nine a person would count.

    Crests in neighbouring rays are then linked into chains, so a ridgeline is one object
    with a depth rather than a scatter of independent points. Linking on range alone is
    not enough -- two crests 15 km away on opposite sides of a valley are unrelated, and
    joining them draws a vertical spike through the frame -- so continuity in elevation
    angle is required as well. `min_km` drops the near field, where a 30 m surface model
    is reporting the canopy and the rooftop the camera is bolted to, not landmarks.
    """
    half = cam["fov"] / 2.0 + half_fov_pad
    az = np.arange(cam["az"] - half, cam["az"] + half + 1e-9, step_deg)
    d_m = np.arange(step_m, max_km * 1000.0, step_m)

    lat0, lon0 = cam["lat"], cam["lon"]
    h_cam = cam["elev"] + (cam.get("agl") or 0.0)
    band, transform = dem.window(lat0, lon0, max_km / 111.0 + 0.05)

    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    A = np.radians(az)[:, None]
    D = d_m[None, :]
    lats = lat0 + (D * np.cos(A)) / m_per_deg_lat
    lons = lon0 + (D * np.sin(A)) / m_per_deg_lon
    h = Dem.sample(band, transform, lats.ravel(), lons.ravel()).reshape(lats.shape)
    ang = np.degrees(np.arctan2(h - h_cam - D ** 2 / (2.0 * R_EFF), D))

    run = np.maximum.accumulate(ang, axis=1)
    rec = ang >= run - 1e-12                      # visible: nothing nearer stands higher
    desc = np.ones_like(rec)
    desc[:, :-1] = ang[:, 1:] < ang[:, :-1]       # and the profile turns over here
    crest = rec & desc

    # Chains are grown ray by ray: each crest either continues the nearest open chain at
    # a similar range, or starts a new one. Ratio rather than absolute tolerance, because
    # a 2 km discrepancy is a different ridge at 5 km and the same one at 60.
    a_out, e_out, r_out, p_out, l_out = [], [], [], [], []
    open_chains: list[tuple[int, float, float]] = []   # (id, last range km, last elev)
    next_id = 0
    for j in range(len(az)):
        idx = np.flatnonzero(crest[j])
        if idx.size == 0:
            open_chains = []
            continue
        levels = ang[j, idx]
        keep = idx[np.concatenate(([True], np.diff(levels) >= min_step_deg))]
        keep = keep[d_m[keep] / 1000.0 >= min_km]
        if keep.size == 0:
            open_chains = []
            continue
        rng = d_m[keep] / 1000.0
        new_open = []
        taken: set[int] = set()
        for i, rk in zip(keep, rng):
            ev = ang[j, i]
            best, best_d = None, link_tol
            for cid, prev, pev in open_chains:
                if cid in taken or abs(ev - pev) > link_d_elev:
                    continue
                rel = abs(rk - prev) / max(rk, prev)
                if rel < best_d:
                    best, best_d = cid, rel
            if best is None:
                best = next_id
                next_id += 1
            taken.add(best)
            new_open.append((best, rk, ev))
            a_out.append(az[j]); e_out.append(ang[j, i])
            r_out.append(rk); p_out.append(h[j, i]); l_out.append(best)
        open_chains = new_open

    return RidgeField(np.array(a_out), np.array(e_out), np.array(r_out),
                      np.array(p_out), np.array(l_out, dtype=int))


def summits(field: RidgeField, min_prominence_deg: float = 0.15,
            min_chain_pts: int = 8, max_per_chain: int = 6) -> np.ndarray:
    """Indices of true summits: local maxima *along* a ridgeline, by prominence.

    A summit is a peak in two senses at once -- a crest along the ray, which membership
    in a RidgeField already guarantees, and a high point across azimuth relative to the
    ridge it sits on. Requiring both is what distinguishes a landmark from a point on a
    smooth skyline, and each one carries a range, so it is a 3-D landmark rather than a
    bearing. Short chains are dropped: a ridge glimpsed over four rays has no summit.
    """
    out: list[int] = []
    for cid in np.unique(field.layer):
        sel = np.flatnonzero(field.layer == cid)
        if sel.size < min_chain_pts:
            continue
        sel = sel[np.argsort(field.az_deg[sel])]
        e = field.elev_deg[sel]
        n = len(e)
        cand = []
        for i in range(1, n - 1):
            if not (e[i] >= e[i - 1] and e[i] > e[i + 1]):
                continue
            left = i
            while left > 0 and e[left - 1] <= e[i]:
                left -= 1
            right = i
            while right < n - 1 and e[right + 1] <= e[i]:
                right += 1
            prom = e[i] - max(e[left:i + 1].min(), e[i:right + 1].min())
            if prom >= min_prominence_deg:
                cand.append((prom, sel[i]))
        cand.sort(reverse=True)
        out += [i for _p, i in cand[:max_per_chain]]
    return np.array(sorted(out), dtype=int)
