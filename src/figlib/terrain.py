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
