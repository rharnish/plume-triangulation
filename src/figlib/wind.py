"""Wind at the time and place of each fire, and what it does to a bearing.

A detection box does not sit over the fire. Smoke rises and drifts, so by the time a
plume is big enough to detect, the bright part of it has moved downwind of its source --
which is why cameras in this dataset recorded smoke whose origin lay outside their own
field of view. Taking the bearing to the box centre therefore carries a *systematic*
error, always downwind, and systematic error is what makes an estimate confidently
wrong rather than merely imprecise.

Only the component of that drift across the line of sight matters. Smoke blowing
directly away from a camera lengthens the plume without moving its bearing; smoke
blowing across the view moves the bearing by the full displacement. So the correction is
to take the *upwind* horizontal edge of the box instead of its centre, and which edge
that is depends on the sign of the crosswind -- which needs to be looked up, not guessed.

Winds come from Open-Meteo's historical archive: free, unauthenticated, hourly, at 10 m
and 100 m. The 100 m level is the better guide, since a plume detectable from tens of
kilometres away has risen well clear of the surface layer.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "meta" / "wind_cache.json"
API = "https://archive-api.open-meteo.com/v1/archive"


def _load() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save(c: dict) -> None:
    CACHE.write_text(json.dumps(c, indent=1) + "\n")


def wind_at(lat: float, lon: float, epoch: int, cache: dict | None = None) -> dict | None:
    """Wind at the hour containing `epoch`. Direction is meteorological (blowing FROM)."""
    own = cache is None
    cache = _load() if own else cache
    dt = datetime.fromtimestamp(epoch, UTC)
    key = f"{lat:.2f},{lon:.2f},{dt:%Y-%m-%d}"
    if key not in cache:
        q = urllib.parse.urlencode({
            "latitude": round(lat, 2), "longitude": round(lon, 2),
            "start_date": f"{dt:%Y-%m-%d}", "end_date": f"{dt:%Y-%m-%d}",
            "hourly": "wind_speed_10m,wind_direction_10m,"
                      "wind_speed_100m,wind_direction_100m",
            "timezone": "UTC"})
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=45) as fh:
                cache[key] = json.load(fh).get("hourly")
        except Exception:
            cache[key] = None
        if own:
            _save(cache)
    h = cache.get(key)
    if not h:
        return None
    stamp = f"{dt:%Y-%m-%dT%H:00}"
    if stamp not in h["time"]:
        return None
    i = h["time"].index(stamp)

    def pick(a, b):
        v = h[a][i]
        return v if v is not None else h[b][i]

    return {
        "from_deg_100m": pick("wind_direction_100m", "wind_direction_10m"),
        "speed_kmh_100m": pick("wind_speed_100m", "wind_speed_10m"),
        "from_deg_10m": h["wind_direction_10m"][i],
        "speed_kmh_10m": h["wind_speed_10m"][i],
        "time": stamp,
    }


def crosswind_sign(view_az_deg: float, wind_from_deg: float) -> float:
    """+1 if the plume is pushed toward increasing image x, -1 if decreasing, 0 if along view.

    Wind *toward* is the reciprocal of the reported *from* direction. Its component
    across the line of sight is sin(toward - view); the sign of that says which way the
    plume leans in frame, and therefore which edge of the box is closest to the source.
    """
    toward = (wind_from_deg + 180.0) % 360.0
    return math.sin(math.radians(toward - view_az_deg))


def upwind_x(x0: float, x1: float, view_az_deg: float,
             wind_from_deg: float | None, dead_zone: float = 0.25) -> float:
    """Horizontal position of the box edge nearest the source.

    With no wind information, or with the wind blowing nearly along the line of sight
    (where the crosswind component is small and the edge choice would be arbitrary),
    fall back to the box centre.
    """
    if wind_from_deg is None:
        return (x0 + x1) / 2.0
    s = crosswind_sign(view_az_deg, wind_from_deg)
    if abs(s) < dead_zone:
        return (x0 + x1) / 2.0
    return x0 if s > 0 else x1
