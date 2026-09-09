# Data

Nothing large is committed here. `meta/` is checked in; `tgz/` is fetched.

## Fetching

```sh
./fetch.sh          # ~13 GB, 189 archives, idempotent
```

## What is here

**`meta/cams.json`** — 505 cameras across 82 HPWREN sites, derived from
`https://www.hpwren.ucsd.edu/cameras/sites.js`. Per site: latitude, longitude, elevation.
Per camera: azimuth, horizontal field of view, roll/pitch/yaw corrections, height above ground
level, and imager type (199 color, 187 monochrome, 109 PTZ, plus VNIR, SWIR and thermal singles).

This is what makes bearing triangulation tractable — a detection's horizontal pixel offset maps
to a true bearing through azimuth and FOV, and bearings from two sites intersect at the ignition.

**`meta/sites.js`** — the upstream source, kept verbatim for provenance.

**`meta/tgz_multicam.txt`** — the 189 archives `fetch.sh` pulls: every FIgLib sequence belonging
to a fire observed by three or more cameras.

**`meta/seqs.txt`, `meta/multi.txt`** — full FIgLib inventory (453 unique sequences) and the
per-event camera counts used to select the above.

**`tgz/`** — FIgLib sequence archives, ~80 JPEG frames each.

## The filename clock

Frames are named `<unix_timestamp>_<signed_offset_seconds>.jpg`:

```
1569359693_-02400.jpg     40 min before the plume becomes visible
1569362093_+00000.jpg     plume appearance
1569364493_+02400.jpg     40 min after
```

Two consequences this project leans on:

1. The signed offset is a ground-truth clock, so detection can be scored as **seconds-to-alert**
   rather than mAP.
2. The 40 minutes preceding every ignition are negatives from the *same camera, scene and
   lighting* — which is what makes a false-alarms-per-camera-day figure meaningful.

## Coverage limits

Camera pose resolves for **175 of 189** sequences. The 14 that do not (`*-iqeye`, `lo-*`, `ml-*`,
`so-*`, `smer-tcs9/10`) are retired hardware absent from `sites.js`, which lists only currently
active cameras while FIgLib reaches back to 2016. Expected, not a defect — the 45 triangulable
events were counted using resolved cameras only.

## Attribution

FIgLib is conceived, created and maintained by Hans-Werner Braun for HPWREN at UC San Diego.
Use of this data requires a credit reference to <https://www.hpwren.ucsd.edu/> in derivative work.
The data is provided as-is with no guarantees.
