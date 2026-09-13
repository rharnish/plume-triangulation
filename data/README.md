# Data

Nothing large is committed here. `meta/` is checked in; `tgz/` is fetched.

## Fetching

```sh
./fetch.sh          # ~13 GB, 189 archives, idempotent
```

## What is here

**`meta/cams.json`** — 505 cameras across 82 HPWREN sites, derived from
`https://www.hpwren.ucsd.edu/cameras/sites.js`. Per site: latitude, longitude, elevation.
Per camera: azimuth, horizontal field of view, roll/pitch/yaw, height above ground level, and
imager type (199 color, 187 monochrome, 109 PTZ, plus VNIR, SWIR and thermal singles).

This is what makes bearing triangulation tractable — a detection's horizontal pixel offset maps
to a true bearing through azimuth and FOV, and bearings from two sites intersect at the ignition.

**Read the orientation fields carefully.** Only position is a survey. Across all 505 cameras,
`az` is exactly 0/90/180/270 on **482** and `fov` exactly 90 or 60 on **483** — a cardinal
heading and a spec sheet, not a calibration. `pitch`, `roll` and `yaw` exist as columns and are
non-zero on only **9**, **14** and **3** cameras respectively; everywhere else they are literal
`0.0` placeholders. There is no focal length, principal point or distortion coefficient
anywhere. HPWREN built this network to give people pictures, not to do photogrammetry, and the
metadata is entirely adequate for that — but any pose claim in this project rests on the numbers
above, and `NOTES.md` records what happened when they were taken at face value.

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

Camera pose resolves for **167 of 191** sequences (189 archives, two of which carry two
annotation passes and are split). The 24 that do not — 14 distinct camera IDs (`*-iqeye`,
`lo-*`, `ml-*`, `mw-e`, `so-*`, `smer-tcs9/10`) — are retired hardware absent from `sites.js`,
which lists only currently active cameras while FIgLib reaches back to 2016. Expected, not a
defect; the **42** triangulable fires were counted using resolved cameras only.

## Corpora and provenance

Published numbers come from the 189 archives above, the `core` corpus. The rest of FIgLib is
not needed to reproduce anything, and `fetch.sh` deliberately does not pull it — another 19 GB
from a research network's CDN is a cost to HPWREN as well as to you. Where the remainder has
been scored, it was fetched separately into `out/figlib_extra/tgz/` and run as its own corpus.

Every stage takes `FIGLIB_CORPUS` (`core` by default, `extra`, or `all`) and writes to its own
metadata and results directories, so extra archives on disk cannot change a core result.
`FIGLIB_TIER` restricts scoring by what the detector could have trained on — `possibly_seen`
(fire on or before 2025-04-14, pyronear's FIgLib snapshot), `likely_unseen` (after the snapshot,
up to the model's 2026-05-25 release) or `unseen` (after the release). See
`src/figlib/corpus.py`.

**`meta/manifests/core.json`, `meta/manifests/extra.json`** — one entry per archive: source URL,
size and SHA-256 of the copy that was scored, the server's `Content-Length`, `Last-Modified` and
`ETag` when checked, frame count per annotation pass, fire date and contamination tier. Check a
local copy against it with `python -m src.figlib.provenance verify core`.

Three archives are listed but unusable, and the manifest says why rather than dropping them:
`20250123_GilmanFire_tdllns-mobo-c` and `20260909_GettyFire_wilson-ws-mobo-c` are served as
empty placeholders (150 and 258 bytes), and `20200831_FIRE_wc-n-mobo-c` holds 180 frames named
by epoch alone, with no plume-appearance offset and so no clock to score against.

**`meta/runs.jsonl`, `meta/<corpus>/runs.jsonl`** — one line per pipeline run: git commit and a
hash of any uncommitted diff, package versions, the model pin, the manifest hashes it read,
parameters, and the SHA-256 of every file it wrote. A result whose hash is not in the log did
not come from a recorded run.

## Attribution

FIgLib is conceived, created and maintained by Hans-Werner Braun for HPWREN at UC San Diego.
Use of this data requires a credit reference to <https://www.hpwren.ucsd.edu/> in derivative work.
The data is provided as-is with no guarantees.
