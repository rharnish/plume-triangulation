# Working notes

Running state of the investigation. Findings that shape the design, and open questions
that could still move the numbers.

## Pipeline state

`ingest.py` -> `fires.py` -> `truth.py` -> `resolve.py` -> `detect_yolo.py` -> `viz.py`

| Quantity | Value |
|---|---|
| Archives / sequences | 189 / **191** (two archives carry two annotation passes and are split) |
| Frames | 14,827 (7,316 pre-ignition, usable as negatives) |
| Sequences with camera pose | 167 / 191 |
| Distinct fires after t0 clustering | **69** (from only 48 event labels) |
| Triangulable (>=2 posed sites) | 42, stable at 42/42/44 for 900/1800/3600 s thresholds |
| Fires with usable ground truth | **33** (10 name-confirmed, 23 probable) |
| **Ground truth AND triangulable** | **26** <- the geolocation scoring set |

## Findings

**FIgLib event labels are not fires.** An unnamed `..._FIRE` label covers every unnamed
fire the network saw that date. `20180603_FIRE` is three ignitions: three cameras at
20:20-20:22 UTC, two at 23:09, one at 01:24 next morning. Clustering on the per-camera
plume clock at 1800 s recovers real fires; grouping by name overcounts multi-camera
coverage and misattributes ground truth.

**Official discovery coincides with plume appearance.** Across the 33 resolved fires,
`FireDiscoveryDateTime` minus annotated plume appearance has median **+1.0 min**
(IQR -4.2 to +9.1); on the 10 name-confirmed alone, median -0.6 min. Humans reported 7
of those 10 *before* the plume was annotated as visible.

*Consequence:* "minutes of warning gained over official reporting" is not a claim this
data supports. What it does support is stronger, because it is measured rather than
assumed: the FIgLib clock is a **validated proxy for when a human knew**, so alerting
N minutes before t0 is N minutes of real warning. The WFIGS join is the evidence for
the metric, not the metric itself.

**Camera pose can be checked without a detector.** A fixed camera sees a 90 deg wedge,
so an incident outside it cannot be the fire. That filter cut 151 candidates to 87 and
discarded none of the ten name-confirmed truths.

**Plumes are visible from cameras that cannot see their source.** Validating pose against
name-confirmed truth, 35 of 40 camera-fire pairs put the official coordinate inside the
field of view. The five that miss are physical, not erroneous: at `20260629_JunctionFire`,
`vo-w` holds the fire at +35.9 deg while `vo-n` at the same site is 54 deg off axis. The
annotator saw smoke in both because the plume drifted into a frame whose wedge never
contained its origin. This is the same displacement that makes plume-axis correction
necessary, and it is why the visibility filter carries a margin.

**Three independent estimates agree.** On the Valley fire, pose plus the WFIGS coordinate
put the smoke at x=0.636 across `pi-w-mobo-c`; the detector, from pixels alone, fires at
0.63 with confidence 0.759.

**Confidence does not separate cloud from plume; bearing does.** On the 8-acre Junction
fire, `mg-e-mobo-c` returns **0.81 on a cumulus** -- more confident than most true
detections. No threshold removes it. It sits ~0.2 of a frame from where the other cameras
agree the fire is, so geometric consistency does. This makes triangulation a false-alarm
filter, not only a locator, and that is measurable: false alarms per camera-day before
and after requiring cross-site agreement.

**The training-free differencer does not work yet.** On the Valley fire it fails to
separate pre-ignition from post (max 202.7 vs 189.0). Absolute rather than signed
difference, a 20-minute background lag, and a rootedness term measuring descent below the
terrain silhouette each helped; none sufficed. Cumulus drifting along the skyline changes
as much as the plume does. Camera jitter was ruled out by phase correlation (max 0.21 px
over a whole sequence). Kept as the uncontaminated floor, and the difficulty is itself
the argument for a learned detector.

## Geolocation results (2026-09-09, full detection pass, 189/189 sequences)

Likelihood field over the ground, bearings from the most confident detection per camera.

| Set | n | median error | max | <=1 km | <=2 km | <=5 km |
|---|---|---|---|---|---|---|
| all scoring fires | 26 | 2.58 km | 62.10 | 6 | 12 | 19 |
| **confirmed tier** | 10 | **1.90 km** | **3.61 km** | | | |
| probable tier | 16 | 4.37 km | 62.10 km | | | |

Best: `20240701_Kitchenfire` at **0.08 km** from the official ignition point, four sites.
Every confirmed-tier fire lands within 3.61 km; the probable tier carries the whole tail.

**The tier split is the finding.** Geometry is not the limiting factor when the ground
truth is right. Where probable-tier fires fail they fail in a specific shape -- bearings
agreeing among themselves to within a few km2 while sitting 20-60 km from the assigned
incident -- `20191006_FIRE` is 27.9 km out with a 2.7 km2 95% region. Independent cameras
do not agree by accident, so **error much larger than sqrt(area95) indicts the ground
truth, not the geometry**. **Geolocation error is therefore an audit of the weaker resolution tier**,
and the 23 probable fires should be re-examined against it rather than trusted.

**Uncertainty earns its place.** On `20200813_Ranch2Fire` the three cameras are nearly
collinear with the fire, so the surface is a long ridge with a 27.8 km2 95% region. The
point estimate is 2.10 km out, but the truth lies well inside the contour. A crossing
point would report only the 2.10 km and hide the fact that the geometry never supported
better.

**Wind correction is a wash.** Taking the upwind box edge rather than the centre moves
the median from 2.58 to 2.55 km, better on 10 fires and worse on 11. Preferring early
detections, which have drifted less, is worse (3.68 km).
Reported as measured. The likely reason is that a box edge is a crude stand-in for the
plume base; a mask would give the real axis, which is why that was deferred rather than
dropped.

## Does the estimate evolve? (2026-09-09)

Yes, and not in the direction expected. Locating the fire from every detection up to time
`t` after the plume appeared:

| since plume | fires locatable | median error | p75 | median area95 |
|---|---|---|---|---|
| 180 s | 17/26 | 4.73 km | 8.20 | 3.5 km2 |
| 360 s | 24/26 | 2.88 km | 7.98 | 4.0 km2 |
| 600 s | 25/26 | 2.79 km | 4.76 | 3.7 km2 |
| **900 s** | 25/26 | **2.26 km** | 4.76 | 3.7 km2 |
| 1800 s | **26/26** | 2.26 km | 6.00 | 3.7 km2 |
| 2400 s | 26/26 | 2.58 km | 6.00 | 3.7 km2 |

Confirmed tier alone: 2.91 km at three minutes from 7 of 10 fires, 1.90 km at fifteen
minutes from all ten.

**Accuracy improves rather than decaying, then plateaus.** Waiting buys coverage and
precision up to about fifteen minutes and nothing after; the slight worsening by forty
minutes is the plume growing away from its source. So the operational tradeoff is
coverage against latency, not accuracy against latency -- at three minutes two thirds of
fires are locatable at 4.7 km, by fifteen minutes nearly all of them at 2.3 km.

**Uncertainty does not tighten.** area95 sits near 3.7 km2 throughout. Extra cameras
arrive roughly as fast as the region would otherwise shrink.

**The estimate wanders, and neither wind nor geometry explains where.** Holding the
camera set fixed across windows -- necessary, since a camera joining the solution can
move it kilometres on its own -- the estimate still moves a median 1.79 km. The angle
between that motion and the wind direction has median 45.9 deg, and against the major
axis of its own uncertainty ellipse 41.4 deg. Both are indistinguishable from random
(n=11). The motion is the detector's box wandering over a growing diffuse plume, not
advection, which is why correcting bearings with wind was a wash.

## Open questions

**Lens distortion — the biggest threat to kilometre accuracy.** `geom.py` assumes a
rectilinear 90 deg camera. Rendered frames show heavy vignetting and visibly bowed
horizons on some units (`vo-n`, `bm-e`). If real barrel distortion is present, pixel to
bearing carries systematic error that grows toward frame edges -- and detections do land
there (`vo-w` at x=0.861). Check by fitting horizon curvature across frames, and against
HPWREN's per-site `peakfinder` horizon profiles as an independent reference. Until this
is settled, treat kilometre errors as provisional.

**Contamination.** Every pyronear model, and SmokeyNet, trains on FIgLib -- see
`models/README.md`. Detection and timing numbers are a labelled reference point, never a
generalisation claim. Geolocation is unaffected: kilometre error against official
coordinates tests geometry, and a memorised detection still yields a valid bearing.

**Not yet validated:** the 23 `probable`-tier fires are resolved by geometry and timing
but not by name. Nothing has yet confirmed one visually the way the confirmed tier was.
They are 23 of the 33 ground-truth fires, so they carry real weight.

## Deliberately deferred

Monochrome/NIR sequences (11 of them, paired with colour views of the same fires) --
"does NIR see smoke earlier" is a real question, saved for later. Terrain: flat-earth
triangulation first, ray-terrain intersection against Copernicus DEM GLO-30 as a
refinement if time allows.
