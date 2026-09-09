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
agree the fire is, so geometric consistency does -- which suggested triangulation is a
false-alarm filter and not only a locator. **Measured, that does not hold** (see the
sweep below): geometry removes that particular cumulus, but across the corpus simply
raising the threshold buys more per false alarm than requiring cross-site agreement
does. The anecdote was real and the generalisation from it was wrong.

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

## Accumulating evidence rather than picking the best detection (2026-09-09)

Taking each camera's single most confident detection discards nearly everything seen, and
in particular discards the *agreement* between a faint early detection and a solid later
one at the same bearing -- which is the part that says the bearing is real rather than a
passing cloud.

Summing log-likelihoods over all detections is not the fix either: one confident false
positive at the wrong bearing then contributes an unbounded penalty and drags the peak.
These detectors do produce those (0.81 on a cumulus). So each detection is a mixture,

    P(detection | fire at x) = pi * Normal(bearing | bearing_to_x, sigma)
                             + (1 - pi) * Uniform(field of view)

with `pi` rising with confidence. Agreement contributes evidence; disagreement falls back
on the uniform term and its influence is **bounded** -- an outlier stops mattering instead
of dominating. Frames are not independent, so per-camera evidence is scaled to grow as
n**alpha rather than n; alpha was swept, and between 0 and 0.75 it barely matters
(2.28 km throughout), with full summation at alpha=1 measurably worse (2.75 km).

**The gain is concentrated early, exactly where evidence is weakest.**

| since plume | max-confidence | accumulated |
|---|---|---|
| **3 min** | 17/26 fires, 4.73 km | **21/26 fires, 3.93 km** |
| 6 min | 24/26, 2.88 km | 25/26, 3.33 km |
| 15 min | 25/26, 2.26 km | 25/26, 3.09 km |
| 40 min | 26/26, 2.58 km | 26/26, 2.28 km |

Four more fires locatable at three minutes and 17% lower median error there. Mid-window
it is slightly worse, which is consistent with weak detections adding noise once strong
ones exist. Raising the confidence floor trades exactly that way -- floor 0.10 gives
21 fires at three minutes, 0.20 gives 19, 0.30 gives 15, while mid-window median improves
from 3.09 to 2.41 km. No single setting wins everywhere; a low floor is right because the
operational question is how fast a fire can be located, not how well it can be located
once it is obvious.

## Animations (2026-09-09)

`python -m src.figlib.animate <fire_id>` writes an mp4 to `out/videos/`: camera frames
with detection boxes on the left, the accumulated posterior with bearings and the
estimate's track on the right, error-vs-time underneath. Five rendered so far.

Worth watching for: on the Kitchen fire a **second bright intersection** appears early,
a spurious crossing of two false-positive rays, and is extinguished as further evidence
accumulates -- the robust mixture visibly outvoting an outlier. On Ranch2 the posterior
is a long ellipse throughout and the truth sits inside it while the point estimate is
~1.8 km out.

Two bugs the animation exposed, both of which had been silently degrading results:

* **Grid centred on the camera centroid.** Ranch2 has sites 78 and 81 km from the fire,
  so a 35 km window about their centroid did not contain the true peak at all and argmax
  returned an edge cell -- reported as a 19 km error that was pure artefact. Fixed with a
  coarse pass followed by a fine grid about its peak. The correct answer is 1.88 km.
* **Camera selection ignored sites.** Ranking by detection count picked four cameras from
  one site on the Club fire, two of them the same camera twice (that archive holds two
  annotation passes), leaving one site and nothing to triangulate. Now takes the best
  camera per site before a second from any site.

## Seconds-to-alert vs false alarms per camera-day (2026-09-09)

`python -m src.figlib.falsealarm` -> `out/falsealarm.json`, `out/figures/falsealarm.png`.

The operational question is not mAP. A network of 505 cameras asks "is anything burning"
1,440 times per camera per day, so the only number that means anything is **how fast a
real ignition is called at a false-alarm rate an operator can live with**. FIgLib is
well suited to measuring it, because the ~40 min of pre-ignition frames in every sequence
are negatives from the *same camera under the same light* -- the same haze, cumulus and
glint the positives sit in, not unrelated images.

Frames within 120 s of annotated plume appearance are discarded rather than counted as
negatives: the annotation is a human judgement, so the minute before it is genuinely
ambiguous, and scoring a detection there as a false alarm would flatter latency at the
expense of the rate. Repeat alarms within 600 s are one alarm, since an operator
dismisses an alarm once.

**Single camera, one frame over threshold** (189 sequences, 4.99 camera-days of negatives):

| tau | FA/camera-day | fires alerted | median s | p75 |
|---|---|---|---|---|
| 0.25 | 25.8 | 98% | 180 | 360 |
| 0.40 | 8.0 | 94% | 240 | 540 |
| 0.50 | 3.6 | 90% | 360 | 734 |
| 0.60 | 1.6 | 76% | 540 | 840 |
| 0.70 | 0.40 | 59% | 660 | 1080 |

**The gap between what this detector does and what a network needs is three orders of
magnitude.** At tau=0.25 -- an unremarkable default -- 25.8 false alarms per camera-day
is **13,000 alerts a day** across 505 cameras. One false alarm per camera-week, which is
roughly what a human dispatcher could absorb, is 0.14/camera-day; the detector reaches
that only past tau=0.7, where it misses 41% of fires and takes 11 minutes on the rest.
Nothing here is a knock on pyronear -- it is what per-frame operation costs at scale, and
it is the argument for spending the effort on temporal and cross-camera structure rather
than on another point of AP.

**Persistence is cheap and it works.** Requiring 2 of the last 3 frames over threshold
cuts the rate by about half at a matched threshold (8.0 -> 4.2 FA/camera-day at tau=0.40)
and by about a quarter at matched *recall*, which is the fair comparison: at ~91% recall,
5.6 FA/camera-day single-frame against 4.2 for 2-of-3, for one extra minute of latency
(300 s -> 360 s). 3-of-5 goes further and costs more; the frontier is smooth, so the choice is
genuinely an operator's to make. Isolated frame-scale flicker dominates the negatives.

**Cross-site agreement does NOT beat thresholding -- the negative result.** Scored on the
42 triangulable fires with the same cameras and the same negative time, requiring two
sites to detect within 180 s with bearings passing within 3 km of each other halves the
false-alarm rate at fixed threshold. But so does raising the threshold, and more cheaply:
the any-camera curve dominates the cross-site curve at every operating point measured.

| budget | any camera | two sites agree |
|---|---|---|
| ~1.35 FA/cam-day | 98% alerted, 301 s | 90% alerted, 270 s |
| ~0.81 FA/cam-day | ~92% alerted, ~426 s (interp.) | 88% alerted, **240 s** |
| ~0.54 FA/cam-day | 88% alerted, 480 s | 83% alerted, 480 s |

Two structural costs explain it: 2 of the 42 fires never produce a cross-site coincidence
at any threshold, capping recall at 95%, and a second site adds observation time to the
denominator as well as a constraint. What survives is narrower and worth stating as such
-- **geometry buys latency, not recall.** Because agreement carries the evidence, the
detector can be run at tau=0.45 instead of 0.7, and a low threshold is a fast one: at a
matched 0.81 FA/camera-day the cross-site rule alerts at a median 240 s against roughly
426 s. That is the honest version of the Junction-cumulus anecdote.

**The corpus cannot resolve the region that matters, and that is itself the finding.**
Forty minutes of negatives per sequence is 4.99 camera-days in total (3.71 for the
triangulable subset). Rates below ~1 FA/camera-day therefore rest on between zero and
three events; the tail of every curve is one alarm wide, and the Poisson upper bound at
zero observed alarms is still 0.74/camera-day. **A rate of one false alarm per camera-week
cannot be measured on FIgLib at all** -- it would need on the order of a hundred
camera-days of negatives, which is exactly the unglamorous data a fielded network already
has and a public benchmark does not. `fa_hi95` is carried in the JSON for every row so
the bare ratios are never read as more than they are.

Caveat on extrapolation: FIgLib negatives are daytime, drawn from the hour before a real
ignition, so they oversample fire weather and undersample night -- optimistic in one
direction and pessimistic in the other. The 1,440 frames/camera-day figure assumes the
60 s cadence holds around the clock.

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

## Housekeeping that will bite later

**Figures and videos live under `out/`, which is gitignored.** That is right for 21 MB of mp4 and
for anything regenerable, but it means a README cannot yet link to any of them. Before publishing,
either move the handful of README figures into a tracked `figures/` directory, or generate them in
CI. The videos are too large for git regardless and need hosting or conversion to short animated
GIFs.

**Monochrome/NIR data is already in hand.** The Club fire animation shows colour *and* monochrome
views of the same two sites, so the deferred "does NIR see smoke earlier" question has usable
paired data sitting in the existing download -- no new fetch needed.

## Deliberately deferred

Monochrome/NIR sequences (11 of them, paired with colour views of the same fires) --
"does NIR see smoke earlier" is a real question, saved for later. Terrain: flat-earth
triangulation first, ray-terrain intersection against Copernicus DEM GLO-30 as a
refinement if time allows.
