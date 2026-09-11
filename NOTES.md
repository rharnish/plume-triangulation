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

**Wind correction is a wash.** Taking the upwind box edge rather than the center moves
the median from 2.58 to 2.55 km, better on 10 fires and worse on 11. Preferring early
detections, which have drifted less, is worse (3.68 km).
Reported as measured. The likely reason looked like the box being a crude stand-in for the
plume base -- so a mask should give the real axis. Tested 2026-09-10 and it does not: see
"Segmentation and per-plume wind fits" below. No mask-derived bearing beats the box edge.

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
move it kilometers on its own -- the estimate still moves a median 1.79 km. The angle
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

* **Grid centered on the camera centroid.** Ranch2 has sites 78 and 81 km from the fire,
  so a 35 km window about their centroid did not contain the true peak at all and argmax
  returned an edge cell -- reported as a 19 km error that was pure artifact. Fixed with a
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

## Edge deployment: Apple M3, measured (2026-09-09)

**Full write-up: [`docs/edge-m3.md`](docs/edge-m3.md)** (committed, with the figure, so a
README can link it -- unlike everything under `out/`). Summary below.

`src/figlib/detect_coreml.py`, `bench_edge.py`, `power.py`, `quantization.py`.
Figure `out/figures/edge_m3.png`; raw in `out/bench_latency.json`,
`out/bench_thermal_fp16_ane.json`, `out/quantization.json`.

The M3 MacBook Air is a defensible stand-in for a fielded camera in the ways that matter:
real ARM64, a real NPU, per-unit wattage from `powermetrics`, and -- being fanless -- a
genuine sustained-throughput curve rather than a burst number. Fielded fire cameras sit in
sealed enclosures on mountaintops; a laptop that cannot dump heat is closer to that than
any rented GPU. Host: M3, 8 GB, macOS 15.6, AC power, Low Power Mode off, batch 1.

`detect_coreml` is a drop-in for `detect_yolo` -- same letterbox, same NMS, same JSON --
so bearings, the likelihood field, evidence accumulation and the false-alarm sweep all run
against Core ML detections unchanged. Core ML FP32 reproduces the ONNX detections
**exactly** on a sampled sequence (14/14 boxes, zero delta at stored precision), so the
export contributes no error of its own.

**A requested compute unit is only a request.** This is the result that justified the
exercise, and it is invisible without instrumentation:

| variant | ANE ops | GPU | CPU | ask for ANE -> | ms | ANE mW |
|---|---|---|---|---|---|---|
| FP32 | **0** | 241 | 0 | lands on **CPU** | 97.3 | **0** |
| FP16 | 226 | 2 | 15 | ANE | **11.0** | 4857 |
| INT8-weight | 226 | 2 | 15 | ANE | 10.4 | 4978 |

The Neural Engine cannot run FP32. Ask for `CPU_AND_NE` with an FP32 model and every
operation lands on the CPU at 97 ms a frame, drawing zero ANE power, and **Core ML reports
no error and returns correct results throughout**. Two independent instruments agree --
`MLComputePlan` op placement and the ANE wattage rail. A benchmark that ran FP32
and called the result "ANE performance" would be 9x wrong and would never find out.

**Latency and energy, model-only, batch 1:**

| placement | FP16 ms | fps | frames/joule |
|---|---|---|---|
| CPU | 48.6 | 20.6 | 2.3 |
| GPU | 24.0 | 41.6 | 5.6 |
| **ANE** | **11.0** | **91.0** | **12.0** |

Total draw barely moves across all of these (7.0-9.6 W); what changes is work done per
watt. **13.3 frames/joule on the ANE against 2.3 on the CPU** -- 5.8x -- and for a
solar-powered enclosure that ratio is the design constraint, not latency.

**Weight-only INT8 buys size, not speed -- confirmed rather than cited.** Identical op
placement, 10.4 ms against 11.0, 12.8 frames/joule against 12.0: all inside the noise.
Through the M3 generation the ANE does weight-only INT8; full INT8 activation compute
arrives with A17 Pro / M4. The measured win is **9.3 MB against 18.2 MB** of model. On an
A17 Pro the same export should show a real speedup, and that is a prediction this rig
cannot test.

**Preprocessing becomes the bottleneck once the model leaves the CPU.** End-to-end is
20.2 ms against 11.0 ms model-only, so JPEG decode plus letterbox costs ~9.4 ms -- about
47% of the frame budget. Move inference to the NPU and the next thing to optimize is
image handling, which no model-only benchmark would ever reveal.

*Capacity, since the false-alarm sweep showed alert latency is partly sampling-bound:* 49
fps end-to-end means one M3 could serve roughly **49 cameras at 1 fps** at about 7.4 W, or
one camera at 1 fps for a small fraction of a watt. Running at video rate is affordable at
the edge in a way that streaming frames to a datacenter is not.

**Fanless sustained throughput: -9.8% over twenty minutes, as a step rather than a decay.**

| minutes | fps | ANE W | total W | thermal pressure |
|---|---|---|---|---|
| 0-3 | 94.0 | 5.13 | 7.86 | Nominal -> Moderate |
| 3-6 | 94.3 | 5.16 | 7.72 | -> **Heavy** |
| 6-10 | 90.2 | 4.79 | 7.34 | Heavy |
| 10-13 | 87.1 | 4.68 | 7.29 | Heavy |
| 13-16 | 85.9 | 4.54 | 7.08 | Heavy |
| 16-20 | 85.2 | 4.36 | 6.88 | Heavy |

Throughput holds at 94 fps for about eight minutes, steps down roughly 10%, and stays
there. Two things worth noticing. **Thermal pressure reads Heavy at ~4 minutes, four
minutes before throughput moves** -- a three-minute burst benchmark would have reported 94
fps and missed the effect entirely, which is why every condition here runs for a fixed
duration rather than a fixed iteration count. And **throttling costs throughput but not
efficiency**: frames/joule goes 12.0 -> 12.4 as fps falls 94 -> 85, because the part holds
its thermal ceiling by dropping clocks and power together. For a battery- or
solar-constrained enclosure that is the benign form of throttling -- you lose frame rate,
not energy per frame.

## What did quantization cost? In kilometers and seconds (2026-09-09)

"INT8 lost 0.4 mAP" tells an operator nothing. Every variant was therefore run through the
same pipeline over the same 93 sequences -- the FP32 reference restricted to that identical
set, since comparing a subset against a full-corpus baseline would confound quantization
with the choice of fires.

| variant | 3 min: n, median | 40 min: n, median | <=2 km | FA/cam-day @0.25 | recall | median s |
|---|---|---|---|---|---|---|
| FP32 | 21/26, 3.93 km | 26/26, 2.28 km | 12 | 25.9 | 0.979 | 240 |
| FP16 | 21/26, 4.01 km | 26/26, **2.28 km** | 12 | 25.9 | 0.979 | 240 |
| INT8w | 21/26, **3.57 km** | 26/26, 2.67 km | 13 | 24.7 | 0.979 | 180 |

**FP16 is free.** Identical median error, identical recall, identical false-alarm rate,
half the model. There is no argument for shipping FP32 to this hardware -- it is slower,
less accurate per watt, and cannot even reach the NPU.

**INT8-weight was expected to cost false alarms, and did not.** On a 72-frame sample it
invented 17 detections and shifted confidences by up to 0.39, so the hypothesis was that
its price would show up in the alarm rate. Measured across all 93 sequences that is wrong,
and the confidence histogram says why:

| tau | FP32 | FP16 | INT8w |
|---|---|---|---|
| 0.05 | 8731 | 8741 | 10698 (+22%) |
| 0.25 | 3436 | 3413 | 3975 (+16%) |
| 0.50 | 1594 | 1598 | 1719 (+8%) |
| **0.70** | **603** | **603** | **603** |

The extra detections are **low-confidence and they thin out with threshold, reaching
exactly parity at 0.7**. Quantization noise perturbs marginal detections and leaves
confident ones untouched, so nothing that would ever raise an alarm changes.

What those extra weak detections do change is early localisation, and they *help*:
**3.57 km against 3.93 km at three minutes, with 9 fires inside 2 km against 6.** That is
the robust mixture doing its job -- disagreement falls back on the uniform term and is
bounded, so extra noisy bearings cost little while extra true ones accumulate. It is worth
being honest that this is a small sample and the 40-minute figure moves the other way
(2.67 vs 2.28 km); the defensible claim is that weight-only INT8 is not measurably worse
where it matters operationally, not that it is better.

*Throughput note:* the 93-sequence pass takes **182 s on the ANE** against roughly 90
minutes for the same work on four x86 cores.

## Terrain-refined pose: a four-fold better fit that made everything worse (2026-09-09)

`src/figlib/calibrate.py`, `pose_validate.py`, `viz_terrain.py`.
Figure `docs/figures/pose_fit.png`; raw in `out/terrain_audit.json`, `out/pose_fit.json`,
`out/pose_fit_staged.json`, `out/pose_geo_compare.json`.

**First: the earlier claim was wrong.** "Camera pose validated against a DEM-synthesised
skyline to within 13 px" came from **one** camera, computed ad hoc, and `viz_terrain.py`
had no `main()` so it could not even be re-run. Made reproducible and swept over all 71
posed cameras:

| | |
|---|---|
| cameras with usable sky | 61 (the 10 monochrome units fail by construction -- the sky test is blue dominance) |
| median absolute residual | **106 px** |
| p90 / max | 223 px / 674 px |
| within 20 px | **6 of 61** |

The 13 px camera was the best in the set.

**The obvious next step, and why it failed.** Fit four parameters per camera against the
observed skyline -- azimuth, pitch, roll, and one radial distortion coefficient -- under a
Huber loss with multi-start. Skyline MAD improved **106 -> 24 px**, four-fold.

Then the held-out test: re-run geolocation against WFIGS coordinates, which the fit never
saw (it only ever looked at ridgelines in pre-ignition frames).

| | median error | <=2 km | |
|---|---|---|---|
| published pose | 2.16 km | 12/26 | |
| **terrain-refined** | **3.38 km** | **6/26** | worse on 17 of 26 |

**The surface metric improved four-fold while the metric that matters got 56% worse.**
Held-out evaluation is the only reason this was caught; on the fitted objective it looked
like a triumph.

**The diagnosis is quantitative.** Perturbing each parameter by one degree and measuring
the loss response:

| parameter | delta loss per degree |
|---|---|
| azimuth | ~0.5 |
| pitch | ~50 |

**An 80:1 ratio.** A ridgeline is nearly horizontal, so sliding it sideways barely changes
it -- azimuth is close to unidentifiable from dense skyline matching. Given free rein the
optimizer spent azimuth on noise: **24 of 61 cameras pegged at the +-6 deg azimuth bound
and 34 of 61 at the k1 bound.** Parameters at their bounds are the tell.

**Second attempt: separate the identifiable from the not.** Azimuth information lives in
*distinctive* ridgeline features, not in the skyline's overall height, so fit shape
(pitch/roll/k1) first with azimuth pinned, then estimate azimuth alone by correlating
skyline gradients, and accept it only where the correlation actually picks a shift out.

Estimates immediately became plausible -- **median +0.40 deg, sd 2.01 deg, centered on
zero** rather than +-6 deg. But the correlation is weak: **azimuth is identifiable on 1 of
61 cameras**, and that one wants 0.30 deg. Applying the staged fit (azimuth pinned, k1
applied, since distortion *does* feed bearings through `undistort_x`) is a wash: median
2.16 -> 2.28 km, better on 7 fires and worse on 9.

A real bug surfaced while chasing this: the gradients were being taken at sigma = 9 px,
which is skyline-extraction jitter rather than terrain. On `vo-n` the predicted and
observed **rows** correlated at 0.910 while their **gradients** correlated at 0.011.
At sigma = 101 px it reaches 0.24 -- still under the gate, but the fix was real and the
symptom was diagnostic.

**What this actually establishes**, which is more useful than the refinement would have
been:

* **The published azimuths hold up.** An independent, feature-based estimate proposes
  corrections centered on zero with sd 2.01 deg -- comparable to the 2 deg sigma already
  assumed per bearing. This is the pose parameter that reaches a bearing, and it is fine.
* **The published pitch often does not hold up, and it does not matter.** A 106 px vertical
  residual is a pitch/elevation error, and pitch does not enter a bearing at all. That
  reconciles the two facts that looked contradictory: badly misaligned skylines alongside
  1.90 km geolocation on the confirmed tier.
* **Terrain is an audit instrument, not a correction.** It catches bad pitch and elevation
  metadata, and it can say when azimuth is unconstrained. It cannot improve azimuth here.

**Deliberately not done:** the fitted poses are kept in `out/` rather than beside the
published metadata, because they are not an improvement and should not be mistaken for one.

### The audit number is an upper bound, not a measurement (corrected 2026-09-09)

Looking at the rendered overlays (`python -m src.figlib.fig_peaks png` regenerates them
into `out/terrain_png/`, sorted by residual; `docs/figures/peaks_examples.png` is six
spanning the range, and is committed) undermines part of the audit
above. On the two *worst* cameras -- `sm-n` at -614 px and `bh-n` at -674 px -- the orange
predicted skyline traces the distant crest about right and its summits land on visible
peaks. It is the **blue observed skyline that is wrong**, locked onto a nearer ridge, a
haze band, or in `bh-n` the roof of the building the camera sits on.

The signed statistics say the same thing: **median -93 px, and predicted sits above
observed on 48 of 61 cameras.** A one-sided bias in exactly the direction a nearer,
lower ridge would produce. Within-frame scatter (median IQR 61 px) is large too, which is
not what a rigid pose error looks like -- a pose error offsets the whole curve.

So **106 px conflates pose error with skyline-extraction error and is an upper bound on
the former.** It cannot be separated until the extractor is fixed. This also supplies a
second and probably more damning reason the pose fit failed: it was regressing onto a
target that is frequently not the skyline. Degeneracy in azimuth explains why the
optimizer *could* wander; an unreliable target explains why it was *rewarded* for it.

**`observed_skyline` is the weak link, and it is weak by construction.** It segments sky
by blue dominance and brightness. That fails on haze, on backlit scenes, on layered
ridges (it has no notion of *which* ridge is the skyline), and completely on the ten
monochrome units, where "blue dominant" is not expressible. Replacing it is the highest-
value next step in this thread -- see Open questions.

**Peaks are now selected by topographic prominence** (`terrain.prominent_peaks`) rather
than by local maximum. A smooth crest carries twenty local maxima that are a tenth of a
degree of noise apiece; prominence keeps the handful a person would point at, typically
1-6 per camera. This does not change any residual, but it makes the overlays legible and
it is a precondition for any real peak-to-peak matching. `bm-w` is instructive: exactly
**one** prominent peak in a 90 deg field, and a +1 px residual that means only "a flat
line matches a flat line" -- a good score carrying no azimuth information at all.

## There are no intrinsics, and the extrinsics are a nameplate (2026-09-09)

Worth stating plainly, because every pose result in this file rests on it. `cams.json`
carries eleven fields per camera. Counting how many are actually populated across all
505:

| field | populated | what it really is |
|---|---|---|
| `lat` / `lon` / `elev` / `agl` | 505 / 505 | surveyed, varied, credible |
| `az` | 505, but **482 are exactly 0/90/180/270** | the cardinal direction in the camera's own name |
| `fov` | 505, but **483 are exactly 90 or 60** | a spec-sheet number |
| `pitch` | **9 non-zero** | placeholder |
| `roll` | **14 non-zero** | placeholder |
| `yaw` | **3 non-zero** | placeholder |

So there is no focal length, no principal point, no distortion coefficients, no sensor
size -- and no measured orientation. `pitch`, `roll` and `yaw` exist as columns and are
literal `0.0` for 96%, 97% and 99% of cameras respectively. What the code has been
calling "published pose" is a position, a compass heading rounded to a quadrant, and a
lens model assumed to be the rectilinear ideal.

That is not a complaint about HPWREN -- the network was built to give people pictures,
not to be photogrammetry. But it does mean the pose numbers were never measurements to
begin with, and the honest framing of the earlier fit is not "the published pose is
wrong" but "there was nothing published to be wrong."

### Why the skyline could never have calibrated it

The failed four-parameter fit was diagnosed as an 80:1 conditioning problem between
azimuth and pitch. That was true but shallow. The real reason, measured:

| camera | skyline points | vertical spread | ridge points | vertical spread | gain |
|---|---|---|---|---|---|
| bh-n | 531 | 43 px | 2 397 | 226 px | 5.3x |
| sm-s | 531 | 42 px | 4 982 | 247 px | 5.8x |
| vo-n | 531 | 34 px | 1 879 | 137 px | 4.1x |
| stgo-e | 531 | 46 px | 5 985 | 442 px | 9.5x |
| wc-n | 531 | 43 px | 5 287 | 268 px | 6.3x |

**The skyline occupies 34-46 px of a 1536 px frame.** It is, to within a couple of
percent of frame height, a horizontal line. Fitting pitch, roll, focal length and radial
distortion to a horizontal line is degenerate no matter how good the extractor is: every
correspondence sits at essentially the same image row and the same range, so a pitch
error, a camera-height error and a focal-length error all produce very nearly the same
residual. No amount of care with the blue curve fixes that, which is why replacing the
extractor was the wrong first move.

The ridge field spans 137-442 px and 2-24 km. The vertical lever arm is 4-10x longer,
and -- more important -- correspondences now sit at *different ranges*, so a camera-height
error (which falls off with range) separates from a pitch error (which does not).

## Monocular depth does not reach these ranges (2026-09-09)

Tested, because layered ridges at different distances is exactly what a depth model
claims to give, and it would have replaced the whole skyline-extraction problem.

### The control that matters

The obvious test -- does predicted depth correlate with the DEM's range? -- is worthless
here, and finding that out was the useful part. Elevation angle already predicts range in
these scenes, so **image row alone scores Spearman +0.84 to +0.98** against DEM range.
Any monotone function of height passes. The test that means something is the **partial**
correlation with image row held fixed: within a horizontal band, do the pixels the DEM
calls distant look further than the ones it calls near? That is the only kind of range
information that can separate a near ridge from a far one at the same elevation angle.

| camera | row-only rho | dark-channel haze, partial | Depth Anything V2, partial |
|---|---|---|---|
| sm-s | +0.934 | -0.162 | +0.129 |
| sm-n | +0.984 | -0.120 | -0.195 |
| bh-n | +0.839 | +0.277 | +0.634 |
| stgo-e | +0.840 | -0.249 | -0.305 |
| wc-n | +0.973 | -0.072 | -0.061 |
| vo-n | +0.394 | -- | +0.296 |

### Dark channel prior: a confounded +0.83 that is really +0.00

He et al.'s transmission estimate is a physically motivated depth proxy -- Koschmieder
gives t = exp(-beta*d), so -log t is proportional to range, and haze is the cue a person
actually uses on these scenes. Raw correlation looked convincing at +0.48 to +0.89.
Controlled for row it collapses to between -0.25 and +0.28, mostly slightly negative. It
was measuring "higher in the frame is further", which the DEM already states exactly.

### Depth Anything V2: correct, and useless here

`onnx-community/depth-anything-v2-small`, run through the onnxruntime already in the
project -- no torch, 99 MB, ~900 ms/frame on four x86 cores. Partial correlation -0.31 to
+0.63, median near zero. Not better than the free physics baseline.

The depth map says why, and says it more clearly than the statistic. On `sm-s` the model
resolves the antenna masts, the equipment huts and the foreground scrub beautifully --
sharp, correctly ordered, genuinely impressive. **Everything past about a kilometer is one
saturated value, indistinguishable from the sky.** The entire ridge stack from 2 to 24 km
is flat. Its dynamic range is spent on the 0-100 m foreground, which is what MDE training
sets contain: NYU tops out around 10 m, KITTI around 80. Our shortest useful ridge is
twenty times KITTI's longest.

Metric models (Depth Pro, ZoeDepth) are worse candidates for the same reason: they emit
meters calibrated on scenes a thousand times closer. Video variants would fix flicker,
which is not the problem.

### The honest caveat

Pose is wrong, so the predicted ridge polylines do not land exactly on the ridges they
name, and both methods are being scored on partly mismatched pixels. The correlations are
a lower bound. But the saturation visible in the depth map is not a pose artifact, and no
correspondence fix recovers information the model never encoded.

### What this changes about the open question

The requirement was never metric depth -- the DEM already gives exact ranges. What is
missing is which *image* pixels belong to which layer, i.e. **ordinal layer segmentation**,
and a saturated depth map cannot supply an ordering it does not represent. Options that
remain: contrast/texture statistics computed *within* a band rather than across the frame,
so the row confound cannot leak in; or a segmentation model asked for boundaries rather
than depth. Either way the partial-correlation control above is the acceptance test, and
it is now written down.

## Classical edge detection: the same confound, and a flat alignment surface (2026-09-09)

Tried after the depth models failed, on the reasonable argument that a silhouette is an
edge and that edge *contrast* has a physical claim depth did not: airlight adds a constant
to both sides of a ridge line, so it cancels in the difference, while intrinsic contrast is
attenuated by exp(-beta*d). Edge magnitude should therefore fall off with range while being
immune to the additive haze offset.

Vertical Sobel on a horizontally smoothed gray image -- smoothing along rows because ridges
are near-horizontal, which lifts a long faint crest above noise while leaving masts and
poles unreinforced.

### Three tests, three negatives

**Does edge strength rank range?** No. Partial Spearman with image row held fixed:
-0.04 to +0.19 across six cameras. Same result as the dark channel and Depth Anything.

**Are edges even present at predicted crests?** The first measurement said yes -- median
gradient at predicted crest pixels was **1.7 to 3.6x** the column median. That figure is
wrong, and wrong in exactly the way this file had just finished warning about. Crests
concentrate in the middle rows where terrain texture lives, while a column median is
dragged down by flat sky. Against a **row-matched** control -- the same rows, random
columns -- the ratio is **0.76 to 1.09**. No better than chance, and below it on three
cameras.

| camera | vs column median | vs same-row random |
|---|---|---|
| sm-s | 2.04 | 0.76 |
| sm-n | 2.22 | 0.89 |
| bh-n | 1.89 | 0.77 |
| stgo-e | 3.56 | 1.09 |
| wc-n | 1.69 | 1.08 |
| vo-n | 2.31 | 1.07 |

**Does the whole stack align anywhere?** This is the only pose-robust test of the three,
and the one worth having run. Render every predicted ridge as a soft mask, correlate it
against the edge map by FFT, and look across every shift in +-400 px of pitch and +-260 px
of azimuth. If the layered structure is in the image, one shift should line the whole stack
up at once and stand clear of the rest.

It does not. Peak z is 1.3 to 2.0, and the peak stands clear of the best rival shift by
**0.01 to 0.08** -- a flat surface with no unique solution. Four of six pegged near the dy
search bound, which is the usual signature of no peak at all rather than a peak outside the
window.

### What this does and does not establish

It does not establish that ridges are invisible to classical CV. Two limitations are real.
The search was a **rigid translation**, but the actual pose error is pitch, roll, azimuth
and distortion together, which is not a pure shift -- a stack that would align under the
right four-parameter warp can look unalignable under two. And raw gradient magnitude is
dominated by terrain texture, roads and vegetation boundaries, which are as horizontal as
the ridges and far more numerous.

What it does establish is that **nothing tried so far -- learned depth, haze physics, or
edge magnitude -- carries range information once image row is controlled for.** Three
independent methods, one control, same answer. The next attempt should extract *long
coherent* structures rather than per-pixel response (dynamic programming along a path,
or matching the count and spacing of layer boundaries per column rather than pixel
overlap), and it must clear the row-matched control before anything else is claimed.

### Why none of it worked: the step is at the noise floor (2026-09-09)

The edge maps make the statistics unnecessary. On `sm-s` a bright continuous line marks the
near ridge on the left of frame, and the hazy right two-thirds -- where the layered ridges
actually are -- is a near-uniform wash with no gradient structure at all. The brightest
features in the whole image are the antenna masts.

Measured directly: the gray-level step across a predicted crest, smoothed along the ridge
only, against the standard deviation of a flat sky patch put through the same filter.

**Noise floor: 4.48 gray levels.**

| range | n | median step | p75 | below 2 levels |
|---|---|---|---|---|
| 2-4 km | 9 230 | 5.99 | 13.89 | 22% |
| 4-8 km | 6 548 | 6.70 | 13.64 | 19% |
| 8-15 km | 4 844 | 5.68 | 11.09 | 24% |
| 15-25 km | 1 152 | 4.49 | 9.88 | 28% |
| **25-45 km** | 611 | **1.52** | 2.74 | **62%** |
| 45-90 km | 120 | 3.95 | 4.79 | 22% |

A ridge step is about **6 gray levels against a 4.48-level floor -- SNR near 1.3** even in
the near field, and at 25-45 km it is 1.5 levels, three times *below* the floor, with 62%
of crests under two levels. The last row is 120 samples and should not be read as a
recovery; it is small-n, and those crests are mostly true skyline against bright sky.

That is the answer to why learned depth, haze inversion and edge magnitude all returned the
same nothing. It is not that the methods are unsuited. **Beyond roughly 20 km the ridge
step is not in the pixels** -- haze has taken it below what an 8-bit JPEG preserves. No
filter, classical or learned, recovers information that was destroyed before the file was
written.

Two caveats hold the claim honest. Pose error means these samples are taken slightly off
the true crests, which *understates* the step, so the numbers are a lower bound. And the
noise floor measured on sky includes JPEG blocking, which is the relevant floor for this
data but not a property of the scene.

What survives is narrow and worth keeping: the **outermost** silhouette does produce a
strong, clean, continuous edge, on every camera looked at. That is a much better skyline
extractor than `observed_skyline`'s blue-dominance heuristic, and it needs no model and no
sky segmentation. It just cannot deliver the *layers*, which was the thing wanted.

Diagnostic figures: `python -m src.figlib.fig_edges [cameras]` -> `out/edges/`.

## Segmentation and per-plume wind fits: measured, and none beat the box (2026-09-10)

`wind.py` corrects the downwind centroid bias by taking the box's upwind *edge*. That edge
is set by whichever plume pixel reaches furthest, which is almost always high in the column
where the smoke has drifted longest -- so it over-corrects in the axis the centroid
under-corrects. A pixel mask can do better in principle: the **foot** of the mask, the
lowest visible smoke, is the least-drifted part of the plume and the closest thing in the
image to the source. Two mask sources, both training-free so neither adds a FIgLib
contamination asterisk (`masks.py`): **diff**, the temporal-differencing mask `detect_diff`
already builds; **sam**, Segment Anything prompted with the pyronear box.

Five ways to turn a mask into a source column (`plumefit.py`), each scored as its own
geolocation variant against `upwind`, pairwise on the fires where both solved, split by
tier:

| variant | masks used | confirmed n=10 | probable n=16 | all n=26 (better/worse) |
|---|---|---|---|---|
| **upwind (box)** | -- | **1.86** | **3.81** | **2.53** |
| foot (diff) | 75/93 | 1.97 | 4.16 | 2.89  (11 / 12) |
| foot (sam) | 93/93 | 2.00 | 4.50 | 2.49  (9 / 13) |
| axis PCA (diff) | 33/93 | 1.86 | 4.07 | 2.85  (3 / 10) |
| axis PCA (sam) | 43/93 | 1.84 | 4.77 | 2.87  (6 / 10) |
| wedge apex (diff) | 25/93 | 2.25 | 3.81 | 2.75  (1 / 10) |
| wedge apex (sam) | 32/93 | 2.15 | 3.81 | 2.75  (2 / 12) |
| sequence apex (diff) | 60/93 | 1.90 | 6.61 | 3.09  (8 / 12) |
| sequence apex (sam) | 68/93 | 1.76 | 4.67 | 3.15  (6 / 13) |
| field log-lik curve (diff) | 68/93 | 2.83 | 5.34 | 3.52  (6 / 14) |
| field log-lik curve (sam) | 92/93 | 3.00 | 4.93 | 3.36  (8 / 17) |

**Not one variant wins.** The closest is `sequence apex (sam)` at 1.76 km on the confirmed
tier -- but 2 fires better, 5 worse, and it costs the probable tier badly. On the full 26
every variant is worse or within noise, and the more the method commits to the pixels the
worse it does: the `field` mode, which hands `geolocate.py` a whole log-likelihood curve
over direction instead of a bearing and a sigma, is the worst of all (2.53 -> 3.36).

The reason is visible in `out/masks/*.jpg` and measured in `plumefit.fit_report`. By the
time the detector is most confident the plume has flattened into a horizontal sheet:

- **the foot degenerates into the mask centroid.** The lowest fifth of a horizontal sheet
  is most of the sheet, so `foot_x` loses exactly the vertical selectivity it was for,
  with extra variance on top.
- **the cone fits refuse.** `wedge_fit` needs width to shrink downward and `axis_fit` needs
  a non-horizontal principal axis; a flattened sheet has neither, so they return `None` on
  60-75% of frames and the variant falls back to the box on most of its bearings. The few
  frames where they do fit are the early, still-vertical ones -- which is the argument for
  the sequence fit, and the sequence fit is the one that blows up the probable tier.
- **SAM's failure mode is worse than diff's.** SAM always returns *a* mask (93/93), but on a
  diffuse plume against haze it snaps to the ridgeline or fills the prompt box, so its extra
  coverage is extra noise, not extra signal.

This is the same shape as the terrain/depth/edge negatives: the information wanted is not in
the pixels at the range and JPEG quality this data has. The box edge is a crude estimator,
but it is crude in a bounded way, and nothing built here improves on it.

Kept opt-in, not deleted: `masks.build_all` builds the cache (~25 min on the GTX 1070,
hours on CPU), then `FIGLIB_VARIANTS=upwind,foot_diff,...` on `geolocate.py` re-scores the
comparison. `python -m src.figlib.plumefit [diff|sam]` prints why each fit refused.
Diagnostic stills: `python -m src.figlib.masks <fire_id>` -> `out/masks/`; mask video
`python -c "from src.figlib.masks import mask_video; mask_video('<fire_id>')"`.

## Open questions

**Lens distortion and pose -- settled 2026-09-09, and not the way it first looked.**
See "Terrain-refined pose" below. Short version: the published *azimuths* hold up, which
is the only part of the pose that reaches a bearing; the published pitch does not, and
does not matter; and no distortion coefficient recoverable from terrain improves
geolocation. Kilometer errors are no longer provisional on this.

**Contamination.** Every pyronear model, and SmokeyNet, trains on FIgLib -- see
`models/README.md`. Detection and timing numbers are a labeled reference point, never a
generalisation claim. Geolocation is unaffected: kilometer error against official
coordinates tests geometry, and a memorised detection still yields a valid bearing.

**Replace the skyline extractor -- the highest-value open item in the terrain thread.**
`observed_skyline`'s blue-dominance heuristic is the limiting factor on everything above.
Options, cheapest first: (1) a **sky-segmentation CNN pretrained on ADE20K**, which has a
`sky` class -- SegFormer-B0/B2 runs on CPU or the M3's ANE, no training, and would work on
the monochrome units too; (2) classical **dynamic-programming skyline extraction** over an
edge map, exploiting the constraint that the skyline is single-valued in each column --
cheap and no dependency; (3) the **mountain-skyline CNNs from the geolocalisation
literature** (PeakLens and the Baatz/Saurer line of work), which are trained for exactly
this and handle layered ridges. Until one of these is in, the 106 px audit figure and the
whole pose question stay unresolved.

**Match layered ridges, not one skyline.** The frames plainly show three to five nested
ridgelines, and `horizon()` returns only the outermost. Local maxima of elevation angle
*along each ray* would give the secondary silhouettes too. Matching a set of ridges rather
than a single curve is far more constraining, and it attacks the azimuth-identifiability
problem directly -- nested ridges at different ranges parallax against each other, which a
single skyline cannot.

**Not yet validated:** the 23 `probable`-tier fires are resolved by geometry and timing
but not by name. Nothing has yet confirmed one visually the way the confirmed tier was.
They are 23 of the 33 ground-truth fires, so they carry real weight.

## Housekeeping that will bite later

**Partly addressed 2026-09-09:** `docs/figures/` now holds committed copies of
`edge_m3.png` and `falsealarm.png`, and `docs/edge-m3.md` is a linkable write-up.

**Addressed 2026-09-10:** the Kitchen fire animation is committed as a 3 MB GIF
(`docs/figures/triangulate_kitchenfire.gif`, 760 px, thinned frames) and linked from the
README. The other four mp4s in `out/videos/` (21 MB) still have nowhere to live; a GIF
each is the cheapest fix if they are wanted.

**Tests, 2026-09-10.** `tests/` runs the geometry against committed metadata and one
fire's detections in `tests/fixtures/yolo/` — no 13 GB download. `test_geom.py` pins the
bearing/projection/field maths; `test_geolocation.py` and `test_accumulate.py` are
end-to-end (Kitchen fire to 0.08 km, plus the bounded-influence property of the mixture).
`requirements-dev.txt` is numpy + pytest only. GitHub Actions (`.github/workflows/ci.yml`)
runs `compileall` and `pytest` on every push.


**Figures and videos live under `out/`, which is gitignored.** That is right for 21 MB of mp4 and
for anything regenerable, but it means a README cannot yet link to any of them. Before publishing,
either move the handful of README figures into a tracked `figures/` directory, or generate them in
CI. The videos are too large for git regardless and need hosting or conversion to short animated
GIFs.

**Monochrome/NIR data is already in hand.** The Club fire animation shows color *and* monochrome
views of the same two sites, so the deferred "does NIR see smoke earlier" question has usable
paired data sitting in the existing download -- no new fetch needed.

## Deliberately deferred

Monochrome/NIR sequences (11 of them, paired with color views of the same fires) --
"does NIR see smoke earlier" is a real question, saved for later. Terrain: flat-earth
triangulation first, ray-terrain intersection against Copernicus DEM GLO-30 as a
refinement if time allows.
