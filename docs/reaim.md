# Re-aim check: did the camera move between the fire and its star solve?

A star solve measures a camera's pose on the night it was solved. The pose ledger
(`star_calibration.ledger`) carries that pose to a fire up to a year away on either side,
assuming the camera didn't move in between. Two solves either side of a fire catch a re-aim,
because they disagree. One solve can't. This tool checks the assumption directly: it compares
a daytime frame from the fire with a daytime frame from the solve's date.

It started as a guard. It could also become a way to recover calibration for old daytime
footage, where only today's camera can be star-solved ([below](#where-this-could-go-calibrating-old-daytime-footage)).

## Why it's needed

**mlo-s-mobo-c (Mt. Laguna Observatory South).** It star-solves at −23.3° from its published
azimuth on every night of 2026-07-21 to −24 (nights agree to 0.04°). The Posta fire,
2025-07-24, is 362 days earlier. That's inside the ledger's one-sided window, so the solve
would carry. Two things say it shouldn't:

- The camera's Posta and Kitchen bearings already agree with the official points to within 1°
  under the *published* azimuth.
- The daytime frames from the two dates are the same scene turned about 18°.

The camera was re-aimed between July 2025 and July 2026. Carrying the solve back would have
broken a bearing that was right.

## The tool

```
python -m src.figlib.stars.reaim                                     # the default pairs
python -m src.figlib.stars.reaim <camera> <fire> <solve-date> ...    # others
cd out/reaim && python3 -m http.server 8010                          # then /viewer.html
```

For each (camera, fire, solve date), `src/figlib/stars/reaim.py`:

1. Takes frame **A**, the FIgLib sequence's first frame (40 min before ignition).
2. Fetches frame **B**, from HPWREN's public CDN on the solve date, at the same local time of
   day so the shadows fall the same way.
3. Estimates the shift that lays B over A:
   - ORB feature matches inside a scene mask. The banner, logos, watermark and lens rim sit
     still in every frame and would otherwise vote for zero.
   - The median displacement of the matches that agree on one translation.
   - Converted to degrees near the frame's middle.

`reaim_viewer.html` overlays the two frames:

- **Compare:** an opacity slider, plus blend, difference and flicker modes.
- **Align:** a shift slider for B, read out in pixels and degrees, and "use estimate".
- **Verdict:** same / moved / unsure, exported as JSON.

Judged pairs are kept in `data/meta/reaim_verdicts.json`.

## First results (2026-10-01)

The seven pairs behind the CDN star solves of 2026-10-01, and tp-w, whose re-aim was already
known:

| camera | fire → solve date | verdict | aligned by hand | estimate |
|---|---|---|---|---|
| mlo-s-mobo-c | Posta 2025-07-24 → 2026-07-21 | **moved** | −18.1° | −18.1° (30% of matches agree) |
| tp-w-mobo-c | Grove 2024-07-24 → 2026-09-12 | **moved** | +2.1° | +2.1° (69%) |
| ws-w-mobo-c | Posta → 2026-07-21 | same | 0 | 0.0° (90%) |
| starr-n-mobo-c | Clubfire 2025-08-23 → 2026-08-20 | same | 0 | −0.07° (11%) |
| starr-n-mobo-m | Clubfire → 2026-08-20 | same | 0 | 0.0° (2%) |
| mpo-w-mobo-c | Crosley 2026-03-19 → 2026-07-04 | same | 0 | 0.0° (39%) |
| mp-n-mobo-m | Scissors 2025-11-02 → 2026-07-03 | unsure (counts as same) | +0.07° | +0.09° (10%) |

- **tp-w agrees with earlier work.** The README's skyline match put tp-w's Grove re-aim at
  2.4°.
- **Weak agreement isn't wrong.** On the "same" pairs a low agreement share mostly reflects
  changed light, cloud and vegetation, not a disagreement about the shift. The hand
  alignments confirm zero.

## What the estimate does not yet do

The estimate works in pixels. It applies no lens model: not the camera's solved lens scale
and radial term, not its optical centre. Its degree figure uses one fixed scale, the shared
lens's pixels per degree at the frame's middle. That has three consequences:

- **It's accurate for small shifts near the middle.** That covers every "same" pair above.
- **It's approximate for large re-aims.** mlo-s's 18° moves most features well off the
  middle, where the fisheye compresses.
- **A rotation isn't a translation in a fisheye.** Turning the camera curves and stretches
  the scene differently across the frame, and a pitch or roll change adds vertical and
  rotational motion. So no single shift lays the two frames over each other everywhere, by
  slider or by estimate. The viewer says to align on features near the middle line for this
  reason.

## Next steps

1. **A lens-aware rotation estimate.**
   - Unproject each matched feature into a viewing direction, through the camera's solved lens
     and optical centre (`star_calibration.pole.pixel_to_cam`; the ledger's `k_ratio`, `k1`,
     `cx`, `cy`).
   - Fit the 3D rotation that maps one set of directions onto the other, robustly (Wahba's
     problem, as `star_calibration.intrinsics` already fits poses).
   - The result is the re-aim as a true change in azimuth, pitch and roll, valid at any size.
     Show it in the viewer, and report a residual so a poor fit is visible.
2. **"Approve estimate" in the viewer.** A verdict that records "moved, by this rotation",
   alongside same / moved / unsure.
3. **The ledger guard.**
   - When a solve would carry to a fire, look up the (camera, fire, solve date) verdict.
   - "Same" or "unsure": carry it.
   - "Moved": refuse it, or with an approved rotation, carry it with the rotation taken off
     (below).
   - Pairs with no verdict carry as today, until someone checks them.
4. **Import.** Add the CDN solves of 2026-10-01 through the guard, plus the library's
   unimported bm-s and bm-e solves, and re-run (`rerun_results.sh`).

## Where this could go: calibrating old daytime footage

Star solves need night frames, and HPWREN's CDN keeps only ~90 days public. So most of the
archive's daytime fire footage has no solve within reach. The camera as it is today can be
solved, though, and its daytime view today can be compared with its daytime view on any old
date:

    pose(old date) = pose(tonight, from the stars) ∘ rotation(today's daytime view → old daytime view)

The star solve fixes the camera absolutely. The daytime registration carries it back through
any re-aim, with no night frames from the old date. With step 1 in place, the "moved" verdict
for mlo-s at Posta becomes a calibrated Posta pose instead of a fallback to the published
azimuth.

**Where it holds.** The camera must be the same unit with the same lens, so that only its
aim changed. A replaced unit changes the frame size, the lens scale or the optical centre, and
the registration residual should show it. Seasons, snow, haze and new structures cut the
matchable features. Low match agreement and the fit's residual are the signals to fall back.

**How to test it before trusting it.** Use cameras with star solves on both sides of a re-aim.

- **tp-w-mobo-c:**
  - It solves at −6.4° (read at the frame's middle) on a FIgLib night, 2021-01-15 (Bonita),
    and at +1.0° in September 2026: a 7.4° re-aim.
  - Registering a 2021-01-15 daytime frame against a 2026-09 daytime frame should recover that
    7.4° from the daytime frames alone. The 2026 solve plus the rotation should land on the
    2021 solve.
  - The disagreement is the method's error.
- **Same-unit pairs, for the floor.** Any camera with FIgLib night solves and CDN solves on
  both sides of a quiet period should register to zero.
- **An independent check.** The DEM skyline (`stars.skyline_check`) can test the transferred
  pose on the old daytime frame directly.

## Files

- `src/figlib/stars/reaim.py`: pairs, frames, the estimate
- `src/figlib/stars/reaim_viewer.html`: the viewer, copied into `out/reaim/`
- `data/meta/reaim_verdicts.json`: judged pairs
- NOTES.md, 2026-10-01: how mlo-s's re-aim was found
