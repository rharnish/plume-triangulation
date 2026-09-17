# Camera calibration from the night sky

The published HPWREN camera table is a nameplate. Azimuths are rounded to a compass quadrant,
the field of view is nominal, and there is no lens model. A bearing is only as good as that
table, so every kilometer figure in this project depends on it. Stars measure the real pose
without a site visit, from frames the cameras already record every night.

![Star tracks against catalog stars under the published and the star-solved pose](figures/star_pose_correction.jpg)

*Green: a star's track over the night. Magenta: that catalog star under the solved pose,
running inside the track. Orange: the same star under the published pose; the yellow arrows
run from one to the other.*

## How a solve works

A point that drifts at the sidereal rate is a star. Matching a night of such tracks to a star
catalog under one shared pose measures a camera's azimuth, pitch, roll and lens together. No
constellation is guessed first: the pose search scores how many catalog stars land on *any*
track, then assigns stars to whole tracks and refits.

## What the solves found

- **86 solves on 52 cameras** in the [ledger](../data/meta/pose_ledger.json), from FIgLib's
  night sequences and moonless nights pulled from HPWREN's public CDN, at a median residual of
  1.3 px and ~22 stars per solve. [Every solve, with its lens scale](figures/star_ledger.png).
- **33 of the 52 cameras point more than 1° from their published azimuth**, mlo-s-mobo-c by
  23°. Consecutive nights agree to 0.01°, except one weak lp-n-mobo-c solve (12 stars at
  2.6 px) that sits 0.25° from the nights either side. Nights two months apart agree to 0.15°.
- **Cameras do get re-aimed.** Otay Mountain's south cameras solve 10.5° off in 2019 and 0.4°
  in 2024, and Toro Peak West moved 7.5° between 2021 and 2026. So corrections are kept per
  camera *and* date, in a ledger that refuses to bridge a re-aim or a sensor change.
- **Not the lens the pipeline assumed.** Every ledger solve puts the focal scale at
  0.877–0.893 of nameplate: an equidistant fisheye spanning about ±55°, not a rectilinear ±45°.
  Near the frame edge that was worth up to 8° of bearing. Three Big Black Mountain cameras turn
  out to be a second lens group, at 0.774–0.779 ([below](#the-poles-length-measures-the-lens)).

## An independent check

A sea horizon's dip below level depends only on the camera's height. On om-w-mobo-c it lies
along the star-solved tilt, not the published level one.

![Sea horizon under the star-solved and the published pose](figures/sea_horizon_check.jpg)

## Pose from the trails, with no star named

Every trail in a frame is carried by one rotation of the sky, `ḋ = ω (p × d)`. Map the track
points into the camera's own frame through the lens alone, which needs no pose, and that
equation is linear in the celestial pole `p`. Least squares gives the pole in closed form,
with no search and no catalog. Against the 86 ledger solves the pole lands a median **0.29°**
from where they put it (81 of 86 within 2°).

The pole fixes two of the three angles. The last one, the turn about the pole, is invisible to
a flow field, so it becomes a 1-D scan at 0.1° instead of a 3-D grid search at 1°.

![From star trails to camera pose, on a camera the grid search never solved](figures/star_solve_process.jpg)

*bm-s-mobo-c, which had never solved. 1: tracks, with nothing named. 2: their velocities and the
closed-form pole. 3: the scan about the pole, where the peak clears the best rival angle by
5.79 to 4.74, a much thinner margin than an easy camera gets. 4: 23 stars at 0.78 px.*

### The pole's length measures the lens

The fit is given the sidereal rate, so `|p|` comes out at 1 only when the pixel-to-angle map is
right. That is why Big Black Mountain's south, west and east cameras never solved: they need
0.774–0.779 of nameplate, not the shared 0.886, a 150 px error at the frame edge.

The same number rejects frames with no coherent rotation. marconi-n, starr-n and sjh-n return
residuals at about 100% of the sidereal rate, so on the nights they were tried they show lit
cloud and noise, not a faint star field.

The measurement is weaker near the pole. Trails close to it carry little scale information, so
a north-facing camera's lens comes out 3–4% low. The solver never imposes the pole's lens: it
runs the published grid, the pole under the shared lens and the pole under the measured lens as
separate attempts, and keeps the best finished solve.

### More solves, none lost

On the 111 sequences the grid search attempted, running both methods takes **86 solves to 91
and 52 cameras to 55**. The 86 already solved keep their poses (median change 0.000°, max
0.094°). The five new solves are not yet in the ledger, so the kilometer figures in the README do
not use them.

## Moonlight does not matter

Three cameras that solve cleanly in the dark were re-solved on 23 night blocks from full moon
to new, all within two weeks. All 23 solve, star counts do not fall, and every pose stays
within 0.07° of that camera's dark-night pose.

![Stars matched and pose change against moonlight](figures/star_moon_ladder.jpg)

That roughly quadruples the usable nights inside HPWREN's 90-day public window. One
qualification: the moon was above the frame on 20 of the 23 blocks, so a full moon *inside* the
frame is still untested.

## A second night is the better acceptance test

Two nights of one camera are independent measurements of one pose, and noise does not repeat
across them. Solves with 20+ stars agree with another night of the same camera to 0.01°, and
8–11-star solves to 0.3°. Star count does track reliability, but the ≥ 8 cutoff is a blunt
stand-in for checking that another night agrees.

## Running it

```sh
python -m src.figlib.stars.nights 20260911 @cams.json   # moonless-night frames from HPWREN's public CDN
python -m src.figlib.stars.run_nights                   # track, star-solve, rebuild data/meta/pose_ledger.json
python -m src.figlib.stars.moon_test                    # the moon-phase ladder
python -m src.figlib.stars.cross_night                  # night-against-night agreement
python -m src.figlib.stars.fig_solve_process            # the four-stage figure
```

The full account, including the sign convention that had to be measured rather than derived, is
in [NOTES.md](../NOTES.md).
