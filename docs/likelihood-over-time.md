# Likelihood over time: what other fields already know

This project builds a likelihood over the ground from camera bearings, then asks two
time-dependent questions of it: *when* to raise an alarm, and *where* the fire is as
evidence arrives. Both questions have long histories elsewhere — robotics, sonar,
seismology, radar, statistical process control. This note maps those methods onto the
specific problems here and ranks the ones worth trying.

> Citations are from memory and have not been re-checked against the sources. Verify
> before quoting any of them.

## Where the project stands

| concern | current approach | code |
|---|---|---|
| location, reported | each camera's single most confident detection in 0–2400 s | [geolocate.py](../src/figlib/geolocate.py) |
| location over time | every detection up to t, added into one log-likelihood surface | [accumulate.py](../src/figlib/accumulate.py) |
| outliers | uniform mixture term, α = 0.25 | [accumulate.py](../src/figlib/accumulate.py) |
| alarm decision | per-frame threshold τ, k-of-m persistence, 600 s refractory | [falsealarm.py](../src/figlib/falsealarm.py) |
| cross-camera alarm | two sites within 3 km and 180 s, each detection already above τ | [falsealarm.py](../src/figlib/falsealarm.py) |
| plume drift | upwind box edge from hourly wind; `pick="earliest"` | [wind.py](../src/figlib/wind.py), [evolve.py](../src/figlib/evolve.py) |

## 1. Accumulating evidence on a grid

**Occupancy grids** in robotics (Moravec & Elfes, 1985) are the closest analogue to
`accumulate.posterior`: every cell holds a log-odds value and each measurement adds to
it. Mature implementations add two safeguards:

- **Clamping** (OctoMap, Hornung et al., 2013) bounds each cell's log-odds, so a long run
  of observations cannot drive a cell to certainty.
- **Decay** pulls old evidence back toward the prior in maps of scenes that change.

Here, log-likelihoods are added without either. A 40-minute sequence contributes roughly
40 near-identical bearings from one camera, and the surface treats each as new
information.

## 2. Repeated measurements from one sensor are not independent

This is the lesson that matters most for this project.

**Bearings-only target motion analysis** in sonar (Nardone & Aidala, 1981, and the
literature after it) found that repeated bearings from one sensor share its bias.
Averaging them shrinks the apparent uncertainty and leaves the error in place.

**Seismic event location** addresses it directly. **Bayesloc** (Myers, Johannesson &
Hanley, 2007) estimates a correction for each station jointly with the event location
and marginalizes over it, rather than assuming stations are perfectly known.

That is this project's situation. Camera azimuths are nameplate values: `az` is exactly
0/90/180/270 on 482 of 505 cameras, which is a per-station pointing bias.

**The symptom is already visible.** In the Kitchen fire animation the 95% region holds at
about 2.6 km² — an equivalent radius near 0.9 km — while the error sits at 1.33 km for
most of the sequence. The official ignition point is outside the stated 95% region for
most of the fire's duration. The uncertainty is overconfident.

Standard remedies:

- **A bias term per camera.** A pointing offset with a prior of a few degrees,
  marginalized out, so detections from one camera share it instead of acting independent.
- **Effective sample size.** Down-weight n correlated detections to count as a few.
- **Covariance intersection** (Julier & Uhlmann, 1997), for fusing estimates whose
  correlation is unknown.

## 3. Deciding when to alarm

**Quickest change detection** is the formal theory behind the seconds-to-alert against
false alarms per camera-day curve.

- **CUSUM** (Page, 1954) and the **sequential probability ratio test** (Wald, 1945) are
  optimal in a precise sense (Lorden, 1971) for the trade between detection delay and
  false-alarm rate.
- The field's standard summary — expected detection delay against average run length to
  false alarm — is the same plot as [falsealarm.png](figures/falsealarm.png), in different
  vocabulary.

CUSUM keeps one number per camera,

```
S = max(0, S + log-likelihood-ratio(frame))
```

and alarms when S crosses a threshold. It is the principled form of k-of-m persistence:

- **Partial credit.** A frame at 0.45 confidence contributes part of an alarm. Under
  k-of-m it contributes only if it clears τ.
- **Self-resetting.** S falls back to zero through clear sky, with no fixed window to tune.

Persistence already cuts the false-alarm rate by about a quarter at matched recall for one
minute of latency, which suggests CUSUM has room to do better.

## 4. Using evidence below the threshold

**Track-before-detect**, from radar and infrared search-and-track (for example Barniv,
1985), does not threshold individual frames. It integrates weak returns over time and
space, and thresholds only the integrated evidence.

A plume at 180 s produces weak detections. The current alarm rules discard everything
under τ, but two sites' weak evidence along crossing bearings may clear a joint threshold
before either camera does alone. The existing cross-site rule cannot see this, because it
requires each detection to pass τ first. Track-before-detect is therefore the fairer test
of the finding that geometry buys latency but not recall.

## 5. An observation model that changes with time

Kalman filters absorb change through process noise, and tracking systems let the
measurement model depend on elapsed time. For a wildfire the source is fixed at ignition,
but the visible plume drifts downwind as it rises, so bearing bias *grows* with time since
appearance.

- **Widen σ with plume age**, so early bearings carry more weight than late ones. The
  existing `pick="earliest"` option is a coarse version of this.
- **Model the bias explicitly** as a function of plume age and crosswind. This turns the
  question in [evolve.py](../src/figlib/evolve.py) into a term in the likelihood.

## 6. False detections and multiple sources

**Robust back-ends in SLAM** — max-mixtures (Olson & Agarwal, 2012) and switchable
constraints (Sünderhauf & Protzel, 2012) — are the temporal relatives of the α = 0.25
uniform outlier term. They handle a transient false positive well: in the Kitchen fire
animation, a bird in front of `lp-e-mobo-c` at 0.45 confidence moves the estimate from
1.33 km to 1.66 km error, and it recovers when a better detection replaces it.

**Multi-target tracking** handles several fires, or a recurring false plume, at once:

- multiple hypothesis tracking (Reid, 1979)
- joint probabilistic data association (Bar-Shalom)
- probability hypothesis density filters (Mahler, 2003)

These keep competing hypotheses alive over time instead of committing to a single peak on
the surface. That matters for days such as `20180603_FIRE`, which holds three separate
ignitions under one label.

## 7. Space-time scan statistics

Disease surveillance raises alarms with Kulldorff's **prospective space-time permutation
scan statistic** (SaTScan, 2005), which corrects for testing many locations and windows at
once. At 505 cameras × 1,440 frames per day, that multiple-testing burden is the same one
this network carries. It is more a framing than a drop-in method here.

## Wildfire systems specifically

- **SmokeyNet** (Dewangan et al., 2022, the paper that introduced FIgLib) uses time
  *inside* the model, stacking consecutive frames through a CNN, an LSTM and a vision
  transformer. It does no multi-camera localization.
- **pyronear's engine**, to the best of recollection, requires detections over
  consecutive frames before alerting: a persistence rule like k-of-m.
- **Operational camera networks** (Pano AI, ALERTWildfire operators) triangulate, but no
  published probabilistic model of evidence accumulating over time is known here. That is
  a gap this project can reasonably occupy.

## What to try, cheapest first

1. **CUSUM per camera** in `falsealarm.py`, plotted against the single-frame and 2-of-3
   curves. A small change that competes directly with a headline result.
2. **A coverage check.** Across the confirmed fires, how often does the 95% region contain
   the official ignition point? If well under 95%, the regions are overconfident — a
   clean, honest result either way.
3. **A per-camera bias term** in `solve`, Bayesloc-style. It should repair coverage, it
   reuses the robust-likelihood machinery, and it is the principled response to nameplate
   azimuths where the terrain pose fit made error worse.
4. **Cross-site track-before-detect**, as the fair test of whether geometry buys latency.
