"""Star calibration as this project uses it: FIgLib and CDN sequences, by name.

The calibration itself -- catalog, lens model, track extraction, pole, solver, ledger rules --
is the star-calibration library (`star_calibration`, github.com/rharnish/star-calibration),
split out with its history after the 2026-09-26 re-run. HPWREN's camera table and CDN night
frames come from its `star_calibration.hpwren` package and the shared cache it keeps.

What stays here is what depends on this project's data or its terrain work:

  solve, tracks   sequence names -> the library's Night: FIgLib archives and CDN blocks alike,
                  tracks and results cached in out/sky/data/star_tracks
  run_nights      track and solve every CDN block, then rebuild data/meta/pose_ledger.json
  resolve_ledger  re-solve a ledger's sequences under the current sky model
  skyline_check   the DEM skyline under each star pose, against the image
  window_ablation how much of a night a solve needs
  fig_*           the figures in README.md and docs/star-calibration.md
"""
