# Models

Weights are fetched, not committed.

## pyronear yolo11s (rapid-raccoon v8.1.0) — Apache 2.0

```sh
curl -sL -o rr.tar.gz https://huggingface.co/pyronear/yolo11s_rapid-raccoon_v8.1.0/resolve/main/onnx_cpu.tar.gz
tar xzf rr.tar.gz && mv best.onnx pyronear_rr_v8.1.0.onnx && rm rr.tar.gz
```

**This model was trained on FIgLib.** `pyro-dataset`'s raw-data README lists
`FIGLIB_ANNOTATED_RESIZED` — "re-annotated dataset from the Fire Ignition images
Library" — as its first source. Detection and timing figures produced with it on FIgLib
therefore measure memorization as well as detection, and are reported here as a labeled
reference point rather than a generalization claim. The same applies to every other
pyronear release and to SmokeyNet, which is the FIgLib paper's own model.

Geolocation results are unaffected: kilometer error against official ignition
coordinates tests geometry, not generalization, and a memorized detection still yields a
valid bearing.

## Core ML variants — built, not fetched

The M3 measurements in [`docs/edge-m3.md`](../docs/edge-m3.md) run three exports of the same
weights, produced on macOS with `requirements-edge.txt` installed:

| variant | file | size |
|---|---|---|
| FP32 | `pyronear_rr_v8.1.0.mlpackage` | 36.3 MB |
| FP16 | `pyronear_rr_v8.1.0_fp16.mlpackage` | 18.2 MB |
| INT8-weight | `pyronear_rr_v8.1.0_int8w.mlpackage` | 9.3 MB |

They live flat in `models/`, next to the `.pt`/`.onnx` weights — `detect_coreml.py` and
`bench_edge.py` both resolve `MODELS / "pyronear_rr_v8.1.0[_variant].mlpackage"` with no
`coreml/` prefix, so that's where the export step needs to put them.

Sizes and op placement are recorded in `out/edge_provenance.json`, alongside a parity check
against ONNX (24 frames, 14/14 detections matched, max Δconfidence 0.0).

`.mlpackage` is gitignored — the packages are reproducible from the ONNX weights above, and
the measurements they produced are committed instead (`data/edge/pm-m3-20260909.txt.gz`, and
the figure).

Two traps worth knowing before re-exporting:

- **Ultralytics writes every export to the same path.** Exporting FP16 silently moves the FP32
  `.mlpackage` aside. Rename each one before exporting the next.
- **`MLComputePlan.load_from_path` aborts the process** when handed an `.mlpackage` — a C++
  `libc++abi` abort that no Python `except` can catch, and it surfaces as a macOS crash dialog.
  Load `ct.models.MLModel(path)` first and pass `.get_compiled_model_path()` instead.
