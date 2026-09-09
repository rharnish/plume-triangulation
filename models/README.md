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
therefore measure memorisation as well as detection, and are reported here as a labelled
reference point rather than a generalisation claim. The same applies to every other
pyronear release and to SmokeyNet, which is the FIgLib paper's own model.

Geolocation results are unaffected: kilometre error against official ignition
coordinates tests geometry, not generalisation, and a memorised detection still yields a
valid bearing.
