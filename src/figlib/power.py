"""Slice a `powermetrics` log by wall clock and attribute watts to benchmark runs.

`powermetrics` needs root, so it runs as a separate long-lived logger rather than
something the harness starts and stops. That makes wall-clock alignment the whole
problem: each condition in `bench_edge` records the epoch seconds it started and
finished, and each powermetrics sample carries a timestamp, so the join is a filter.

Reported per condition: mean CPU, GPU and ANE power over the samples that fall inside
the window, and frames per joule -- which is the number that actually matters for a
solar-powered enclosure, and which no latency figure alone can tell you.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"

STAMP = re.compile(r"\*\*\* Sampled system activity \((.+?)\)")
POWER = re.compile(r"^(CPU|GPU|ANE|Combined) Power(?: \(.*?\))?: (\d+) mW", re.M)
PRESSURE = re.compile(r"Current pressure level: (\w+)")


def _open(path: Path):
    """Read the log whether or not it is compressed.

    The raw `powermetrics` output is the one artifact here that cannot be regenerated --
    the Mac session that produced it is over -- so the committed copy is gzipped, at
    1.5 MB against 28 MB plain.
    """
    import gzip
    return (gzip.open(path, "rt") if path.suffix == ".gz" else open(path))


def _default_log() -> Path:
    """Prefer a local uncompressed run, fall back to the committed gzip."""
    local = OUT / "pm.txt"
    return local if local.exists() else ROOT / "data" / "edge" / "pm-m3-20260909.txt.gz"


def parse(path: Path) -> list[dict]:
    """One record per powermetrics sample: epoch seconds, mW per unit, pressure."""
    with _open(path) as fh:
        text = fh.read()
    blocks = text.split("*** Sampled system activity")
    out = []
    for b in blocks[1:]:
        m = re.match(r" \((.+?)\)", b)
        if not m:
            continue
        try:
            t = datetime.strptime(m.group(1).strip(), "%a %b %d %H:%M:%S %Y %z")
        except ValueError:
            continue
        rec = {"t": t.timestamp()}
        for unit, mw in POWER.findall(b):
            rec[unit.lower()] = int(mw)
        p = PRESSURE.search(b)
        if p:
            rec["pressure"] = p.group(1)
        out.append(rec)
    return out


def window(samples: list[dict], t0: float, t1: float) -> dict:
    """Mean power over samples inside [t0, t1]; the first sample is usually partial."""
    sel = [s for s in samples if t0 <= s["t"] <= t1]
    if len(sel) > 2:
        sel = sel[1:-1]                      # drop the ramp in and the ramp out
    if not sel:
        return {}
    def mean(k):
        v = [s[k] for s in sel if k in s]
        return round(sum(v) / len(v)) if v else None
    return dict(n=len(sel), cpu_mw=mean("cpu"), gpu_mw=mean("gpu"),
                ane_mw=mean("ane"), combined_mw=mean("combined"),
                pressure=sorted({s.get("pressure", "?") for s in sel}))


def figure(pm_path: Path) -> None:
    """Two panels: energy efficiency by placement, and the fanless throttle curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lat = json.loads((OUT / "bench_latency.json").read_text())
    samples = parse(pm_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- efficiency by variant x placement
    variants = ["fp32", "fp16", "int8w"]
    units = ["cpu", "gpu", "ane", "all"]
    color = {"cpu": "#8c8c8c", "gpu": "#1f77b4", "ane": "#d62728", "all": "#2ca02c"}
    w = 0.2
    by = {(r["variant"], r["units"]): r for r in lat["rows"]}
    for j, u in enumerate(units):
        vals = [by[(v, u)]["frames_per_joule"] or 0 for v in variants]
        x = np.arange(len(variants)) + (j - 1.5) * w
        axes[0].bar(x, vals, w, label=u.upper(), color=color[u])
        for xi, val, v in zip(x, vals, variants):
            ane = by[(v, u)]["power_model_only"].get("ane_mw")
            axes[0].text(xi, val + 0.25, f"{val:.1f}", ha="center", fontsize=7.5)
            # Only worth flagging where the Neural Engine was *asked for* and drew
            # nothing. CPU-only and GPU-only reading 0 mW is not a surprise.
            if ane == 0 and u in ("ane", "all"):
                axes[0].text(xi, 0.35, "ANE\n0 mW", ha="center", fontsize=7,
                             color="white", weight="bold")
    axes[0].set_xticks(np.arange(len(variants)))
    axes[0].set_xticklabels(["FP32\n36 MB", "FP16\n18 MB", "INT8-weight\n9.3 MB"])
    axes[0].set_ylabel("frames per joule")
    axes[0].set_title("Energy efficiency by precision and requested placement")
    axes[0].legend(title="requested", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].text(0.02, 0.97,
                 "Ask for the Neural Engine with an FP32 model and it draws 0 mW:\n"
                 "the ANE cannot run FP32, and Core ML reports no error.\n"
                 "INT8 weights change nothing on M3 -- weight-only quantization,\n"
                 "so the win is 9.3 MB, not throughput.",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8,
                 color="0.25")
    axes[0].set_ylim(0, 17)

    # --- throttle curve
    th_path = OUT / "bench_thermal_fp16_ane.json"
    if th_path.exists():
        th = json.loads(th_path.read_text())
        f = th["fps_per_10s"]
        t = np.arange(len(f)) * 10 / 60.0
        axes[1].plot(t, f, color="#d62728", lw=1.4, label="throughput (fps)")
        axes[1].set_xlabel("minutes of sustained inference")
        axes[1].set_ylabel("frames per second", color="#d62728")
        axes[1].set_ylim(0, max(f) * 1.15)
        t0, t1 = th["t"]
        pw = [((s["t"] - t0) / 60, s.get("ane", 0) / 1000)
              for s in samples if t0 <= s["t"] <= t1]
        ax2 = axes[1].twinx()
        px = [p[0] for p in pw]
        py = [p[1] for p in pw]
        ax2.plot(px, py, color="#1f77b4", lw=0.6, alpha=0.3)
        k = 15                                    # rolling median, readability only
        if len(py) > k:
            sm_y = [float(np.median(py[max(0, i - k):i + 1])) for i in range(len(py))]
            ax2.plot(px, sm_y, color="#1f77b4", lw=1.6, label="ANE power (W)")
        ax2.set_ylabel("ANE power (W)", color="#1f77b4")
        ax2.set_ylim(0, 6)
        first = np.mean(f[:3])
        axes[1].axhline(first, ls=":", color="0.4", lw=1)
        axes[1].annotate(
            f"holds {first:.0f} fps for ~8 min, then steps down\n"
            f"{100 * (1 - np.mean(f[-3:]) / first):.1f}% and stays there",
            xy=(12, np.mean(f[-3:])), xytext=(6.0, first * 0.55), fontsize=8.5,
            color="0.25", arrowprops=dict(arrowstyle="->", color="0.5"))
        axes[1].set_ylim(0, max(f) * 1.25)
        axes[1].set_title("Fanless sustained throughput, FP16 on the Neural Engine")

    fig.text(0.5, 0.015, "Apple M3, 8 GB, macOS 15.6, on AC power, Low Power Mode off. "
             "Batch 1. Model-only latency; end-to-end adds ~9.4 ms of decode and letterbox.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    dest = OUT / "figures" / "edge_m3.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    print(f"wrote {dest}")


def main(argv: list[str]) -> None:
    pm = Path(argv[0]) if argv else _default_log()
    samples = parse(pm)
    print(f"{len(samples)} powermetrics samples, "
          f"{datetime.fromtimestamp(samples[0]['t'])} .. "
          f"{datetime.fromtimestamp(samples[-1]['t'])}")

    lat = json.loads((OUT / "bench_latency.json").read_text())
    rows = []
    print(f"\n{'variant':7s} {'unit':5s} {'ms':>7} {'fps':>6} "
          f"{'CPU mW':>7} {'GPU mW':>7} {'ANE mW':>7} {'tot mW':>7} {'frames/J':>9}")
    for r in lat["rows"]:
        w = window(samples, *r["t_model"])
        fps = r["model_only"]["fps"]
        tot = w.get("combined_mw")
        fpj = round(fps / (tot / 1000), 1) if tot else None
        r["power_model_only"] = w
        r["frames_per_joule"] = fpj
        rows.append(r)
        print(f"{r['variant']:7s} {r['units']:5s} "
              f"{r['model_only']['median_ms']:7.2f} {fps:6.1f} "
              f"{str(w.get('cpu_mw')):>7} {str(w.get('gpu_mw')):>7} "
              f"{str(w.get('ane_mw')):>7} {str(tot):>7} {str(fpj):>9}")

    lat["rows"] = rows
    (OUT / "bench_latency.json").write_text(json.dumps(lat, indent=1) + "\n")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
