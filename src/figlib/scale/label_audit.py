"""pyronear's released model against the pyro-sdis labels: how well they agree, and where not.

Two uses. Scoring: how the model the project runs on FIgLib does on its own kind of data,
at box level and image level. Auditing: the places the two disagree, ranked, because the
scale comparison trains on these labels. The model very probably saw these images in
training (pyro-dataset draws on the same SDIS cameras), so the scores are optimistic --
and a disagreement it still has with a label it may have trained on makes that label
more suspect, not less.

Box level. Predictions are matched to labels greedily in descending confidence, each
label at most once, to the unmatched label with the highest IoU that passes the rule.
Because the order is by confidence, the matches of the predictions above any threshold t
are exactly the matches a run at t would make, so one matching at the confidence floor
serves every threshold. TP = matched prediction, FP = unmatched prediction, FN = labels
minus TP. There is no box-level TN.

Rules, because a smoke box has no crisp edge and IoU >= 0.5 mostly scores agreement about
extent rather than existence:
    iou50    IoU >= 0.5, the COCO / ultralytics convention (AP50)
    iou10    IoU >= 0.1, clear overlap -- the headline rule
    center   either box's center lies inside the other
    ioa50    intersection / smaller area >= 0.5, forgives a big box around a small one

Image level. Positive = any label; score = max detection confidence (0 if none). This is
the one place TN exists (the unlabeled negatives). `located_recall` additionally requires
a labeled box to be matched, so a positive image called for a detection elsewhere in the
frame is not counted as found.

Intervals: 95% percentile bootstrap over camera-days, not images -- consecutive frames of
one camera are near-duplicates, and resampling them independently would overstate
precision by treating one event as dozens of observations.

    python -m src.figlib.scale.label_audit val|train [--sheets]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / "data" / "sdis" / "yolo" / "labels"
IMAGES = ROOT / "data" / "sdis" / "yolo" / "images"
PRED = ROOT / "out" / "scale" / "sdis_rr"
OUT = ROOT / "out" / "scale" / "sdis_audit"

T_PROD = 0.2          # pyronear's production confidence threshold
T_AUDIT = 0.5         # "confident" for the unlabeled-detection list
RULES = ("iou50", "iou10", "center", "ioa50")
N_BOOT = 1000


# ------------------------------------------------------------------ loading

def meta(stem: str) -> dict:
    partner, camera, date = stem.split("_", 2)
    return {"partner": partner, "camera": camera, "group": f"{camera}/{date[:10]}"}


def load(split: str) -> list[dict]:
    rows = []
    with open(PRED / f"{split}.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            lab = []
            for l in (LABELS / split / f"{r['image']}.txt").read_text().splitlines():
                if l.strip():
                    _, xc, yc, w, h = map(float, l.split())
                    lab.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
            pred = sorted(((d["conf"], (d["x0"], d["y0"], d["x1"], d["y1"])) for d in r["dets"]),
                          key=lambda p: -p[0])
            rows.append({"image": r["image"], **meta(r["image"]),
                         "labels": np.array(lab, float).reshape(-1, 4),
                         "conf": np.array([p[0] for p in pred], float),
                         "pred": np.array([p[1] for p in pred], float).reshape(-1, 4)})
    return rows


# ------------------------------------------------------------------ geometry

def _area(b):
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def pairwise(a: np.ndarray, b: np.ndarray) -> dict:
    """IoU, intersection-over-smaller and center containment for every (a_i, b_j)."""
    A, B = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]), 0, None)
    ih = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]), 0, None)
    inter = iw * ih
    aa, ab = _area(a)[:, None], _area(b)[None, :]
    iou = inter / (aa + ab - inter + 1e-12)
    ioa = inter / (np.minimum(aa, ab) + 1e-12)

    def inside(c, box):
        return ((c[..., 0] >= box[..., 0]) & (c[..., 0] <= box[..., 2]) &
                (c[..., 1] >= box[..., 1]) & (c[..., 1] <= box[..., 3]))

    ca = np.stack([(A[..., 0] + A[..., 2]) / 2, (A[..., 1] + A[..., 3]) / 2], -1)
    cb = np.stack([(B[..., 0] + B[..., 2]) / 2, (B[..., 1] + B[..., 3]) / 2], -1)
    center = inside(ca, B) | inside(cb, A)
    return {"iou": iou, "ioa": ioa, "center": center}


def passes(p: dict, rule: str, iou_thr: float | None = None) -> np.ndarray:
    if iou_thr is not None:
        return p["iou"] >= iou_thr
    return {"iou50": p["iou"] >= 0.5, "iou10": p["iou"] >= 0.1,
            "center": p["center"], "ioa50": p["ioa"] >= 0.5}[rule]


def match(row: dict, rule: str, iou_thr: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(label index per prediction or -1, IoU of that match) under greedy conf order."""
    n_p, n_l = len(row["pred"]), len(row["labels"])
    m, miou = np.full(n_p, -1), np.zeros(n_p)
    if not n_p or not n_l:
        return m, miou
    p = row.get("_pair") or pairwise(row["pred"], row["labels"])
    row["_pair"] = p
    ok = passes(p, rule, iou_thr)
    used = np.zeros(n_l, bool)
    for i in range(n_p):                       # already sorted by descending conf
        cand = np.where(ok[i] & ~used)[0]
        if cand.size:
            j = cand[np.argmax(p["iou"][i, cand])]
            m[i], miou[i], used[j] = j, p["iou"][i, j], True
    return m, miou


# ------------------------------------------------------------------ metrics

def ap(conf: np.ndarray, tp: np.ndarray, n_labels: int) -> float:
    """COCO 101-point interpolated average precision."""
    if n_labels == 0:
        return float("nan")
    o = np.argsort(-conf, kind="stable")
    ctp = np.cumsum(tp[o])
    rec = ctp / n_labels
    prec = ctp / np.arange(1, len(o) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side="left")
    return float(np.mean(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0)))


def prf(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return {"precision": p, "recall": r, "f1": f}


def binary(tp, fp, tn, fn) -> dict:
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"recall": rec, "fpr": 1 - spec,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "balanced_accuracy": (rec + spec) / 2,
            "mcc": (tp * tn - fp * fn) / den if den else float("nan")}


def roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(groups: dict, fn, seed: int = 0) -> list[float]:
    """95% CI of fn(summed counts) resampling camera-day groups with replacement."""
    keys = list(groups)
    mat = np.array([groups[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    vals = [fn(mat[rng.integers(0, len(keys), len(keys))].sum(0)) for _ in range(N_BOOT)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def rnd(d):
    if isinstance(d, dict):
        return {k: rnd(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [rnd(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else round(float(d), 4)
    if isinstance(d, np.integer):
        return int(d)
    return d


def evaluate(rows: list[dict]) -> tuple[dict, dict]:
    n_lab = sum(len(r["labels"]) for r in rows)
    res: dict = {"images": len(rows), "labels": n_lab,
                 "negative_images": sum(not len(r["labels"]) for r in rows),
                 "predictions_at_floor": int(sum(len(r["pred"]) for r in rows))}

    # --- box level, per rule
    conf_all = np.concatenate([r["conf"] for r in rows])
    matches = {}
    res["box"] = {}
    for rule in RULES:
        mm = [match(r, rule) for r in rows]
        matches[rule] = mm
        tp_flag = np.concatenate([m[0] >= 0 for m in mm]).astype(float)
        keep = conf_all >= T_PROD
        tp = int(tp_flag[keep].sum()); fp = int(keep.sum()) - tp
        res["box"][rule] = {"at_0.2": {"tp": tp, "fp": fp, "fn": n_lab - tp,
                                       **prf(tp, fp, n_lab - tp)},
                            "ap": ap(conf_all, tp_flag, n_lab)}
    res["box"]["ap50_95"] = float(np.mean([
        ap(conf_all, np.concatenate([match(r, None, t)[0] >= 0 for r in rows]).astype(float), n_lab)
        for t in np.arange(0.5, 0.96, 0.05)]))

    # threshold sweep, headline rule
    tp_flag = np.concatenate([m[0] >= 0 for m in matches["iou10"]])
    sweep = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        k = conf_all >= t
        tp = int(tp_flag[k].sum())
        sweep.append({"conf": float(t), **prf(tp, int(k.sum()) - tp, n_lab - tp)})
    res["box"]["iou10_sweep"] = sweep
    res["box"]["iou10_best_f1"] = max(sweep, key=lambda s: s["f1"])

    # --- image level
    y = np.array([len(r["labels"]) > 0 for r in rows])
    score = np.array([r["conf"][0] if len(r["conf"]) else 0.0 for r in rows])
    called = score >= T_PROD
    located = np.array([bool(((m[0] >= 0) & (r["conf"] >= T_PROD)).any())
                        for r, m in zip(rows, matches["iou10"])])
    tp, fp = int((called & y).sum()), int((called & ~y).sum())
    tn, fn = int((~called & ~y).sum()), int((~called & y).sum())
    res["image"] = {"at_0.2": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, **binary(tp, fp, tn, fn),
                               "located_recall": located[y].mean()},
                    "roc_auc": roc_auc(score, y),
                    "pr_auc": ap(score, y.astype(float), int(y.sum()))}

    # --- localization on iou10 matches at 0.2
    ious, ratio, off = [], [], []
    for r, (m, miou) in zip(rows, matches["iou10"]):
        for i in np.where((m >= 0) & (r["conf"] >= T_PROD))[0]:
            p, l = r["pred"][i], r["labels"][m[i]]
            ious.append(miou[i])
            ratio.append(np.log2(_area(p) / _area(l)))
            scale = np.sqrt(_area(l))
            off.append(np.hypot((p[0] + p[2] - l[0] - l[2]) / 2, (p[1] + p[3] - l[1] - l[3]) / 2) / scale)
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)}
    res["localization_iou10_at_0.2"] = {"n": len(ious), "iou": q(ious),
                                        "log2_area_ratio_pred_over_label": q(ratio),
                                        "center_offset_over_label_size": q(off)}

    # --- strata, iou10 at 0.2
    def strat(key_label, key_pred=None):
        lab_tot, lab_hit = defaultdict(int), defaultdict(int)
        pr_tot, pr_hit = defaultdict(int), defaultdict(int)
        for r, (m, _) in zip(rows, matches["iou10"]):
            hit = set(m[(m >= 0) & (r["conf"] >= T_PROD)].tolist())
            for j, l in enumerate(r["labels"]):
                k = key_label(r, l)
                lab_tot[k] += 1; lab_hit[k] += j in hit
            if key_pred:
                for i in np.where(r["conf"] >= T_PROD)[0]:
                    k = key_pred(r, r["pred"][i])
                    pr_tot[k] += 1; pr_hit[k] += m[i] >= 0
        out = {k: {"labels": lab_tot[k], "recall": lab_hit[k] / lab_tot[k]} for k in sorted(lab_tot)}
        for k in pr_tot:
            out.setdefault(k, {}).update({"predictions": pr_tot[k], "precision": pr_hit[k] / pr_tot[k]})
        return out

    def hbin(_r, b):
        h = b[3] - b[1]
        return ("a <0.02" if h < 0.02 else "b 0.02-0.05" if h < 0.05 else
                "c 0.05-0.10" if h < 0.10 else "d >=0.10")

    def ybin(_r, b):
        return f"{min(int(b[3] * 10), 9) / 10:.1f}"

    res["strata"] = {"box_height": strat(hbin, hbin), "partner": strat(lambda r, b: r["partner"],
                                                                       lambda r, b: r["partner"]),
                     "box_bottom_y": strat(ybin, ybin)}
    cams = strat(lambda r, b: r["camera"], lambda r, b: r["camera"])
    res["strata"]["camera_top15"] = dict(sorted(cams.items(), key=lambda kv: -kv[1].get("labels", 0))[:15])
    neg = defaultdict(lambda: [0, 0])
    for r, c in zip(rows, called):
        if not len(r["labels"]):
            neg[r["partner"]][0] += 1; neg[r["partner"]][1] += c
    res["strata"]["image_fpr_by_partner"] = {k: {"negatives": v[0], "fpr": v[1] / v[0]} for k, v in neg.items()}

    # --- camera-day bootstrap
    g = defaultdict(lambda: np.zeros(8))
    for r, (m, _), c in zip(rows, matches["iou10"], called):
        k = r["conf"] >= T_PROD
        tpb = int((m[k] >= 0).sum())
        has = len(r["labels"]) > 0
        g[r["group"]] += [tpb, int(k.sum()) - tpb, len(r["labels"]) - tpb,
                          c and has, c and not has, (not c) and not has, (not c) and has, 1]
    f1 = lambda s: 2 * s[0] / (2 * s[0] + s[1] + s[2])
    res["ci95_camera_day"] = {
        "groups": len(g),
        "box_iou10_precision": boot(g, lambda s: s[0] / (s[0] + s[1])),
        "box_iou10_recall": boot(g, lambda s: s[0] / (s[0] + s[2])),
        "box_iou10_f1": boot(g, f1),
        "image_recall": boot(g, lambda s: s[3] / (s[3] + s[6])),
        "image_fpr": boot(g, lambda s: s[4] / (s[4] + s[5])),
        "image_mcc": boot(g, lambda s: binary(*s[3:7])["mcc"]),
    }

    # --- audit lists (loosest agreement: any overlap or center containment)
    unlabeled, unsupported = [], []
    for r in rows:
        p = pairwise(r["pred"], r["labels"]) if len(r["pred"]) and len(r["labels"]) else None
        near = (p["iou"] > 0) | p["center"] if p is not None else None
        for i, c in enumerate(r["conf"]):
            if c >= T_AUDIT and (near is None or not near[i].any()):
                unlabeled.append({"image": r["image"], "conf": float(c), "box": r["pred"][i].tolist(),
                                  "labels_in_image": len(r["labels"])})
        for j, l in enumerate(r["labels"]):
            if near is None or not near[:, j].any():
                unsupported.append({"image": r["image"], "box": l.tolist(),
                                    "height": float(l[3] - l[1]),
                                    "image_max_conf": float(r["conf"][0]) if len(r["conf"]) else 0.0})
    unlabeled.sort(key=lambda d: -d["conf"])
    unsupported.sort(key=lambda d: -d["height"])
    res["audit_counts"] = {"unlabeled_detections_conf>=0.5": len(unlabeled),
                           "images_with_unlabeled_detection": len({d["image"] for d in unlabeled}),
                           "unsupported_labels_no_overlap_at_conf>=0.05": len(unsupported)}
    return rnd(res), {"unlabeled": unlabeled, "unsupported": unsupported}


# ------------------------------------------------------------------ contact sheets

def sheet(split: str, items: list[dict], rows_by_image: dict, dest: Path, title: str,
          n: int = 24, per_row: int = 4) -> None:
    from PIL import Image, ImageDraw
    cw, ch = 480, 320
    items = items[:n]
    out = Image.new("RGB", (cw * per_row, ch * ((len(items) + per_row - 1) // per_row)), "black")
    for k, it in enumerate(items):
        im = Image.open(IMAGES / split / f"{it['image']}.jpg").convert("RGB")
        W, H = im.size
        r = rows_by_image[it["image"]]
        x0, y0, x1, y1 = it["box"]
        cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
        half = max((x1 - x0) * W, (y1 - y0) * H * 1.5, 90) * 1.6
        box = (int(max(0, cx - half)), int(max(0, cy - half / 1.5)),
               int(min(W, cx + half)), int(min(H, cy + half / 1.5)))
        d = ImageDraw.Draw(im)
        for l in r["labels"]:
            d.rectangle([l[0] * W - 2, l[1] * H - 2, l[2] * W + 2, l[3] * H + 2], outline=(255, 40, 40), width=2)
        for c, p in zip(r["conf"], r["pred"]):
            if c >= 0.05:
                d.rectangle([p[0] * W, p[1] * H, p[2] * W, p[3] * H], outline=(0, 220, 255), width=1 if c < T_PROD else 2)
        tile = im.crop(box).resize((cw, ch), Image.LANCZOS)
        td = ImageDraw.Draw(tile)
        cam = it["image"].split("_")[1]
        tag = f"{cam}  conf {it['conf']:.2f}" if "conf" in it else f"{cam}  max {it['image_max_conf']:.2f}"
        td.rectangle([0, 0, cw, 24], fill="black")
        td.text((4, 3), f"{k + 1}. {tag}", fill="white", font_size=17)
        out.paste(tile, ((k % per_row) * cw, (k // per_row) * ch))
    out.save(dest, quality=88)
    print(f"{title}: {dest}")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("split", choices=("val", "train"))
    ap_.add_argument("--sheets", action="store_true")
    a = ap_.parse_args()
    rows = load(a.split)
    res, lists = evaluate(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.split}_metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in lists.items():
        (OUT / f"{a.split}_{k}.json").write_text(json.dumps(v, indent=0) + "\n")
    print(json.dumps({k: res[k] for k in ("images", "labels", "box", "image", "ci95_camera_day",
                                          "audit_counts")}, indent=1)[:6000])
    if a.sheets:
        by = {r["image"]: r for r in rows}
        sheet(a.split, lists["unlabeled"], by, OUT / f"{a.split}_unlabeled.jpg", "unlabeled")
        sheet(a.split, lists["unsupported"], by, OUT / f"{a.split}_unsupported.jpg", "unsupported")


if __name__ == "__main__":
    main()
