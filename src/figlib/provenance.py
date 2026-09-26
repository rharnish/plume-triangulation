"""Where every archive came from, and what produced every result.

Two records, both committed, both small:

* **Archive manifests** (`data/meta/manifests/<corpus>.json`). One entry per archive:
  source URL, byte size and SHA-256 of the local copy, the server's size, `Last-Modified`
  and `ETag` at the time of the check, frame count, fire date and contamination tier.
  Upstream files do change -- `20260909_GettyFire_wilson-ws-mobo-c` is served as a
  258-byte placeholder with no frames -- so the hash is what pins a result to its input.
* **Run log** (`<corpus metadata>/runs.jsonl`). One line appended per pipeline stage:
  git commit and a hash of any uncommitted diff, package versions, the model pin, input
  manifest hashes, parameters, every FIGLIB_* setting as resolved (`settings.py`, with the
  profile's hash), and the SHA-256 of every output. A result file whose hash
  is not in the log did not come from a recorded run.

The detector weights are pinned here, and `check_model` refuses to run on anything else.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from . import corpus as C
from . import settings

ROOT = C.ROOT
MANIFESTS = C.SHARED_META / "manifests"
SOURCE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar"

MODEL = {
    "file": "models/pyronear_rr_v8.1.0.onnx",
    "sha256": "613db7510865c48b3c1a9a4f6f2307a2822810490b121e5bb103ea8f1bf1f10b",
    "source": "https://huggingface.co/pyronear/yolo11s_rapid-raccoon_v8.1.0",
    "revision": "81a1f6bd060ca7100e496ff5dedeb2d327d135e0",
    "archive": "onnx_cpu.tar.gz",
    "archive_sha256": "77bb91413277893ff70c2e57df3aad5e6d2cf40d989bea689f6b7b252b705976",
    "license": "Apache-2.0",
}


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ model pin

def check_model(path: Path | None = None) -> dict:
    """Verify the local weights are the pinned ones; raise if not.

    FIGLIB_ALLOW_UNPINNED_MODEL=1 runs anyway and records the actual hash, for deliberate
    experiments with other weights -- the run log then says so.
    """
    path = path or ROOT / MODEL["file"]
    actual = sha256_file(path)
    if actual != MODEL["sha256"]:
        if not settings.flag("FIGLIB_ALLOW_UNPINNED_MODEL"):
            raise RuntimeError(
                f"{rel(path)} has sha256 {actual}, not the pinned {MODEL['sha256']} "
                f"({MODEL['source']} @ {MODEL['revision'][:8]}). Refetch it per "
                "models/README.md, or set FIGLIB_ALLOW_UNPINNED_MODEL=1 on purpose.")
        return {**MODEL, "file": rel(path), "sha256": actual, "pinned": False}
    return {**MODEL, "pinned": True}


# -------------------------------------------------------------------- run log

def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def code_state() -> dict:
    diff = _git("diff", "HEAD", "--", "src", "tests", "configs", "run_detect.sh",
                "data/fetch.sh")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "src", "configs")
    return {
        "commit": (_git("rev-parse", "HEAD") or "").strip() or None,
        "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "untracked_src": sorted(untracked.split()) if untracked else [],
    }


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod, attr in (("numpy", "__version__"), ("cv2", "__version__"),
                      ("onnxruntime", "__version__"), ("scipy", "__version__"),
                      ("matplotlib", "__version__")):
        if mod in sys.modules:
            out[mod] = getattr(sys.modules[mod], attr, None)
    return out


def digest_outputs(paths) -> dict:
    """SHA-256 per output file; a directory is summarized as a count and one digest over
    its sorted (name, sha256) pairs, with the per-file hashes kept alongside."""
    out = {}
    for p in map(Path, paths):
        if p.is_dir():
            files = {f.name: sha256_file(f) for f in sorted(p.glob("*.json"))}
            joined = "\n".join(f"{k} {v}" for k, v in files.items())
            out[rel(p)] = {"n_files": len(files),
                           "digest": hashlib.sha256(joined.encode()).hexdigest(),
                           "files": files}
        elif p.exists():
            out[rel(p)] = sha256_file(p)
        else:
            out[rel(p)] = None
    return out


def record(stage: str, outputs, params: dict | None = None, model: dict | None = None,
           started: str | None = None, extra_inputs=()) -> dict:
    """Append one run to the corpus's run log and return it."""
    c = C.current()
    manifests = {}
    for name in (("core", "extra") if c.name == "all" else (c.name,)):
        m = MANIFESTS / f"{name}.json"
        manifests[rel(m)] = sha256_file(m) if m.exists() else None
    rec = {
        "stage": stage,
        "corpus": c.name,
        "tiers": C.tier_filter(),
        "started_utc": started,
        "finished_utc": utc_now(),
        "argv": sys.argv,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FIGLIB_")},
        # Every FIGLIB_* setting as the run saw it, defaults included, and the profile's hash.
        "settings": settings.resolved(),
        "code": code_state(),
        "versions": _versions(),
        "platform": platform.platform(),
        "model": model,
        "inputs": {**manifests, **digest_outputs(extra_inputs)},
        "params": params or {},
        "outputs": digest_outputs(outputs),
    }
    log = c.meta / "runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


# ---------------------------------------------------------- archive manifests

class _HashingReader:
    """File wrapper that hashes every byte read, so one pass both parses and hashes."""

    def __init__(self, fh):
        self.fh, self.h, self.n = fh, hashlib.sha256(), 0

    def read(self, size=-1):
        b = self.fh.read(size)
        self.h.update(b)
        self.n += len(b)
        return b


def _inspect(path: str) -> dict:
    from .ingest import resolve_frame_names
    p = Path(path)
    names: list[str] = []
    status = "ok"
    with open(p, "rb") as raw:
        hr = _HashingReader(raw)
        try:
            with tarfile.open(fileobj=hr, mode="r|gz") as tf:
                names = [m.name for m in tf]
        except (tarfile.TarError, OSError, EOFError) as exc:
            status = f"unreadable: {type(exc).__name__}: {exc}"
        while hr.read(1 << 20):          # hash any trailing bytes the tar reader skipped
            pass
    resolved, repairs = resolve_frame_names(names)
    t0s: dict[int, int] = {}
    for e, o in resolved.values():
        t0s[e - o] = t0s.get(e - o, 0) + 1
    unannotated = sum(1 for n in names if n.endswith(".jpg")) - len(resolved) \
        - repairs["duplicates"] - repairs["unresolved"]
    n_frames = sum(t0s.values())
    if status == "ok" and n_frames == 0:
        # Frames named by epoch alone carry no plume-appearance offset, so there is no
        # t0 to score latency or separate negatives against; ingest cannot use them.
        status = (f"unannotated: {unannotated} frames with no offset in the name"
                  if unannotated else "no frames")
    return {"file": p.name, "bytes": p.stat().st_size, "sha256": hr.h.hexdigest(),
            "local_mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, UTC)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_frames": n_frames, "t0_frames": {str(k): v for k, v in sorted(t0s.items())},
            "frame_name_repairs": {k: v for k, v in repairs.items() if v},
            "status": status}


def _head(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "plume-triangulation provenance"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {"http_status": r.status,
                    "content_length": int(r.headers.get("Content-Length") or -1),
                    "last_modified": r.headers.get("Last-Modified"),
                    "etag": (r.headers.get("ETag") or "").strip('"') or None}
    except Exception as exc:                          # network, not logic
        return {"http_status": None, "error": f"{type(exc).__name__}: {exc}"}


def build_manifest(name: str, workers: int = 4, check_remote: bool = True) -> Path:
    c = C.CORPORA[name]
    paths = C.tgz_paths(c)
    started = utc_now()
    with ProcessPoolExecutor(workers) as pool:
        entries = list(pool.map(_inspect, map(str, paths), chunksize=2))
    for k, e in enumerate(entries, 1):
        e["url"] = f"{SOURCE}/{e['file']}"
        e["fire_date"] = e["file"][:8]
        e["contamination"] = C.contamination(e["file"])
        if check_remote:
            e["remote"] = _head(e["url"])
            e["matches_remote_size"] = e["remote"].get("content_length") == e["bytes"]
            time.sleep(0.05)                          # one polite request at a time
        if k % 50 == 0:
            print(f"  {k}/{len(entries)}", flush=True)

    from collections import Counter
    doc = {
        "corpus": name,
        "source": SOURCE,
        "credit": "HPWREN, https://www.hpwren.ucsd.edu/ (FIgLib)",
        "local_dirs": [rel(d) for d in c.tgz_dirs],
        "started_utc": started,
        "generated_utc": utc_now(),
        "code": code_state(),
        "n_archives": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "status": dict(Counter(e["status"] for e in entries)),
        "contamination": dict(Counter(e["contamination"] for e in entries)),
        "contamination_cutoffs": {"figlib_snapshot": C.FIGLIB_SNAPSHOT,
                                  "model_release": C.MODEL_RELEASE},
        "size_mismatches": [e["file"] for e in entries
                            if check_remote and not e.get("matches_remote_size")],
        "archives": entries,
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    dest = MANIFESTS / f"{name}.json"
    dest.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{name}: {len(entries)} archives, {doc['total_bytes'] / 1e9:.1f} GB, "
          f"status {doc['status']}, size mismatches {len(doc['size_mismatches'])} "
          f"-> {rel(dest)}")
    return dest


def verify_manifest(name: str) -> list[str]:
    """Re-hash local archives against the manifest; return the problems found."""
    doc = json.loads((MANIFESTS / f"{name}.json").read_text())
    have = {p.name: p for p in C.tgz_paths(C.CORPORA[name])}
    problems = []
    for e in doc["archives"]:
        p = have.pop(e["file"], None)
        if p is None:
            problems.append(f"missing {e['file']}")
        elif sha256_file(p) != e["sha256"]:
            problems.append(f"changed {e['file']}")
    problems += [f"not in manifest {n}" for n in sorted(have)]
    return problems


def main(argv: list[str]) -> None:
    cmd = argv[:1]
    if cmd == ["manifest"]:
        build_manifest(argv[1], check_remote="--no-remote" not in argv)
    elif cmd == ["verify"]:
        problems = verify_manifest(argv[1])
        print("\n".join(problems) or f"{argv[1]}: every archive matches its manifest")
        sys.exit(1 if problems else 0)
    elif cmd == ["record-detect"]:
        # Called once after a detection pass, so the run log carries one entry for the
        # whole pass rather than one per worker process.
        c = C.current()
        record("detect_yolo", [c.dets], model=check_model(),
               params={"imgsz": 1024, "providers": ["CPUExecutionProvider"]},
               started=argv[1] if len(argv) > 1 else None)
        print(f"recorded detection pass in {rel(c.meta / 'runs.jsonl')}")
    else:
        print("usage: python -m src.figlib.provenance manifest|verify <corpus> | "
              "record-detect [started_utc]")


if __name__ == "__main__":
    main(sys.argv[1:])
