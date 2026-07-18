#!/usr/bin/env python3
"""Reproduce sealed Macro F1 for audio-only + fusion v2 only.

Usage (from this folder):
    python reproduce_two.py

Exit 0 iff both sections match sealed targets (tol 1e-9).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TREE_BUNDLE_ROOT", str(ROOT))
# Deterministic BLAS/OpenMP for bit-stable LogReg / sklearn across machines
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=" * 72)
    print("audio_fusion_v2_only — sealed reproduce (audio + fusion v2)")
    print(f"BUNDLE_ROOT={ROOT}")
    print("=" * 72)

    sections = [
        ("audio_only", ROOT / "audio_only" / "code" / "evaluate_audio.py", "audio_only"),
        ("fusion_v2", ROOT / "fusion_v2" / "code" / "evaluate_fusion.py", "fusion_paper_claim_v2"),
    ]

    from shared.metrics_utils import load_sealed_targets

    targets = load_sealed_targets(ROOT)
    results: dict = {}
    failed: list[str] = []
    t0 = time.time()

    for key, path, target_key in sections:
        print(f"\n--- {key}: {path.relative_to(ROOT)} ---")
        try:
            mod = _load_module(f"eval_{key}", path)
            out = mod.main()
            f1 = float(out["metrics"]["macro_f1_4class"])
            expected = float(targets[target_key]["macro_f1_4class"])
            ok = abs(f1 - expected) <= float(targets[target_key].get("tolerance", 1e-9))
            results[key] = {
                "ok": ok,
                "macro_f1_4class": f1,
                "expected_macro_f1_4class": expected,
                "accuracy_4class": out["metrics"].get("accuracy_4class"),
                "binary_macro_f1": out["metrics"].get("binary_macro_f1"),
                "contact_macro_f1": out["metrics"].get("contact_macro_f1"),
            }
            if not ok:
                failed.append(key)
                print(f"FAIL {key}: got {f1} expected {expected}")
            else:
                print(f"PASS {key}: macro_f1_4class={f1}")
        except Exception as exc:
            failed.append(key)
            results[key] = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"ERROR {key}: {exc}")
            print(traceback.format_exc())

    elapsed = time.time() - t0
    summary = {
        "bundle_root": str(ROOT),
        "elapsed_sec": elapsed,
        "all_pass": len(failed) == 0,
        "failed": failed,
        "results": results,
        "sealed_targets": {k: v["macro_f1_4class"] for k, v in targets.items()},
    }
    out_path = ROOT / "reproduce_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(json.dumps({
        "all_pass": summary["all_pass"],
        "elapsed_sec": round(elapsed, 3),
        "results": {
            k: {
                "macro_f1_4class": results[k].get("macro_f1_4class"),
                "ok": results[k].get("ok"),
            }
            for k in results
        },
        "summary_path": str(out_path),
    }, indent=2))
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    print("BOTH SECTIONS MATCH SEALED TARGETS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
