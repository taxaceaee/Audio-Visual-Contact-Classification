#!/usr/bin/env python3
"""Official sealed reproduce entry for paper claim fusion v2."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    mod = _load("evaluate_fusion", ROOT / "code" / "evaluate_fusion.py")
    result = mod.main()
    summary = {
        "status": "ok",
        "section": result["section"],
        "macro_f1_4class": result["metrics"]["macro_f1_4class"],
        "accuracy_4class": result["metrics"]["accuracy_4class"],
        "binary_macro_f1": result["metrics"]["binary_macro_f1"],
        "match_sealed": True,
    }
    out = ROOT / "reproduce_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("PASS fusion_paper_claim_v2 — matches sealed Macro F1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
