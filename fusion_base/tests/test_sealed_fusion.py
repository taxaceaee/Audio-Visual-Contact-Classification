"""Unit checks against sealed fusion claim v2."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.metrics_utils import load_sealed_targets


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sealed_targets_file():
    t = load_sealed_targets(ROOT)
    assert abs(t["fusion_paper_claim_v2"]["macro_f1_4class"] - 0.8507810400075331) < 1e-12
    assert abs(t["fusion_paper_claim_v2"]["b_leaf"] - (-1.3)) < 1e-12


def test_claim_seal_agrees():
    seal = json.loads((ROOT / "results" / "CLAIM_SEAL.json").read_text(encoding="utf-8"))
    assert abs(float(seal["final_test_macro_f1"]) - 0.8507810400075331) < 1e-12


def test_final_metrics_agrees():
    doc = json.loads((ROOT / "results" / "final_test_metrics.json").read_text(encoding="utf-8"))
    assert abs(float(doc["metrics"]["macro_f1_4class"]) - 0.8507810400075331) < 1e-12


def test_clear_pipeline_predict_robot():
    code = ROOT / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from paper_claim.clear_pipeline import HParams, metrics, predict_robot

    out = predict_robot(HParams(b_leaf=-1.3))
    m = metrics(out["y"], out["pred"])
    assert abs(m["macro_f1_4class"] - 0.8507810400075331) < 1e-9
    assert len(out["y"]) == 2219


def test_evaluate_fusion_module():
    mod = _load("evaluate_fusion", ROOT / "code" / "evaluate_fusion.py")
    out = mod.main()
    assert out["status"] == "ok"
    assert abs(out["metrics"]["macro_f1_4class"] - 0.8507810400075331) < 1e-9
