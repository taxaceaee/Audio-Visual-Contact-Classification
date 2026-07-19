"""Run clear_pipeline on bundled features; assert sealed Macro F1."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CODE = Path(__file__).resolve().parent
os.environ.setdefault("TREE_BUNDLE_ROOT", str(_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from paper_claim.clear_pipeline import HParams, metrics, predict_robot  # noqa: E402
from shared.metrics_utils import assert_close, load_sealed_targets  # noqa: E402


def main() -> dict:
    targets = load_sealed_targets(_ROOT)["fusion_paper_claim_v2"]
    seal_path = _ROOT / "results" / "CLAIM_SEAL.json"
    sealed_metrics_path = _ROOT / "results" / "final_test_metrics.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    sealed_doc = json.loads(sealed_metrics_path.read_text(encoding="utf-8"))
    sealed_f1 = float(seal["final_test_macro_f1"])
    sealed_doc_f1 = float(sealed_doc["metrics"]["macro_f1_4class"])

    hp = HParams(b_leaf=float(targets.get("b_leaf", -1.3)))
    out = predict_robot(hp)
    m = metrics(out["y"], out["pred"])

    assert_close(
        m["macro_f1_4class"],
        targets["macro_f1_4class"],
        name="fusion macro_f1_4class vs target",
        tol=targets.get("tolerance", 1e-9),
    )
    assert_close(
        m["macro_f1_4class"],
        sealed_f1,
        name="fusion macro_f1_4class vs CLAIM_SEAL",
        tol=1e-9,
    )
    assert_close(
        m["macro_f1_4class"],
        sealed_doc_f1,
        name="fusion macro_f1_4class vs final_test_metrics.json",
        tol=1e-9,
    )
    assert int(len(out["y"])) == int(targets["n"])

    result = {
        "section": "fusion_paper_claim_v2",
        "status": "ok",
        "architecture": out["architecture"],
        "metrics": m,
        "sealed_target_macro_f1_4class": targets["macro_f1_4class"],
        "feature_stack_order": ["total240_or_robot_mix", "wav2vec2", "clip"],
        "cascade_order": [
            "1_v2_base_hier_trunk_meta_else",
            "2_amb_lift_ts_cs_bin",
            "3_secondary_clip_trunk",
            "4_material_leaf_bias_b_leaf",
        ],
    }
    print(
        json.dumps(
            {
                "section": "fusion_paper_claim_v2",
                "macro_f1_4class": m["macro_f1_4class"],
                "accuracy_4class": m["accuracy_4class"],
                "binary_macro_f1": m["binary_macro_f1"],
                "match_sealed": True,
                "architecture": out["architecture"],
                "feature_stack_order": result["feature_stack_order"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    main()
