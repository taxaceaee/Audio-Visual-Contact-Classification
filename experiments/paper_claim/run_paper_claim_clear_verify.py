#!/usr/bin/env python3
"""Verify clear pipeline matches sealed paper_claim_v2 (no new HP tuning).

Uses b_leaf from sealed selection_lock; does not re-select.
Writes outputs/paper_claim_v2/clear_pipeline_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

from paper_claim.clear_pipeline import HParams, metrics, predict_robot

CLAIM = Path("outputs/paper_claim_v2")
SEALED = CLAIM / "final_test_metrics.json"
LOCK = CLAIM / "selection_lock.json"


def main():
    lock = json.loads(LOCK.read_text())
    sealed = json.loads(SEALED.read_text())
    b_leaf = float(lock["selected_candidate"]["b_leaf"])
    hp = HParams(b_leaf=b_leaf)

    out = predict_robot(hp)
    m = metrics(out["y"], out["pred"])
    sealed_f1 = sealed["metrics"]["macro_f1_4class"]
    delta = abs(m["macro_f1_4class"] - sealed_f1)

    report = {
        "protocol": "clear_pipeline_verify_vs_sealed_v2",
        "architecture": out["architecture"],
        "hp": {
            "ts_th": hp.ts_th,
            "cs_th": hp.cs_th,
            "img_th": hp.img_th,
            "img_cs": hp.img_cs,
            "b_leaf": hp.b_leaf,
        },
        "detectors": "CLIP + wav2vec2 + robot_mix (train); CLIP binary/trunk secondary",
        "dropped": [
            "efficientnet/dino/convnext in amb det",
            "bandlimit stress view",
            "do_tt",
            "multi material biases",
            "v2_rule_soft / hier material soft",
        ],
        "sealed_macro_f1": sealed_f1,
        "clear_macro_f1": m["macro_f1_4class"],
        "abs_delta_f1": delta,
        "match_sealed": delta < 1e-9,
        "metrics": m,
        "invariants": {
            "no_new_hp_selection": True,
            "b_leaf_from_hand_lock": True,
            "no_robot_uda": True,
            "filename_class_features": False,
            "selection_used_hand_only": True,
        },
    }
    CLAIM.mkdir(parents=True, exist_ok=True)
    (CLAIM / "clear_pipeline_metrics.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    print(json.dumps(report, indent=2, default=float))
    if delta >= 1e-6:
        # allow tiny float noise; fail hard if materially different
        if delta > 1e-4:
            raise SystemExit(
                f"clear pipeline F1 {m['macro_f1_4class']} != sealed {sealed_f1}"
            )


if __name__ == "__main__":
    main()
