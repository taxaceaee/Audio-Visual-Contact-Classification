#!/usr/bin/env python3
"""PAPER CLAIM gt09 — pure single-shot robot open + seal.

Requires hand selection_lock; refuses second open without PAPER_CLAIM_FORCE_REOPEN=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import paper_claim_protocol as claim
from paper_claim import gt09_core as core

REPO = Path(__file__).resolve().parent
RECIPE_PATH = Path(
    os.environ.get("PAPER_CLAIM_RECIPE", REPO / "paper_claim/RECIPE_gt09_material_multibb.json")
)


def main():
    recipe = claim.load_recipe(RECIPE_PATH)
    out_env = os.environ.get("PAPER_CLAIM_OUT")
    out_dir = Path(out_env) if out_env else (REPO / recipe["artifacts"]["claim_out"])
    if not out_dir.is_absolute():
        out_dir = (REPO / out_dir).resolve()
    claim.assert_not_sealed(out_dir)
    lock = claim.load_hand_selection_lock(out_dir, recipe_path=RECIPE_PATH)
    sc = lock["selected_candidate"]
    primary = recipe["frozen_contact_stack"]["primary"]
    secondary = recipe["frozen_contact_stack"]["secondary"]

    alpha = float(sc.get("alpha_multibb", 0.0))
    b_leaf = float(sc.get("b_leaf", 0.0))
    b_trunk = float(sc.get("b_trunk", 0.0))
    b_twig = float(sc.get("b_twig", 0.0))
    T = float(sc.get("T", 1.0))

    # Fit multibb on hand only
    clf, _, _, _ = core.fit_hand_multibb_full(C=0.05)

    # --- first robot load for this claim dir ---
    contact_pack = core.robot_contact_and_meta(primary, secondary)
    tu = contact_pack["segment_ids"]
    contact = contact_pack["contact"]
    soft_meta = contact_pack["soft_meta"]
    r_paths = contact_pack["paths"]
    yt = contact_pack["y_window"]

    ru, _, soft_mm, _ = core.robot_multibb_proba(clf)
    # align multibb segs to contact segs
    order_mm = {str(k): i for i, k in enumerate(ru)}
    soft_mm_aligned = np.array([soft_mm[order_mm[str(k)]] for k in tu], dtype=np.float64)
    soft = core.blend_soft(soft_meta, soft_mm_aligned, alpha)
    pred_seg = core.decode_material(contact, soft, b_leaf, b_trunk, b_twig, T)
    pred = core.expand_segment_pred(tu, pred_seg, r_paths)

    metrics = {
        "accuracy_4class": float(accuracy_score(yt, pred)),
        "macro_precision_4class": float(
            precision_score(yt, pred, average="macro", zero_division=0)
        ),
        "macro_recall_4class": float(recall_score(yt, pred, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "weighted_f1_4class": float(f1_score(yt, pred, average="weighted", zero_division=0)),
        "binary_macro_f1": float(f1_score(yt > 0, pred > 0, average="macro", zero_division=0)),
    }
    result = {
        "split": "robot_test_final",
        "protocol": "paper_claim_gt09_material_multibb_pure_single_shot",
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": lock["recipe_sha256"],
        "locked_candidate": sc,
        "n": int(len(yt)),
        "metrics": metrics,
        "per_class_4class": classification_report(
            yt,
            pred,
            labels=[0, 1, 2, 3],
            target_names=["ambient", "leaf", "trunk", "twig"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_4class": confusion_matrix(yt, pred, labels=[0, 1, 2, 3]).tolist(),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "no_test_hp_tuning": True,
            "filename_class_features": False,
            "robot_uda": False,
            "pure_single_shot_architecture": True,
            "robot_opened_once_for_claim": True,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    seal = claim.seal_claim(out_dir, lock, result)
    print(
        json.dumps(
            {
                "step": "final_test_sealed",
                "macro_f1_4class": metrics["macro_f1_4class"],
                "n": result["n"],
                "seal": str(out_dir / "CLAIM_SEAL.json"),
                "target_gt_0.9": metrics["macro_f1_4class"] > 0.9,
            },
            indent=2,
        ),
        flush=True,
    )
    return result, seal


if __name__ == "__main__":
    main()
