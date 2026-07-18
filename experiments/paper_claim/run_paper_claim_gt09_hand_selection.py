#!/usr/bin/env python3
"""PAPER CLAIM gt09 — hand-only material multi-bb selection. Never loads robot labels/features for scoring.

Usage:
  PAPER_CLAIM_RECIPE=paper_claim/RECIPE_gt09_material_multibb.json \\
  PAPER_CLAIM_OUT=outputs/paper_claim_gt09 \\
  python run_paper_claim_gt09_hand_selection.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

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

    primary = recipe["frozen_contact_stack"]["primary"]
    secondary = recipe["frozen_contact_stack"]["secondary"]
    rule = recipe["material_decode"]["selection_rule"]
    elig = rule["eligible"]
    cv_cfg = rule["cv"]

    # --- hand only ---
    bundle = np.load(core.HAND_BUNDLE)
    sy = bundle["sy"].astype(np.int64)
    print(
        json.dumps(
            {
                "step": "build_hand_multibb_oof",
                "n_segments": int(len(sy)),
                "robot_loaded": False,
            }
        ),
        flush=True,
    )
    mm = core.build_hand_multibb_oof(
        n_splits=cv_cfg["n_splits"], random_state=cv_cfg["random_state"], C=0.05
    )
    assert np.array_equal(mm["sy"], sy)
    specs = mm["specs"]
    oof_mm = mm["oof_multibb4"]
    oof_meta = bundle["oof_meta"].astype(np.float64)
    contact = core.apply_contact(
        bundle["pred_v2"],
        bundle["oof_ts"],
        bundle["oof_cs"],
        bundle["oof_bin"],
        bundle["oof_tr_img"],
        primary,
        secondary,
    )
    base_f1 = core.macro_f1(sy, contact)
    base_tr = float(np.sum((sy == 2) & (contact == 2)) / max((sy == 2).sum(), 1))

    folds = list(
        StratifiedGroupKFold(
            cv_cfg["n_splits"],
            shuffle=cv_cfg["shuffle"],
            random_state=cv_cfg["random_state"],
        ).split(np.arange(len(sy)), sy, specs)
    )
    cascades = [tuple(x) for x in rule["cascade_views"]]
    a_tt_g, a_tl_g = rule["cascade_gain_eval"]

    # Practical pure-paper grid (full recipe linspace is dense; coarse then refine)
    # Two-stage hand grid for speed: stage-1 alpha on identity bias; stage-2 bias around winners.
    alphas = np.linspace(0.0, 1.0, 11)
    stage1 = []
    for alpha in alphas:
        soft0 = core.blend_soft(oof_meta, oof_mm, alpha)
        fold_scores = []
        oof_pred = np.zeros(len(sy), np.int64)
        for _, va in folds:
            vs = [
                core.macro_f1(
                    sy[va],
                    core.decode_material(
                        contact, core.cascade_contaminate(soft0, sy, a_tt, a_tl), 0, 0, 0, 1.0
                    )[va],
                )
                for a_tt, a_tl in cascades
            ]
            fold_scores.append(float(np.min(vs)))
            oof_pred[va] = core.decode_material(contact, soft0, 0, 0, 0, 1.0)[va]
        stage1.append(
            {
                "alpha_multibb": float(alpha),
                "worst": float(np.min(fold_scores)),
                "clean": core.macro_f1(sy, oof_pred),
            }
        )
    stage1 = sorted(stage1, key=lambda r: (r["worst"], r["clean"]), reverse=True)
    top_alphas = [r["alpha_multibb"] for r in stage1[:4]]

    b_leafs = np.linspace(-2.0, 0.5, 11)
    b_trunks = np.array([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    b_twigs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    Ts = np.array([0.75, 1.0, 1.25])

    rows = []
    best = None
    n_cand = 0
    for alpha in top_alphas:
        soft0 = core.blend_soft(oof_meta, oof_mm, alpha)
        for bl in b_leafs:
            for bt in b_trunks:
                for bw in b_twigs:
                    for T in Ts:
                        n_cand += 1
                        fold_scores = []
                        oof_pred = np.zeros(len(sy), np.int64)
                        for _, va in folds:
                            vs = []
                            for a_tt, a_tl in cascades:
                                soft_v = core.cascade_contaminate(soft0, sy, a_tt, a_tl)
                                pred_v = core.decode_material(
                                    contact, soft_v, bl, bt, bw, T
                                )
                                vs.append(core.macro_f1(sy[va], pred_v[va]))
                            fold_scores.append(float(np.min(vs)))
                            oof_pred[va] = core.decode_material(
                                contact, soft0, bl, bt, bw, T
                            )[va]
                        worst = float(np.min(fold_scores))
                        clean = core.macro_f1(sy, oof_pred)
                        soft_c = core.cascade_contaminate(soft0, sy, a_tt_g, a_tl_g)
                        pred_c0 = core.decode_material(contact, soft_c, 0, 0, 0, 1.0)
                        pred_c1 = core.decode_material(contact, soft_c, bl, bt, bw, T)
                        casc_gain = core.macro_f1(sy, pred_c1) - core.macro_f1(sy, pred_c0)
                        tr_rec = float(
                            np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1)
                        )
                        false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
                        nflip = int(np.sum(oof_pred != contact))
                        row = {
                            "alpha_multibb": float(alpha),
                            "b_leaf": float(bl),
                            "b_trunk": float(bt),
                            "b_twig": float(bw),
                            "T": float(T),
                            "worst": worst,
                            "clean": clean,
                            "casc_gain": float(casc_gain),
                            "trunk_rec": tr_rec,
                            "false_tr": false_tr,
                            "nflip": nflip,
                            "base_contact_f1": base_f1,
                        }
                        rows.append(row)
                        ok = (
                            casc_gain >= elig["casc_gain_min"]
                            and clean >= elig["clean_macro_min"]
                            and (base_tr - tr_rec) <= elig["trunk_rec_drop_max"]
                            and false_tr <= elig["false_trunk_max"]
                            and nflip >= elig["nflip_min"]
                        )
                        if not ok:
                            continue
                        key = (casc_gain, worst, clean, tr_rec)
                        if best is None or key > best[0]:
                            best = (key, row)

    out_dir.mkdir(parents=True, exist_ok=True)
    lb = pd.DataFrame(rows).sort_values(
        ["casc_gain", "worst", "clean"], ascending=False
    )
    lb_path = out_dir / "hand_selection_leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    if best is None:
        # fallback: max worst among clean>=0.94 (still hand-only; may be identity-ish)
        elig_rows = [r for r in rows if r["clean"] >= 0.94 and r["nflip"] >= 0]
        if not elig_rows:
            elig_rows = rows
        elig_rows = sorted(
            elig_rows, key=lambda r: (r["worst"], r["clean"], r["casc_gain"]), reverse=True
        )
        sel = elig_rows[0]
        sel_mode = "fallback_worst_clean"
    else:
        sel = best[1]
        sel_mode = "cascade_gain_eligible"

    # cache hand multibb OOF for audit
    np.savez_compressed(
        out_dir / "hand_multibb_oof.npz",
        sy=sy,
        specs=specs,
        oof_multibb4=oof_mm,
        contact=contact,
        oof_meta=oof_meta,
    )

    payload = {
        "protocol": "paper_claim_gt09_hand_only_material_multibb",
        "selected_candidate": {**sel, "mode": sel_mode},
        "selection_summary": {
            "n_candidates_scored": n_cand,
            "n_leaderboard_rows": len(rows),
            "base_contact_hand_macro": base_f1,
            "base_trunk_rec": base_tr,
            "multibb_oof_macro_alone": core.macro_f1(sy, oof_mm.argmax(1)),
        },
        "leaderboard_path": str(lb_path),
    }
    lock = claim.write_hand_selection_lock(
        out_dir, payload, recipe, recipe_path=RECIPE_PATH
    )
    print(
        json.dumps(
            {
                "step": "hand_selection_done",
                "claim_out": str(out_dir),
                "selected": lock["selected_candidate"],
                "test_loaded": lock["test_loaded"],
                "recipe_sha256": lock["recipe_sha256"],
            },
            indent=2,
            default=float,
        ),
        flush=True,
    )
    return lock


if __name__ == "__main__":
    main()
