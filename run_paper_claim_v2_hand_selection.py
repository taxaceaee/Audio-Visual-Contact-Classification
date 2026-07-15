#!/usr/bin/env python3
"""PAPER CLAIM v2 (simple) — hand-only selection. Never loads robot.

Selects single free HP: b_leaf, under pre-registered cascade-gain ranking.
Contact stack frozen (amb-lift + secondary CLIP; no do_tt).

Usage:
  python run_paper_claim_v2_hand_selection.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import paper_claim_protocol as claim
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
RECIPE_PATH = claim.RECIPE_V2_PATH


def apply_contact(pred_v2, oof_ts, oof_cs, oof_bin, oof_tr, primary, secondary):
    out = pred_v2.copy()
    out[
        (out == 0)
        & (oof_ts >= primary["ts_th"])
        & (np.maximum(oof_cs, oof_bin) >= primary["cs_th"])
    ] = 2
    # do_tt intentionally omitted in v2
    out[
        (out == 0)
        & (oof_tr >= secondary["img_th"])
        & (oof_bin >= secondary["img_cs"])
    ] = 2
    return out


def cascade(soft, y, a_tt, a_tl):
    out = soft.copy()
    m = y == 2
    if m.any() and a_tt > 0:
        out[m, 3] = out[m, 3] + a_tt * out[m, 2]
        out[m, 2] = (1.0 - a_tt) * out[m, 2]
        out[m] = claim.normalize(out[m])
    m = y == 3
    if m.any() and a_tl > 0:
        out[m, 1] = out[m, 1] + a_tl * out[m, 3]
        out[m, 3] = (1.0 - a_tl) * out[m, 3]
        out[m] = claim.normalize(out[m])
    return out


def decode_leaf_bias(contact, soft_meta, b_leaf):
    """Protect-trunk: redecode only leaf/twig with single leaf logit bias on meta."""
    out = contact.copy()
    m = (out == 1) | (out == 3)
    if not m.any():
        return out
    logits = np.log(np.clip(soft_meta[m][:, 1:4], 1e-12, None))
    logits = logits + np.array([float(b_leaf), 0.0, 0.0], dtype=np.float64)
    out[m] = 1 + logits.argmax(1)
    return out


def main():
    recipe = claim.load_recipe(RECIPE_PATH)
    out_dir = claim.claim_out_dir(recipe)
    claim.assert_not_sealed(out_dir)

    contact_cfg = recipe["frozen_contact_stack"]
    primary = contact_cfg["primary"]
    secondary = contact_cfg["secondary"]
    mat_cfg = recipe["material_decode"]
    rule = mat_cfg["selection_rule"]
    elig = rule["eligible"]
    cv_cfg = rule["cv"]
    leaf_bs = claim.parse_linspace(rule["grid"]["b_leaf"])
    cascades = [tuple(x) for x in rule["cascade_views"]]
    a_tt_g, a_tl_g = rule["cascade_gain_eval"]

    # --- hand data only ---
    fr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y = fr.y.to_numpy(np.int64)
    files = fr.audio_file.astype(str)
    sk = sel.seg(files).to_numpy()
    sg = sel.spec(files).to_numpy()
    useg, codes = np.unique(sk, return_inverse=True)
    n = len(useg)
    sy = np.array([y[codes == i][0] for i in range(n)], np.int64)
    ss = np.array([sg[codes == i][0] for i in range(n)])

    bundle_path = Path(recipe["artifacts"]["hand_oof_bundle"])
    z = np.load(bundle_path)
    assert len(z["sy"]) == n
    assert np.array_equal(z["sy"], sy)

    contact = apply_contact(
        z["pred_v2"],
        z["oof_ts"],
        z["oof_cs"],
        z["oof_bin"],
        z["oof_tr_img"],
        primary,
        secondary,
    )
    soft_meta = z["oof_meta"].astype(np.float64)
    base = claim.fast_macro_f1(sy, contact)
    base_tr = float(np.sum((sy == 2) & (contact == 2)) / max((sy == 2).sum(), 1))
    print(
        json.dumps(
            {
                "step": "hand_selection_v2",
                "recipe_id": recipe["recipe_id"],
                "recipe_sha256": claim.recipe_sha256(path=RECIPE_PATH),
                "claim_out": str(out_dir),
                "base_contact_hand_macro": base,
                "base_trunk_rec": base_tr,
                "note": "robot not loaded",
            },
            indent=2,
        ),
        flush=True,
    )

    folds = list(
        StratifiedGroupKFold(
            cv_cfg["n_splits"],
            shuffle=cv_cfg["shuffle"],
            random_state=cv_cfg["random_state"],
        ).split(np.arange(n), sy, ss)
    )

    rows = []
    for bl in leaf_bs:
        oof = np.zeros(n, np.int64)
        fold_w = []
        for _, va in folds:
            vs = []
            for a_tt, a_tl in cascades:
                soft_v = cascade(soft_meta, sy, a_tt, a_tl)
                pred_v = decode_leaf_bias(contact, soft_v, bl)
                vs.append(claim.fast_macro_f1(sy[va], pred_v[va]))
            fold_w.append(min(vs))
            oof[va] = decode_leaf_bias(contact, soft_meta, bl)[va]
        clean = claim.fast_macro_f1(sy, oof)
        if clean < elig["clean_macro_min"]:
            continue
        trunk_rec = float(np.sum((sy == 2) & (oof == 2)) / max((sy == 2).sum(), 1))
        if trunk_rec < base_tr - elig["trunk_rec_drop_max"]:
            continue
        twig_rec = float(np.sum((sy == 3) & (oof == 3)) / max((sy == 3).sum(), 1))
        leaf_rec = float(np.sum((sy == 1) & (oof == 1)) / max((sy == 1).sum(), 1))
        false_tr = int(np.sum((sy == 0) & (oof == 2)))
        if false_tr > elig["false_trunk_max"]:
            continue
        nflip = int(np.sum(oof != contact))
        soft_c = cascade(soft_meta, sy, a_tt_g, a_tl_g)
        pred_id = decode_leaf_bias(contact, soft_c, 0.0)
        pred_fx = decode_leaf_bias(contact, soft_c, bl)
        casc_gain = claim.fast_macro_f1(sy, pred_fx) - claim.fast_macro_f1(sy, pred_id)
        if nflip >= elig["nflip_min"] and casc_gain < 0.005:
            continue
        worst = float(np.min(fold_w))
        rows.append(
            dict(
                mode="protect_trunk_leaf_bias_only",
                b_leaf=float(bl),
                T=1.0,
                b_trunk=0.0,
                b_twig=0.0,
                soft_source="meta",
                worst=worst,
                clean=float(clean),
                casc_gain=float(casc_gain),
                trunk_rec=trunk_rec,
                twig_rec=twig_rec,
                leaf_rec=leaf_rec,
                nflip=nflip,
                false_tr=false_tr,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        chosen = dict(
            mode="none",
            b_leaf=0.0,
            T=1.0,
            b_trunk=0.0,
            b_twig=0.0,
            soft_source="meta",
            worst=base,
            clean=base,
            casc_gain=0.0,
            trunk_rec=base_tr,
            twig_rec=float(np.sum((sy == 3) & (contact == 3)) / max((sy == 3).sum(), 1)),
            leaf_rec=float(np.sum((sy == 1) & (contact == 1)) / max((sy == 1).sum(), 1)),
            nflip=0,
            false_tr=int(np.sum((sy == 0) & (contact == 2))),
        )
        chosen_note = "fallback_identity"
        df = pd.DataFrame([chosen])
    else:
        rank_keys = rule["rank_keys_desc"]
        df = df.sort_values(rank_keys, ascending=False).reset_index(drop=True)
        elig_df = df[
            (df.casc_gain >= elig["casc_gain_min"])
            & (df.clean >= elig["clean_macro_min"])
            & (df.nflip >= elig["nflip_min"])
        ]
        if len(elig_df) == 0:
            chosen = dict(
                mode="none",
                b_leaf=0.0,
                T=1.0,
                b_trunk=0.0,
                b_twig=0.0,
                soft_source="meta",
                worst=base,
                clean=base,
                casc_gain=0.0,
                trunk_rec=base_tr,
                twig_rec=float(
                    np.sum((sy == 3) & (contact == 3)) / max((sy == 3).sum(), 1)
                ),
                leaf_rec=float(
                    np.sum((sy == 1) & (contact == 1)) / max((sy == 1).sum(), 1)
                ),
                nflip=0,
                false_tr=int(np.sum((sy == 0) & (contact == 2))),
            )
            chosen_note = "fallback_identity_no_eligible"
        else:
            chosen = elig_df.iloc[0].to_dict()
            chosen_note = "eligible_cascade_gain_1d_rank"

    df.to_csv(out_dir / "hand_selection_leaderboard.csv", index=False)
    print(df.head(10).to_string(index=False), flush=True)
    print("SELECTED", chosen, chosen_note, flush=True)

    # static leakage guards (bundle must be hand OOF)
    assert "hand" in str(bundle_path).lower()

    lock = claim.write_hand_selection_lock(
        out_dir,
        {
            "protocol": "paper_claim_v2_simple_hand_selection",
            "frozen_contact_stack": contact_cfg,
            "material_decode": {
                "mode": mat_cfg["mode"],
                "soft_source": mat_cfg["soft_source"],
                "T": mat_cfg["T"],
                "b_trunk": mat_cfg["b_trunk"],
                "b_twig": mat_cfg["b_twig"],
                "apply_only_when_pred_in": mat_cfg["apply_only_when_pred_in"],
            },
            "selection_rule": rule["name"],
            "selected_candidate": chosen,
            "selection_note": chosen_note,
            "base_contact_hand_macro": base,
            "base_trunk_rec": base_tr,
            "n_candidates_scored": int(len(df)),
            "hand_oof_bundle": str(bundle_path),
            "leakage_audit": {
                "robot_loaded": False,
                "selection_split": "hand_train only",
                "group_column": "specimen_group",
                "class_from_path_features": False,
            },
        },
        recipe,
        recipe_path=RECIPE_PATH,
    )
    print(
        json.dumps(
            {
                "status": "selection_locked",
                "selection_lock": str(out_dir / "selection_lock.json"),
                "recipe_sha256": lock["recipe_sha256"],
                "selected_candidate": chosen,
                "next_step": "python run_paper_claim_v2_final_test_once.py",
            },
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
