#!/usr/bin/env python3
"""PAPER CLAIM — hand-only selection (never loads robot).

Pure single-shot path step 1/2:
  - Load frozen RECIPE.json
  - Select material bias on hand under pre-registered cascade-gain rule
  - Write outputs/paper_claim_v1/selection_lock.json (test_loaded=false)
  - Refuse if claim already sealed

Usage:
  python run_paper_claim_hand_selection.py
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


def soft_v2(meta: np.ndarray, hier: np.ndarray) -> np.ndarray:
    out = meta.copy()
    h = hier.argmax(1)
    out[h == 2] = hier[h == 2]
    return out


def apply_contact(pred_v2, oof_ts, oof_cs, oof_bin, oof_tt, oof_tr, primary, secondary):
    out = pred_v2.copy()
    out[
        (out == 0)
        & (oof_ts >= primary["ts_th"])
        & (np.maximum(oof_cs, oof_bin) >= primary["cs_th"])
    ] = 2
    if primary.get("do_tt", True):
        out[
            (out == 3)
            & (oof_tt >= primary["tt_th"])
            & (oof_ts >= primary["ts_th"] * 0.9)
        ] = 2
    out[
        (out == 0)
        & (oof_tr >= secondary["img_th"])
        & (oof_bin >= secondary["img_cs"])
    ] = 2
    return out


def decode_protect(contact, soft, T, bl, bt, bg):
    out = contact.copy()
    m = (out == 1) | (out == 3)
    if not m.any():
        return out
    logits = np.log(np.clip(soft[m][:, 1:4], 1e-12, None)) / float(T)
    logits = logits + np.array([bl, bt, bg], dtype=np.float64)
    out[m] = 1 + logits.argmax(1)
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


def main():
    recipe = claim.load_recipe()
    out_dir = claim.claim_out_dir(recipe)
    claim.assert_not_sealed(out_dir)

    contact_cfg = recipe["frozen_contact_stack"]
    primary = contact_cfg["primary"]
    secondary = contact_cfg["secondary"]
    mat_cfg = recipe["material_decode"]
    rule = mat_cfg["selection_rule"]
    grid = rule["grid"]
    temps = list(grid["T"])
    leaf_bs = claim.parse_linspace(grid["b_leaf"])
    trunk_bs = claim.parse_linspace(grid["b_trunk"])
    twig_bs = claim.parse_linspace(grid["b_twig"])
    cascades = [tuple(x) for x in rule["cascade_views"]]
    a_tt_g, a_tl_g = rule["cascade_gain_eval"]
    elig = rule["eligible"]
    cv_cfg = rule["cv"]

    bundle_path = Path(recipe["artifacts"]["hand_oof_bundle"])
    z = np.load(bundle_path)
    # rebuild labels/groups from hand manifest (specimen groups)
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
    assert len(z["sy"]) == n
    assert np.array_equal(z["sy"], sy)

    contact = apply_contact(
        z["pred_v2"],
        z["oof_ts"],
        z["oof_cs"],
        z["oof_bin"],
        z["oof_tt"],
        z["oof_tr_img"],
        primary,
        secondary,
    )
    soft0 = soft_v2(z["oof_meta"].astype(np.float64), z["oof_hier"].astype(np.float64))
    base = claim.fast_macro_f1(sy, contact)
    base_tr = float(np.sum((sy == 2) & (contact == 2)) / max((sy == 2).sum(), 1))
    print(
        json.dumps(
            {
                "step": "hand_selection",
                "recipe_id": recipe["recipe_id"],
                "recipe_sha256": claim.recipe_sha256(),
                "claim_out": str(out_dir),
                "base_contact_hand_macro": base,
                "base_trunk_rec": base_tr,
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
    for T in temps:
        for bl in leaf_bs:
            for bt in trunk_bs:
                for bg in twig_bs:
                    oof = np.zeros(n, np.int64)
                    fold_w = []
                    for _, va in folds:
                        vs = []
                        for a_tt, a_tl in cascades:
                            soft = cascade(soft0, sy, a_tt, a_tl)
                            pred = decode_protect(contact, soft, T, bl, bt, bg)
                            vs.append(claim.fast_macro_f1(sy[va], pred[va]))
                        fold_w.append(min(vs))
                        oof[va] = decode_protect(contact, soft0, T, bl, bt, bg)[va]
                    clean = claim.fast_macro_f1(sy, oof)
                    if clean < elig["clean_macro_min"]:
                        continue
                    trunk_rec = float(
                        np.sum((sy == 2) & (oof == 2)) / max((sy == 2).sum(), 1)
                    )
                    if trunk_rec < base_tr - elig["trunk_rec_drop_max"]:
                        continue
                    twig_rec = float(
                        np.sum((sy == 3) & (oof == 3)) / max((sy == 3).sum(), 1)
                    )
                    leaf_rec = float(
                        np.sum((sy == 1) & (oof == 1)) / max((sy == 1).sum(), 1)
                    )
                    false_tr = int(np.sum((sy == 0) & (oof == 2)))
                    if false_tr > elig["false_trunk_max"]:
                        continue
                    nflip = int(np.sum(oof != contact))
                    soft_c = cascade(soft0, sy, a_tt_g, a_tl_g)
                    pred_id = decode_protect(contact, soft_c, 1.0, 0.0, 0.0, 0.0)
                    pred_fx = decode_protect(contact, soft_c, T, bl, bt, bg)
                    casc_gain = claim.fast_macro_f1(sy, pred_fx) - claim.fast_macro_f1(
                        sy, pred_id
                    )
                    if nflip >= elig["nflip_min"] and casc_gain < 0.005:
                        continue
                    worst = float(np.min(fold_w))
                    score = (
                        1.0 * casc_gain
                        + 0.05 * (clean - base)
                        + 0.02 * (twig_rec - 0.89)
                        + 0.01 * min(nflip, 20) / 20
                    )
                    rows.append(
                        dict(
                            mode="protect_trunk",
                            T=float(T),
                            b_leaf=float(bl),
                            b_trunk=float(bt),
                            b_twig=float(bg),
                            worst=worst,
                            clean=float(clean),
                            casc_gain=float(casc_gain),
                            trunk_rec=trunk_rec,
                            twig_rec=twig_rec,
                            leaf_rec=leaf_rec,
                            nflip=nflip,
                            false_tr=false_tr,
                            score=float(score),
                        )
                    )

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(
            [
                dict(
                    mode="none",
                    T=1.0,
                    b_leaf=0.0,
                    b_trunk=0.0,
                    b_twig=0.0,
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
                    score=0.0,
                )
            ]
        )
    # stable sort (pre-registered rank keys)
    rank_keys = rule["rank_keys_desc"]
    df = df.sort_values(rank_keys, ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "hand_selection_leaderboard.csv", index=False)

    elig_df = df[
        (df.casc_gain >= elig["casc_gain_min"])
        & (df.clean >= elig["clean_macro_min"])
        & (df.nflip >= elig["nflip_min"])
    ]
    if len(elig_df) == 0:
        chosen = dict(
            mode="none",
            T=1.0,
            b_leaf=0.0,
            b_trunk=0.0,
            b_twig=0.0,
            worst=base,
            clean=base,
            casc_gain=0.0,
            trunk_rec=base_tr,
            twig_rec=float(np.sum((sy == 3) & (contact == 3)) / max((sy == 3).sum(), 1)),
            leaf_rec=float(np.sum((sy == 1) & (contact == 1)) / max((sy == 1).sum(), 1)),
            nflip=0,
            false_tr=int(np.sum((sy == 0) & (contact == 2))),
            score=0.0,
        )
        chosen_note = "fallback_identity_no_material_change"
    else:
        chosen = elig_df.iloc[0].to_dict()
        chosen_note = "eligible_cascade_gain_rank"

    print("TOP5", flush=True)
    print(df.head(5).to_string(index=False), flush=True)
    print("SELECTED", chosen, chosen_note, flush=True)

    lock = claim.write_hand_selection_lock(
        out_dir,
        {
            "protocol": "paper_claim_v1_hand_selection",
            "frozen_contact_stack": contact_cfg,
            "material_decode": {
                "mode": mat_cfg["mode"],
                "soft_source": mat_cfg["soft_source"],
                "apply_only_when_pred_in": mat_cfg["apply_only_when_pred_in"],
            },
            "selection_rule": rule["name"],
            "selected_candidate": chosen,
            "selection_note": chosen_note,
            "base_contact_hand_macro": base,
            "base_trunk_rec": base_tr,
            "n_candidates_scored": int(len(df)),
            "n_eligible": int(len(elig_df)),
            "hand_oof_bundle": str(bundle_path),
        },
        recipe,
    )
    print(
        json.dumps(
            {
                "status": "selection_locked",
                "selection_lock": str(out_dir / "selection_lock.json"),
                "recipe_sha256": lock["recipe_sha256"],
                "selected_candidate": chosen,
                "next_step": "python run_paper_claim_final_test_once.py",
            },
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
