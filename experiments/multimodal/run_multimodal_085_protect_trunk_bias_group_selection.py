"""Hand-only: protect-trunk material bias on locked 0.837 contact stack.

Material logit bias is applied only when the current hard pred is leaf or twig
(trunk hard labels are protected). Soft source is v2_rule_soft (hier when hier
says trunk else meta). Selection maximizes cascade recovery gain under hand-only
trunk→twig / twig→leaf soft contaminants, with clean F1 and trunk_rec floors.

Paper-pure: specimen_group OOF; never loads robot/test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import multimodal_085_protocol as proto
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_protect_trunk_bias_group_selection",
    )
)
BUNDLE = Path(
    "outputs/audio_feature_benchmarks/multimodal_085_material_logit_bias_group_selection/"
    "hand_oof_bundle.npz"
)

PRIMARY = dict(ts_th=0.5, cs_th=0.1, tt_th=0.45, do_tt=True)
SECONDARY = dict(th=0.625, cth=0.75)


def apply_contact(pred_v2, oof_ts, oof_cs, oof_bin, oof_tt, oof_tr):
    out = pred_v2.copy()
    out[
        (out == 0)
        & (oof_ts >= PRIMARY["ts_th"])
        & (np.maximum(oof_cs, oof_bin) >= PRIMARY["cs_th"])
    ] = 2
    if PRIMARY["do_tt"]:
        out[
            (out == 3)
            & (oof_tt >= PRIMARY["tt_th"])
            & (oof_ts >= PRIMARY["ts_th"] * 0.9)
        ] = 2
    out[(out == 0) & (oof_tr >= SECONDARY["th"]) & (oof_bin >= SECONDARY["cth"])] = 2
    return out


def soft_v2(meta, hier):
    out = meta.copy()
    h = hier.argmax(1)
    out[h == 2] = hier[h == 2]
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
        out[m] = proto.normalize(out[m])
    m = y == 3
    if m.any() and a_tl > 0:
        out[m, 1] = out[m, 1] + a_tl * out[m, 3]
        out[m, 3] = (1.0 - a_tl) * out[m, 3]
        out[m] = proto.normalize(out[m])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(n), sy, ss)
    )

    z = np.load(BUNDLE)
    contact = apply_contact(
        z["pred_v2"], z["oof_ts"], z["oof_cs"], z["oof_bin"], z["oof_tt"], z["oof_tr_img"]
    )
    meta = z["oof_meta"].astype(np.float64)
    hier = z["oof_hier"].astype(np.float64)
    soft0 = soft_v2(meta, hier)
    base = proto.fast_macro_f1(sy, contact)
    base_tr = float(np.sum((sy == 2) & (contact == 2)) / max((sy == 2).sum(), 1))
    print("base", base, "trunk", base_tr, flush=True)

    temps = [0.6, 0.8, 1.0, 1.25]
    leaf_bs = np.linspace(-1.6, 0.2, 10)
    trunk_bs = np.linspace(0.0, 1.6, 9)
    twig_bs = np.linspace(-0.4, 1.0, 8)
    cascades = [(0.0, 0.0), (0.35, 0.35), (0.5, 0.4)]

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
                            vs.append(proto.fast_macro_f1(sy[va], pred[va]))
                        fold_w.append(min(vs))
                        oof[va] = decode_protect(contact, soft0, T, bl, bt, bg)[va]
                    clean = proto.fast_macro_f1(sy, oof)
                    if clean < 0.93:
                        continue
                    trunk_rec = float(
                        np.sum((sy == 2) & (oof == 2)) / max((sy == 2).sum(), 1)
                    )
                    if trunk_rec < base_tr - 0.01:
                        continue
                    twig_rec = float(
                        np.sum((sy == 3) & (oof == 3)) / max((sy == 3).sum(), 1)
                    )
                    leaf_rec = float(
                        np.sum((sy == 1) & (oof == 1)) / max((sy == 1).sum(), 1)
                    )
                    false_tr = int(np.sum((sy == 0) & (oof == 2)))
                    if false_tr > 12:
                        continue
                    nflip = int(np.sum(oof != contact))
                    soft_c = cascade(soft0, sy, 0.45, 0.4)
                    pred_id = decode_protect(contact, soft_c, 1.0, 0.0, 0.0, 0.0)
                    pred_fx = decode_protect(contact, soft_c, T, bl, bt, bg)
                    casc_gain = proto.fast_macro_f1(sy, pred_fx) - proto.fast_macro_f1(
                        sy, pred_id
                    )
                    if nflip > 0 and casc_gain < 0.005:
                        continue
                    worst = float(np.min(fold_w))
                    bal = min(leaf_rec, trunk_rec, twig_rec)
                    # Primary selection score: cascade recovery under floors
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
                            clean=clean,
                            casc_gain=float(casc_gain),
                            trunk_rec=trunk_rec,
                            twig_rec=twig_rec,
                            leaf_rec=leaf_rec,
                            nflip=nflip,
                            false_tr=false_tr,
                            score=float(score),
                        )
                    )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("no candidates passed floors")
    df = df.sort_values(
        ["casc_gain", "twig_rec", "clean", "score"], ascending=False
    ).reset_index(drop=True)
    df.to_csv(OUT / "hand_protect_trunk_bias_leaderboard.csv", index=False)

    # Selection rule (paper-pure, pre-registered style):
    # among protect_trunk candidates with casc_gain>=0.02, clean>=0.94, nflip>0,
    # maximize casc_gain then twig_rec then clean.
    elig = df[(df.casc_gain >= 0.02) & (df.clean >= 0.94) & (df.nflip > 0)]
    if len(elig) == 0:
        # fall back to identity (no material change)
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
    else:
        chosen = elig.iloc[0].to_dict()

    print(df.head(15).to_string(index=False), flush=True)
    print("SELECTED", chosen, flush=True)

    proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_protect_trunk_bias_group_OOF",
            "primary": PRIMARY,
            "secondary": SECONDARY,
            "selected_candidate": chosen,
            "source": "v2_rule_soft",
            "protect_trunk": True,
            "base_contact_hand_macro": base,
            "base_trunk_rec": base_tr,
            "selection_recipe": (
                "protect-trunk leaf/twig redecode with v2_rule_soft+bias; "
                "eligible if casc_gain>=0.02 clean>=0.94 nflip>0 trunk_rec floor; "
                "rank by casc_gain, twig_rec, clean"
            ),
            "base_robot_artifact": (
                "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
                "segment_rule_stack_v2_base_segment_outputs.npz"
            ),
            "n_candidates_scored": int(len(df)),
        },
    )


if __name__ == "__main__":
    main()
