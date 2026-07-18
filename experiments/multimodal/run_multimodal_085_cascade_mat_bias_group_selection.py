"""Hand-only: material logit bias under class-cascade surrogate on locked 0.837.

Surrogate (hand labels only): push soft mass trunk→twig and twig→leaf to mimic the
residual robot confusion cascade, then select temperature + material logit biases that
recover worst-view macro F1 while preserving clean hand F1 and trunk_rec floor.

Decode is bias-only on locked meta/hier soft (same family as v2), not full expert redecode.

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
        "outputs/audio_feature_benchmarks/multimodal_085_cascade_mat_bias_group_selection",
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


def cascade_contaminate(soft4, y, a_tt, a_tl, contact_mask):
    """Push true-trunk mass toward twig; true-twig mass toward leaf (hand y only)."""
    out = soft4.copy()
    # trunk -> twig
    m = contact_mask & (y == 2) & (a_tt > 0)
    if m.any():
        out[m, 3] = out[m, 3] + a_tt * out[m, 2]
        out[m, 2] = (1.0 - a_tt) * out[m, 2]
        out[m] = proto.normalize(out[m])
    # twig -> leaf
    m = contact_mask & (y == 3) & (a_tl > 0)
    if m.any():
        out[m, 1] = out[m, 1] + a_tl * out[m, 3]
        out[m, 3] = (1.0 - a_tl) * out[m, 3]
        out[m] = proto.normalize(out[m])
    return out


def leaf_contaminate(soft4, alpha, contact_mask):
    out = soft4.copy()
    if alpha <= 0:
        return out
    leaf = np.zeros(4)
    leaf[1] = 1.0
    out[contact_mask] = (1 - alpha) * out[contact_mask] + alpha * leaf
    out[contact_mask] = proto.normalize(out[contact_mask])
    return out


def material_decode(contact, soft4, T, bl, bt, bg):
    out = contact.copy()
    m = out > 0
    if not m.any():
        return out
    logits = np.log(np.clip(soft4[m][:, 1:4], 1e-12, None)) / float(T)
    logits = logits + np.array([bl, bt, bg], dtype=np.float64)
    out[m] = 1 + logits.argmax(1)
    return out


def soft_for_source(name, oof_meta, oof_hier):
    if name == "meta":
        return oof_meta
    if name == "hier":
        return oof_hier
    if name == "blend_mh":
        return proto.normalize(0.5 * oof_meta + 0.5 * oof_hier)
    if name == "blend_h_heavy":
        return proto.normalize(0.7 * oof_hier + 0.3 * oof_meta)
    if name == "blend_m_heavy":
        return proto.normalize(0.7 * oof_meta + 0.3 * oof_hier)
    if name == "v2_rule_soft":
        # approximate hier_trunk_meta_else in soft: where hier argmax==trunk use hier else meta
        h = oof_hier.argmax(1)
        out = oof_meta.copy()
        out[h == 2] = oof_hier[h == 2]
        return out
    raise KeyError(name)


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
    oof_meta = z["oof_meta"].astype(np.float64)
    oof_hier = z["oof_hier"].astype(np.float64)
    contact = apply_contact(
        z["pred_v2"], z["oof_ts"], z["oof_cs"], z["oof_bin"], z["oof_tt"], z["oof_tr_img"]
    )
    base_f1 = proto.fast_macro_f1(sy, contact)
    base_trunk = float(np.sum((sy == 2) & (contact == 2)) / max((sy == 2).sum(), 1))
    print("base", base_f1, "trunk", base_trunk, flush=True)

    contact_mask = contact > 0
    sources = ["v2_rule_soft", "meta", "hier", "blend_mh", "blend_h_heavy", "blend_m_heavy"]
    temps = [0.6, 0.8, 1.0, 1.25, 1.5]
    leaf_bs = np.linspace(-1.5, 0.3, 10)
    trunk_bs = np.linspace(-0.3, 1.5, 10)
    twig_bs = np.linspace(-0.6, 1.0, 9)

    # cascade strengths for views
    cascades = [(0.0, 0.0), (0.25, 0.25), (0.4, 0.35), (0.55, 0.45), (0.35, 0.55)]
    leaf_alphas = [0.0, 0.25, 0.4]

    rows = []
    best = None
    best_comp = -1e9

    id_row = dict(
        source="identity",
        T=1.0,
        b_leaf=0.0,
        b_trunk=0.0,
        b_twig=0.0,
        worst=base_f1,
        clean=base_f1,
        cascade_worst=base_f1,
        trunk_rec=base_trunk,
        twig_rec=float(np.sum((sy == 3) & (contact == 3)) / max((sy == 3).sum(), 1)),
        leaf_rec=float(np.sum((sy == 1) & (contact == 1)) / max((sy == 1).sum(), 1)),
        nflip=0,
        casc_gain=0.0,
        false_tr=int(np.sum((sy == 0) & (contact == 2))),
        amb=float(np.sum((sy == 0) & (contact == 0)) / max((sy == 0).sum(), 1)),
        comp=0.5 * base_f1 + 0.3 * base_f1 + 0.15 * base_trunk,
    )
    rows.append(id_row)
    best = id_row
    best_comp = id_row["comp"]

    print("grid cascade bias...", flush=True)
    for src in sources:
        soft0 = soft_for_source(src, oof_meta, oof_hier)
        for T in temps:
            for bl in leaf_bs[::2]:
                for bt in trunk_bs[::2]:
                    for bg in twig_bs[::2]:
                        fold_w = []
                        casc_fold = []
                        oof_pred = np.zeros(n, np.int64)
                        for _, va in folds:
                            vs = []
                            for a_tt, a_tl in cascades:
                                soft_v = cascade_contaminate(
                                    soft0, sy, a_tt, a_tl, contact_mask
                                )
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            for a in leaf_alphas:
                                soft_v = leaf_contaminate(soft0, a, contact_mask)
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            fold_w.append(min(vs))
                            # strong cascade view score
                            soft_c = cascade_contaminate(soft0, sy, 0.45, 0.4, contact_mask)
                            pred_c = material_decode(contact, soft_c, T, bl, bt, bg)
                            casc_fold.append(proto.fast_macro_f1(sy[va], pred_c[va]))
                            oof_pred[va] = material_decode(contact, soft0, T, bl, bt, bg)[va]
                        worst = float(np.min(fold_w))
                        casc_worst = float(np.min(casc_fold))
                        clean = proto.fast_macro_f1(sy, oof_pred)
                        if clean < 0.92:
                            continue
                        trunk_rec = float(
                            np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1)
                        )
                        if trunk_rec < base_trunk - 0.04:
                            continue
                        twig_rec = float(
                            np.sum((sy == 3) & (oof_pred == 3)) / max((sy == 3).sum(), 1)
                        )
                        leaf_rec = float(
                            np.sum((sy == 1) & (oof_pred == 1)) / max((sy == 1).sum(), 1)
                        )
                        amb = float(
                            np.sum((sy == 0) & (oof_pred == 0)) / max((sy == 0).sum(), 1)
                        )
                        if amb < 0.97:
                            continue
                        false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
                        if false_tr > 15:
                            continue
                        nflip = int(np.sum(oof_pred != contact))
                        soft_c = cascade_contaminate(soft0, sy, 0.45, 0.4, contact_mask)
                        pred_id_c = material_decode(contact, soft_c, 1.0, 0, 0, 0)
                        pred_fix_c = material_decode(contact, soft_c, T, bl, bt, bg)
                        casc_gain = proto.fast_macro_f1(sy, pred_fix_c) - proto.fast_macro_f1(
                            sy, pred_id_c
                        )
                        # must show cascade recovery if flipping
                        if nflip > 0 and casc_gain < 0.008:
                            continue
                        if nflip > 120:
                            continue
                        bal = min(leaf_rec, trunk_rec, twig_rec)
                        # prefer leaf-down trunk-up if cascade gain positive
                        direction = 0.0
                        if bl < -0.15 and bt > 0.15 and casc_gain > 0.01:
                            direction = 0.01
                        # twig should not dominate trunk bias
                        if bg > bt + 0.3:
                            direction -= 0.005
                        comp = (
                            0.32 * worst
                            + 0.18 * clean
                            + 0.18 * casc_worst
                            + 0.15 * max(casc_gain, 0)
                            + 0.10 * trunk_rec
                            + 0.05 * bal
                            + direction
                            - 0.01 * false_tr
                            - 0.015 * max(nflip - 50, 0) / 50
                        )
                        row = dict(
                            source=src,
                            T=float(T),
                            b_leaf=float(bl),
                            b_trunk=float(bt),
                            b_twig=float(bg),
                            worst=worst,
                            clean=clean,
                            cascade_worst=casc_worst,
                            trunk_rec=trunk_rec,
                            twig_rec=twig_rec,
                            leaf_rec=leaf_rec,
                            nflip=nflip,
                            casc_gain=float(casc_gain),
                            false_tr=false_tr,
                            amb=amb,
                            comp=float(comp),
                        )
                        rows.append(row)
                        if comp > best_comp:
                            best_comp = comp
                            best = row

    # local refine
    if best is not None and best["source"] != "identity":
        print("refine", best, flush=True)
        src = best["source"]
        soft0 = soft_for_source(src, oof_meta, oof_hier)
        for T in np.linspace(max(0.5, best["T"] - 0.3), best["T"] + 0.3, 5):
            for bl in np.linspace(best["b_leaf"] - 0.35, best["b_leaf"] + 0.35, 7):
                for bt in np.linspace(best["b_trunk"] - 0.35, best["b_trunk"] + 0.35, 7):
                    for bg in np.linspace(best["b_twig"] - 0.3, best["b_twig"] + 0.3, 5):
                        fold_w = []
                        casc_fold = []
                        oof_pred = np.zeros(n, np.int64)
                        for _, va in folds:
                            vs = []
                            for a_tt, a_tl in cascades:
                                soft_v = cascade_contaminate(
                                    soft0, sy, a_tt, a_tl, contact_mask
                                )
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            for a in leaf_alphas:
                                soft_v = leaf_contaminate(soft0, a, contact_mask)
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            fold_w.append(min(vs))
                            soft_c = cascade_contaminate(soft0, sy, 0.45, 0.4, contact_mask)
                            pred_c = material_decode(contact, soft_c, T, bl, bt, bg)
                            casc_fold.append(proto.fast_macro_f1(sy[va], pred_c[va]))
                            oof_pred[va] = material_decode(contact, soft0, T, bl, bt, bg)[va]
                        worst = float(np.min(fold_w))
                        casc_worst = float(np.min(casc_fold))
                        clean = proto.fast_macro_f1(sy, oof_pred)
                        if clean < 0.92:
                            continue
                        trunk_rec = float(
                            np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1)
                        )
                        if trunk_rec < base_trunk - 0.04:
                            continue
                        twig_rec = float(
                            np.sum((sy == 3) & (oof_pred == 3)) / max((sy == 3).sum(), 1)
                        )
                        leaf_rec = float(
                            np.sum((sy == 1) & (oof_pred == 1)) / max((sy == 1).sum(), 1)
                        )
                        amb = float(
                            np.sum((sy == 0) & (oof_pred == 0)) / max((sy == 0).sum(), 1)
                        )
                        if amb < 0.97:
                            continue
                        false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
                        if false_tr > 15:
                            continue
                        nflip = int(np.sum(oof_pred != contact))
                        soft_c = cascade_contaminate(soft0, sy, 0.45, 0.4, contact_mask)
                        pred_id_c = material_decode(contact, soft_c, 1.0, 0, 0, 0)
                        pred_fix_c = material_decode(contact, soft_c, T, bl, bt, bg)
                        casc_gain = proto.fast_macro_f1(sy, pred_fix_c) - proto.fast_macro_f1(
                            sy, pred_id_c
                        )
                        if nflip > 0 and casc_gain < 0.008:
                            continue
                        if nflip > 120:
                            continue
                        bal = min(leaf_rec, trunk_rec, twig_rec)
                        direction = 0.0
                        if bl < -0.15 and bt > 0.15 and casc_gain > 0.01:
                            direction = 0.01
                        if bg > bt + 0.3:
                            direction -= 0.005
                        comp = (
                            0.32 * worst
                            + 0.18 * clean
                            + 0.18 * casc_worst
                            + 0.15 * max(casc_gain, 0)
                            + 0.10 * trunk_rec
                            + 0.05 * bal
                            + direction
                            - 0.01 * false_tr
                            - 0.015 * max(nflip - 50, 0) / 50
                        )
                        row = dict(
                            source=src,
                            T=float(T),
                            b_leaf=float(bl),
                            b_trunk=float(bt),
                            b_twig=float(bg),
                            worst=worst,
                            clean=clean,
                            cascade_worst=casc_worst,
                            trunk_rec=trunk_rec,
                            twig_rec=twig_rec,
                            leaf_rec=leaf_rec,
                            nflip=nflip,
                            casc_gain=float(casc_gain),
                            false_tr=false_tr,
                            amb=amb,
                            comp=float(comp),
                        )
                        rows.append(row)
                        if comp > best_comp:
                            best_comp = comp
                            best = row

    df = pd.DataFrame(rows).sort_values("comp", ascending=False)
    df.to_csv(OUT / "hand_cascade_mat_bias_leaderboard.csv", index=False)
    print(df.head(15).to_string(index=False), flush=True)
    print("SELECTED", best, flush=True)

    proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_cascade_mat_bias_group_OOF",
            "primary": PRIMARY,
            "secondary": SECONDARY,
            "selected_candidate": best,
            "base_contact_hand_macro": base_f1,
            "base_trunk_rec": base_trunk,
            "selection_recipe": (
                "bias-only material decode on meta/hier soft; select under trunk→twig / "
                "twig→leaf cascade contaminant + leaf mix; trunk floor; hand only"
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
