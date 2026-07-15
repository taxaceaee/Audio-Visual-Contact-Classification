"""Hand-only gated residual material rescue on locked 0.837 contact.

Unlike full material redecode (which destroyed robot trunk), only flip contact
predictions when soft confidence is low / margin small, under leaf-contaminant
and leave-backbone stress views. Requires trunk_rec floor vs identity base.

Paper-pure: specimen_group OOF; never loads robot/test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import multimodal_085_protocol as proto
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_gated_mat_rescue_group_selection",
    )
)
BUNDLE = Path(
    "outputs/audio_feature_benchmarks/multimodal_085_material_logit_bias_group_selection/"
    "hand_oof_bundle.npz"
)
EXP = Path(
    "outputs/audio_feature_benchmarks/multimodal_085_leaf_contaminant_mat_group_selection/"
    "hand_oof_mat_experts.npz"
)
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
}
CLAP = Path("outputs/audio_clap_features/hand_train_full_X.npy")
W2V = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")
AUDIO_CLEAN = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"
)

PRIMARY = dict(ts_th=0.5, cs_th=0.1, tt_th=0.45, do_tt=True)
SECONDARY = dict(th=0.625, cth=0.75)


def pool(x, codes, n):
    return np.vstack([x[codes == i].mean(0) for i in range(n)]).astype(np.float32)


def fit_multi(x, y, c=0.1):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=2500,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42,
        ),
    )
    m.fit(x, y)
    return m


def proba_aligned(m, x, n_cls=3):
    p = m.predict_proba(x)
    out = np.zeros((len(x), n_cls), dtype=np.float64)
    for j, c in enumerate(m.classes_):
        if 0 <= int(c) < n_cls:
            out[:, int(c)] = p[:, j]
    s = out.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    return out / s


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


def soft_bias_mat3(mat3, T, bl, bt, bg):
    logits = np.log(np.clip(mat3, 1e-12, None)) / float(T)
    logits = logits + np.array([bl, bt, bg], dtype=np.float64)
    # softmax
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def apply_gate(contact, mat3, conf_th, margin_th, T, bl, bt, bg, modes):
    """modes: set of allowed flips as (from,to) class pairs in {1,2,3}."""
    out = contact.copy()
    m = out > 0
    if not m.any():
        return out, 0
    p = soft_bias_mat3(mat3[m], T, bl, bt, bg)
    conf = p.max(1)
    # top2 margin
    part = np.partition(p, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    arg = p.argmax(1) + 1  # class 1..3
    cur = out[m]
    flip = (conf >= conf_th) & (margin >= margin_th) & (arg != cur)
    # restrict flip types
    allowed = np.zeros(flip.shape, dtype=bool)
    for a, b in modes:
        allowed |= (cur == a) & (arg == b)
    flip &= allowed
    cur2 = cur.copy()
    cur2[flip] = arg[flip]
    out[m] = cur2
    return out, int(flip.sum())


def contaminate_mat3(mat3, alpha):
    if alpha <= 0:
        return mat3
    leaf = np.zeros_like(mat3)
    leaf[:, 0] = 1.0
    out = (1 - alpha) * mat3 + alpha * leaf
    return out / out.sum(1, keepdims=True)


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
    base_f1 = proto.fast_macro_f1(sy, contact)
    base_trunk = float(np.sum((sy == 2) & (contact == 2)) / max((sy == 2).sum(), 1))
    base_twig = float(np.sum((sy == 3) & (contact == 3)) / max((sy == 3).sum(), 1))
    base_leaf = float(np.sum((sy == 1) & (contact == 1)) / max((sy == 1).sum(), 1))
    print("base", base_f1, "trunk", base_trunk, flush=True)

    # load or build mat experts
    if EXP.exists():
        e = np.load(EXP)
        mats = {
            "multi": e["mat_multi"],
            "av": e["mat_av"],
            "clip": e["mat_clip"],
            "meta3": e["mat_meta3"],
            "hier3": e["mat_hier3"],
            "clap": e["mat_clap"],
            "blend_img_av": e["mat_blend_img_av"] if "mat_blend_img_av" in e else None,
        }
        if mats["blend_img_av"] is None:
            mats["blend_img_av"] = proto.normalize(0.5 * mats["multi"] + 0.5 * mats["av"])
        mats["blend_mh"] = proto.normalize(0.5 * mats["meta3"] + 0.5 * mats["hier3"])
        mats["blend_all"] = proto.normalize(
            (mats["multi"] + mats["av"] + mats["meta3"] + mats["clap"]) / 4.0
        )
        print("reused mat experts", flush=True)
    else:
        raise SystemExit("need hand_oof_mat_experts.npz from leaf_contaminant run")

    # mode families (from,to)
    mode_sets = {
        "none": [],
        "lt_only": [(1, 3)],  # leaf->twig
        "tl_only": [(3, 1)],
        "ttw_only": [(2, 3)],  # trunk->twig
        "twt_only": [(3, 2)],  # twig->trunk
        "wood_swap": [(2, 3), (3, 2)],
        "leaf_twig": [(1, 3), (3, 1)],
        "leaf_to_wood": [(1, 2), (1, 3)],
        "twig_leaf_trunk": [(3, 1), (3, 2), (1, 3)],
        "all_mat": [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)],
        # residual-targeted: fix leafy bias (leaf<-twig, twig<-trunk recovery)
        "anti_leafy": [(1, 3), (3, 2), (1, 2)],
        "anti_leafy_soft": [(1, 3), (3, 2)],
    }

    # Compact high-ROI grid (full product is too large)
    sources = ["blend_mh", "multi", "av", "blend_all", "meta3", "clip"]
    confs = [0.45, 0.55, 0.65, 0.75, 0.85]
    margins = [0.0, 0.08, 0.15, 0.25]
    temps = [1.0]
    # mild anti-leafy biases + identity
    bias_triples = [
        (0.0, 0.0, 0.0),
        (-0.5, 0.4, 0.0),
        (-0.8, 0.6, 0.2),
        (-1.0, 0.8, 0.0),
        (-0.4, 0.6, -0.2),
        (0.0, 0.5, -0.3),
        (-0.6, 0.3, 0.4),
        (0.0, 0.0, 0.5),
    ]
    mode_priority = [
        "anti_leafy_soft",
        "anti_leafy",
        "lt_only",
        "twt_only",
        "wood_swap",
        "leaf_twig",
        "leaf_to_wood",
        "all_mat",
    ]

    alphas = [0.0, 0.25, 0.4]
    rows = []
    best = None
    best_comp = -1e9

    # identity always
    id_row = dict(
        mode="none",
        source="blend_mh",
        conf_th=1.0,
        margin_th=1.0,
        T=1.0,
        b_leaf=0.0,
        b_trunk=0.0,
        b_twig=0.0,
        worst=base_f1,
        clean=base_f1,
        cont_worst=base_f1,
        trunk_rec=base_trunk,
        twig_rec=base_twig,
        leaf_rec=base_leaf,
        nflip=0,
        cont_gain=0.0,
        false_tr=int(np.sum((sy == 0) & (contact == 2))),
        amb=float(np.sum((sy == 0) & (contact == 0)) / max((sy == 0).sum(), 1)),
        comp=0.55 * base_f1 + 0.25 * base_f1 + 0.1 * base_trunk,
    )
    rows.append(id_row)
    best = id_row
    best_comp = id_row["comp"]

    print("grid gated rescue...", flush=True)
    n_scored = 0
    for mode_name in mode_priority:
        modes = mode_sets[mode_name]
        for src in sources:
            mat0 = mats[src]
            for conf_th in confs:
                for margin_th in margins:
                    for T in temps:
                        for bl, bt, bg in bias_triples:
                            fold_w = []
                            cont_fold = []
                            oof_pred = contact.copy()
                            for _, va in folds:
                                vs = []
                                for a in alphas:
                                    mat_v = contaminate_mat3(mat0, a)
                                    pred_v, _ = apply_gate(
                                        contact,
                                        mat_v,
                                        conf_th,
                                        margin_th,
                                        T,
                                        bl,
                                        bt,
                                        bg,
                                        modes,
                                    )
                                    vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                                for bb in ("clip", "multi", "meta3"):
                                    pred_v, _ = apply_gate(
                                        contact,
                                        mats[bb],
                                        conf_th,
                                        margin_th,
                                        T,
                                        bl,
                                        bt,
                                        bg,
                                        modes,
                                    )
                                    vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                                fold_w.append(min(vs))
                                cont_fold.append(vs[2])  # a=0.4
                                pred_c, _ = apply_gate(
                                    contact, mat0, conf_th, margin_th, T, bl, bt, bg, modes
                                )
                                oof_pred[va] = pred_c[va]
                            n_scored += 1
                            worst = float(np.min(fold_w))
                            cont_worst = float(np.min(cont_fold))
                            clean = proto.fast_macro_f1(sy, oof_pred)
                            if clean < 0.92:
                                continue
                            trunk_rec = float(
                                np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1)
                            )
                            if trunk_rec < base_trunk - 0.03:
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
                            if nflip > 80:
                                continue
                            mat_c = contaminate_mat3(mat0, 0.35)
                            pred_id_c = contact  # identity under cont uses same hard labels
                            pred_fix_c, _ = apply_gate(
                                contact, mat_c, conf_th, margin_th, T, bl, bt, bg, modes
                            )
                            cont_gain = proto.fast_macro_f1(sy, pred_fix_c) - proto.fast_macro_f1(
                                sy, pred_id_c
                            )
                            clean_gain = clean - base_f1
                            if cont_gain < 0.005 and clean_gain < 0.002 and nflip > 0:
                                continue
                            bal = min(leaf_rec, trunk_rec, twig_rec)
                            comp = (
                                0.35 * worst
                                + 0.20 * clean
                                + 0.15 * cont_worst
                                + 0.12 * max(cont_gain, 0)
                                + 0.10 * trunk_rec
                                + 0.05 * bal
                                + 0.03 * min(nflip, 30) / 30
                                - 0.02 * max(nflip - 40, 0) / 40
                                - 0.01 * false_tr
                            )
                            if mode_name.startswith("anti_leafy") and cont_gain > 0.01:
                                comp += 0.008
                            row = dict(
                                mode=mode_name,
                                source=src,
                                conf_th=float(conf_th),
                                margin_th=float(margin_th),
                                T=float(T),
                                b_leaf=float(bl),
                                b_trunk=float(bt),
                                b_twig=float(bg),
                                worst=worst,
                                clean=clean,
                                cont_worst=cont_worst,
                                trunk_rec=trunk_rec,
                                twig_rec=twig_rec,
                                leaf_rec=leaf_rec,
                                nflip=nflip,
                                cont_gain=float(cont_gain),
                                false_tr=false_tr,
                                amb=amb,
                                comp=float(comp),
                            )
                            rows.append(row)
                            if comp > best_comp:
                                best_comp = comp
                                best = row
    print(f"scored {n_scored} candidates, kept {len(rows)}", flush=True)

    df = pd.DataFrame(rows).sort_values("comp", ascending=False)
    df.to_csv(OUT / "hand_gated_mat_rescue_leaderboard.csv", index=False)
    print(df.head(20).to_string(index=False), flush=True)
    print("SELECTED", best, flush=True)

    # store mode pairs for final test
    modes_selected = mode_sets[best["mode"]]
    proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_gated_mat_rescue_group_OOF",
            "primary": PRIMARY,
            "secondary": SECONDARY,
            "selected_candidate": best,
            "selected_modes": modes_selected,
            "base_contact_hand_macro": base_f1,
            "base_trunk_rec": base_trunk,
            "selection_recipe": (
                "gated material flips only (conf+margin+allowed pairs) under leaf-contaminant "
                "and leave-backbone views; trunk_rec floor vs 0.837 identity"
            ),
            "base_robot_artifact": (
                "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
                "segment_rule_stack_v2_base_segment_outputs.npz"
            ),
            "n_candidates_scored": int(len(df)),
        },
    )
    print("lock written", flush=True)


if __name__ == "__main__":
    main()
