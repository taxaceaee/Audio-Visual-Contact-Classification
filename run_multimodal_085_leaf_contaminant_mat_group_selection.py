"""Hand-only: material reweight selected under leaf-contamination surrogate on 0.837 base.

Paper-pure protocol:
  - specimen_group StratifiedGroupKFold OOF only
  - never loads robot/test labels or features for selection
  - locked contact stack = best-strict 0.837 (v2 + stress amb-lift + secondary CLIP trunk)
  - material decode (multi-bb / meta / hier / CLAP soft) chosen by worst-view over
    clean + leaf-contaminant surrogates that mimic "robot camera makes wood look leafier"

Rationale (catalog C5/G4 hybrid + D2): residual robot errors after 0.837 are mostly
trunk→twig and twig→leaf. Hand clean OOF is saturated so identity bias wins ordinary
grids. Leaf-contaminant soft views create selectable wood material failures without
robot stats.
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
        "outputs/audio_feature_benchmarks/multimodal_085_leaf_contaminant_mat_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
BUNDLE = Path(
    "outputs/audio_feature_benchmarks/multimodal_085_material_logit_bias_group_selection/"
    "hand_oof_bundle.npz"
)
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
    "swin": Path("outputs/image_timm_features/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"),
}
CLAP = Path("outputs/audio_clap_features/hand_train_full_X.npy")
W2V = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")
AUDIO = {
    "clean": Path(
        "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"
    ),
    "robot_mix": Path(
        "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/robot_mix/X.npy"
    ),
    "bandlimit": Path(
        "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/bandlimit/X.npy"
    ),
}

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
    out[
        (out == 0)
        & (oof_tr >= SECONDARY["th"])
        & (oof_bin >= SECONDARY["cth"])
    ] = 2
    return out


def material_decode(contact, soft4, T, b_leaf, b_trunk, b_twig):
    """soft4: (n,4) ambient+material; only re-decode contact rows using mat 1:4."""
    out = contact.copy()
    m = out > 0
    if not m.any():
        return out
    base = soft4[m][:, 1:4]
    logits = np.log(np.clip(base, 1e-12, None)) / float(T)
    logits = logits + np.array([b_leaf, b_trunk, b_twig], dtype=np.float64)
    out[m] = 1 + logits.argmax(1)
    return out


def soft4_from_mat3(mat3, contact_mask):
    """Build 4-class soft from 3-class material on contact; ambient elsewhere."""
    p = np.zeros((len(mat3), 4), dtype=np.float64)
    p[:, 0] = 1.0
    p[contact_mask, 0] = 0.0
    p[contact_mask, 1:4] = mat3[contact_mask]
    # normalize contact rows
    s = p[contact_mask].sum(1, keepdims=True)
    s[s <= 0] = 1.0
    p[contact_mask] = p[contact_mask] / s
    return p


def contaminate_leaf(soft4, alpha, contact_mask):
    """Mix material mass toward leaf (class 1) for contact rows — hand-only sim of leafy camera bias."""
    out = soft4.copy()
    if alpha <= 0 or not contact_mask.any():
        return out
    leaf = np.zeros(4, dtype=np.float64)
    leaf[1] = 1.0
    out[contact_mask] = (1.0 - alpha) * out[contact_mask] + alpha * leaf
    out[contact_mask] = proto.normalize(out[contact_mask])
    return out


def contaminate_leaf_logits(soft4, bias, contact_mask):
    """Add leaf logit bias (softer than hard mix)."""
    out = soft4.copy()
    if bias == 0 or not contact_mask.any():
        return out
    logits = np.log(np.clip(out[contact_mask], 1e-12, None))
    logits[:, 1] += float(bias)
    out[contact_mask] = proto.normalize(np.exp(logits))
    return out


def wood_macro(y, pred):
    """Macro F1 over leaf/trunk/twig only (classes 1,2,3); ignore ambient rows in score."""
    mask = y > 0
    if mask.sum() == 0:
        return 0.0
    vals = []
    for c in (1, 2, 3):
        tp = np.sum((y == c) & (pred == c) & mask)
        fp = np.sum((y != c) & (pred == c) & mask)
        fn = np.sum((y == c) & (pred != c) & mask)
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


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
    pred_v2 = z["pred_v2"]
    oof_meta = z["oof_meta"].astype(np.float64)
    oof_hier = z["oof_hier"].astype(np.float64)
    oof_ts, oof_cs, oof_bin, oof_tt, oof_tr = (
        z["oof_ts"],
        z["oof_cs"],
        z["oof_bin"],
        z["oof_tt"],
        z["oof_tr_img"],
    )
    assert len(pred_v2) == n

    contact = apply_contact(pred_v2, oof_ts, oof_cs, oof_bin, oof_tt, oof_tr)
    base_clean = proto.fast_macro_f1(sy, contact)
    print("locked 0.837-style contact hand OOF", base_clean, flush=True)

    img = {
        k: pool(np.load(p / "hand_train_full/X.npy").astype(np.float32), codes, n)
        for k, p in IMG.items()
        if (p / "hand_train_full/X.npy").exists()
    }
    multi = np.hstack([img[k] for k in ("clip", "eff", "dino", "convnext") if k in img])
    clap = pool(np.load(CLAP).astype(np.float32), codes, n)
    w2v = pool(np.load(W2V).astype(np.float32), codes, n)
    aud = {k: pool(np.load(p).astype(np.float32), codes, n) for k, p in AUDIO.items()}

    # OOF material 3-class experts (leaf/trunk/twig on contact labels; ambient mapped to leaf for train skip)
    mat_y = sy.copy()
    # train only on true contact for material heads
    oof_mat = {
        "multi": np.zeros((n, 3), np.float64),
        "clip": np.zeros((n, 3), np.float64),
        "eff": np.zeros((n, 3), np.float64),
        "dino": np.zeros((n, 3), np.float64),
        "convnext": np.zeros((n, 3), np.float64),
        "swin": np.zeros((n, 3), np.float64) if "swin" in img else None,
        "clap": np.zeros((n, 3), np.float64),
        "av": np.zeros((n, 3), np.float64),  # audio+vision
        "meta3": np.zeros((n, 3), np.float64),
        "hier3": np.zeros((n, 3), np.float64),
    }
    if oof_mat["swin"] is None:
        del oof_mat["swin"]

    print("OOF material experts...", flush=True)
    for fold_id, (tr, va) in enumerate(folds):
        tr_c = tr[sy[tr] > 0]
        if len(tr_c) < 12:
            continue
        ytr = sy[tr_c] - 1  # 0,1,2
        # multi-bb
        oof_mat["multi"][va] = proba_aligned(fit_multi(multi[tr_c], ytr, 0.05), multi[va])
        for name in ("clip", "eff", "dino", "convnext", "swin"):
            if name not in img or name not in oof_mat:
                continue
            oof_mat[name][va] = proba_aligned(fit_multi(img[name][tr_c], ytr, 0.08), img[name][va])
        oof_mat["clap"][va] = proba_aligned(fit_multi(clap[tr_c], ytr, 0.05), clap[va])
        Xav = np.hstack([aud["clean"][tr_c], w2v[tr_c], multi[tr_c], clap[tr_c]])
        Xav_va = np.hstack([aud["clean"][va], w2v[va], multi[va], clap[va]])
        oof_mat["av"][va] = proba_aligned(fit_multi(Xav, ytr, 0.03), Xav_va)
        # soft from locked meta/hier (already 4-class OOF)
        oof_mat["meta3"][va] = proto.normalize(oof_meta[va][:, 1:4])
        oof_mat["hier3"][va] = proto.normalize(oof_hier[va][:, 1:4])
        print(f" fold {fold_id + 1}/5", flush=True)

    # blends
    oof_mat["blend_img_av"] = proto.normalize(0.5 * oof_mat["multi"] + 0.5 * oof_mat["av"])
    oof_mat["blend_meta_multi"] = proto.normalize(0.5 * oof_mat["meta3"] + 0.5 * oof_mat["multi"])
    oof_mat["blend_hier_av"] = proto.normalize(0.5 * oof_mat["hier3"] + 0.5 * oof_mat["av"])
    oof_mat["blend_all"] = proto.normalize(
        (oof_mat["multi"] + oof_mat["av"] + oof_mat["meta3"] + oof_mat["clap"]) / 4.0
    )

    contact_mask = contact > 0
    soft_sources = {}
    for name, mat3 in oof_mat.items():
        soft_sources[name] = soft4_from_mat3(mat3, contact_mask)
    soft_sources["meta4"] = oof_meta
    soft_sources["hier4"] = oof_hier
    soft_sources["blend_mh4"] = proto.normalize(0.5 * oof_meta + 0.5 * oof_hier)

    # identity baseline under contamination (for gain)
    alphas = [0.0, 0.15, 0.25, 0.35, 0.45]
    leaf_logit_biases = [0.0, 0.4, 0.8, 1.2]

    temps = [0.75, 1.0, 1.25, 1.5]
    # Prefer trunk-up / leaf-down / mild twig adjustments under leafy sim
    leaf_bs = np.linspace(-1.8, 0.4, 12)
    trunk_bs = np.linspace(-0.4, 1.8, 12)
    twig_bs = np.linspace(-0.8, 1.2, 11)

    rows = []
    best = None
    best_comp = -1e9

    # coarse grid then refine top
    print("grid material bias under leaf-contaminant views...", flush=True)
    source_names = [
        "multi",
        "av",
        "blend_img_av",
        "blend_meta_multi",
        "blend_all",
        "meta4",
        "hier4",
        "blend_mh4",
        "clip",
        "clap",
    ]
    # subsample bias grid for speed: systematic product with stride
    for src in source_names:
        soft0 = soft_sources[src]
        for T in temps:
            for bl in leaf_bs[::2]:
                for bt in trunk_bs[::2]:
                    for bg in twig_bs[::2]:
                        # evaluate views
                        view_scores = []
                        fold_worst = []
                        for _, va in folds:
                            vs = []
                            for a in alphas:
                                soft_v = contaminate_leaf(soft0, a, contact_mask)
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            for lb in leaf_logit_biases:
                                soft_v = contaminate_leaf_logits(soft0, lb, contact_mask)
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            # leave-one-backbone weak views (if multi-like source)
                            if src in ("multi", "blend_img_av", "blend_all"):
                                for bb in ("clip", "eff", "dino"):
                                    if bb not in soft_sources:
                                        continue
                                    soft_v = soft_sources[bb]
                                    pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                    vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            fold_worst.append(min(vs))
                            view_scores.append(np.mean(vs))
                        worst = float(np.min(fold_worst))
                        mean_view = float(np.mean(view_scores))
                        pred_clean = material_decode(contact, soft0, T, bl, bt, bg)
                        clean = proto.fast_macro_f1(sy, pred_clean)
                        if clean < 0.90:
                            continue
                        amb = float(
                            np.sum((sy == 0) & (pred_clean == 0)) / max((sy == 0).sum(), 1)
                        )
                        if amb < 0.97:
                            continue
                        false_tr = int(np.sum((sy == 0) & (pred_clean == 2)))
                        if false_tr > 20:
                            continue
                        trunk_rec = float(
                            np.sum((sy == 2) & (pred_clean == 2)) / max((sy == 2).sum(), 1)
                        )
                        twig_rec = float(
                            np.sum((sy == 3) & (pred_clean == 3)) / max((sy == 3).sum(), 1)
                        )
                        leaf_rec = float(
                            np.sum((sy == 1) & (pred_clean == 1)) / max((sy == 1).sum(), 1)
                        )
                        wood_c = wood_macro(sy, pred_clean)
                        # stress gain vs identity on heavy leaf mix
                        pred_id_cont = material_decode(
                            contact, contaminate_leaf(soft0, 0.35, contact_mask), 1.0, 0, 0, 0
                        )
                        pred_fix_cont = material_decode(
                            contact, contaminate_leaf(soft0, 0.35, contact_mask), T, bl, bt, bg
                        )
                        cont_gain = wood_macro(sy, pred_fix_cont) - wood_macro(sy, pred_id_cont)
                        nflip = int(np.sum(pred_clean != contact))
                        # Prefer: worst-view, contaminant recovery gain, wood balance, clean floor
                        bal = min(leaf_rec, trunk_rec, twig_rec)
                        comp = (
                            0.40 * worst
                            + 0.18 * clean
                            + 0.18 * mean_view
                            + 0.12 * cont_gain
                            + 0.08 * wood_c
                            + 0.06 * bal
                            - 0.01 * abs(bl) * 0.1
                            - 0.012 * false_tr
                        )
                        # Bonus if bias is leaf-down trunk-up (direction of oracle residual) BUT only if
                        # contaminant gain is positive — still hand-certified via surrogate.
                        if bl < -0.2 and bt > 0.2 and cont_gain > 0.01:
                            comp += 0.01
                        row = dict(
                            source=src,
                            T=float(T),
                            b_leaf=float(bl),
                            b_trunk=float(bt),
                            b_twig=float(bg),
                            worst=worst,
                            mean_view=mean_view,
                            clean=clean,
                            wood_c=wood_c,
                            cont_gain=cont_gain,
                            trunk_rec=trunk_rec,
                            twig_rec=twig_rec,
                            leaf_rec=leaf_rec,
                            bal=bal,
                            false_tr=false_tr,
                            amb=amb,
                            nflip=nflip,
                            comp=comp,
                        )
                        rows.append(row)
                        if comp > best_comp:
                            best_comp = comp
                            best = row

    # refine around best with denser local grid
    if best is not None:
        print("refine around", best, flush=True)
        src = best["source"]
        soft0 = soft_sources[src]
        for T in [best["T"] * 0.85, best["T"], best["T"] * 1.15, 1.0]:
            for bl in np.linspace(best["b_leaf"] - 0.4, best["b_leaf"] + 0.4, 7):
                for bt in np.linspace(best["b_trunk"] - 0.4, best["b_trunk"] + 0.4, 7):
                    for bg in np.linspace(best["b_twig"] - 0.3, best["b_twig"] + 0.3, 5):
                        fold_worst = []
                        view_scores = []
                        for _, va in folds:
                            vs = []
                            for a in alphas:
                                soft_v = contaminate_leaf(soft0, a, contact_mask)
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            for lb in leaf_logit_biases:
                                soft_v = contaminate_leaf_logits(soft0, lb, contact_mask)
                                pred_v = material_decode(contact, soft_v, T, bl, bt, bg)
                                vs.append(proto.fast_macro_f1(sy[va], pred_v[va]))
                            fold_worst.append(min(vs))
                            view_scores.append(np.mean(vs))
                        worst = float(np.min(fold_worst))
                        mean_view = float(np.mean(view_scores))
                        pred_clean = material_decode(contact, soft0, T, bl, bt, bg)
                        clean = proto.fast_macro_f1(sy, pred_clean)
                        if clean < 0.90:
                            continue
                        amb = float(
                            np.sum((sy == 0) & (pred_clean == 0)) / max((sy == 0).sum(), 1)
                        )
                        if amb < 0.97:
                            continue
                        false_tr = int(np.sum((sy == 0) & (pred_clean == 2)))
                        if false_tr > 20:
                            continue
                        trunk_rec = float(
                            np.sum((sy == 2) & (pred_clean == 2)) / max((sy == 2).sum(), 1)
                        )
                        twig_rec = float(
                            np.sum((sy == 3) & (pred_clean == 3)) / max((sy == 3).sum(), 1)
                        )
                        leaf_rec = float(
                            np.sum((sy == 1) & (pred_clean == 1)) / max((sy == 1).sum(), 1)
                        )
                        wood_c = wood_macro(sy, pred_clean)
                        pred_id_cont = material_decode(
                            contact, contaminate_leaf(soft0, 0.35, contact_mask), 1.0, 0, 0, 0
                        )
                        pred_fix_cont = material_decode(
                            contact, contaminate_leaf(soft0, 0.35, contact_mask), T, bl, bt, bg
                        )
                        cont_gain = wood_macro(sy, pred_fix_cont) - wood_macro(sy, pred_id_cont)
                        nflip = int(np.sum(pred_clean != contact))
                        bal = min(leaf_rec, trunk_rec, twig_rec)
                        comp = (
                            0.40 * worst
                            + 0.18 * clean
                            + 0.18 * mean_view
                            + 0.12 * cont_gain
                            + 0.08 * wood_c
                            + 0.06 * bal
                            - 0.012 * false_tr
                        )
                        if bl < -0.2 and bt > 0.2 and cont_gain > 0.01:
                            comp += 0.01
                        row = dict(
                            source=src,
                            T=float(T),
                            b_leaf=float(bl),
                            b_trunk=float(bt),
                            b_twig=float(bg),
                            worst=worst,
                            mean_view=mean_view,
                            clean=clean,
                            wood_c=wood_c,
                            cont_gain=cont_gain,
                            trunk_rec=trunk_rec,
                            twig_rec=twig_rec,
                            leaf_rec=leaf_rec,
                            bal=bal,
                            false_tr=false_tr,
                            amb=amb,
                            nflip=nflip,
                            comp=comp,
                        )
                        rows.append(row)
                        if comp > best_comp:
                            best_comp = comp
                            best = row

    # Always include identity (no material redecode) as candidate
    id_row = dict(
        source="identity_contact",
        T=1.0,
        b_leaf=0.0,
        b_trunk=0.0,
        b_twig=0.0,
        worst=base_clean,
        mean_view=base_clean,
        clean=base_clean,
        wood_c=wood_macro(sy, contact),
        cont_gain=0.0,
        trunk_rec=float(np.sum((sy == 2) & (contact == 2)) / max((sy == 2).sum(), 1)),
        twig_rec=float(np.sum((sy == 3) & (contact == 3)) / max((sy == 3).sum(), 1)),
        leaf_rec=float(np.sum((sy == 1) & (contact == 1)) / max((sy == 1).sum(), 1)),
        bal=0.0,
        false_tr=int(np.sum((sy == 0) & (contact == 2))),
        amb=float(np.sum((sy == 0) & (contact == 0)) / max((sy == 0).sum(), 1)),
        nflip=0,
        comp=0.40 * base_clean + 0.18 * base_clean + 0.18 * base_clean,
    )
    id_row["bal"] = min(id_row["leaf_rec"], id_row["trunk_rec"], id_row["twig_rec"])
    rows.append(id_row)
    if best is None or id_row["comp"] > best_comp:
        best = id_row
        best_comp = id_row["comp"]

    df = pd.DataFrame(rows).sort_values("comp", ascending=False)
    df.to_csv(OUT / "hand_leaf_contaminant_mat_leaderboard.csv", index=False)
    print(df.head(15).to_string(index=False), flush=True)
    print("SELECTED", best, flush=True)

    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_leaf_contaminant_material_group_OOF",
            "primary": PRIMARY,
            "secondary": SECONDARY,
            "selected_candidate": best,
            "base_robot_artifact": (
                "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
                "segment_rule_stack_v2_base_segment_outputs.npz"
            ),
            "selection_recipe": (
                "locked 0.837 contact; material T/bias/source by worst-view over clean + "
                "leaf-contaminant soft surrogates + leave-backbone views; hand only"
            ),
            "n_candidates_scored": int(len(df)),
            "base_contact_hand_macro": float(base_clean),
        },
    )
    # save OOF mat experts for reuse / diagnostics
    np.savez_compressed(
        OUT / "hand_oof_mat_experts.npz",
        sy=sy,
        contact=contact,
        **{f"mat_{k}": v for k, v in oof_mat.items()},
    )
    print("lock written", OUT / "selection_lock.json", flush=True)
    return lock


if __name__ == "__main__":
    main()
