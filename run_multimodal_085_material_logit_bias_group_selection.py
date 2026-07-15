"""Hand-only selection: locked v2 contact + stress amb-lift + material logit bias.

Paper-pure: specimen_group OOF, never loads robot/test. Selects temperature and
per-class material logit biases (leaf/trunk/twig) to maximize worst-fold macro F1
while preserving clean hand F1 and ambient purity.

Rationale: residual robot errors after amb-lift are mostly material (trunk↔twig,
twig↔leaf). Cost-sensitive material decode (catalog D2) can reweight soft masses
without changing contact decisions.
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
        "outputs/audio_feature_benchmarks/multimodal_085_material_logit_bias_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
IMG_MULTI = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
}
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
W2V = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")

# Locked amb-lift family from prior hand-selected best-effort (not robot-tuned).
LIFT_CANDIDATES = [
    dict(tscore_mode="stress", ts_th=0.5, cs_th=0.1, tt_th=0.45, do_amb_lift=True, do_twig_to_trunk=True),
    dict(tscore_mode="stress", ts_th=0.4, cs_th=0.55, tt_th=0.5, do_amb_lift=True, do_twig_to_trunk=False),
    dict(tscore_mode="max", ts_th=0.55, cs_th=0.25, tt_th=0.5, do_amb_lift=True, do_twig_to_trunk=True),
    dict(tscore_mode="img", ts_th=0.65, cs_th=0.35, tt_th=0.55, do_amb_lift=True, do_twig_to_trunk=False),
    dict(tscore_mode="stress", ts_th=0.5, cs_th=0.1, tt_th=0.45, do_amb_lift=True, do_twig_to_trunk=False),
]


def pool(x, codes, n):
    return np.vstack([x[codes == i].mean(0) for i in range(n)]).astype(np.float32)


def fit_bin(x, y, c=0.05):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42),
    )
    m.fit(x, y)
    return m


def pos(m, x):
    p = m.predict_proba(x)
    cl = list(m.classes_)
    return p[:, cl.index(1)] if 1 in cl else p[:, -1]


def apply_lift(pred, tscore, cs, bins, tt, cfg):
    out = pred.copy()
    if cfg["do_amb_lift"]:
        m = (out == 0) & (tscore >= cfg["ts_th"]) & (np.maximum(cs, bins) >= cfg["cs_th"])
        out[m] = 2
    if cfg["do_twig_to_trunk"]:
        m = (out == 3) & (tt >= cfg["tt_th"]) & (tscore >= cfg["ts_th"] * 0.9)
        out[m] = 2
    return out


def material_decode(p_soft, contact_pred, T, b_leaf, b_trunk, b_twig, source="meta"):
    """Keep ambient/contact from contact_pred; re-decode material with logit bias."""
    out = contact_pred.copy()
    contact = out > 0
    if not contact.any():
        return out
    if source == "meta":
        base = p_soft
    elif source == "hier":
        base = p_soft
    else:
        base = p_soft
    logits = np.log(np.clip(base[contact][:, 1:4], 1e-12, None)) / float(T)
    logits = logits + np.array([b_leaf, b_trunk, b_twig], dtype=np.float64)
    out[contact] = 1 + logits.argmax(1)
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

    oof_meta = np.load(STACK / "hand_oof_meta.npy")
    oof_hier = np.load(STACK / "hand_oof_hier.npy")
    assert len(oof_meta) == n
    pred_v2 = np.where(oof_hier.argmax(1) == 2, oof_hier.argmax(1), oof_meta.argmax(1))
    print("base v2-like", proto.fast_macro_f1(sy, pred_v2), flush=True)

    img = {
        k: pool(np.load(p / "hand_train_full/X.npy").astype(np.float32), codes, n)
        for k, p in IMG_MULTI.items()
    }
    multi = np.hstack([img[k] for k in ("clip", "eff", "dino", "convnext")])
    aud = {k: pool(np.load(p).astype(np.float32), codes, n) for k, p in AUDIO.items()}
    w2v = pool(np.load(W2V).astype(np.float32), codes, n)
    sx_c = img["clip"]

    oof_bin = np.zeros(n)
    oof_ts = np.zeros(n)
    oof_cs = np.zeros(n)
    oof_tt = np.zeros(n)
    oof_tr_img = np.zeros(n)

    print("OOF detectors...", flush=True)
    for fold_id, (tr, va) in enumerate(folds):
        oof_bin[va] = pos(fit_bin(sx_c[tr], (sy[tr] > 0).astype(int), 0.03), sx_c[va])
        oof_tr_img[va] = pos(fit_bin(sx_c[tr], (sy[tr] == 2).astype(int), 0.1), sx_c[va])
        Xst = np.vstack(
            [
                np.hstack([aud["robot_mix"][tr], w2v[tr], multi[tr]]),
                np.hstack([aud["bandlimit"][tr], w2v[tr], multi[tr]]),
            ]
        )
        m_ts = fit_bin(Xst, np.concatenate([(sy[tr] == 2).astype(int)] * 2), 0.05)
        m_cs = fit_bin(Xst, np.concatenate([(sy[tr] > 0).astype(int)] * 2), 0.05)
        Xva = np.hstack([aud["clean"][va], w2v[va], multi[va]])
        Xvm = np.hstack([aud["robot_mix"][va], w2v[va], multi[va]])
        oof_ts[va] = 0.5 * (pos(m_ts, Xva) + pos(m_ts, Xvm))
        oof_cs[va] = 0.5 * (pos(m_cs, Xva) + pos(m_cs, Xvm))
        wood = tr[(sy[tr] == 2) | (sy[tr] == 3)]
        if len(wood) > 8:
            Xw = np.vstack(
                [
                    np.hstack([aud[v][wood], w2v[wood], multi[wood]])
                    for v in ("clean", "robot_mix", "bandlimit")
                ]
            )
            yw = np.concatenate([(sy[wood] == 2).astype(int)] * 3)
            m_tt = fit_bin(Xw, yw, 0.1)
            oof_tt[va] = pos(m_tt, Xva)
        print(f" fold {fold_id + 1}/5", flush=True)

    def tscore_for(mode):
        if mode == "stress":
            return oof_ts
        if mode == "max":
            return np.maximum(oof_ts, oof_tr_img)
        return oof_tr_img

    # Soft sources for material decode
    soft_sources = {
        "meta": oof_meta,
        "hier": oof_hier,
        "blend_mh": proto.normalize(0.5 * oof_meta + 0.5 * oof_hier),
        "blend_h_heavy": proto.normalize(0.7 * oof_hier + 0.3 * oof_meta),
        "blend_m_heavy": proto.normalize(0.7 * oof_meta + 0.3 * oof_hier),
    }

    temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    # Biases: leaf negative / trunk positive often help macro under imbalance; search broad
    leaf_biases = np.linspace(-1.5, 0.5, 9)
    trunk_biases = np.linspace(-0.5, 2.0, 11)
    twig_biases = np.linspace(-1.0, 1.0, 9)

    rows = []
    print("Grid search lift × material bias...", flush=True)
    for lift in LIFT_CANDIDATES:
        tscore = tscore_for(lift["tscore_mode"])
        contact_pred = apply_lift(pred_v2, tscore, oof_cs, oof_bin, oof_tt, lift)
        base_clean = proto.fast_macro_f1(sy, contact_pred)
        if base_clean < 0.90:
            continue
        for src_name, psoft in soft_sources.items():
            for T in temps:
                for b_leaf in leaf_biases:
                    for b_trunk in trunk_biases:
                        for b_twig in twig_biases:
                            # skip near-identity early when all zero-ish and T=1
                            fold_scores = []
                            oof_pred = np.zeros(n, np.int64)
                            for _, va in folds:
                                pred = material_decode(
                                    psoft, contact_pred, T, b_leaf, b_trunk, b_twig, src_name
                                )
                                fold_scores.append(proto.fast_macro_f1(sy[va], pred[va]))
                                oof_pred[va] = pred[va]
                            clean = proto.fast_macro_f1(sy, oof_pred)
                            if clean < 0.92:
                                continue
                            false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
                            if false_tr > 40:
                                continue
                            amb_rec = float(
                                np.sum((sy == 0) & (oof_pred == 0)) / max((sy == 0).sum(), 1)
                            )
                            if amb_rec < 0.96:
                                continue
                            trunk_rec = float(
                                np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1)
                            )
                            twig_rec = float(
                                np.sum((sy == 3) & (oof_pred == 3)) / max((sy == 3).sum(), 1)
                            )
                            leaf_rec = float(
                                np.sum((sy == 1) & (oof_pred == 1)) / max((sy == 1).sum(), 1)
                            )
                            worst = float(np.min(fold_scores))
                            mean_f = float(np.mean(fold_scores))
                            # Prefer worst-fold, then mean, trunk, clean; penalize false trunk
                            composite = (
                                0.5 * worst
                                + 0.2 * mean_f
                                + 0.15 * clean
                                + 0.1 * trunk_rec
                                + 0.05 * twig_rec
                                - 0.01 * false_tr
                            )
                            rows.append(
                                dict(
                                    **{f"lift_{k}": v for k, v in lift.items()},
                                    soft_source=src_name,
                                    T=float(T),
                                    b_leaf=float(b_leaf),
                                    b_trunk=float(b_trunk),
                                    b_twig=float(b_twig),
                                    worst_fold_macro_f1=worst,
                                    mean_fold_macro_f1=mean_f,
                                    clean_macro_f1=clean,
                                    trunk_recall=trunk_rec,
                                    twig_recall=twig_rec,
                                    leaf_recall=leaf_rec,
                                    ambient_recall=amb_rec,
                                    false_trunk=false_tr,
                                    composite=composite,
                                )
                            )

    if not rows:
        raise RuntimeError("No candidates passed hand gates")

    board = pd.DataFrame(rows).sort_values(
        ["composite", "worst_fold_macro_f1", "trunk_recall", "clean_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_material_logit_bias_leaderboard.csv", index=False)
    selected = board.iloc[0].to_dict()

    # identity baseline row for reference
    id_pred = material_decode(oof_meta, pred_v2, 1.0, 0.0, 0.0, 0.0)
    print("identity meta on v2 contact", proto.fast_macro_f1(sy, id_pred), flush=True)
    print("selected", selected, flush=True)
    print(board.head(20).to_string(index=False), flush=True)

    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_material_logit_bias_hand_group_OOF",
            "base_robot_artifact": (
                "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
                "segment_rule_stack_v2_base_segment_outputs.npz"
            ),
            "selected_candidate": selected,
            "base_hand_macro_f1_v2": proto.fast_macro_f1(sy, pred_v2),
            "n_candidates": int(len(board)),
            "selection_recipe": "worst-fold × material logit bias on hand OOF soft probs; lift family discrete",
        },
    )
    np.savez_compressed(
        OUT / "hand_oof_bundle.npz",
        sy=sy,
        pred_v2=pred_v2,
        oof_meta=oof_meta,
        oof_hier=oof_hier,
        oof_ts=oof_ts,
        oof_cs=oof_cs,
        oof_tt=oof_tt,
        oof_bin=oof_bin,
        oof_tr_img=oof_tr_img,
    )
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
