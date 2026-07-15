"""Surrogate contact-collapse stress selection (hand only).

Problem: robot trunk→ambient is driven by audio contact mass collapsing; this
error almost never appears on clean hand OOF, so normal grids cannot select a
rescue.

Approach (still hand-only, no robot):
  1. Rebuild nested image heads + audio contact/material like hier v2.
  2. Evaluate candidates under clean AND surrogate views that *force*
     audio contact attenuation (pc *= factor) and material entropy noise.
  3. Select alpha/tb/th/mat weights/rule by mean over folds of
     min(clean_macro, worst_surrogate_macro), with trunk-recall tie-break.
  4. Diversity vs locked v2 params is logged.

Never loads robot/test.
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

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_surrogate_contact_stress_group_selection",
    )
)
META_FEAT = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_features.npy")
META_Y = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_y.npy")
IMG_CLIP = Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k")
IMG_EFF = Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k")


def fast_macro(y, p):
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def fit_lr4(x, y, c=0.03):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
        ),
    )
    m.fit(x, y)
    return m


def fit_bin(x, y, c=0.03):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=1500, class_weight="balanced", random_state=42),
    )
    m.fit(x, y)
    return m


def fit_mat(x, y3, c=0.03):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
        ),
    )
    m.fit(x, y3)
    return m


def decode_hier(contact, mat, th):
    pred = np.zeros(len(contact), np.int64)
    for i in range(len(contact)):
        if contact[i] >= th:
            pred[i] = 1 + int(np.argmax(mat[i]))
        else:
            pred[i] = 0
    return pred


def make_mat(sa_mat, imat, w, tb, trunk_score):
    mat = suite.normalize(sa_mat * (1.0 - w) + imat * w)
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1.0))
        logits[:, 1] += tb * trunk_score
        mat = suite.normalize(np.exp(logits))
    return mat


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set("total240")
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
    nseg = len(useg)
    sy = np.array([y[codes == i][0] for i in range(nseg)], np.int64)
    ss = np.array([sg[codes == i][0] for i in range(nseg)])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(nseg), sy, ss)
    )

    # audio OOF locked
    af = audio_base.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train")
    high = group_audio.load_highsr_oof(Path("outputs"))
    pair = specimen.load_pairwise_oof(
        Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        )
    )
    anchor = lift.anchor_lift_proba(af, high, pair)
    assignment = np.full(len(y), -1, np.int64)
    for k, (_, va) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(y)), y, sg)
    ):
        assignment[va] = k
    src = broad.load_train_sources(Path("outputs"), y, assignment, 42)
    audio = lift.postprocess(
        af, lift.normalize(0.95 * anchor + 0.05 * src["report_gate_onehot"]), "segment_lift"
    )
    pc, lg = avr.avr_inputs(audio)
    seg_pc = np.array([pc[codes == i].mean() for i in range(nseg)], np.float64)
    sa_mat = np.vstack(
        [suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0] for i in range(nseg)]
    )

    x_clip = np.load(IMG_CLIP / "hand_train_full/X.npy").astype(np.float32)
    x_eff = np.load(IMG_EFF / "hand_train_full/X.npy").astype(np.float32)
    sx_c = np.vstack([x_clip[codes == i].mean(0) for i in range(nseg)])
    sx_e = np.vstack([x_eff[codes == i].mean(0) for i in range(nseg)])
    sx_cat = np.hstack([sx_c, sx_e])

    # nested OOF image heads + meta
    oof_bin = np.zeros(nseg, np.float64)
    oof_trunk = np.zeros(nseg, np.float64)
    oof_mat = np.zeros((nseg, 3), np.float64)
    oof_meta = np.zeros((nseg, 4), np.float64)

    # meta model fitted on stored nested features (prior hand OOF)
    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=4000,
            class_weight={0: 1.0, 1: 1.0, 2: 1.2, 3: 1.2},
            multi_class="multinomial",
            random_state=42,
        ),
    )
    meta.fit(np.load(META_FEAT), np.load(META_Y))

    print("building nested image OOF...", flush=True)
    for fold_id, (tr, va) in enumerate(folds):
        m_bin = fit_bin(sx_c[tr], (sy[tr] > 0).astype(int))
        m_trunk = fit_bin(sx_c[tr], (sy[tr] == 2).astype(int), 0.1)
        ct = tr[sy[tr] > 0]
        m_mat = fit_mat(sx_cat[ct], sy[ct] - 1)
        m_img4 = fit_lr4(sx_c[tr], sy[tr])
        oof_bin[va] = m_bin.predict_proba(sx_c[va])[:, 1]
        oof_trunk[va] = m_trunk.predict_proba(sx_c[va])[:, 1]
        oof_mat[va] = suite.normalize(m_mat.predict_proba(sx_cat[va]))
        img_mat3 = suite.normalize(m_img4.predict_proba(sx_c[va])[:, 1:])
        z = sel.feat_meta(sa_mat[va], img_mat3, seg_pc[va])
        oof_meta[va] = suite.normalize(meta.predict_proba(z))
        print(f" fold {fold_id+1}/5", flush=True)

    np.save(OUT / "hand_oof_meta.npy", oof_meta)
    np.save(OUT / "hand_oof_bin.npy", oof_bin)
    np.save(OUT / "hand_oof_trunk.npy", oof_trunk)
    np.save(OUT / "hand_oof_mat.npy", oof_mat)
    np.save(OUT / "hand_seg_pc.npy", seg_pc)
    np.save(OUT / "hand_sa_mat.npy", sa_mat)
    np.save(OUT / "hand_oof_y.npy", sy)

    # surrogate contact collapse factors + optional material noise
    collapse = [1.0, 0.55, 0.35, 0.20, 0.10]
    mat_noise = [0.0, 0.15]
    alphas = [0.15, 0.25, 0.35, 0.4, 0.5, 0.65]
    tbs = [0.0, 0.3, 0.5, 0.8, 1.2, 1.6]
    ths = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    # material image weights [leaf, trunk, twig]
    mat_ws = [
        np.array([0.4, 0.55, 0.4]),
        np.array([0.5, 0.7, 0.5]),
        np.array([0.35, 0.8, 0.35]),
        np.array([0.45, 0.9, 0.45]),
        np.array([0.3, 1.0, 0.3]),
    ]
    rules = ["hier_only", "hier_trunk_meta_else", "hier_if_contact_else_meta"]

    # v2 reference
    w0 = np.array([0.4, 0.55, 0.4])
    contact_v2 = 0.4 * seg_pc + 0.6 * oof_bin
    mat_v2 = make_mat(sa_mat, oof_mat, w0, 0.3, oof_trunk)
    pred_h_v2 = decode_hier(contact_v2, mat_v2, 0.55)
    pred_v2 = np.where(pred_h_v2 == 2, pred_h_v2, oof_meta.argmax(1))
    v2_clean = fast_macro(sy, pred_v2)

    rows = []
    rng = np.random.default_rng(0)
    print("grid search under surrogate contact collapse...", flush=True)
    for alpha in alphas:
        for tb in tbs:
            for th in ths:
                for mw in mat_ws:
                    for rule in rules:
                        fold_clean = []
                        fold_surr = []
                        oof_pred_clean = np.zeros(nseg, np.int64)
                        for fold_id, (tr, va) in enumerate(folds):
                            # clean
                            contact = alpha * seg_pc + (1 - alpha) * oof_bin
                            mat = make_mat(sa_mat, oof_mat, mw, tb, oof_trunk)
                            pred_h = decode_hier(contact, mat, th)
                            if rule == "hier_only":
                                pred = pred_h
                            elif rule == "hier_trunk_meta_else":
                                pred = np.where(pred_h == 2, pred_h, oof_meta.argmax(1))
                            else:  # hier_if_contact_else_meta
                                pred = np.where(contact >= th, pred_h, oof_meta.argmax(1))
                            oof_pred_clean[va] = pred[va]
                            fold_clean.append(fast_macro(sy[va], pred[va]))

                            # surrogate views: attenuate audio contact + noise on sa_mat
                            s_scores = []
                            for fac in collapse[1:]:  # skip clean
                                for noise in mat_noise:
                                    pc_s = seg_pc * fac
                                    contact_s = alpha * pc_s + (1 - alpha) * oof_bin
                                    sa_s = sa_mat.copy()
                                    if noise > 0:
                                        sa_s = suite.normalize(
                                            np.exp(
                                                np.log(np.clip(sa_s, 1e-8, 1))
                                                + noise * rng.standard_normal(sa_s.shape)
                                            )
                                        )
                                    mat_s = make_mat(sa_s, oof_mat, mw, tb, oof_trunk)
                                    pred_hs = decode_hier(contact_s, mat_s, th)
                                    if rule == "hier_only":
                                        pred_s = pred_hs
                                    elif rule == "hier_trunk_meta_else":
                                        pred_s = np.where(
                                            pred_hs == 2, pred_hs, oof_meta.argmax(1)
                                        )
                                    else:
                                        pred_s = np.where(
                                            contact_s >= th, pred_hs, oof_meta.argmax(1)
                                        )
                                    s_scores.append(fast_macro(sy[va], pred_s[va]))
                            fold_surr.append(min(s_scores))
                        fold_min = [min(c, s) for c, s in zip(fold_clean, fold_surr)]
                        # also measure trunk recall under strongest collapse on full set
                        fac = 0.10
                        contact_s = alpha * (seg_pc * fac) + (1 - alpha) * oof_bin
                        mat_s = make_mat(sa_mat, oof_mat, mw, tb, oof_trunk)
                        pred_hs = decode_hier(contact_s, mat_s, th)
                        if rule == "hier_only":
                            pred_s = pred_hs
                        elif rule == "hier_trunk_meta_else":
                            pred_s = np.where(pred_hs == 2, pred_hs, oof_meta.argmax(1))
                        else:
                            pred_s = np.where(contact_s >= th, pred_hs, oof_meta.argmax(1))
                        trunk_surr = float(
                            np.sum((sy == 2) & (pred_s == 2)) / max(np.sum(sy == 2), 1)
                        )
                        rows.append(
                            {
                                "alpha": alpha,
                                "tb": tb,
                                "th": th,
                                "mat_w": mw.tolist(),
                                "rule": rule,
                                "score_mean_min": float(np.mean(fold_min)),
                                "score_worst_min": float(np.min(fold_min)),
                                "mean_clean": float(np.mean(fold_clean)),
                                "worst_clean": float(np.min(fold_clean)),
                                "mean_surr": float(np.mean(fold_surr)),
                                "worst_surr": float(np.min(fold_surr)),
                                "global_clean": fast_macro(sy, oof_pred_clean),
                                "trunk_recall_clean": float(
                                    np.sum((sy == 2) & (oof_pred_clean == 2))
                                    / max(np.sum(sy == 2), 1)
                                ),
                                "trunk_recall_collapse0.1": trunk_surr,
                                "disagreement_vs_v2": float(
                                    np.mean(oof_pred_clean != pred_v2)
                                ),
                            }
                        )

    board = pd.DataFrame(rows).sort_values(
        [
            "score_mean_min",
            "trunk_recall_collapse0.1",
            "score_worst_min",
            "mean_clean",
        ],
        ascending=False,
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_surrogate_contact_leaderboard.csv", index=False)

    # require reasonable clean performance (not collapse clean quality)
    viable = board[(board["mean_clean"] >= 0.90) & (board["trunk_recall_collapse0.1"] >= 0.70)]
    if len(viable) == 0:
        viable = board[board["mean_clean"] >= 0.88]
    selected = viable.iloc[0].to_dict() if len(viable) else board.iloc[0].to_dict()

    # also report best by pure collapse trunk recall with clean>=0.92
    rescue = board[(board["mean_clean"] >= 0.92)].sort_values(
        ["trunk_recall_collapse0.1", "score_mean_min"], ascending=False
    )
    rescue_sel = rescue.iloc[0].to_dict() if len(rescue) else selected

    lock = {
        "protocol": "multimodal_surrogate_audio_contact_collapse_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "test_loaded": False,
        "v2_reference_clean_macro_f1": v2_clean,
        "selected_candidate": selected,
        "selected_collapse_rescue_candidate": rescue_sel,
        "surrogate_views": {
            "audio_contact_multipliers": collapse,
            "material_log_noise": mat_noise,
            "rationale": "simulate robot trunk→ambient via audio contact collapse; select image-heavy contact gate + trunk bias",
        },
        "invariants": {
            "robot_not_used": True,
            "filename_class_features": False,
            "nested_image_oof": True,
        },
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print("\nTOP 12 by score_mean_min:")
    cols = [
        "alpha",
        "tb",
        "th",
        "rule",
        "score_mean_min",
        "mean_clean",
        "mean_surr",
        "trunk_recall_clean",
        "trunk_recall_collapse0.1",
        "disagreement_vs_v2",
    ]
    print(board.head(12)[cols].to_string(index=False))
    print("\nTOP collapse rescue (clean>=0.92):")
    print(rescue.head(8)[cols].to_string(index=False) if len(rescue) else "none")


if __name__ == "__main__":
    main()
