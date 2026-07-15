"""Dual-collapse surrogate selection: attenuate BOTH audio contact and image binary.

Hand OOF image binary is near-perfect, so audio-only collapse does not create
trunk→ambient. This script also soft-collapses image contact and boosts image
trunk head, selecting hier params by min(clean, dual_collapse) macro F1.

Uses OOF artifacts from multimodal_surrogate_contact_stress_group_selection.
Never loads robot/test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_multimodal_val_locked_suite as suite

OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_surrogate_dual_collapse_group_selection",
    )
)
SRC = Path(
    "outputs/audio_feature_benchmarks/multimodal_surrogate_contact_stress_group_selection"
)


def fast_macro(y, p):
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def decode_hier(contact, mat, th):
    pred = np.zeros(len(contact), np.int64)
    hit = contact >= th
    pred[hit] = 1 + mat[hit].argmax(1)
    return pred


def make_mat(sa_mat, imat, w, tb, trunk_score):
    mat = suite.normalize(sa_mat * (1.0 - w) + imat * w)
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1.0))
        logits[:, 1] += tb * trunk_score
        mat = suite.normalize(np.exp(logits))
    return mat


def apply(rule, pred_h, meta_arg, contact, th):
    if rule == "hier_only":
        return pred_h
    if rule == "hier_trunk_meta_else":
        return np.where(pred_h == 2, pred_h, meta_arg)
    if rule == "image_trunk_force":
        # if trunk head high and contact moderate, force trunk
        return pred_h
    return np.where(contact >= th, pred_h, meta_arg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sy = np.load(SRC / "hand_oof_y.npy")
    seg_pc = np.load(SRC / "hand_seg_pc.npy")
    sa_mat = np.load(SRC / "hand_sa_mat.npy")
    oof_bin = np.load(SRC / "hand_oof_bin.npy")
    oof_trunk = np.load(SRC / "hand_oof_trunk.npy")
    oof_mat = np.load(SRC / "hand_oof_mat.npy")
    oof_meta = np.load(SRC / "hand_oof_meta.npy")
    meta_arg = oof_meta.argmax(1)
    n = len(sy)
    ss = np.load(
        "outputs/audio_feature_benchmarks/multimodal_phase1_softstack_group_selection/hand_specimen.npy",
        allow_pickle=True,
    )
    # specimen for folds
    from sklearn.model_selection import StratifiedGroupKFold

    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(n), sy, ss
        )
    )

    # dual collapse schedules (audio_fac, img_bin_fac, trunk_boost)
    schedules = [
        ("clean", 1.0, 1.0, 0.0),
        ("audio_hard", 0.15, 1.0, 0.0),
        ("img_hard", 1.0, 0.25, 0.0),
        ("dual_hard", 0.15, 0.25, 0.0),
        ("dual_hard_trunkboost", 0.15, 0.25, 0.5),
        ("dual_extreme", 0.05, 0.15, 0.0),
        ("dual_extreme_trunkboost", 0.05, 0.15, 0.8),
    ]

    alphas = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    tbs = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
    ths = [0.25, 0.35, 0.45, 0.55, 0.65]
    mat_ws = [
        np.array([0.4, 0.55, 0.4]),
        np.array([0.5, 0.8, 0.5]),
        np.array([0.4, 1.0, 0.4]),
        np.array([0.3, 1.0, 0.3]),
    ]
    rules = ["hier_only", "hier_trunk_meta_else", "hier_if_contact_else_meta"]
    # extra: force trunk if oof_trunk high even when hier ambient
    force_trunk_ths = [None, 0.55, 0.65, 0.75]

    rows = []
    for alpha in alphas:
        for tb in tbs:
            for th in ths:
                for mw in mat_ws:
                    for rule in rules:
                        for fth in force_trunk_ths:
                            # score each schedule globally + fold min of dual_hard
                            sched_scores = {}
                            pred_clean = None
                            for sname, af, bf, tboost in schedules:
                                pc = seg_pc * af
                                bin_s = np.clip(oof_bin * bf, 0, 1)
                                trunk_s = np.clip(oof_trunk + tboost * oof_trunk, 0, 1)
                                contact = alpha * pc + (1 - alpha) * bin_s
                                mat = make_mat(sa_mat, oof_mat, mw, tb, trunk_s)
                                pred_h = decode_hier(contact, mat, th)
                                if rule == "hier_only":
                                    pred = pred_h.copy()
                                elif rule == "hier_trunk_meta_else":
                                    pred = np.where(pred_h == 2, pred_h, meta_arg)
                                else:
                                    pred = np.where(contact >= th, pred_h, meta_arg)
                                if fth is not None:
                                    # force trunk when trunk head confident
                                    pred = pred.copy()
                                    pred[oof_trunk >= fth] = 2
                                sched_scores[sname] = fast_macro(sy, pred)
                                if sname == "clean":
                                    pred_clean = pred
                                    trunk_clean = float(
                                        np.sum((sy == 2) & (pred == 2))
                                        / max(np.sum(sy == 2), 1)
                                    )
                                if sname == "dual_hard":
                                    trunk_dual = float(
                                        np.sum((sy == 2) & (pred == 2))
                                        / max(np.sum(sy == 2), 1)
                                    )
                                    # count remaining trunk→ambient under dual
                                    t2a = int(np.sum((sy == 2) & (pred == 0)))

                            # fold min on dual_hard and dual_extreme
                            fold_mins = []
                            for _, va in folds:
                                vs = []
                                for sname, af, bf, tboost in schedules:
                                    if sname == "clean":
                                        continue
                                    pc = seg_pc * af
                                    bin_s = np.clip(oof_bin * bf, 0, 1)
                                    trunk_s = np.clip(oof_trunk + tboost * oof_trunk, 0, 1)
                                    contact = alpha * pc + (1 - alpha) * bin_s
                                    mat = make_mat(sa_mat, oof_mat, mw, tb, trunk_s)
                                    pred_h = decode_hier(contact, mat, th)
                                    if rule == "hier_only":
                                        pred = pred_h
                                    elif rule == "hier_trunk_meta_else":
                                        pred = np.where(pred_h == 2, pred_h, meta_arg)
                                    else:
                                        pred = np.where(contact >= th, pred_h, meta_arg)
                                    if fth is not None:
                                        pred = pred.copy()
                                        pred[oof_trunk >= fth] = 2
                                    vs.append(fast_macro(sy[va], pred[va]))
                                fold_mins.append(min(vs))
                            clean_fold = []
                            for _, va in folds:
                                clean_fold.append(fast_macro(sy[va], pred_clean[va]))
                            fold_min = [min(c, s) for c, s in zip(clean_fold, fold_mins)]
                            rows.append(
                                {
                                    "alpha": alpha,
                                    "tb": tb,
                                    "th": th,
                                    "mat_w": mw.tolist(),
                                    "rule": rule,
                                    "force_trunk_th": -1 if fth is None else fth,
                                    "score_mean_min": float(np.mean(fold_min)),
                                    "score_worst_min": float(np.min(fold_min)),
                                    "clean_macro": sched_scores["clean"],
                                    "dual_hard_macro": sched_scores["dual_hard"],
                                    "dual_extreme_macro": sched_scores["dual_extreme"],
                                    "dual_hard_tb_macro": sched_scores["dual_hard_trunkboost"],
                                    "trunk_recall_clean": trunk_clean,
                                    "trunk_recall_dual_hard": trunk_dual,
                                    "trunk_to_ambient_dual_hard": t2a,
                                    "mean_clean_fold": float(np.mean(clean_fold)),
                                }
                            )

    board = pd.DataFrame(rows).sort_values(
        [
            "score_mean_min",
            "trunk_recall_dual_hard",
            "dual_hard_macro",
            "clean_macro",
        ],
        ascending=False,
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_dual_collapse_leaderboard.csv", index=False)

    # primary: good clean + strong dual-hard trunk rescue
    viable = board[
        (board["clean_macro"] >= 0.90)
        & (board["trunk_recall_dual_hard"] >= 0.75)
        & (board["mean_clean_fold"] >= 0.88)
    ]
    if len(viable) == 0:
        viable = board[board["clean_macro"] >= 0.88]
    selected = viable.iloc[0].to_dict()

    # secondary: maximize dual-hard trunk recall with clean>=0.90
    rescue = board[board["clean_macro"] >= 0.90].sort_values(
        ["trunk_recall_dual_hard", "dual_hard_macro", "clean_macro"], ascending=False
    )
    rescue_sel = rescue.iloc[0].to_dict() if len(rescue) else selected

    lock = {
        "protocol": "multimodal_surrogate_dual_contact_collapse_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "test_loaded": False,
        "source_oof": str(SRC),
        "selected_candidate": selected,
        "selected_max_trunk_under_dual": rescue_sel,
        "schedules": [s[0] for s in schedules],
        "invariants": {
            "robot_not_used": True,
            "filename_class_features": False,
            "surrogate_dual_collapse": True,
        },
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    cols = [
        "alpha",
        "tb",
        "th",
        "rule",
        "force_trunk_th",
        "score_mean_min",
        "clean_macro",
        "dual_hard_macro",
        "trunk_recall_clean",
        "trunk_recall_dual_hard",
        "trunk_to_ambient_dual_hard",
    ]
    print("\nTOP score_mean_min:")
    print(board.head(15)[cols].to_string(index=False))
    print("\nTOP trunk dual-hard (clean>=0.90):")
    print(rescue.head(10)[cols].to_string(index=False) if len(rescue) else "none")


if __name__ == "__main__":
    main()
