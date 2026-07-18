"""Hierarchical contact-rescue fusion: image-assisted gate + classwise material.

Hand-only StratifiedGroupKFold selection. Never loads robot/test.
"""
from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_hier_contact_rescue_group_selection",
    )
)
IMG_CLIP = Path(
    "outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"
)
IMG_EFF = Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k")


def seg(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)


def spec(s: pd.Series) -> pd.Series:
    return seg(s).str.replace(r"_segment_.*$", "", regex=True)


def fast_macro(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def oof_probs(
    x: np.ndarray,
    y: np.ndarray,
    folds: list,
    *,
    binary: bool = False,
    contact_only_material: bool = False,
    C: float = 0.03,
) -> np.ndarray:
    if binary:
        oof = np.zeros(len(y), dtype=np.float32)
    elif contact_only_material:
        oof = np.zeros((len(y), 3), dtype=np.float32)
    else:
        oof = np.zeros((len(y), 4), dtype=np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=1500,
                class_weight="balanced",
                multi_class="multinomial" if not binary else "auto",
                random_state=42,
            ),
        )
        if binary:
            m.fit(x[tr], (y[tr] > 0).astype(int))
            oof[va] = m.predict_proba(x[va])[:, 1]
        elif contact_only_material:
            ci = tr[y[tr] > 0]
            if len(ci) < 3 or len(np.unique(y[ci])) < 2:
                oof[va] = 1.0 / 3.0
                continue
            m.fit(x[ci], y[ci] - 1)
            oof[va] = m.predict_proba(x[va])
        else:
            m.fit(x[tr], y[tr])
            oof[va] = suite.normalize(m.predict_proba(x[va]))
    return oof


def best_threshold(contact: np.ndarray, mat: np.ndarray, y: np.ndarray, ths: np.ndarray):
    best = (-1.0, -1.0, 0.5, None)
    mat_pred = mat.argmax(1) + 1
    for th in ths:
        pred = np.where(contact >= th, mat_pred, 0)
        f1 = fast_macro(y, pred)
        bf1 = float(f1_score(y > 0, pred > 0, average="macro", zero_division=0))
        if (f1, bf1) > (best[0], best[1]):
            best = (f1, bf1, float(th), pred)
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set("total240")
    fr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y = fr.y.to_numpy(np.int64)
    files = fr.audio_file.astype(str)
    sk = seg(files).to_numpy()
    sg = spec(files).to_numpy()
    useg, codes = np.unique(sk, return_inverse=True)
    sy = np.array([y[codes == i][0] for i in range(len(useg))])
    ss = np.array([sg[codes == i][0] for i in range(len(useg))])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(sy)), sy, ss
        )
    )

    af = audio_base.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train"
    )
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
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(y)), y, sg
        )
    ):
        assignment[va] = k
    src = broad.load_train_sources(Path("outputs"), y, assignment, 42)
    audio = lift.postprocess(
        af,
        lift.normalize(0.95 * anchor + 0.05 * src["report_gate_onehot"]),
        "segment_lift",
    )
    pc, lg = avr.avr_inputs(audio)
    seg_pc = np.array([pc[codes == i].mean() for i in range(len(useg))])
    seg_audio_mat = np.vstack(
        [
            suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0]
            for i in range(len(useg))
        ]
    )
    seg_audio4 = np.vstack(
        [
            suite.normalize(audio[codes == i].mean(0, keepdims=True))[0]
            for i in range(len(useg))
        ]
    )

    x_clip = np.load(IMG_CLIP / "hand_train_full/X.npy").astype(np.float32)
    x_eff = np.load(IMG_EFF / "hand_train_full/X.npy").astype(np.float32)
    sx_clip = np.vstack([x_clip[codes == i].mean(0) for i in range(len(useg))])
    sx_eff = np.vstack([x_eff[codes == i].mean(0) for i in range(len(useg))])
    sx_cat = np.hstack([sx_clip, sx_eff])

    print("OOF image binary contact + material...", flush=True)
    img_contact_clip = oof_probs(sx_clip, sy, folds, binary=True)
    img_contact_cat = oof_probs(sx_cat, sy, folds, binary=True)
    img_mat_clip = oof_probs(sx_clip, sy, folds, contact_only_material=True)
    img_mat_eff = oof_probs(sx_eff, sy, folds, contact_only_material=True)
    img_mat_cat = oof_probs(sx_cat, sy, folds, contact_only_material=True)
    img4_clip = oof_probs(sx_clip, sy, folds, binary=False)
    trunk_oof = np.zeros(len(sy), dtype=np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, max_iter=1200, class_weight="balanced", random_state=42
            ),
        )
        m.fit(sx_clip[tr], (sy[tr] == 2).astype(int))
        trunk_oof[va] = m.predict_proba(sx_clip[va])[:, 1]

    np.savez(
        OUT / "hand_oof_arrays.npz",
        seg_pc=seg_pc,
        seg_audio_mat=seg_audio_mat,
        seg_audio4=seg_audio4,
        img_contact_clip=img_contact_clip,
        img_contact_cat=img_contact_cat,
        img_mat_clip=img_mat_clip,
        img_mat_eff=img_mat_eff,
        img_mat_cat=img_mat_cat,
        img4_clip=img4_clip,
        trunk_oof=trunk_oof,
        sy=sy,
        ss=ss,
    )

    ths = np.linspace(0.30, 0.70, 21)
    mat_keys = {
        "clip": img_mat_clip,
        "eff": img_mat_eff,
        "cat": img_mat_cat,
        "clip4mat": suite.normalize(img4_clip[:, 1:]),
        "audio": seg_audio_mat,
    }
    contact_cfgs = [("audio", 1.0, seg_pc)]
    for alpha in (0.5, 0.7, 0.85, 0.95):
        contact_cfgs.append(
            ("clip", alpha, alpha * seg_pc + (1 - alpha) * img_contact_clip)
        )
        contact_cfgs.append(
            ("cat", alpha, alpha * seg_pc + (1 - alpha) * img_contact_cat)
        )

    # Coarse classwise weights then refine top
    weight_grid = list(product([0.0, 0.4, 0.7, 1.0], repeat=3))
    rows = []
    for cname, alpha, contact in contact_cfgs:
        for mname, imat in mat_keys.items():
            for wl, wt, ww in weight_grid:
                w = np.array([wl, wt, ww], dtype=np.float64)
                if mname == "audio":
                    mat0 = seg_audio_mat
                else:
                    mat0 = suite.normalize(seg_audio_mat * (1.0 - w) + imat * w)
                for trunk_bias in (0.0, 0.3, 0.6):
                    if trunk_bias == 0.0:
                        mat = mat0
                    else:
                        logits = np.log(np.clip(mat0, 1e-8, 1.0))
                        logits[:, 1] = logits[:, 1] + trunk_bias * trunk_oof
                        mat = suite.normalize(np.exp(logits))
                    f1, bf1, th, _ = best_threshold(contact, mat, sy, ths)
                    rows.append(
                        {
                            "contact_source": cname,
                            "contact_alpha_audio": float(alpha),
                            "material_source": mname,
                            "leaf_w": float(wl),
                            "trunk_w": float(wt),
                            "twig_w": float(ww),
                            "trunk_bias": float(trunk_bias),
                            "contact_threshold": float(th),
                            "macro_f1_4class": float(f1),
                            "binary_macro_f1": float(bf1),
                            "trunk_recall": float(
                                np.sum((sy == 2) & ((contact >= th) & (mat.argmax(1) + 1 == 2)))
                                / max(np.sum(sy == 2), 1)
                            ),
                        }
                    )
    board = (
        pd.DataFrame(rows)
        .sort_values(
            ["macro_f1_4class", "binary_macro_f1", "trunk_recall"], ascending=False
        )
        .reset_index(drop=True)
    )
    # Local refine around top-5 weight settings
    refine_rows = []
    tops = board.head(8)
    for _, row in tops.iterrows():
        cname = row["contact_source"]
        alpha = float(row["contact_alpha_audio"])
        if cname == "audio":
            contact = seg_pc
        elif cname == "clip":
            contact = alpha * seg_pc + (1 - alpha) * img_contact_clip
        else:
            contact = alpha * seg_pc + (1 - alpha) * img_contact_cat
        mname = row["material_source"]
        imat = mat_keys[mname]
        base_w = np.array([row["leaf_w"], row["trunk_w"], row["twig_w"]], dtype=np.float64)
        for d in product([-0.15, 0.0, 0.15], repeat=3):
            w = np.clip(base_w + np.array(d), 0.0, 1.0)
            if mname == "audio":
                mat0 = seg_audio_mat
            else:
                mat0 = suite.normalize(seg_audio_mat * (1.0 - w) + imat * w)
            for trunk_bias in (
                max(0.0, row["trunk_bias"] - 0.15),
                row["trunk_bias"],
                min(1.0, row["trunk_bias"] + 0.15),
            ):
                if trunk_bias == 0.0:
                    mat = mat0
                else:
                    logits = np.log(np.clip(mat0, 1e-8, 1.0))
                    logits[:, 1] = logits[:, 1] + trunk_bias * trunk_oof
                    mat = suite.normalize(np.exp(logits))
                f1, bf1, th, _ = best_threshold(contact, mat, sy, ths)
                refine_rows.append(
                    {
                        "contact_source": cname,
                        "contact_alpha_audio": float(alpha),
                        "material_source": mname,
                        "leaf_w": float(w[0]),
                        "trunk_w": float(w[1]),
                        "twig_w": float(w[2]),
                        "trunk_bias": float(trunk_bias),
                        "contact_threshold": float(th),
                        "macro_f1_4class": float(f1),
                        "binary_macro_f1": float(bf1),
                        "trunk_recall": float(
                            np.sum(
                                (sy == 2)
                                & ((contact >= th) & (mat.argmax(1) + 1 == 2))
                            )
                            / max(np.sum(sy == 2), 1)
                        ),
                    }
                )
    board2 = (
        pd.concat([board, pd.DataFrame(refine_rows)], ignore_index=True)
        .sort_values(
            ["macro_f1_4class", "binary_macro_f1", "trunk_recall"], ascending=False
        )
        .drop_duplicates(
            subset=[
                "contact_source",
                "contact_alpha_audio",
                "material_source",
                "leaf_w",
                "trunk_w",
                "twig_w",
                "trunk_bias",
                "contact_threshold",
            ]
        )
        .reset_index(drop=True)
    )
    board2.to_csv(OUT / "hand_hier_contact_rescue_leaderboard.csv", index=False)
    best = board2.iloc[0].to_dict()
    lock = {
        "protocol": "segment_hier_contact_image_rescue_classwise_material_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "audio_source": "locked group-aware OOF audio",
        "image_source": "CLIP (+EfficientNet cat) segment mean OOF contact/material",
        "test_loaded": False,
        "selected_candidate": best,
        "n_candidates": int(len(board2)),
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print(board2.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
