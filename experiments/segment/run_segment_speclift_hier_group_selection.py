"""Spec-lift audio + hierarchical image material (segment-pure labels).

Hand-only selection. Uses specimen contact consensus only on audio probs
(already locked in audio pipeline), not specimen y[0] for training.
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
import train_audio_group_consistency_pair_blend_select_final_test as ga
import train_audio_specimen_contact_consensus_select_final_test as sp
import train_audio_specimen_contact_lift_select_final_test as sl
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_speclift_hier_group_selection",
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


def fast_macro(y, p):
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


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

    # Spec-lift style audio OOF (group-aware highsr + pairwise)
    af = audio_base.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train"
    )
    high = ga.load_highsr_oof(Path("outputs"))
    pair = sp.load_pairwise_oof(
        Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        )
    )
    raw_audio = sl.normalize(0.8 * high + 0.2 * pair)
    seg_audio, w2s = sp.segment_proba_from_window(af, raw_audio)
    codes_audio = sp.specimen_codes_for_segments(af, w2s)
    locked = json.load(
        open(
            "outputs/audio_feature_benchmarks/audio_specimen_contact_lift_select/reports/"
            "audio_specimen_contact_lift_select_selected_without_test.json"
        )
    )["selected_without_test"]
    seg_audio = sl.consensus_and_lift(
        seg_audio,
        codes_audio,
        float(locked["consensus_threshold"]),
        int(locked["min_contact_segments"]),
        locked["lift_min_mass"],
        float(locked["lift_floor"]),
        float(locked["lift_confidence"]),
    )
    audio = seg_audio[w2s]
    assert len(audio) == len(y)
    pc, lg = avr.avr_inputs(audio)
    seg_pc = np.array([pc[codes == i].mean() for i in range(len(useg))])
    sa_mat = np.vstack(
        [
            suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0]
            for i in range(len(useg))
        ]
    )

    x_clip = np.load(IMG_CLIP / "hand_train_full/X.npy").astype(np.float32)
    x_eff = np.load(IMG_EFF / "hand_train_full/X.npy").astype(np.float32)
    sx_c = np.vstack([x_clip[codes == i].mean(0) for i in range(len(useg))])
    sx_e = np.vstack([x_eff[codes == i].mean(0) for i in range(len(useg))])
    sx_cat = np.hstack([sx_c, sx_e])

    def oof_mat(x):
        o = np.zeros((len(sy), 3), np.float32)
        for tr, va in folds:
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.03,
                    max_iter=1500,
                    class_weight="balanced",
                    multi_class="multinomial",
                    random_state=42,
                ),
            )
            ci = tr[sy[tr] > 0]
            m.fit(x[ci], sy[ci] - 1)
            o[va] = m.predict_proba(x[va])
        return o

    def oof_bin(x):
        o = np.zeros(len(sy), np.float32)
        for tr, va in folds:
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.03, max_iter=1500, class_weight="balanced", random_state=42
                ),
            )
            m.fit(x[tr], (sy[tr] > 0).astype(int))
            o[va] = m.predict_proba(x[va])[:, 1]
        return o

    print("OOF material/contact...", flush=True)
    mats = {
        "clip": oof_mat(sx_c),
        "eff": oof_mat(sx_e),
        "cat": oof_mat(sx_cat),
        "audio": sa_mat,
    }
    bins = {"clip": oof_bin(sx_c), "cat": oof_bin(sx_cat)}
    trunk = np.zeros(len(sy), np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, max_iter=1200, class_weight="balanced", random_state=42
            ),
        )
        m.fit(sx_c[tr], (sy[tr] == 2).astype(int))
        trunk[va] = m.predict_proba(sx_c[va])[:, 1]

    np.savez(
        OUT / "hand_oof.npz",
        sy=sy,
        ss=ss,
        seg_pc=seg_pc,
        sa_mat=sa_mat,
        trunk=trunk,
        **{f"mat_{k}": v for k, v in mats.items()},
        **{f"bin_{k}": v for k, v in bins.items()},
    )

    ths = np.linspace(0.32, 0.68, 19)
    rows = []
    for csrc, alpha in [
        ("audio", 1.0),
        ("clip", 0.5),
        ("clip", 0.7),
        ("cat", 0.5),
        ("cat", 0.7),
    ]:
        if csrc == "audio":
            contact = seg_pc
        else:
            contact = alpha * seg_pc + (1 - alpha) * bins[csrc]
        for msrc in mats:
            for wl, wt, ww in product([0.2, 0.4, 0.6, 0.8], repeat=3):
                w = np.array([wl, wt, ww], dtype=np.float64)
                if msrc == "audio":
                    mat0 = sa_mat
                else:
                    mat0 = suite.normalize(sa_mat * (1 - w) + mats[msrc] * w)
                for tb in (0.0, 0.3, 0.5):
                    if tb:
                        logits = np.log(np.clip(mat0, 1e-8, 1))
                        logits[:, 1] += tb * trunk
                        mat = suite.normalize(np.exp(logits))
                    else:
                        mat = mat0
                    mat_pred = mat.argmax(1) + 1
                    best_f1, best_th, best_bf = -1, 0.5, -1
                    for th in ths:
                        pred = np.where(contact >= th, mat_pred, 0)
                        f1 = fast_macro(sy, pred)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_th = float(th)
                            best_bf = float(
                                f1_score(
                                    sy > 0, pred > 0, average="macro", zero_division=0
                                )
                            )
                    rows.append(
                        {
                            "contact_source": csrc,
                            "contact_alpha_audio": float(alpha),
                            "material_source": msrc,
                            "leaf_w": float(wl),
                            "trunk_w": float(wt),
                            "twig_w": float(ww),
                            "trunk_bias": float(tb),
                            "contact_threshold": best_th,
                            "macro_f1_4class": float(best_f1),
                            "binary_macro_f1": float(best_bf),
                        }
                    )
    board = (
        pd.DataFrame(rows)
        .sort_values(["macro_f1_4class", "binary_macro_f1"], ascending=False)
        .reset_index(drop=True)
    )
    board.to_csv(OUT / "hand_speclift_hier_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    lock = {
        "protocol": "segment_speclift_audio_hier_image_material_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "audio_source": "spec-lift consensus highsr+pairwise OOF (locked HP)",
        "image_source": "CLIP/Eff segment-pure contact material OOF",
        "label_unit": "segment (pure); no specimen y[0]",
        "test_loaded": False,
        "selected_candidate": best,
        "audio_lift_lock": locked,
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
