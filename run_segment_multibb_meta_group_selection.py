"""Hand-only multi-backbone segment meta fusion selection (no robot/test load)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
        "outputs/audio_feature_benchmarks/segment_multibb_meta_group_selection",
    )
)

# Backbones with both hand_train_full and robot_test caches.
BACKBONES = [
    Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    Path("outputs/image_deep_features/resnet18_224"),
    Path("outputs/image_deep_features/resnet34_224"),
    Path("outputs/image_deep_features/vit_b16_224"),
]


def seg(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)


def spec(s: pd.Series) -> pd.Series:
    return seg(s).str.replace(r"_segment_.*$", "", regex=True)


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    return (-p * np.log(p)).sum(axis=1, keepdims=True).astype(np.float32)


def build_meta_features(
    audio4: np.ndarray,
    img4_list: list[np.ndarray],
    contact: np.ndarray,
) -> np.ndarray:
    """Stack audio 4-class + multi-backbone 4-class image probs + gates/margins."""
    parts: list[np.ndarray] = [
        np.log(np.clip(audio4, 1e-6, 1.0)).astype(np.float32),
        audio4.astype(np.float32),
        contact[:, None].astype(np.float32),
        (1.0 - audio4[:, 0:1]).astype(np.float32),
        audio4.max(1, keepdims=True).astype(np.float32),
        entropy(audio4),
    ]
    # top-2 margin on audio
    s = np.sort(audio4, axis=1)
    parts.append((s[:, -1] - s[:, -2])[:, None].astype(np.float32))

    ens = np.zeros_like(audio4, dtype=np.float64)
    for im in img4_list:
        parts.append(np.log(np.clip(im, 1e-6, 1.0)).astype(np.float32))
        parts.append(im.astype(np.float32))
        parts.append(im.max(1, keepdims=True).astype(np.float32))
        parts.append(entropy(im))
        parts.append((1.0 - im[:, 0:1]).astype(np.float32))
        ens += im
    ens = suite.normalize(ens / max(len(img4_list), 1))
    parts.append(ens.astype(np.float32))
    parts.append(np.log(np.clip(ens, 1e-6, 1.0)).astype(np.float32))
    # blend audio/image ensemble at fixed exploratory weights as features (meta learns mix)
    for w in (0.3, 0.5, 0.7):
        b = suite.normalize((1 - w) * audio4 + w * ens)
        parts.append(b.astype(np.float32))
    return np.hstack(parts).astype(np.float32)


def oof_image_4class(
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    folds: list,
    C: float = 0.03,
) -> np.ndarray:
    oof = np.zeros((len(y_seg), 4), dtype=np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=1200,
                class_weight="balanced",
                multi_class="multinomial",
                random_state=42,
            ),
        )
        m.fit(x_seg[tr], y_seg[tr])
        oof[va] = suite.normalize(m.predict_proba(x_seg[va]))
    return oof


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

    # Locked OOF audio (hand only).
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
    # segment-mean 4-class audio
    sa4 = np.vstack(
        [
            suite.normalize(audio[codes == i].mean(0, keepdims=True))[0]
            for i in range(len(useg))
        ]
    )
    pc, _ = avr.avr_inputs(audio)
    spc = np.array([pc[codes == i].mean() for i in range(len(useg))], dtype=np.float64)

    # Multi-backbone OOF 4-class image probs at segment level.
    img4_list: list[np.ndarray] = []
    for bb in BACKBONES:
        x = np.load(bb / "hand_train_full/X.npy").astype(np.float32)
        assert len(x) == len(y), f"align {bb}"
        x_seg = np.vstack([x[codes == i].mean(0) for i in range(len(useg))])
        print(f"OOF image backbone {bb.name} ...", flush=True)
        img4_list.append(oof_image_4class(x_seg, sy, folds, C=0.03))

    z = build_meta_features(sa4, img4_list, spc)
    np.save(OUT / "hand_meta_oof_features.npy", z)
    np.save(OUT / "hand_meta_oof_y.npy", sy)
    np.save(OUT / "hand_meta_oof_groups.npy", ss)

    rows = []
    # Logistic meta with class-weight grid (focus trunk/twig).
    for C in (0.03, 0.1, 0.3, 1.0):
        for tw in (1.0, 1.5, 2.0, 2.5, 3.0):
            for gw in (1.0, 1.2, 1.5, 2.0):
                cw = {0: 1.0, 1: 1.0, 2: tw, 3: gw}
                fs = []
                for tr, va in folds:
                    m = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=C,
                            max_iter=4000,
                            class_weight=cw,
                            multi_class="multinomial",
                            random_state=42,
                        ),
                    )
                    m.fit(z[tr], sy[tr])
                    fs.append(
                        f1_score(
                            sy[va], m.predict(z[va]), average="macro", zero_division=0
                        )
                    )
                rows.append(
                    {
                        "model": "logreg",
                        "C": C,
                        "trunk_class_weight": tw,
                        "twig_class_weight": gw,
                        "max_depth": -1,
                        "lr": -1.0,
                        "mean_cv_macro_f1": float(np.mean(fs)),
                        "worst_cv_macro_f1": float(np.min(fs)),
                    }
                )

    # HistGradientBoosting meta (class-weight via sample_weight).
    for max_depth in (3, 5, None):
        for lr in (0.05, 0.1):
            fs = []
            for tr, va in folds:
                # mild trunk/twig upweight via sample weights
                sw = np.ones(len(tr), dtype=np.float64)
                sw[sy[tr] == 2] = 2.0
                sw[sy[tr] == 3] = 1.5
                m = HistGradientBoostingClassifier(
                    max_depth=max_depth,
                    learning_rate=lr,
                    max_iter=200,
                    random_state=42,
                )
                m.fit(z[tr], sy[tr], sample_weight=sw)
                fs.append(
                    f1_score(sy[va], m.predict(z[va]), average="macro", zero_division=0)
                )
            rows.append(
                {
                    "model": "hgb",
                    "C": -1.0,
                    "trunk_class_weight": 2.0,
                    "twig_class_weight": 1.5,
                    "max_depth": -1 if max_depth is None else int(max_depth),
                    "lr": float(lr),
                    "mean_cv_macro_f1": float(np.mean(fs)),
                    "worst_cv_macro_f1": float(np.min(fs)),
                }
            )

    board = (
        pd.DataFrame(rows)
        .sort_values(["mean_cv_macro_f1", "worst_cv_macro_f1"], ascending=False)
        .reset_index(drop=True)
    )
    board.to_csv(OUT / "hand_multibb_meta_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    lock = {
        "protocol": "segment_multibb_image_audio_meta_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "audio_source": "locked group-aware OOF audio 4-class segment mean",
        "image_source": "CLIP+EfficientNet+ResNet18/34+ViT-B16 segment-mean 4-class OOF",
        "backbones": [str(b) for b in BACKBONES],
        "test_loaded": False,
        "selected_candidate": best,
    }
    (OUT / "selection_lock.json").write_text(
        json.dumps(lock, indent=2, default=float)
    )
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
