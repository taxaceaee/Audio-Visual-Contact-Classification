from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_stress_robust_group_selection as image_base
import run_multimodal_val_locked_suite as suite
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base


OUT = Path("outputs")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")


def segment_only(files: pd.Series, p: np.ndarray) -> np.ndarray:
    codes, _ = suite.audio_keys(files)
    n_groups = int(codes.max()) + 1
    logp = np.log(suite.normalize(p))
    sums = np.vstack([np.bincount(codes, weights=logp[:, c], minlength=n_groups) for c in range(4)]).T
    seg = suite.normalize(np.exp(sums - sums.max(axis=1, keepdims=True)))
    return seg[codes]


def main() -> None:
    report = OUT / "audio_feature_benchmarks/multimodal_multiview_aggregation_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set("total240")
    frame = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    audio_frame = audio_base.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train")
    y = frame.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y, audio_frame.y.to_numpy(dtype=np.int64)):
        raise AssertionError("label alignment")
    high = group_audio.load_highsr_oof(OUT)
    pair = specimen.load_pairwise_oof(OUT / "audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy")
    p_audio = lift.anchor_lift_proba(audio_frame, high, pair)
    groups = frame.specimen_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))

    X = np.load(OUT / "image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy")
    image_oof = np.zeros((len(y), 4), dtype=np.float64)
    for tr, va in folds:
        m = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=900, random_state=42))]).fit(X[tr], y[tr])
        image_oof[va] = image_base.image_proba(m, X[va])
    image_seg = segment_only(frame.audio_file, image_oof)
    image_spec = suite.segment_lift(frame.audio_file, image_oof)
    rows = []
    for image_name, pi in [("raw", image_oof), ("segment_only", image_seg), ("segment_lift", image_spec)]:
        for order in ["fuse_raw", "fuse_then_segment"]:
            for iw in np.linspace(0.0, 0.6, 25):
                p_raw = suite.normalize(np.exp((1-iw)*np.log(suite.normalize(p_audio)) + iw*np.log(suite.normalize(pi))))
                p = p_raw if order == "fuse_raw" else suite.segment_lift(frame.audio_file, p_raw)
                fs = [f1_score(y[va], p[va].argmax(1), average="macro", zero_division=0) for _, va in folds]
                rows.append({"image_aggregation": image_name, "order": order, "image_weight": float(iw), "mean_fold_macro_f1": float(np.mean(fs)), "worst_fold_macro_f1": float(np.min(fs)), "std_fold_macro_f1": float(np.std(fs))})
    lb = pd.DataFrame(rows).sort_values(["worst_fold_macro_f1", "mean_fold_macro_f1", "std_fold_macro_f1"], ascending=[False, False, True]).reset_index(drop=True)
    lb.to_csv(report / "hand_multiview_aggregation_leaderboard.csv", index=False)
    best = lb.iloc[0].to_dict()
    lock = {"protocol": "multimodal_multiview_aggregation_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
