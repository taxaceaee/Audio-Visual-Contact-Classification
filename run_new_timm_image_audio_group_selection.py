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

import run_multimodal_stress_robust_group_selection as base
import run_multimodal_val_locked_suite as suite


OUT = Path("outputs")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
BACKBONES = {
    "dino_vits14": OUT / "image_timm_features/vit_small_patch14_dinov2.lvd142m/hand_train_full",
    "convnext_tiny": OUT / "image_timm_features/convnext_tiny.fb_in22k_ft_in1k/hand_train_full",
    "swin_tiny": OUT / "image_timm_features/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k/hand_train_full",
    "efficientnet_b3": OUT / "image_timm_features/efficientnet_b3.ra2_in1k/hand_train_full",
}


def main() -> None:
    report = OUT / "audio_feature_benchmarks/new_timm_image_audio_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    frame = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    y = frame.y.to_numpy(dtype=np.int64)
    run = OUT / "audio_feature_benchmarks/audio_highsr_temporal_tta_select"
    lock = json.loads((run / "reports/audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    pair = suite.normalize(np.load(OUT / "audio_feature_benchmarks/audio_group_consistency_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy"))
    audio = suite.segment_lift(frame.audio_file, suite.normalize(0.8 * np.load(run / "oof_proba" / lock["selected_candidate"] / "clean_oof_proba.npy") + 0.2 * pair))
    groups = frame.specimen_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))
    rows = []
    for name, folder in BACKBONES.items():
        X = np.load(folder / "X.npy")
        y_cache = np.load(folder / "y.npy").astype(np.int64)
        if X.shape[0] != len(y) or not np.array_equal(y, y_cache):
            raise AssertionError(f"{name} alignment")
        image_oof = np.zeros((len(y), 4), dtype=np.float64)
        for tr, va in folds:
            model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X[tr], y[tr])
            image_oof[va] = base.image_proba(model, X[va])
        for iw in np.linspace(0.0, 1.0, 21):
            for kind in ["linear", "log"]:
                p = suite.normalize((1-iw)*audio + iw*image_oof) if kind == "linear" else suite.normalize(np.exp((1-iw)*np.log(suite.normalize(audio)) + iw*np.log(suite.normalize(image_oof))))
                fs = [f1_score(y[va], p[va].argmax(1), average="macro", zero_division=0) for _, va in folds]
                rows.append({"backbone": name, "image_weight": float(iw), "kind": kind, "mean_macro_f1": float(np.mean(fs)), "worst_fold_macro_f1": float(np.min(fs)), "std_macro_f1": float(np.std(fs))})
    leaderboard = pd.DataFrame(rows).sort_values(["worst_fold_macro_f1", "mean_macro_f1", "std_macro_f1"], ascending=[False, False, True]).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_new_timm_image_audio_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "new_timm_image_audio_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
