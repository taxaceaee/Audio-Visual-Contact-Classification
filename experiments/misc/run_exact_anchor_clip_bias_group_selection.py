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

import run_multimodal_val_locked_suite as suite
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base


OUT = Path("outputs")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")


def main() -> None:
    report = OUT / "audio_feature_benchmarks/exact_anchor_clip_bias_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set("total240")
    af = audio_base.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train")
    frame = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    y = frame.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y, af.y.to_numpy(dtype=np.int64)):
        raise AssertionError("alignment")
    high = group_audio.load_highsr_oof(OUT)
    pair = specimen.load_pairwise_oof(OUT / "audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy")
    pa = lift.anchor_lift_proba(af, high, pair)
    X = np.load(OUT / "image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy")
    groups = frame.specimen_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))
    pi = np.zeros((len(y), 4), dtype=np.float64)
    for tr, va in folds:
        m = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=900, random_state=42))]).fit(X[tr], y[tr])
        raw = m.predict_proba(X[va])
        for col, cls in enumerate(m.classes_): pi[va, int(cls)] = raw[:, col]
    pi = suite.normalize(pi)
    rows = []
    for iw in np.linspace(0.0, 0.5, 21):
        p0 = suite.normalize(np.exp((1-iw)*np.log(suite.normalize(pa)) + iw*np.log(suite.normalize(pi))))
        for trunk_bias in np.linspace(-1.2, 1.2, 25):
            for twig_bias in np.linspace(-1.0, 1.0, 21):
                bias = np.array([0.0, 0.0, trunk_bias, twig_bias])
                p = suite.normalize(np.exp(np.log(p0) + bias[None, :]))
                fs = [f1_score(y[va], p[va].argmax(1), average="macro", zero_division=0) for _, va in folds]
                rows.append({"image_weight": float(iw), "trunk_bias": float(trunk_bias), "twig_bias": float(twig_bias), "mean_fold_macro_f1": float(np.mean(fs)), "worst_fold_macro_f1": float(np.min(fs)), "std_fold_macro_f1": float(np.std(fs))})
    lb = pd.DataFrame(rows).sort_values(["worst_fold_macro_f1", "mean_fold_macro_f1", "std_fold_macro_f1"], ascending=[False, False, True]).reset_index(drop=True)
    lb.to_csv(report / "hand_anchor_clip_bias_leaderboard.csv", index=False)
    best = lb.iloc[0].to_dict()
    lock = {"protocol": "exact_anchor_clip_class_bias_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
