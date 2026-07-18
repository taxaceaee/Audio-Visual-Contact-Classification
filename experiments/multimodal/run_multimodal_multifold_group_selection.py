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


def main() -> None:
    output = Path("outputs")
    report = output / "audio_feature_benchmarks" / "multimodal_multifold_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    root = Path("/home/ttung05/Desktop/tree_base/tree_structures")
    frame = suite.load_manifest(root / "audio_visual_dataset_default" / "dataset.csv", root / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    y = frame.y.to_numpy(dtype=np.int64)
    groups = frame.specimen_group.to_numpy()
    X_img, y_img = suite.image_cache(output, "hand_train_full")
    if not np.array_equal(y, y_img):
        raise AssertionError("image/audio hand rows are not aligned")
    pair = suite.normalize(np.load(output / "audio_feature_benchmarks" / "audio_group_consistency_pair_blend_select" / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"))
    p_audio = {}
    for view in ["clean", "robot_mix", "bandlimit"]:
        p_audio[view] = suite.segment_lift(frame.audio_file, suite.normalize(0.8 * base.oof_view(output, view) + 0.2 * pair))

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(np.arange(len(frame)), y, groups))
    fold_summary = [{"fold": i, "train_rows": len(tr), "val_rows": len(va), "train_groups": int(pd.Series(groups[tr]).nunique()), "val_groups": int(pd.Series(groups[va]).nunique()), "overlap": len(set(groups[tr]) & set(groups[va]))} for i, (tr, va) in enumerate(folds)]
    if any(x["overlap"] for x in fold_summary):
        raise AssertionError("specimen group overlap in multifold split")
    (report / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")

    # Fit each fold's image model once. Candidate fusion weights must not
    # refit or inspect the holdout labels; caching these OOF image predictions
    # also keeps the selection run tractable.
    fold_predictions = []
    for train_idx, val_idx in folds:
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42, solver="lbfgs"))]).fit(X_img[train_idx], y[train_idx])
        fold_predictions.append((val_idx, base.image_proba(model, X_img[val_idx])))

    rows = []
    for alpha in np.linspace(0.0, 1.0, 21):
        for kind in ["linear", "log"]:
            scores = {view: [] for view in p_audio}
            for val_idx, p_img in fold_predictions:
                for view, pa_full in p_audio.items():
                    pa = pa_full[val_idx]
                    if kind == "linear":
                        p = suite.normalize(alpha * pa + (1.0 - alpha) * p_img)
                    else:
                        p = suite.normalize(np.exp(alpha * np.log(suite.normalize(pa)) + (1.0 - alpha) * np.log(suite.normalize(p_img))))
                    scores[view].append(float(f1_score(y[val_idx], p.argmax(axis=1), average="macro", zero_division=0)))
            row = {"kind": kind, "audio_weight": float(alpha)}
            for view, vals in scores.items():
                row[f"mean_macro_f1_{view}"] = float(np.mean(vals))
                row[f"worst_fold_macro_f1_{view}"] = float(np.min(vals))
            row["mean_across_views"] = float(np.mean([row[f"mean_macro_f1_{v}"] for v in scores]))
            row["worst_across_views_folds"] = float(np.min([row[f"worst_fold_macro_f1_{v}"] for v in scores]))
            rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values(["worst_across_views_folds", "mean_across_views"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_multifold_group_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "multimodal_multifold_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
