from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


def image_proba(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, col]
    return suite.apply_bias(p)


def oof_view(output: Path, view: str) -> np.ndarray:
    run = output / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    lock = json.loads((run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    cand = lock["selected_candidate"]
    return suite.normalize(np.load(run / "oof_proba" / cand / f"{view}_oof_proba.npy"))


def score(y, p):
    pred = p.argmax(axis=1)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def main():
    output = Path("outputs")
    report = output / "audio_feature_benchmarks" / "multimodal_stress_robust_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    root = Path("/home/ttung05/Desktop/tree_base/tree_structures")
    full, train_idx, val_idx = suite.build_group_split(root, report)
    X_img, y_img = suite.image_cache(output, "hand_train_full")
    y = full.y.to_numpy(dtype=np.int64)
    image_model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X_img[train_idx], y[train_idx])
    p_img = image_proba(image_model, X_img[val_idx])
    pair = suite.normalize(np.load(output / "audio_feature_benchmarks" / "audio_group_consistency_pair_blend_select" / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"))[val_idx]
    p_audio = {view: suite.segment_lift(full.audio_file.iloc[val_idx], suite.normalize(0.8 * oof_view(output, view)[val_idx] + 0.2 * pair)) for view in ["clean", "robot_mix", "bandlimit"]}
    rows = []
    for alpha in np.linspace(0.0, 1.0, 21):
        for kind in ["linear", "log"]:
            view_scores = {}
            for view, pa in p_audio.items():
                if kind == "linear":
                    p = suite.normalize(alpha * pa + (1 - alpha) * p_img)
                else:
                    p = suite.normalize(np.exp(alpha * np.log(suite.normalize(pa)) + (1 - alpha) * np.log(suite.normalize(p_img))))
                view_scores[view] = score(y[val_idx], p)
            rows.append({"kind": kind, "audio_weight": float(alpha), **{f"macro_f1_{k}": v for k, v in view_scores.items()}, "worst_view_macro_f1": min(view_scores.values()), "mean_view_macro_f1": float(np.mean(list(view_scores.values())))})
    leaderboard = pd.DataFrame(rows).sort_values(["worst_view_macro_f1", "mean_view_macro_f1"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_group_stress_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "multimodal_stress_robust_group_val_only", "group_column": "specimen_group", "group_overlap": 0, "test_loaded": False, "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
