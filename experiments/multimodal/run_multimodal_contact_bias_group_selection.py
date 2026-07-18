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


def main():
    output = Path("outputs")
    report = output / "audio_feature_benchmarks" / "multimodal_contact_bias_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    root = Path("/home/ttung05/Desktop/tree_base/tree_structures")
    full, train_idx, val_idx = suite.build_group_split(root, report)
    X, y = suite.image_cache(output, "hand_train_full")
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X[train_idx], y[train_idx])
    raw = model.predict_proba(X[val_idx])
    pi = np.zeros((len(val_idx), 4))
    for col, cls in enumerate(model.classes_): pi[:, int(cls)] = raw[:, col]
    pi = suite.apply_bias(pi)
    run = output / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    lock = json.loads((run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    pair = suite.normalize(np.load(output / "audio_feature_benchmarks" / "audio_group_consistency_pair_blend_select" / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"))[val_idx]
    views = {}
    for v in ["clean", "robot_mix", "bandlimit"]:
        high = np.load(run / "oof_proba" / lock["selected_candidate"] / f"{v}_oof_proba.npy")[val_idx]
        views[v] = suite.segment_lift(full.audio_file.iloc[val_idx], suite.normalize(0.8 * high + 0.2 * pair))
    rows = []
    for alpha in [0.5, 0.65, 0.75, 0.85, 1.0]:
        for tb in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            for wb in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
                bias = np.asarray([-0.05, 0.0, tb, wb])
                scores = {}
                for v, pa in views.items():
                    p = suite.normalize(np.exp(alpha * np.log(pa) + (1-alpha) * np.log(pi) + bias[None, :]))
                    scores[v] = float(f1_score(y[val_idx], p.argmax(axis=1), average="macro", zero_division=0))
                rows.append({"audio_weight": alpha, "trunk_bias": tb, "twig_bias": wb, **{f"macro_f1_{v}": s for v, s in scores.items()}, "worst_view_macro_f1": min(scores.values()), "mean_view_macro_f1": float(np.mean(list(scores.values())))})
    leaderboard = pd.DataFrame(rows).sort_values(["worst_view_macro_f1", "mean_view_macro_f1"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_group_contact_bias_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "multimodal_contact_bias_group_val_only", "group_column": "specimen_group", "group_overlap": 0, "test_loaded": False, "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
