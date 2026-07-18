from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = np.arange(4, dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = p.argmax(axis=1)
    by, bp = (y > 0).astype(int), (pred > 0).astype(int)
    return {
        "accuracy_4class": float(accuracy_score(y, pred)),
        "macro_precision_4class": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1_4class": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "binary_macro_f1": float(f1_score(by, bp, average="macro", zero_division=0)),
    }


def load_features(output: Path, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(Path("/home/ttung05/Desktop/tree_base/tree_structures/audio_visual_dataset_default/dataset.csv"))
    labels = {name: i for i, name in enumerate(CLASS_NAMES)}
    frame = pd.DataFrame({"audio_file": raw.audio_file.astype(str), "y": raw.category.astype(str).str.lower().map(labels).astype(np.int64)})
    audio240 = np.load(output / "audio_feature_benchmarks" / "total240_trainval_select" / "features" / "hand_train_full" / "X.npy")
    highsr = np.load(output / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select" / "features" / "hand_train_full" / "clean" / "X.npy")
    image = np.load(output / "image_deep_features" / "resnet18_224" / "hand_train_full" / "X.npy")
    y240 = np.load(output / "audio_feature_benchmarks" / "total240_trainval_select" / "features" / "hand_train_full" / "y.npy")
    if not np.array_equal(y240.astype(int), frame.y.to_numpy()):
        raise AssertionError("Cached audio labels do not align with manifest")
    split_dir = output / "audio_feature_benchmarks" / "multimodal_val_locked_suite" / "splits"
    train_files = set(pd.read_csv(split_dir / "hand_group_train.csv").audio_file.astype(str))
    val_files = set(pd.read_csv(split_dir / "hand_group_val.csv").audio_file.astype(str))
    if train_files & val_files:
        raise AssertionError("Group split overlap")
    train_idx = frame.index[frame.audio_file.isin(train_files)].to_numpy()
    val_idx = frame.index[frame.audio_file.isin(val_files)].to_numpy()
    if len(train_idx) + len(val_idx) != len(frame):
        raise AssertionError("Group split does not cover all hand rows")
    return frame, audio240, highsr, image, (train_idx, val_idx)


def main() -> None:
    output = Path("outputs")
    report = output / "audio_feature_benchmarks" / "multimodal_feature_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    frame, audio240, highsr, image, (train_idx, val_idx) = load_features(output, "hand_train_full")
    y = frame.y.to_numpy(dtype=np.int64)
    feature_sets = {
        "audio_total240": audio240,
        "audio_highsr": highsr,
        "image_resnet18": image,
        "audio_total240_image_resnet18": np.hstack([audio240, image]),
        "audio_highsr_image_resnet18": np.hstack([highsr, image]),
        "audio_total240_highsr_image_resnet18": np.hstack([audio240, highsr, image]),
    }
    rows = []
    models = []
    for c in [0.1, 1.0]:
        models.append((f"logreg_c{c:g}_balanced", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=800, random_state=42))])))
    models.append(("extratrees", ExtraTreesClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)))
    for feature_name, X in feature_sets.items():
        for model_name, model in models:
            model.fit(X[train_idx], y[train_idx])
            raw = model.predict_proba(X[val_idx])
            p = np.zeros((len(val_idx), 4), dtype=np.float64)
            for col, cls in enumerate(model.classes_):
                p[:, int(cls)] = raw[:, col]
            row = {"feature_set": feature_name, "model": model_name, **score(y[val_idx], p)}
            rows.append(row)
            if row["macro_f1_4class"] == max(r["macro_f1_4class"] for r in rows):
                joblib.dump(model, report / "selected_val_model.joblib")
                (report / "selected_model_meta.json").write_text(json.dumps({"feature_set": feature_name, "model": model_name}, indent=2), encoding="utf-8")
    leaderboard = pd.DataFrame(rows).sort_values(["macro_f1_4class", "binary_macro_f1", "accuracy_4class"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_group_val_feature_leaderboard.csv", index=False)
    lock = {
        "protocol": "multimodal_feature_fusion_group_val_only",
        "split": "specimen_grouped_hand_group_train_hand_group_val",
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_loaded": False,
        "candidate_count": int(len(rows)),
        "best_val_candidate": leaderboard.iloc[0].to_dict(),
    }
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
