from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = np.arange(4, dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "multimodal_feature_group_selection"
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    by, bp = (y > 0).astype(int), (pred > 0).astype(int)
    return {
        "accuracy_4class": float(accuracy_score(y, pred)),
        "macro_precision_4class": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1_4class": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "binary_accuracy_ambient_noambient": float(accuracy_score(by, bp)),
        "binary_macro_precision_ambient_noambient": float(precision_score(by, bp, average="macro", zero_division=0)),
        "binary_macro_recall_ambient_noambient": float(recall_score(by, bp, average="macro", zero_division=0)),
        "binary_macro_f1_ambient_noambient": float(f1_score(by, bp, average="macro", zero_division=0)),
        "per_class_4class": classification_report(y, pred, labels=LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "per_class_binary": classification_report(by, bp, labels=[0, 1], target_names=["ambient", "noambient"], output_dict=True, zero_division=0),
        "confusion_matrix_4class": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "confusion_matrix_binary": confusion_matrix(by, bp, labels=[0, 1]).tolist(),
    }


def main() -> None:
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    if lock["best_val_candidate"]["feature_set"] != "audio_total240_image_resnet18" or lock["best_val_candidate"]["model"] != "logreg_c0.1_balanced":
        raise AssertionError("This final evaluator is locked to a different candidate than the selection lock")
    raw_train = pd.read_csv(ROOT / "audio_visual_dataset_default" / "dataset.csv")
    raw_test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default" / "dataset.csv")
    labels = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_train = raw_train.category.astype(str).str.lower().map(labels).to_numpy(dtype=np.int64)
    y_test = raw_test.category.astype(str).str.lower().map(labels).to_numpy(dtype=np.int64)
    a_train = np.load(OUT / "audio_feature_benchmarks" / "total240_trainval_select" / "features" / "hand_train_full" / "X.npy")
    a_test_path = OUT / "audio_feature_benchmarks" / "total240_trainval_select" / "features" / "robot_test" / "X.npy"
    a_test = np.load(a_test_path) if a_test_path.exists() else None
    if a_test is None:
        raise FileNotFoundError("Locked total240 robot/test feature cache is missing")
    i_train = np.load(OUT / "image_deep_features" / "resnet18_224" / "hand_train_full" / "X.npy")
    i_test = np.load(OUT / "image_deep_features" / "resnet18_224" / "robot_test" / "X.npy")
    if not np.array_equal(np.load(OUT / "image_deep_features" / "resnet18_224" / "hand_train_full" / "y.npy"), y_train):
        raise AssertionError("Training image labels are not aligned")
    if not np.array_equal(np.load(OUT / "image_deep_features" / "resnet18_224" / "robot_test" / "y.npy"), y_test):
        raise AssertionError("Test image labels are not aligned")
    if len(a_test) != len(y_test):
        raise AssertionError(f"Audio test feature rows {len(a_test)} != test labels {len(y_test)}")
    X_train = np.hstack([a_train, i_train])
    X_test = np.hstack([a_test, i_test])
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(np.int64)
    result = {"split": "robot_test_final", "n": int(len(y_test)), "locked_candidate": lock["best_val_candidate"], **metrics(y_test, pred)}
    (REPORT / "feature_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / "feature_final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(REPORT / "feature_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(REPORT / "feature_final_test_confusion_matrix_binary.csv")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
