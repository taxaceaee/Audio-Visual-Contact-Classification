from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


LABELS = np.arange(4, dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "multimodal_oof_stack_group_selection"
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")


def image_proba(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, col]
    return suite.apply_bias(p)


def meta_features(pa: np.ndarray, pi: np.ndarray) -> np.ndarray:
    pa = suite.normalize(pa)
    pi = suite.normalize(pi)
    return np.hstack([np.log(pa), np.log(pi), pa, pi, -np.sum(pa * np.log(pa), axis=1, keepdims=True), -np.sum(pi * np.log(pi), axis=1, keepdims=True)])


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


def make_image_model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))])


def main() -> None:
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    if lock["best_val_candidate"]["model"] != "meta_logreg_c0.01_none":
        raise AssertionError("Final evaluator is not using the locked meta candidate")
    full, train_idx, _ = suite.build_group_split(ROOT, REPORT)
    _, pa_full = suite.load_audio_oof(ROOT, OUT)
    X_img, y_img = suite.image_cache(OUT, "hand_train_full")
    y = full.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y, y_img):
        raise AssertionError("Hand labels are not aligned")
    inner_groups = full.specimen_group.to_numpy()[train_idx]
    inner_y = y[train_idx]
    image_oof = np.zeros((len(train_idx), 4), dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
    for inner_train, inner_val in splitter.split(np.arange(len(train_idx)), inner_y, inner_groups):
        model = make_image_model().fit(X_img[train_idx[inner_train]], inner_y[inner_train])
        image_oof[inner_val] = image_proba(model, X_img[train_idx[inner_val]])
    meta_X = meta_features(pa_full[train_idx], image_oof)
    meta = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.01, class_weight=None, max_iter=2000, random_state=42))]).fit(meta_X, y[train_idx])

    # Only after the selection lock do we load robot/test artifacts.
    raw_test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default" / "dataset.csv")
    labels = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_test = raw_test.category.astype(str).str.lower().map(labels).to_numpy(dtype=np.int64)
    X_test_img, y_test_img = suite.image_cache(OUT, "robot_test")
    if not np.array_equal(y_test, y_test_img):
        raise AssertionError("Robot/test image labels are not aligned")
    image_full = make_image_model().fit(X_img, y)
    pi_test = image_proba(image_full, X_test_img)
    audio_path = OUT / "audio_feature_benchmarks" / "audio_lift_source_blend_select" / "reports" / "audio_lift_source_blend_select_final_test_predictions.csv"
    audio_frame = pd.read_csv(audio_path)
    if not np.array_equal(audio_frame.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
        raise AssertionError("Audio test probability order is not aligned")
    pa_test = suite.normalize(audio_frame[suite.PROBA_COLUMNS].to_numpy())
    p_test = meta.predict_proba(meta_features(pa_test, pi_test))
    pred = p_test.argmax(axis=1).astype(np.int64)
    result = {"split": "robot_test_final", "n": int(len(y_test)), "locked_candidate": lock["best_val_candidate"], **metrics(y_test, pred)}
    (REPORT / "oof_stack_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / "oof_stack_final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(REPORT / "oof_stack_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(REPORT / "oof_stack_final_test_confusion_matrix_binary.csv")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
