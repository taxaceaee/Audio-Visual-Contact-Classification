"""Shared protocol helpers for strict multimodal 0.85 drive (no robot in selection)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def fast_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def write_selection_lock(path: Path, payload: dict) -> dict:
    """Write hand-only selection lock; forces test_loaded=False."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = dict(payload)
    lock["test_loaded"] = False
    lock.setdefault("selection_data", "hand/default only")
    lock.setdefault("group_column", "specimen_group")
    lock.setdefault(
        "invariants",
        {
            "robot_not_used_in_selection": True,
            "filename_class_features": False,
            "no_test_hp_tuning": True,
        },
    )
    path.write_text(json.dumps(lock, indent=2, default=float))
    return lock


def load_selection_lock(path: Path) -> dict:
    lock = json.loads(Path(path).read_text())
    if lock.get("test_loaded"):
        raise AssertionError(f"selection lock already test-loaded: {path}")
    return lock


def metrics_bundle(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    by = (y > 0).astype(int)
    bp = (pred > 0).astype(int)
    return {
        "n": int(len(y)),
        "metrics": {
            "accuracy_4class": float(accuracy_score(y, pred)),
            "macro_precision_4class": float(
                precision_score(y, pred, average="macro", zero_division=0)
            ),
            "macro_recall_4class": float(
                recall_score(y, pred, average="macro", zero_division=0)
            ),
            "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1_4class": float(
                f1_score(y, pred, average="weighted", zero_division=0)
            ),
            "binary_macro_f1": float(f1_score(by, bp, average="macro", zero_division=0)),
        },
        "per_class_4class": classification_report(
            y,
            pred,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_4class": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
    }


def write_final_metrics(path: Path, result: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float))
    return result
