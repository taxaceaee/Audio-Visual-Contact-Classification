"""Shared metric helpers for sealed robot/test evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]


def bundle_root() -> Path:
    """Bundle root (fusion_base): parent of shared/."""
    return Path(__file__).resolve().parent.parent


def load_sealed_targets(root: Path | None = None) -> dict[str, Any]:
    root = root or bundle_root()
    path = root / "shared" / "sealed_targets.json"
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_y_pred(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    binary_y = (y > 0).astype(np.int64)
    binary_p = (pred > 0).astype(np.int64)
    contact_mask = y > 0
    if contact_mask.any():
        contact_f1 = float(
            f1_score(
                y[contact_mask],
                pred[contact_mask],
                labels=[1, 2, 3],
                average="macro",
                zero_division=0,
            )
        )
    else:
        contact_f1 = float("nan")
    return {
        "n": int(len(y)),
        "accuracy_4class": float(accuracy_score(y, pred)),
        "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
        "binary_macro_f1": float(
            f1_score(binary_y, binary_p, average="macro", zero_division=0)
        ),
        "contact_macro_f1": contact_f1,
        "confusion_matrix_4class": confusion_matrix(
            y, pred, labels=[0, 1, 2, 3]
        ).tolist(),
        "per_class": classification_report(
            y,
            pred,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def metrics_from_predictions_csv(
    path: Path,
    y_col: str = "y",
    pred_col: str = "pred_y",
) -> dict[str, Any]:
    df = pd.read_csv(path)
    if y_col not in df.columns or pred_col not in df.columns:
        raise KeyError(f"{path} must contain columns {y_col!r} and {pred_col!r}")
    return metrics_from_y_pred(df[y_col].to_numpy(), df[pred_col].to_numpy())


def assert_close(
    got: float,
    expected: float,
    *,
    name: str,
    tol: float = 1e-9,
) -> None:
    if abs(float(got) - float(expected)) > tol:
        raise AssertionError(
            f"{name}: got {got!r} expected {expected!r} (tol={tol})"
        )
