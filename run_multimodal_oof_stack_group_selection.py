from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = p.argmax(axis=1)
    by, bp = (y > 0).astype(int), (pred > 0).astype(int)
    return {
        "accuracy_4class": float(accuracy_score(y, pred)),
        "macro_precision_4class": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
        "binary_macro_f1": float(f1_score(by, bp, average="macro", zero_division=0)),
    }


def image_proba(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, col]
    return suite.apply_bias(p)


def meta_features(pa: np.ndarray, pi: np.ndarray) -> np.ndarray:
    pa = suite.normalize(pa)
    pi = suite.normalize(pi)
    entropy_a = -np.sum(pa * np.log(pa), axis=1, keepdims=True)
    entropy_i = -np.sum(pi * np.log(pi), axis=1, keepdims=True)
    return np.hstack([np.log(pa), np.log(pi), pa, pi, entropy_a, entropy_i])


def main() -> None:
    output = Path("outputs")
    report = output / "audio_feature_benchmarks" / "multimodal_oof_stack_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    root = Path("/home/ttung05/Desktop/tree_base/tree_structures")
    full, train_idx, val_idx = suite.build_group_split(root, report)
    _, pa_full = suite.load_audio_oof(root, output)
    X_img, y_img = suite.image_cache(output, "hand_train_full")
    y = full.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y, y_img):
        raise AssertionError("Image labels do not align")

    # Cross-fit image probabilities inside group-train only.
    inner_groups = full.specimen_group.to_numpy()
    inner_assign = inner_groups[train_idx]
    inner_y = y[train_idx]
    image_oof = np.zeros((len(train_idx), 4), dtype=np.float64)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
    for inner_train, inner_val in inner.split(np.arange(len(train_idx)), inner_y, inner_assign):
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))])
        model.fit(X_img[train_idx[inner_train]], inner_y[inner_train])
        image_oof[inner_val] = image_proba(model, X_img[train_idx[inner_val]])
    full_image_model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))])
    full_image_model.fit(X_img[train_idx], y[train_idx])
    image_val = image_proba(full_image_model, X_img[val_idx])

    X_meta_train = meta_features(pa_full[train_idx], image_oof)
    X_meta_val = meta_features(pa_full[val_idx], image_val)
    rows = []
    models = []
    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        for cw in [None, "balanced"]:
            models.append((f"meta_logreg_c{c:g}_{cw or 'none'}", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight=cw, max_iter=2000, random_state=42))])))
    for name, model in models:
        model.fit(X_meta_train, y[train_idx])
        raw = model.predict_proba(X_meta_val)
        p = np.zeros((len(val_idx), 4), dtype=np.float64)
        for col, cls in enumerate(model.classes_):
            p[:, int(cls)] = raw[:, col]
        row = {"model": name, **metrics(y[val_idx], p)}
        rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values(["macro_f1_4class", "binary_macro_f1", "accuracy_4class"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_group_val_oof_stack_leaderboard.csv", index=False)
    lock = {"protocol": "multimodal_oof_stack_group_val_only", "group_column": "specimen_group", "group_overlap": 0, "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "test_loaded": False, "candidate_count": int(len(rows)), "best_val_candidate": leaderboard.iloc[0].to_dict()}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
