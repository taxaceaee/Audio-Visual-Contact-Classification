from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "multimodal_stress_robust_group_selection"
NAMES = ["ambient", "leaf", "trunk", "twig"]
LABELS = np.arange(4)


def main():
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    cand = lock["selected_candidate"]
    full, train_idx, _ = suite.build_group_split(ROOT, REPORT)
    X_train, y_train = suite.image_cache(OUT, "hand_train_full")
    y = full.y.to_numpy(dtype=np.int64)
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X_train, y)
    raw_test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default" / "dataset.csv")
    test = suite.load_manifest(ROOT / "audio_visual_dataset_robo_default" / "dataset.csv", ROOT / "audio_visual_dataset_robo_default", "robot_test")
    X_test, y_test_cache = suite.image_cache(OUT, "robot_test")
    y_test = test.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, y_test_cache):
        raise AssertionError("Test labels are not aligned")
    raw = model.predict_proba(X_test)
    p_img = np.zeros((len(X_test), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p_img[:, int(cls)] = raw[:, col]
    p_img = suite.apply_bias(p_img)
    audio_frame = pd.read_csv(OUT / "audio_feature_benchmarks" / "audio_lift_source_blend_select" / "reports" / "audio_lift_source_blend_select_final_test_predictions.csv")
    if not np.array_equal(audio_frame.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
        raise AssertionError("Audio test probabilities are not aligned")
    p_audio = suite.normalize(audio_frame[suite.PROBA_COLUMNS].to_numpy())
    alpha = float(cand["audio_weight"])
    p = suite.normalize(np.exp(alpha * np.log(p_audio) + (1.0 - alpha) * np.log(p_img)))
    pred = p.argmax(axis=1)
    binary_y, binary_p = (y_test > 0).astype(int), (pred > 0).astype(int)
    result = {
        "split": "robot_test_final",
        "n": int(len(y_test)),
        "locked_candidate": cand,
        "accuracy_4class": float(accuracy_score(y_test, pred)),
        "macro_precision_4class": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1_4class": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "binary_accuracy_ambient_noambient": float(accuracy_score(binary_y, binary_p)),
        "binary_macro_precision_ambient_noambient": float(precision_score(binary_y, binary_p, average="macro", zero_division=0)),
        "binary_macro_recall_ambient_noambient": float(recall_score(binary_y, binary_p, average="macro", zero_division=0)),
        "binary_macro_f1_ambient_noambient": float(f1_score(binary_y, binary_p, average="macro", zero_division=0)),
        "per_class_4class": classification_report(y_test, pred, labels=LABELS, target_names=NAMES, output_dict=True, zero_division=0),
        "per_class_binary": classification_report(binary_y, binary_p, labels=[0, 1], target_names=["ambient", "noambient"], output_dict=True, zero_division=0),
        "confusion_matrix_4class": confusion_matrix(y_test, pred, labels=LABELS).tolist(),
        "confusion_matrix_binary": confusion_matrix(binary_y, binary_p, labels=[0, 1]).tolist(),
    }
    (REPORT / "stress_robust_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / "stress_robust_final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=NAMES, columns=NAMES).to_csv(REPORT / "stress_robust_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(REPORT / "stress_robust_final_test_confusion_matrix_binary.csv")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
