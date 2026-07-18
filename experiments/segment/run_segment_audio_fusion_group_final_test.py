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
from run_segment_audio_fusion_group_selection import segment_only


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "segment_audio_fusion_group_selection"
NAMES = ["ambient", "leaf", "trunk", "twig"]
LABELS = np.arange(4)


def main() -> None:
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    cand = lock["selected_candidate"]
    hand = suite.load_manifest(ROOT / "audio_visual_dataset_default" / "dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    test = suite.load_manifest(ROOT / "audio_visual_dataset_robo_default" / "dataset.csv", ROOT / "audio_visual_dataset_robo_default", "robot_test")
    X_train, y_train = suite.image_cache(OUT, "hand_train_full")
    X_test, y_test_cache = suite.image_cache(OUT, "robot_test")
    if not np.array_equal(hand.y.to_numpy(), y_train) or not np.array_equal(test.y.to_numpy(), y_test_cache):
        raise AssertionError("Image cache labels are not aligned")
    image_model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X_train, y_train)
    raw_image = image_model.predict_proba(X_test)
    p_image = np.zeros((len(X_test), 4), dtype=np.float64)
    for col, cls in enumerate(image_model.classes_):
        p_image[:, int(cls)] = raw_image[:, col]
    p_image = suite.apply_bias(p_image)

    high = pd.read_csv(OUT / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select" / "reports" / "audio_highsr_temporal_tta_select_final_test_predictions.csv")
    pair = pd.read_csv(OUT / "audio_feature_benchmarks" / "audio_pairwise_contact_stress_cv_select" / "reports" / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv")
    for frame in [high, pair]:
        if not np.array_equal(frame.audio_file.astype(str).to_numpy(), test.audio_file.astype(str).to_numpy()):
            raise AssertionError("Audio final rows are not aligned")
    p_audio_window = suite.normalize(0.8 * high[suite.PROBA_COLUMNS].to_numpy() + 0.2 * pair[suite.PROBA_COLUMNS].to_numpy())
    p_audio = segment_only(test.audio_file, p_audio_window)
    alpha = float(cand["audio_weight"])
    p = suite.normalize(np.exp(alpha * np.log(p_audio) + (1.0 - alpha) * np.log(p_image)))
    y = test.y.to_numpy(dtype=np.int64)
    pred = p.argmax(axis=1)
    by, bp = (y > 0).astype(int), (pred > 0).astype(int)
    result = {
        "split": "robot_test_final", "n": int(len(y)), "locked_candidate": cand,
        "accuracy_4class": float(accuracy_score(y, pred)),
        "macro_precision_4class": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1_4class": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "binary_accuracy_ambient_noambient": float(accuracy_score(by, bp)),
        "binary_macro_precision_ambient_noambient": float(precision_score(by, bp, average="macro", zero_division=0)),
        "binary_macro_recall_ambient_noambient": float(recall_score(by, bp, average="macro", zero_division=0)),
        "binary_macro_f1_ambient_noambient": float(f1_score(by, bp, average="macro", zero_division=0)),
        "per_class_4class": classification_report(y, pred, labels=LABELS, target_names=NAMES, output_dict=True, zero_division=0),
        "per_class_binary": classification_report(by, bp, labels=[0, 1], target_names=["ambient", "noambient"], output_dict=True, zero_division=0),
        "confusion_matrix_4class": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "confusion_matrix_binary": confusion_matrix(by, bp, labels=[0, 1]).tolist(),
    }
    (REPORT / "segment_audio_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / "segment_audio_final_test_report.csv", index=False)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
