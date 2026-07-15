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
REPORT = OUT / "audio_feature_benchmarks/multimodal_broad_audio_ensemble_group_selection"
NAMES = ["ambient", "leaf", "trunk", "twig"]
LABELS = np.arange(4)


def main() -> None:
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    if lock.get("test_loaded"):
        raise AssertionError("selection lock already used test data")
    cand = lock["selected_candidate"]
    if cand["audio_sources"] != "highsr_hgb+pairwise":
        raise AssertionError(f"unexpected locked candidate: {cand}")
    source_weight = float(cand["source_weight_first"])
    image_weight = float(cand["image_weight"])

    # Fit image branch on all hand rows only after selection is locked.
    hand = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    X_hand, y_hand_cache = suite.image_cache(OUT, "hand_train_full")
    y_hand = hand.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y_hand, y_hand_cache):
        raise AssertionError("hand labels are not aligned")
    image_model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X_hand, y_hand)

    # Only now open the robot/test manifest and final prediction caches.
    raw_test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    test = suite.load_manifest(ROOT / "audio_visual_dataset_robo_default/dataset.csv", ROOT / "audio_visual_dataset_robo_default", "robot_test")
    X_test, y_test_cache = suite.image_cache(OUT, "robot_test")
    y_test = test.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, y_test_cache):
        raise AssertionError("test labels are not aligned")
    raw_img = image_model.predict_proba(X_test)
    p_img = np.zeros((len(X_test), 4), dtype=np.float64)
    for col, cls in enumerate(image_model.classes_):
        p_img[:, int(cls)] = raw_img[:, col]
    p_img = suite.apply_bias(p_img)

    audio_paths = {
        "highsr_hgb": OUT / "audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv",
        "pairwise": OUT / "audio_feature_benchmarks/audio_group_consistency_pair_blend_select/reports/audio_group_consistency_pair_blend_select_final_test_predictions.csv",
    }
    audio_probs = {}
    for name, path in audio_paths.items():
        df = pd.read_csv(path)
        if not np.array_equal(df.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
            raise AssertionError(f"audio alignment failed for {name}")
        audio_probs[name] = suite.normalize(df[suite.PROBA_COLUMNS].to_numpy())
    p_audio = suite.normalize(source_weight * audio_probs["highsr_hgb"] + (1.0 - source_weight) * audio_probs["pairwise"])
    if cand["kind"] == "log":
        p = suite.normalize(np.exp((1.0 - image_weight) * np.log(suite.normalize(p_audio)) + image_weight * np.log(suite.normalize(p_img))))
    else:
        p = suite.normalize((1.0 - image_weight) * p_audio + image_weight * p_img)
    pred = p.argmax(axis=1)
    binary_y, binary_p = (y_test > 0).astype(int), (pred > 0).astype(int)
    result = {
        "split": "robot_test_final",
        "protocol": "locked_after_5fold_stratified_group_selection",
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
    (REPORT / "broad_audio_ensemble_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / "broad_audio_ensemble_final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=NAMES, columns=NAMES).to_csv(REPORT / "broad_audio_ensemble_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(REPORT / "broad_audio_ensemble_final_test_confusion_matrix_binary.csv")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
