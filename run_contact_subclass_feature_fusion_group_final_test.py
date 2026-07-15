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
REPORT = OUT / "audio_feature_benchmarks/contact_subclass_feature_fusion_group_selection"
NAMES = ["ambient", "leaf", "trunk", "twig"]
LABELS = np.arange(4)


def proba4(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, col]
    return suite.normalize(p)


def main() -> None:
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    cand = lock["selected_candidate"]
    if lock.get("test_loaded"):
        raise AssertionError("test already used")
    c = float(cand["C"])
    threshold = float(cand["contact_threshold"])
    X_hand = np.hstack([
        np.load(OUT / "audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"),
        np.load(OUT / "image_timm_features/efficientnet_b3.ra2_in1k/hand_train_full/X.npy"),
    ]).astype(np.float64)
    hand = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    y_hand = hand.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y_hand, np.load(OUT / "audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/y.npy").astype(np.int64)):
        raise AssertionError("hand alignment")
    m4 = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=900, random_state=42))]).fit(X_hand, y_hand)
    mb = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=900, random_state=42))]).fit(X_hand, (y_hand > 0).astype(int))
    mc = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=900, random_state=42))]).fit(X_hand[y_hand > 0], y_hand[y_hand > 0])

    # Open robot/test after the selection lock.
    raw_test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    test = suite.load_manifest(ROOT / "audio_visual_dataset_robo_default/dataset.csv", ROOT / "audio_visual_dataset_robo_default", "robot_test")
    X_test = np.hstack([
        np.load(OUT / "audio_feature_benchmarks/total240_trainval_select/features/robot_test/X.npy"),
        np.load(OUT / "image_timm_features/efficientnet_b3.ra2_in1k/robot_test/X.npy"),
    ]).astype(np.float64)
    y_test = test.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, np.load(OUT / "audio_feature_benchmarks/total240_trainval_select/features/robot_test/y.npy").astype(np.int64)):
        raise AssertionError("test alignment")
    p_direct = proba4(m4, X_test)
    p_bin = mb.predict_proba(X_test)
    p_contact = np.zeros((len(X_test), 3), dtype=np.float64)
    raw_contact = mc.predict_proba(X_test)
    for col, cls in enumerate(mc.classes_):
        p_contact[:, int(cls) - 1] = raw_contact[:, col]
    pred = p_direct.argmax(1)
    gate = p_bin[:, 1] >= threshold
    pred[gate] = p_contact[gate].argmax(1) + 1
    binary_y, binary_p = (y_test > 0).astype(int), (pred > 0).astype(int)
    result = {
        "split": "robot_test_final",
        "protocol": "locked_after_5fold_specimen_group_hand_only_selection",
        "n": int(len(y_test)),
        "locked_candidate": cand,
        "feature_set": "total240_audio + efficientnet_b3_image",
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
    (REPORT / "contact_subclass_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / "contact_subclass_final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=NAMES, columns=NAMES).to_csv(REPORT / "contact_subclass_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(REPORT / "contact_subclass_final_test_confusion_matrix_binary.csv")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
