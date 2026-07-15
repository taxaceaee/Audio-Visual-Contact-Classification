from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch

import run_finetuned_image_group_fusion as impl
import run_multimodal_val_locked_suite as suite


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "finetuned_image_group_fusion"
NAMES = ["ambient", "leaf", "trunk", "twig"]
LABELS = np.arange(4)


def main():
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_train = pd.read_csv(ROOT / "audio_visual_dataset_default" / "dataset.csv")
    raw_test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default" / "dataset.csv")
    train = suite.load_manifest(ROOT / "audio_visual_dataset_default" / "dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    test = suite.load_manifest(ROOT / "audio_visual_dataset_robo_default" / "dataset.csv", ROOT / "audio_visual_dataset_robo_default", "robot_test")
    model = impl.make_model(device)
    selected = torch.load(REPORT / "selected_group_val_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    # Test is intentionally first accessed here, after the selection lock and full retraining.
    p_img = impl.predict(model, test, device)
    audio_path = OUT / "audio_feature_benchmarks" / "audio_lift_source_blend_select" / "reports" / "audio_lift_source_blend_select_final_test_predictions.csv"
    audio = pd.read_csv(audio_path)
    if not np.array_equal(audio.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
        raise AssertionError("Audio test rows are not aligned")
    p_audio = suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy())
    alpha = float(lock["selected_audio_weight"])
    p = suite.normalize(np.exp(alpha * np.log(p_audio) + (1 - alpha) * np.log(p_img)))
    y = test.y.to_numpy(dtype=np.int64)
    pred = p.argmax(axis=1)
    by, bp = (y > 0).astype(int), (pred > 0).astype(int)
    result = {
        "split": "robot_test_final", "n": int(len(y)), "locked_candidate": lock,
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
    (REPORT / "finetuned_group_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / "finetuned_group_final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=NAMES, columns=NAMES).to_csv(REPORT / "finetuned_group_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(REPORT / "finetuned_group_final_test_confusion_matrix_binary.csv")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
