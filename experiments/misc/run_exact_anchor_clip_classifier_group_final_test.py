from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

import run_multimodal_val_locked_suite as suite
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks/exact_anchor_clip_classifier_group_selection"
NAMES = ["ambient", "leaf", "trunk", "twig"]
LABELS = np.arange(4)


def aligned(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_): p[:, int(cls)] = raw[:, col]
    return suite.normalize(p)


def main() -> None:
    lock = json.loads((REPORT / "selection_lock.json").read_text())
    cand = lock["selected_candidate"]
    if lock.get("test_loaded"):
        raise AssertionError("test already used")
    X_hand = np.load(OUT / "image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy").astype(np.float32)
    hand = audio_base.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train")
    y_hand = hand.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y_hand, np.load(OUT / "image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/y.npy").astype(np.int64)):
        raise AssertionError("hand image alignment")
    model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.1, class_weight="balanced", random_state=42).fit(X_hand, y_hand)

    # Open robot/test only after the hand grouped lock.
    test = audio_base.load_manifest(ROOT / "audio_visual_dataset_robo_default/dataset.csv", "robot_test")
    raw_test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    X_test = np.load(OUT / "image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/X.npy").astype(np.float32)
    y_test = test.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, np.load(OUT / "image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/y.npy").astype(np.int64)):
        raise AssertionError("test image alignment")
    p_img = aligned(model, X_test)
    high = pd.read_csv(OUT / "audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv")
    pair = pd.read_csv(OUT / "audio_feature_benchmarks/audio_group_consistency_pair_blend_select/reports/audio_group_consistency_pair_blend_select_final_test_predictions.csv")
    for name, df in [("high", high), ("pair", pair)]:
        if not np.array_equal(df.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()): raise AssertionError(name)
    p_audio = lift.anchor_lift_proba(test, high[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64), pair[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64))
    bias = np.array([0.0, 0.0, float(cand["trunk_bias"]), float(cand["twig_bias"])])
    iw = float(cand["image_weight"])
    p = suite.normalize(np.exp((1-iw)*np.log(suite.normalize(p_audio)) + iw*np.log(suite.normalize(p_img)) + bias[None, :]))
    pred = p.argmax(1)
    by, bp = (y_test > 0).astype(int), (pred > 0).astype(int)
    result = {
        "split": "robot_test_final", "protocol": "locked_after_5fold_specimen_group_hand_only_classifier_selection", "n": int(len(y_test)), "locked_candidate": cand,
        "audio_pipeline": "anchor_lift_proba(highsr=0.8,pairwise=0.2)", "class_bias": bias.tolist(),
        "accuracy_4class": float(accuracy_score(y_test,pred)), "macro_precision_4class": float(precision_score(y_test,pred,average="macro",zero_division=0)), "macro_recall_4class": float(recall_score(y_test,pred,average="macro",zero_division=0)), "macro_f1_4class": float(f1_score(y_test,pred,average="macro",zero_division=0)), "weighted_f1_4class": float(f1_score(y_test,pred,average="weighted",zero_division=0)),
        "binary_accuracy_ambient_noambient": float(accuracy_score(by,bp)), "binary_macro_precision_ambient_noambient": float(precision_score(by,bp,average="macro",zero_division=0)), "binary_macro_recall_ambient_noambient": float(recall_score(by,bp,average="macro",zero_division=0)), "binary_macro_f1_ambient_noambient": float(f1_score(by,bp,average="macro",zero_division=0)),
        "per_class_4class": classification_report(y_test,pred,labels=LABELS,target_names=NAMES,output_dict=True,zero_division=0), "per_class_binary": classification_report(by,bp,labels=[0,1],target_names=["ambient","noambient"],output_dict=True,zero_division=0), "confusion_matrix_4class": confusion_matrix(y_test,pred,labels=LABELS).tolist(), "confusion_matrix_binary": confusion_matrix(by,bp,labels=[0,1]).tolist(),
    }
    (REPORT / "exact_anchor_clip_classifier_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,(dict,list))}]).to_csv(REPORT / "exact_anchor_clip_classifier_final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=NAMES, columns=NAMES).to_csv(REPORT / "exact_anchor_clip_classifier_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient","noambient"], columns=["ambient","noambient"]).to_csv(REPORT / "exact_anchor_clip_classifier_final_test_confusion_matrix_binary.csv")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__": main()
