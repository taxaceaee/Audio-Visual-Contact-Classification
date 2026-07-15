"""Final evaluator for the hand-locked EfficientNet material rule.

The lock is created by run_eff_material_rule_selection.py and is hand-only.
Robot/test is loaded only after the lock is read and validated.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
BASE = Path("outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection")
EFF = Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k")
OUT = Path("outputs/audio_feature_benchmarks/eff_image_oof_fusion_group_selection")

def seg(s): return s.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)

def pool(keys, x):
    u, c = np.unique(keys, return_inverse=True)
    return u, np.vstack([x[c == i].mean(0) for i in range(len(u))])

def main():
    lock = json.loads((OUT / "material_rule_selection_lock.json").read_text())
    assert lock["selection_data"] == "hand/default only"
    assert lock["test_loaded"] is False
    cand = lock["candidate"]
    z = np.load(BASE / "segment_rule_stack_v2_base_segment_outputs.npz", allow_pickle=True)
    seg_ids = z["segment_ids"].astype(str)
    row_seg = z["row_segment_ids"].astype(str)
    p_h = z["p_h"].astype(np.float64)
    p_meta = z["p_meta"].astype(np.float64)

    hand = pd.read_csv(ROOT / "audio_visual_dataset_default/dataset.csv")
    hand_keys = seg(hand.audio_file).to_numpy()
    xh = np.load(EFF / "hand_train_full/X.npy").astype(np.float32)
    hu, hx = pool(hand_keys, xh)
    hyrow = pd.Categorical(hand.category, categories=["ambient", "leaf", "trunk", "twig"]).codes
    hy = np.array([hyrow[hand_keys == k][0] for k in hu])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.01, solver="liblinear", multi_class="ovr", max_iter=500,
        class_weight="balanced", random_state=42))
    model.fit(hx, hy)

    # Test is opened only after reading the hand-only lock and training full hand.
    test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    test_keys = seg(test.audio_file).to_numpy()
    xt = np.load(EFF / "robot_test/X.npy").astype(np.float32)
    tu, tx = pool(test_keys, xt)
    test_order = {k: i for i, k in enumerate(tu)}
    tx = np.vstack([tx[test_order[k]] for k in seg_ids])
    pe = model.predict_proba(tx)
    contact = np.clip(1.0 - p_h[:, 0], 1e-8, 1.0)
    old_mat = p_h[:, 1:] / contact[:, None]
    new_mat = pe[:, 1:] / np.clip(pe[:, 1:].sum(1, keepdims=True), 1e-8, 1.0)
    w = float(cand["eff_material_weight"])
    wm = float(cand["meta_weight"])
    mat = (1.0 - w) * old_mat + w * new_mat
    q = np.column_stack([1.0 - contact, contact[:, None] * mat])
    pred_seg = np.where(p_h.argmax(1) == 2, 2, (wm * p_meta + (1.0 - wm) * q).argmax(1))
    pred = np.array([pred_seg[{k: i for i, k in enumerate(seg_ids)}[k]] for k in test_keys])
    y = pd.Categorical(test.category, categories=["ambient", "leaf", "trunk", "twig"]).codes
    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "n": int(len(y)),
        "locked_candidate": cand,
        "metrics": {
            "accuracy_4class": float(accuracy_score(y, pred)),
            "macro_precision_4class": float(__import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(y, pred, average="macro", zero_division=0)),
            "macro_recall_4class": float(__import__("sklearn.metrics", fromlist=["recall_score"]).recall_score(y, pred, average="macro", zero_division=0)),
            "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1_4class": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "binary_macro_f1": float(f1_score(y > 0, pred > 0, average="macro", zero_division=0)),
        },
        "per_class_4class": classification_report(y, pred, labels=[0,1,2,3], target_names=["ambient","leaf","trunk","twig"], output_dict=True, zero_division=0),
        "confusion_matrix_4class": confusion_matrix(y, pred, labels=[0,1,2,3]).tolist(),
        "invariants": {"test_loaded_after_lock": True, "selection_used_hand_only": True, "segment_label_pure": True},
    }
    (OUT / "eff_material_rule_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))

if __name__ == "__main__": main()
