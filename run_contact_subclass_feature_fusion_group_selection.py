from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


OUT = Path("outputs")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")


def aligned(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, col]
    return suite.normalize(p)


def main() -> None:
    report = OUT / "audio_feature_benchmarks/contact_subclass_feature_fusion_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    frame = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    y = frame.y.to_numpy(dtype=np.int64)
    audio = np.load(OUT / "audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy").astype(np.float64)
    image = np.load(OUT / "image_timm_features/efficientnet_b3.ra2_in1k/hand_train_full/X.npy").astype(np.float64)
    if not np.array_equal(y, np.load(OUT / "audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/y.npy").astype(np.int64)):
        raise AssertionError("audio alignment")
    if not np.array_equal(y, np.load(OUT / "image_timm_features/efficientnet_b3.ra2_in1k/hand_train_full/y.npy").astype(np.int64)):
        raise AssertionError("image alignment")
    X = np.hstack([audio, image])
    groups = frame.specimen_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))
    rows = []
    for c in [0.01, 0.03, 0.1, 0.3, 1.0]:
        direct = np.zeros((len(y), 4), dtype=np.float64)
        binary = np.zeros((len(y), 2), dtype=np.float64)
        contact = np.zeros((len(y), 3), dtype=np.float64)
        for tr, va in folds:
            m4 = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=700, random_state=42))]).fit(X[tr], y[tr])
            mb = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=700, random_state=42))]).fit(X[tr], (y[tr] > 0).astype(int))
            mc = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=700, random_state=42))]).fit(X[tr][y[tr] > 0], y[tr][y[tr] > 0])
            direct[va] = aligned(m4, X[va])
            binary[va] = mb.predict_proba(X[va])
            rawc = mc.predict_proba(X[va])
            for col, cls in enumerate(mc.classes_):
                contact[va, int(cls) - 1] = rawc[:, col]
        for threshold in np.linspace(0.25, 0.75, 21):
            pred = direct.argmax(1)
            gate = binary[:, 1] >= threshold
            pred[gate] = contact[gate].argmax(1) + 1
            fs = [f1_score(y[va], pred[va], average="macro", zero_division=0) for _, va in folds]
            rows.append({"model": "hierarchical_feature_logreg", "C": c, "contact_threshold": float(threshold), "mean_fold_macro_f1": float(np.mean(fs)), "worst_fold_macro_f1": float(np.min(fs)), "std_fold_macro_f1": float(np.std(fs))})
        for threshold in np.linspace(0.25, 0.75, 21):
            pred = direct.argmax(1)
            gate = binary[:, 1] >= threshold
            pred[gate] = contact[gate].argmax(1) + 1
            # Keep the binary model's ambient decision for low contact mass.
            pred[~gate] = 0
            fs = [f1_score(y[va], pred[va], average="macro", zero_division=0) for _, va in folds]
            rows.append({"model": "binary_gate_contact_logreg", "C": c, "contact_threshold": float(threshold), "mean_fold_macro_f1": float(np.mean(fs)), "worst_fold_macro_f1": float(np.min(fs)), "std_fold_macro_f1": float(np.std(fs))})
    leaderboard = pd.DataFrame(rows).sort_values(["worst_fold_macro_f1", "mean_fold_macro_f1", "std_fold_macro_f1"], ascending=[False, False, True]).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_contact_subclass_feature_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "contact_subclass_audio_total240_plus_efficientnet_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
