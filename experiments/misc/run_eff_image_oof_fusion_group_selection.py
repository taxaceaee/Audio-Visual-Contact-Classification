"""Hand-only grouped OOF experiment: add EfficientNet image probabilities.

This is deliberately separate from the locked v2 baseline.  It never loads
robot/test; the output lock is the only input allowed to the final evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs/audio_feature_benchmarks/eff_image_oof_fusion_group_selection")
BASE = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
EFF = Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k")


def seg(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)


def spec(s: pd.Series) -> pd.Series:
    return seg(s).str.replace(r"_segment_.*$", "", regex=True)


def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 1e-12)
    return x / x.sum(1, keepdims=True)


def macro(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for c in range(4):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        vals.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(vals))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep this selector independent of the multimodal runtime imports.  The
    # manifest is the only source of labels and is hand/default only.
    fr = pd.read_csv(ROOT / "audio_visual_dataset_default/dataset.csv")
    yrow = pd.Categorical(fr["category"], categories=["ambient", "leaf", "trunk", "twig"]).codes.astype(np.int64)
    keys = seg(fr["audio_file"]).to_numpy()
    groups = spec(fr["audio_file"]).to_numpy()
    useg, codes = np.unique(keys, return_inverse=True)
    y = np.array([yrow[codes == i][0] for i in range(len(useg))], dtype=np.int64)
    g = np.array([groups[codes == i][0] for i in range(len(useg))])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(y)), y, g))

    # These are the already-generated, hand-only nested OOF bases of v2.
    p_meta = norm(np.load(BASE / "hand_oof_meta.npy"))
    p_hier = norm(np.load(BASE / "hand_oof_hier.npy"))
    y_saved = np.load(BASE / "hand_stack_oof_y.npy")
    assert np.array_equal(y, y_saved)

    x = np.load(EFF / "hand_train_full/X.npy").astype(np.float32)
    sx = np.vstack([x[codes == i].mean(0) for i in range(len(useg))])
    p_eff = np.zeros((len(y), 4), dtype=np.float64)
    for fold, (tr, va) in enumerate(folds):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.01, solver="liblinear", multi_class="ovr", max_iter=500,
                class_weight="balanced", random_state=42,
            ),
        )
        model.fit(sx[tr], y[tr])
        p_eff[va] = norm(model.predict_proba(sx[va]))
        print(f"fold {fold + 1}/5 done", flush=True)

    rows = []
    # Use fixed grids, select by worst grouped fold then mean grouped fold.
    for wh in np.linspace(0.0, 0.8, 9):
        for we in np.linspace(0.0, 0.8, 9):
            if wh + we > 1.0:
                continue
            wm = 1.0 - wh - we
            p = norm(wm * p_meta + wh * p_hier + we * p_eff)
            scores = [macro(y[va], p[va].argmax(1)) for _, va in folds]
            rows.append({"meta_weight": wm, "hier_weight": wh, "eff_weight": we,
                         "mean_cv_macro_f1": float(np.mean(scores)),
                         "worst_cv_macro_f1": float(np.min(scores))})

    # A small stacked LR over OOF probability/log-probability features.
    z = np.hstack([
        np.log(np.clip(p_meta, 1e-6, 1)), np.log(np.clip(p_hier, 1e-6, 1)),
        np.log(np.clip(p_eff, 1e-6, 1)), p_meta, p_hier, p_eff,
    ]).astype(np.float32)
    board = pd.DataFrame(rows).sort_values(
        ["worst_cv_macro_f1", "mean_cv_macro_f1"], ascending=False
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_oof_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    np.save(OUT / "hand_oof_eff.npy", p_eff.astype(np.float32))
    lock = {
        "protocol": "eff_image_probability_fusion_group_oof",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "base_oof_sources": [str(BASE / "hand_oof_meta.npy"), str(BASE / "hand_oof_hier.npy")],
        "candidate": best,
        "test_loaded": False,
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
