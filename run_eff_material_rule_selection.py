"""Select the EfficientNet material replacement rule on hand grouped OOF only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
EFF = Path("outputs/audio_feature_benchmarks/eff_image_oof_fusion_group_selection")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")

def f1(y, p):
    vals = []
    for c in range(4):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        vals.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(vals))

def main():
    h = np.load(BASE / "hand_oof_hier.npy")
    m = np.load(BASE / "hand_oof_meta.npy")
    e = np.load(EFF / "hand_oof_eff.npy")
    y = np.load(BASE / "hand_stack_oof_y.npy")
    # Reconstruct exactly the segment order used by the OOF cache.
    fr = pd.read_csv(ROOT / "audio_visual_dataset_default/dataset.csv")
    row_y = pd.Categorical(fr.category, categories=["ambient", "leaf", "trunk", "twig"]).codes
    key = fr.audio_file.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)
    group = key.str.replace(r"_segment_.*$", "", regex=True)
    useg, code = np.unique(key.to_numpy(), return_inverse=True)
    yg = np.array([row_y[code == i][0] for i in range(len(useg))])
    gg = np.array([group.to_numpy()[code == i][0] for i in range(len(useg))])
    assert np.array_equal(yg, y)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(y)), y, gg))
    contact = np.clip(1.0 - h[:, 0], 1e-8, 1.0)
    old_mat = h[:, 1:] / contact[:, None]
    new_mat = e[:, 1:] / np.clip(e[:, 1:].sum(1, keepdims=True), 1e-8, 1.0)
    rows = []
    for meta_weight in np.linspace(0.0, 1.0, 11):
        for eff_material_weight in np.linspace(0.0, 1.0, 11):
            mat = (1.0 - eff_material_weight) * old_mat + eff_material_weight * new_mat
            q = np.column_stack([1.0 - contact, contact[:, None] * mat])
            pred = np.where(h.argmax(1) == 2, 2, (meta_weight * m + (1.0 - meta_weight) * q).argmax(1))
            scores = [f1(y[va], pred[va]) for _, va in folds]
            rows.append({"meta_weight": float(meta_weight), "eff_material_weight": float(eff_material_weight),
                         "mean_cv_macro_f1": float(np.mean(scores)), "worst_cv_macro_f1": float(np.min(scores))})
    board = pd.DataFrame(rows).sort_values(["worst_cv_macro_f1", "mean_cv_macro_f1"], ascending=False)
    board.to_csv(EFF / "material_rule_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    lock = {"protocol": "effnet_material_rule_grouped_hand_oof", "selection_data": "hand/default only",
            "group_column": "specimen_group", "candidate": best, "test_loaded": False}
    (EFF / "material_rule_selection_lock.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))
    print(board.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
