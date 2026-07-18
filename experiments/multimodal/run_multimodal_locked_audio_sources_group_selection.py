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

import run_multimodal_stress_robust_group_selection as image_base
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as audio_broad
import train_val_select_final_test as audio_base


OUT = Path("outputs")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
LABELS = np.arange(4, dtype=np.int64)


def main() -> None:
    report = OUT / "audio_feature_benchmarks/multimodal_locked_audio_sources_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    frame = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    y = frame.y.to_numpy(dtype=np.int64)
    X_img, y_img = suite.image_cache(OUT, "hand_train_full")
    if not np.array_equal(y, y_img):
        raise AssertionError("image/audio hand rows are not aligned")

    groups = frame.specimen_group.to_numpy()
    split = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))
    fold_assignment = np.full(len(y), -1, dtype=np.int64)
    summaries = []
    for fid, (tr, va) in enumerate(split):
        fold_assignment[va] = fid
        summaries.append({"fold": fid, "train_rows": len(tr), "val_rows": len(va), "train_groups": int(pd.Series(groups[tr]).nunique()), "val_groups": int(pd.Series(groups[va]).nunique()), "overlap": len(set(groups[tr]) & set(groups[va]))})
    if np.any(fold_assignment < 0) or any(x["overlap"] for x in summaries):
        raise AssertionError("invalid specimen grouped folds")
    (report / "fold_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # Load only locked audio OOF sources. This helper never opens robot/test.
    audio_base.CONFIG["random_state"] = 42
    audio_base.configure_feature_set("total240")
    sources = audio_broad.load_train_sources(OUT, y, fold_assignment, 42)
    sources = {name: suite.normalize(value) for name, value in sources.items()}
    if any(value.shape != (len(y), 4) for value in sources.values()):
        raise AssertionError("bad audio source shape")

    image_oof = np.zeros((len(y), 4), dtype=np.float64)
    for tr, va in split:
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X_img[tr], y[tr])
        image_oof[va] = image_base.image_proba(model, X_img[va])

    def score_by_fold(pred: np.ndarray) -> tuple[float, float, float]:
        vals = [f1_score(y[va], pred[va], average="macro", zero_division=0) for _, va in split]
        return float(np.mean(vals)), float(np.min(vals)), float(np.std(vals))

    candidates: dict[str, np.ndarray] = dict(sources)
    source_names = list(sources)
    # Add transparent pairwise audio combinations; weights are selected on
    # hand grouped folds only and are reused unchanged at final evaluation.
    for i, left in enumerate(source_names):
        for right in source_names[i + 1:]:
            if (left, right) in [("report_gate_onehot", "report_gate_proba")]:
                continue
            for weight in np.linspace(0.2, 0.8, 7):
                candidates[f"{left}+{right}@{weight:.1f}"] = suite.normalize(weight * sources[left] + (1.0 - weight) * sources[right])

    rows = []
    for audio_name, pa in candidates.items():
        for iw in np.linspace(0.0, 1.0, 21):
            for kind in ["linear", "log"]:
                if kind == "linear":
                    p = suite.normalize((1.0 - iw) * pa + iw * image_oof)
                else:
                    p = suite.normalize(np.exp((1.0 - iw) * np.log(suite.normalize(pa)) + iw * np.log(suite.normalize(image_oof))))
                mean, worst, std = score_by_fold(p.argmax(axis=1))
                rows.append({"audio_candidate": audio_name, "image_weight": float(iw), "kind": kind, "mean_fold_macro_f1": mean, "worst_fold_macro_f1": worst, "std_fold_macro_f1": std})
    leaderboard = pd.DataFrame(rows).sort_values(["worst_fold_macro_f1", "mean_fold_macro_f1", "std_fold_macro_f1"], ascending=[False, False, True]).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_locked_audio_multimodal_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "multimodal_locked_audio_sources_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "source_names": sorted(sources), "candidate_count": len(rows), "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
