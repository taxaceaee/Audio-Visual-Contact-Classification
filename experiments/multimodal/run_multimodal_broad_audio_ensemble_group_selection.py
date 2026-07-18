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

import run_multimodal_stress_robust_group_selection as base
import run_multimodal_val_locked_suite as suite


OUT = Path("outputs")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
VIEWS = ["clean", "robot_mix", "bandlimit"]


def norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


def load_candidates() -> dict[str, dict[str, np.ndarray]]:
    specs = {
        "highsr_hgb": OUT / "audio_feature_benchmarks/audio_highsr_temporal_tta_select/oof_proba/highsr_hgb_default__all_aug",
        "highsr_extratrees": OUT / "audio_feature_benchmarks/audio_highsr_temporal_extratrees_select/oof_proba/highsr_extratrees__all_aug",
        "highsr_regularized": OUT / "audio_feature_benchmarks/audio_highsr_temporal_hgb_regularized_select/oof_proba/highsr_hgb_regularized__all_aug",
        "mfcc40": OUT / "audio_feature_benchmarks/audio_mfcc40_tta_grid_hgb_select/oof_proba/grid_hgb_default__all_aug",
        "total120": OUT / "audio_feature_benchmarks/audio_total120_tta_grid_hgb_select/oof_proba/grid_hgb_default__all_aug",
        "total240": OUT / "audio_feature_benchmarks/audio_tta_grid_hgb_select/oof_proba/grid_hgb_default__all_aug",
    }
    result = {}
    for name, folder in specs.items():
        result[name] = {view: norm(np.load(folder / f"{view}_oof_proba.npy")) for view in VIEWS}
    # Pairwise source consistency is a complementary source, not an augmented
    # acoustic classifier. Reuse it in every stress evaluation as its OOF
    # predictions are only available in the clean domain.
    pair = norm(np.load(OUT / "audio_feature_benchmarks/audio_group_consistency_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy"))
    result["pairwise"] = {view: pair for view in VIEWS}
    return result


def main() -> None:
    report = OUT / "audio_feature_benchmarks/multimodal_broad_audio_ensemble_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    frame = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    y = frame.y.to_numpy(dtype=np.int64)
    X_img, y_img = suite.image_cache(OUT, "hand_train_full")
    if not np.array_equal(y, y_img):
        raise AssertionError("image/audio hand rows are not aligned")
    audio_sources = load_candidates()
    n = len(frame)
    if any(next(iter(x.values())).shape[0] != n for x in audio_sources.values()):
        raise AssertionError("OOF source length mismatch")

    groups = frame.specimen_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(n), y, groups))
    summary = [{"fold": i, "train_rows": len(tr), "val_rows": len(va), "train_groups": int(pd.Series(groups[tr]).nunique()), "val_groups": int(pd.Series(groups[va]).nunique()), "overlap": len(set(groups[tr]) & set(groups[va]))} for i, (tr, va) in enumerate(folds)]
    if any(x["overlap"] for x in summary):
        raise AssertionError("group overlap")
    (report / "fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # OOF image probabilities: one classifier per specimen fold.
    image_oof = np.zeros((n, 4), dtype=np.float64)
    for train_idx, val_idx in folds:
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X_img[train_idx], y[train_idx])
        image_oof[val_idx] = base.image_proba(model, X_img[val_idx])

    # Lift each audio window to segment-level while preserving the audio-only
    # pipeline's aggregation rule. Since every specimen is in one fold, the
    # aggregation cannot mix labels across train and validation groups.
    lifted = {}
    for source, views in audio_sources.items():
        lifted[source] = {view: suite.segment_lift(frame.audio_file, p) for view, p in views.items()}

    rows = []
    names = list(audio_sources)
    # Single-source and pairwise blends. Restrict the grid to three sources so
    # this remains a transparent finite candidate family, not a test search.
    blend_sets = [(name,) for name in names]
    blend_sets += [("highsr_hgb", "highsr_extratrees"), ("highsr_hgb", "mfcc40"), ("highsr_hgb", "total120"), ("highsr_hgb", "pairwise"), ("highsr_regularized", "mfcc40"), ("highsr_extratrees", "mfcc40")]
    for blend in blend_sets:
        for source_weight in np.linspace(0.0, 1.0, 11):
            if len(blend) == 1 and source_weight != 1.0:
                continue
            if len(blend) == 1:
                source_p = {v: lifted[blend[0]][v] for v in VIEWS}
            else:
                a, b = blend
                source_p = {v: norm(source_weight * lifted[a][v] + (1.0 - source_weight) * lifted[b][v]) for v in VIEWS}
            for image_weight in np.linspace(0.0, 1.0, 21):
                for kind in ["linear", "log"]:
                    view_scores = {v: [] for v in VIEWS}
                    for train_idx, val_idx in folds:
                        for view in VIEWS:
                            pa, pi = source_p[view][val_idx], image_oof[val_idx]
                            if kind == "linear":
                                p = norm((1.0 - image_weight) * pa + image_weight * pi)
                            else:
                                p = norm(np.exp((1.0 - image_weight) * np.log(norm(pa)) + image_weight * np.log(norm(pi))))
                            view_scores[view].append(float(f1_score(y[val_idx], p.argmax(axis=1), average="macro", zero_division=0)))
                    row = {"audio_sources": "+".join(blend), "source_weight_first": float(source_weight), "image_weight": float(image_weight), "kind": kind}
                    for view in VIEWS:
                        row[f"mean_{view}"] = float(np.mean(view_scores[view]))
                        row[f"worst_fold_{view}"] = float(np.min(view_scores[view]))
                    row["mean_across_views"] = float(np.mean([row[f"mean_{v}"] for v in VIEWS]))
                    row["worst_view_fold"] = float(np.min([row[f"worst_fold_{v}"] for v in VIEWS]))
                    rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values(["worst_view_fold", "mean_across_views"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_broad_audio_ensemble_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "multimodal_broad_audio_ensemble_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "candidate_count": len(rows), "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
