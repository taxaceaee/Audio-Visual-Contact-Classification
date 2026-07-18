from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lightgbm import LGBMClassifier

import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")


@dataclass(frozen=True)
class SegmentCandidate:
    name: str
    kind: str
    model_kind: str = "direct"
    sample_mode: str = "window"
    train_views: tuple[str, ...] = STRESS_VIEWS
    direct_factory: Callable[[], object] | None = None
    binary_factory: Callable[[], object] | None = None
    contact_factory: Callable[[], object] | None = None
    members: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only segment/prototype stress-CV selection. Uses only hand/default "
            "audio features for selection; robot/test is loaded after the selection lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--clean-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument("--force-rebuild-stress", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def label_counts_from_y(y: np.ndarray) -> dict[str, int]:
    return {base.ID2LABEL[int(label)]: int((y == label).sum()) for label in LABELS}


def reduce_by_segment(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode not in {"segment_mean", "segment_median"}:
        raise ValueError(f"Unsupported segment reducer: {mode}")

    group_frame = pd.DataFrame({"group": groups, "y": y})
    rows = []
    labels = []
    counts = []
    for _, positions in group_frame.groupby("group", sort=False).indices.items():
        group_labels = np.unique(y[positions])
        if len(group_labels) != 1:
            raise AssertionError("Segment group has multiple labels inside training fold")
        block = X[positions]
        if mode == "segment_mean":
            rows.append(block.mean(axis=0))
        else:
            rows.append(np.median(block, axis=0))
        labels.append(int(group_labels[0]))
        counts.append(int(len(positions)))

    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(counts, dtype=np.int64),
    )


def make_training_set(
    spec: SegmentCandidate,
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    segment_groups: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []
    for view in spec.train_views:
        X_view = X_by_view[view][train_idx]
        y_view = y[train_idx]
        groups_view = segment_groups[train_idx]
        if spec.sample_mode == "window":
            X_parts.append(X_view)
            y_parts.append(y_view)
        elif spec.sample_mode in {"segment_mean", "segment_median"}:
            X_reduced, y_reduced, _ = reduce_by_segment(
                X_view,
                y_view,
                groups_view,
                spec.sample_mode,
            )
            X_parts.append(X_reduced)
            y_parts.append(y_reduced)
        else:
            raise ValueError(f"Unsupported sample_mode: {spec.sample_mode}")

    return np.vstack(X_parts), np.concatenate(y_parts)


def fit_single_candidate(
    spec: SegmentCandidate,
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    segment_groups: np.ndarray,
    train_idx: np.ndarray,
) -> object:
    X_train, y_train = make_training_set(spec, X_by_view, y, segment_groups, train_idx)

    if spec.model_kind == "direct":
        model = spec.direct_factory()
        model.fit(X_train, y_train)
        return {
            "kind": "single",
            "model_kind": "direct",
            "sample_mode": spec.sample_mode,
            "model": model,
        }

    if spec.model_kind == "hierarchical":
        binary_model = spec.binary_factory()
        binary_model.fit(X_train, (y_train > 0).astype(np.int64))
        contact_idx = np.where(y_train > 0)[0]
        contact_model = spec.contact_factory()
        contact_model.fit(X_train[contact_idx], y_train[contact_idx])
        return {
            "kind": "single",
            "model_kind": "hierarchical",
            "sample_mode": spec.sample_mode,
            "binary_model": binary_model,
            "contact_model": contact_model,
        }

    raise ValueError(f"Unsupported model_kind: {spec.model_kind}")


def fit_candidate(
    spec: SegmentCandidate,
    candidates: dict[str, SegmentCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    segment_groups: np.ndarray,
    train_idx: np.ndarray,
) -> object:
    if spec.kind == "single":
        return fit_single_candidate(spec, X_by_view, y, segment_groups, train_idx)

    if spec.kind == "ensemble":
        return {
            "kind": "ensemble",
            "members": {
                member: fit_candidate(candidates[member], candidates, X_by_view, y, segment_groups, train_idx)
                for member in spec.members
            },
            "weights": list(spec.weights) if spec.weights else [1.0] * len(spec.members),
        }

    raise ValueError(f"Unsupported candidate kind: {spec.kind}")


def predict_artifact(artifact: object, X: np.ndarray) -> np.ndarray:
    if artifact["kind"] == "ensemble":
        weights = np.asarray(artifact["weights"], dtype=np.float64)
        weights = weights / weights.sum()
        member_probas = [
            predict_artifact(member_artifact, X)
            for member_artifact in artifact["members"].values()
        ]
        return np.average(np.stack(member_probas), axis=0, weights=weights)

    if artifact["model_kind"] == "direct":
        return cv.proba_aligned(artifact["model"], X)

    if artifact["model_kind"] == "hierarchical":
        return cv.hierarchical_proba(artifact["binary_model"], artifact["contact_model"], X)

    raise ValueError(f"Unsupported artifact: {artifact}")


def tune_bias_multiview(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if grid is None:
        grid = np.linspace(-1.2, 1.2, 13)

    def score_bias(bias: np.ndarray) -> dict[str, float]:
        scores = {
            view: f1_score(y_true, cv.predict_with_bias(proba, bias), average="macro")
            for view, proba in proba_by_view.items()
        }
        scores["stress_worst_macro_f1"] = min(scores.values())
        scores["stress_mean_macro_f1"] = float(np.mean([scores[view] for view in STRESS_VIEWS]))
        return scores

    best_bias = np.zeros(4, dtype=np.float64)
    best_scores = score_bias(best_bias)
    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        scores = score_bias(bias)
        if (
            scores["stress_worst_macro_f1"] > best_scores["stress_worst_macro_f1"]
            or (
                scores["stress_worst_macro_f1"] == best_scores["stress_worst_macro_f1"]
                and scores["stress_mean_macro_f1"] > best_scores["stress_mean_macro_f1"]
            )
        ):
            best_bias = bias
            best_scores = scores

    for ambient_bias in np.linspace(-0.8, 0.8, 9):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        scores = score_bias(bias)
        if (
            scores["stress_worst_macro_f1"] > best_scores["stress_worst_macro_f1"]
            or (
                scores["stress_worst_macro_f1"] == best_scores["stress_worst_macro_f1"]
                and scores["stress_mean_macro_f1"] > best_scores["stress_mean_macro_f1"]
            )
        ):
            best_bias = bias
            best_scores = scores

    return best_bias, best_scores


def make_candidates(random_state: int) -> dict[str, SegmentCandidate]:
    class_weights = base.CONFIG["class_weights"]

    def hgb_regularized() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=360,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.08,
            class_weight=class_weights,
            random_state=random_state,
        )

    def hgb_shallow() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=420,
            learning_rate=0.03,
            max_leaf_nodes=9,
            min_samples_leaf=35,
            l2_regularization=0.15,
            class_weight=class_weights,
            random_state=random_state,
        )

    def lgbm_regularized() -> LGBMClassifier:
        return LGBMClassifier(
            objective="multiclass",
            num_class=len(base.CLASS_NAMES),
            n_estimators=620,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=28,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def extra_trees() -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
        )

    def logreg() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.5,
                        class_weight=class_weights,
                        max_iter=4000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def rbf_svm() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=5.0,
                        gamma="scale",
                        class_weight=class_weights,
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def knn() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=13, weights="distance", n_jobs=-1)),
            ]
        )

    def binary_hgb() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=random_state,
        )

    def binary_extra() -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def contact_hgb() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=420,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=18,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=random_state,
        )

    def contact_lgbm() -> LGBMClassifier:
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=560,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=22,
            subsample=0.86,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    specs = [
        SegmentCandidate("window_hgb_regularized_all_aug", "single", "direct", "window", STRESS_VIEWS, hgb_regularized),
        SegmentCandidate("window_hgb_shallow_all_aug", "single", "direct", "window", STRESS_VIEWS, hgb_shallow),
        SegmentCandidate("window_lgbm_all_aug", "single", "direct", "window", STRESS_VIEWS, lgbm_regularized),
        SegmentCandidate(
            "window_hier_extra_hgb_all_aug",
            "single",
            "hierarchical",
            "window",
            STRESS_VIEWS,
            None,
            binary_extra,
            contact_hgb,
        ),
        SegmentCandidate("segment_mean_hgb_regularized_all_aug", "single", "direct", "segment_mean", STRESS_VIEWS, hgb_regularized),
        SegmentCandidate("segment_mean_hgb_shallow_all_aug", "single", "direct", "segment_mean", STRESS_VIEWS, hgb_shallow),
        SegmentCandidate("segment_mean_lgbm_all_aug", "single", "direct", "segment_mean", STRESS_VIEWS, lgbm_regularized),
        SegmentCandidate("segment_mean_extra_all_aug", "single", "direct", "segment_mean", STRESS_VIEWS, extra_trees),
        SegmentCandidate("segment_mean_logreg_all_aug", "single", "direct", "segment_mean", STRESS_VIEWS, logreg),
        SegmentCandidate("segment_mean_svm_all_aug", "single", "direct", "segment_mean", STRESS_VIEWS, rbf_svm),
        SegmentCandidate("segment_mean_knn_all_aug", "single", "direct", "segment_mean", STRESS_VIEWS, knn),
        SegmentCandidate(
            "segment_mean_hier_hgb_hgb_all_aug",
            "single",
            "hierarchical",
            "segment_mean",
            STRESS_VIEWS,
            None,
            binary_hgb,
            contact_hgb,
        ),
        SegmentCandidate(
            "segment_mean_hier_hgb_lgbm_all_aug",
            "single",
            "hierarchical",
            "segment_mean",
            STRESS_VIEWS,
            None,
            binary_hgb,
            contact_lgbm,
        ),
        SegmentCandidate(
            "segment_mean_hier_extra_hgb_all_aug",
            "single",
            "hierarchical",
            "segment_mean",
            STRESS_VIEWS,
            None,
            binary_extra,
            contact_hgb,
        ),
        SegmentCandidate("segment_median_hgb_regularized_all_aug", "single", "direct", "segment_median", STRESS_VIEWS, hgb_regularized),
        SegmentCandidate("segment_median_lgbm_all_aug", "single", "direct", "segment_median", STRESS_VIEWS, lgbm_regularized),
    ]
    return {spec.name: spec for spec in specs}


def evaluate_candidate(
    spec: SegmentCandidate,
    candidates: dict[str, SegmentCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    segment_groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    print(
        f"\nAudio candidate: {spec.name} mode={spec.sample_mode} views={spec.train_views or spec.members}",
        flush=True,
    )
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = fit_candidate(spec, candidates, X_by_view, y, segment_groups, train_idx)
        train_time = time.perf_counter() - start
        view_scores = {}
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = predict_artifact(artifact, X_by_view[view][val_idx])
            pred = oof_by_view[view][val_idx].argmax(axis=1)
            view_scores[f"{view}_macro_f1"] = f1_score(y[val_idx], pred, average="macro")
        fold_row = {
            "candidate": spec.name,
            "fold": fold_id,
            "train_time_sec": train_time,
            "val_samples": int(len(val_idx)),
            **view_scores,
            "fold_worst_macro_f1": min(view_scores.values()),
        }
        fold_rows.append(fold_row)
        print(
            f"  fold {fold_id}: "
            + " | ".join(f"{view}={view_scores[f'{view}_macro_f1']:.4f}" for view in STRESS_VIEWS)
            + f" | worst={fold_row['fold_worst_macro_f1']:.4f} time={train_time:.2f}s",
            flush=True,
        )

    bias, tuned_scores = tune_bias_multiview(oof_by_view, y)
    clean_pred = cv.predict_with_bias(oof_by_view["clean"], bias)
    row = base.make_report_row(
        model_name=spec.name,
        split_name="audio_segment_stress_oof_cv",
        y_true=y,
        y_pred=clean_pred,
        train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
        predict_time_sec=0.0,
    )
    row.update(
        {
            "candidate_kind": spec.kind,
            "model_kind": spec.model_kind,
            "sample_mode": spec.sample_mode,
            "train_views_json": json.dumps(list(spec.train_views)),
            "members_json": json.dumps(list(spec.members)),
            "weights_json": json.dumps(list(spec.weights)),
            "class_bias_json": json.dumps(bias.tolist()),
            **{f"tuned_{view}_macro_f1": tuned_scores[view] for view in STRESS_VIEWS},
            "stress_worst_macro_f1": tuned_scores["stress_worst_macro_f1"],
            "stress_mean_macro_f1": tuned_scores["stress_mean_macro_f1"],
        }
    )
    print(
        "  tuned: "
        + " | ".join(f"{view}={tuned_scores[view]:.4f}" for view in STRESS_VIEWS)
        + f" | worst={tuned_scores['stress_worst_macro_f1']:.4f} bias={np.round(bias, 3).tolist()}",
        flush=True,
    )
    return row, oof_by_view, fold_rows


def evaluate_oof_ensemble(
    spec: SegmentCandidate,
    oof_by_candidate: dict[str, dict[str, np.ndarray]],
    y: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    weights = np.asarray(spec.weights if spec.weights else [1.0] * len(spec.members), dtype=np.float64)
    weights = weights / weights.sum()
    oof_by_view = {}
    for view in STRESS_VIEWS:
        member_probas = [oof_by_candidate[member][view] for member in spec.members]
        oof_by_view[view] = np.average(np.stack(member_probas), axis=0, weights=weights)

    bias, tuned_scores = tune_bias_multiview(oof_by_view, y)
    clean_pred = cv.predict_with_bias(oof_by_view["clean"], bias)
    row = base.make_report_row(
        model_name=spec.name,
        split_name="audio_segment_stress_oof_cv",
        y_true=y,
        y_pred=clean_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    row.update(
        {
            "candidate_kind": spec.kind,
            "model_kind": "ensemble",
            "sample_mode": "ensemble",
            "train_views_json": json.dumps([]),
            "members_json": json.dumps(list(spec.members)),
            "weights_json": json.dumps(list(spec.weights)),
            "class_bias_json": json.dumps(bias.tolist()),
            **{f"tuned_{view}_macro_f1": tuned_scores[view] for view in STRESS_VIEWS},
            "stress_worst_macro_f1": tuned_scores["stress_worst_macro_f1"],
            "stress_mean_macro_f1": tuned_scores["stress_mean_macro_f1"],
        }
    )
    print(
        f"\nOOF ensemble: {spec.name} members={spec.members} weights={np.round(weights, 3).tolist()} "
        + " | ".join(f"{view}={tuned_scores[view]:.4f}" for view in STRESS_VIEWS)
        + f" | worst={tuned_scores['stress_worst_macro_f1']:.4f}",
        flush=True,
    )
    return row, oof_by_view


def build_ensemble_candidates(
    leaderboard: pd.DataFrame,
    candidates: dict[str, SegmentCandidate],
) -> dict[str, SegmentCandidate]:
    ranked = leaderboard.sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    )["model"].tolist()
    window_ranked = [name for name in ranked if candidates[name].sample_mode == "window"]
    segment_ranked = [name for name in ranked if candidates[name].sample_mode.startswith("segment")]

    recipes: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {}
    if len(ranked) >= 3:
        recipes["audio_segment_ensemble_top3"] = (tuple(ranked[:3]), (1.0, 1.0, 1.0))
    if len(ranked) >= 5:
        recipes["audio_segment_ensemble_top5"] = (tuple(ranked[:5]), (1.0, 1.0, 1.0, 1.0, 1.0))
    if window_ranked and segment_ranked:
        recipes["audio_segment_ensemble_best_window_segment_50_50"] = (
            (window_ranked[0], segment_ranked[0]),
            (1.0, 1.0),
        )
        recipes["audio_segment_ensemble_best_window_segment_35_65"] = (
            (window_ranked[0], segment_ranked[0]),
            (0.35, 0.65),
        )
    if len(segment_ranked) >= 3:
        recipes["audio_segment_ensemble_segment_top3"] = (
            tuple(segment_ranked[:3]),
            (1.0, 1.0, 1.0),
        )

    return {
        name: SegmentCandidate(
            name=name,
            kind="ensemble",
            members=members,
            weights=weights,
        )
        for name, (members, weights) in recipes.items()
        if len(set(members)) == len(members)
    }


def load_audio_only_train_payloads(
    train_df: pd.DataFrame,
    clean_feature_cache_dir: Path,
    stress_feature_dir: Path,
    force_rebuild_stress: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float | bool]]]:
    clean_feat, clean_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=clean_feature_cache_dir,
        force_rebuild=False,
    )
    payloads = {"clean": clean_feat}
    timing = {"clean": clean_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, view_timing = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=stress_feature_dir,
            force_rebuild=force_rebuild_stress,
        )
        payloads[view] = payload
        timing[view] = view_timing
    return payloads, timing


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_segment_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH               =", root_path.resolve())
    print("RUN_DIR                 =", run_dir.resolve())
    print("CLEAN_FEATURE_CACHE_DIR =", args.clean_feature_cache_dir.resolve())
    print("STRESS_FEATURE_DIR      =", args.stress_feature_dir.resolve())
    print("Feature set             = audio only total240")
    print("Selection               = train-only grouped stress-CV, then final robot/test")

    train_csv = base.require_file(
        root_path / "audio_visual_dataset_default" / "dataset.csv",
        "hand/default dataset.csv",
    )
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)
    train_df["segment_group"] = train_df["group_key"]

    segment_label_counts = train_df.groupby("segment_group")["y"].nunique()
    if int((segment_label_counts > 1).sum()) != 0:
        raise AssertionError("Segment aggregation would mix labels")

    payloads, timing = load_audio_only_train_payloads(
        train_df,
        clean_feature_cache_dir=args.clean_feature_cache_dir,
        stress_feature_dir=args.stress_feature_dir,
        force_rebuild_stress=args.force_rebuild_stress,
    )
    X_by_view = {view: payloads[view]["X"] for view in STRESS_VIEWS}
    y = payloads["clean"]["y"]
    for view in STRESS_VIEWS:
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")

    segment_groups = train_df["segment_group"].to_numpy()
    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_audio_segment_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only total240 segment/prototype candidates; robot/test loaded after selection lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_segment_groups": int(train_df["segment_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "row_label_counts": label_counts_from_y(y),
        "segment_label_counts": {
            base.ID2LABEL[int(label)]: int(count)
            for label, count in train_df.groupby("segment_group")["y"].first().value_counts().sort_index().items()
        },
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    candidates = make_candidates(args.random_state)
    leaderboard_rows = []
    fold_rows = []
    oof_by_candidate = {}
    for spec in candidates.values():
        row, oof_by_view, candidate_fold_rows = evaluate_candidate(
            spec,
            candidates,
            X_by_view,
            y,
            segment_groups,
            splits,
        )
        leaderboard_rows.append(row)
        fold_rows.extend(candidate_fold_rows)
        oof_by_candidate[spec.name] = oof_by_view

    initial_leaderboard = pd.DataFrame(leaderboard_rows)
    ensemble_specs = build_ensemble_candidates(initial_leaderboard, candidates)
    candidates.update(ensemble_specs)
    for spec in ensemble_specs.values():
        row, oof_by_view = evaluate_oof_ensemble(spec, oof_by_candidate, y)
        leaderboard_rows.append(row)
        oof_by_candidate[spec.name] = oof_by_view

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_stress_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidates[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_kind": selected_spec.kind,
        "selected_model_kind": selected_spec.model_kind,
        "selected_sample_mode": selected_spec.sample_mode,
        "selected_train_views": list(selected_spec.train_views),
        "selected_members": list(selected_spec.members),
        "selected_weights": list(selected_spec.weights),
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    # Robot/test is intentionally loaded only after the selection lock above is written.
    test_csv = base.require_file(
        root_path / "audio_visual_dataset_robo_default" / "dataset.csv",
        "robot dataset.csv",
    )
    test_df = base.load_manifest(test_csv, "robot_test")
    test_feat, test_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )

    start = time.perf_counter()
    full_idx = np.arange(len(y))
    final_artifact = fit_candidate(
        selected_spec,
        candidates,
        X_by_view,
        y,
        segment_groups,
        full_idx,
    )
    final_fit_time = time.perf_counter() - start
    final_proba = predict_artifact(final_artifact, test_feat["X"])
    final_pred = cv.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_feat["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_segment_stress_cv",
            "selected_stress_worst_macro_f1": selected["stress_worst_macro_f1"],
            "selected_stress_mean_macro_f1": selected["stress_mean_macro_f1"],
            "selected_clean_oof_macro_f1": selected["tuned_clean_macro_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )

    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]
    ].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_feat["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_segment_stress_cv_select_no_test_until_final",
            "feature_set": base.FEATURE_SET,
            "feature_spec": base.FEATURE_SPEC,
            "feature_names": base.FEATURE_NAMES,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_segment_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "clean_feature_cache_dir": str(args.clean_feature_cache_dir.resolve()),
        "stress_feature_dir": str(args.stress_feature_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "stress_leaderboard": str(leaderboard_path.resolve()),
            "fold_report": str(fold_report_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen audio-only segment stress-CV selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_stress_worst_macro_f1",
                "selected_clean_oof_macro_f1",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Stress leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
