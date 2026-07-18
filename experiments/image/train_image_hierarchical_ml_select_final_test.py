from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_image_handcrafted_ml_select_final_test as img
import train_image_spatial_v2_select_final_test as spatial


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    view: str
    factory: Callable[[int], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only handcrafted hierarchical ML. Hand/default is split into "
            "train/val; ambient-vs-contact and contact-subtype models plus class "
            "bias are selected on val only, then robot/test is loaded after lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_hierarchical_ml_specimen_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["specimen", "segment", "row"], default="specimen")
    parser.add_argument(
        "--v1-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/multimodal_classical_cv_select/image_features"),
    )
    parser.add_argument(
        "--spatial-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/image_spatial_v2_features"),
    )
    parser.add_argument("--force-rebuild-spatial", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["regularized_val_score", "macro_f1_4class", "contact_macro_f1", "binary_macro_f1"],
        default="regularized_val_score",
    )
    parser.add_argument("--bias-penalty", type=float, default=0.03)
    return parser.parse_args()


def make_logreg(c: float, class_weight: object = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=5000,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_selected_logreg(k: int, c: float) -> Callable[[int], Pipeline]:
    def factory(dim: int) -> Pipeline:
        actual_k = min(k, dim)
        return Pipeline(
            [
                ("select", SelectKBest(score_func=f_classif, k=actual_k)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=5000,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

    return factory


def make_candidates(random_state: int) -> dict[str, CandidateSpec]:
    def logreg_factory(c: float) -> Callable[[int], Pipeline]:
        return lambda _dim: make_logreg(c)

    def extra_trees(_dim: int) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=900,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def random_forest(_dim: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    specs = [
        CandidateSpec("hier_v1_compact_logreg_c1", "v1_compact", logreg_factory(1.0)),
        CandidateSpec("hier_v1_compact_logreg_c3", "v1_compact", logreg_factory(3.0)),
        CandidateSpec("hier_v1_compact_extra_trees", "v1_compact", extra_trees),
        CandidateSpec("hier_v1_compact_random_forest", "v1_compact", random_forest),
        CandidateSpec("hier_v1_full_f400_logreg", "v1_full", make_selected_logreg(400, 1.5)),
        CandidateSpec("hier_spatial_f800_logreg", "spatial", make_selected_logreg(800, 1.5)),
        CandidateSpec("hier_spatial_f1600_logreg", "spatial", make_selected_logreg(1600, 1.0)),
        CandidateSpec("hier_v1_spatial_f1000_logreg", "v1_spatial", make_selected_logreg(1000, 1.5)),
        CandidateSpec("hier_v1_spatial_f2200_logreg", "v1_spatial", make_selected_logreg(2200, 1.0)),
    ]
    return {spec.name: spec for spec in specs}


def proba_for_labels(model: object, X: np.ndarray, labels: list[int]) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    output = np.zeros((len(X), len(labels)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        if int(class_id) in labels:
            output[:, labels.index(int(class_id))] = raw[:, column]
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def combine_hierarchical_proba(binary_proba: np.ndarray, subtype_proba: np.ndarray) -> np.ndarray:
    output = np.zeros((len(binary_proba), 4), dtype=np.float64)
    output[:, 0] = binary_proba[:, 0]
    output[:, 1:] = binary_proba[:, 1:2] * subtype_proba
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def fit_hierarchical(spec: CandidateSpec, X: np.ndarray, y: np.ndarray) -> tuple[object, object, float]:
    start = time.perf_counter()
    binary_y = (y > 0).astype(np.int64)
    binary_model = spec.factory(X.shape[1])
    binary_model.fit(X, binary_y)

    contact_mask = y > 0
    subtype_model = spec.factory(X.shape[1])
    subtype_model.fit(X[contact_mask], y[contact_mask])
    return binary_model, subtype_model, time.perf_counter() - start


def predict_hierarchical_proba(binary_model: object, subtype_model: object, X: np.ndarray) -> np.ndarray:
    binary_proba = proba_for_labels(binary_model, X, [0, 1])
    subtype_proba = proba_for_labels(subtype_model, X, [1, 2, 3])
    return combine_hierarchical_proba(binary_proba, subtype_proba)


def metrics_row(
    spec: CandidateSpec,
    y_true: np.ndarray,
    proba: np.ndarray,
    bias: np.ndarray,
    train_time: float,
    predict_time: float,
    bias_mode: str,
    bias_penalty: float,
) -> dict[str, object]:
    pred = img.predict_with_bias(proba, bias)
    row = img.make_report_row(
        model_name=spec.name,
        split_name="hand_val",
        y_true=y_true,
        y_pred=pred,
        feature_set=spec.view,
        feature_name=f"{spec.view}+hierarchical_binary_subtype",
        n_features=0,
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )
    bias_l1 = float(np.abs(bias).sum())
    row.update(
        {
            "view": spec.view,
            "bias_mode": bias_mode,
            "class_bias_json": json.dumps(bias.tolist()),
            "bias_l1": bias_l1,
            "regularized_val_score": float(row["macro_f1_4class"] - bias_penalty * bias_l1),
            "default_window_macro_f1": f1_score(y_true, proba.argmax(axis=1), average="macro", zero_division=0),
        }
    )
    return row


def evaluate_candidate(
    spec: CandidateSpec,
    train_views: dict[str, np.ndarray],
    val_views: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    bias_penalty: float,
) -> tuple[list[dict[str, object]], np.ndarray, tuple[object, object]]:
    print(f"\nVAL hierarchical candidate: {spec.name} view={spec.view}", flush=True)
    binary_model, subtype_model, train_time = fit_hierarchical(spec, train_views[spec.view], y_train)
    start = time.perf_counter()
    val_proba = predict_hierarchical_proba(binary_model, subtype_model, val_views[spec.view])
    predict_time = time.perf_counter() - start
    tuned_bias, _ = img.tune_class_bias(val_proba, y_val)
    rows = [
        metrics_row(spec, y_val, val_proba, np.zeros(4, dtype=np.float64), train_time, predict_time, "none", bias_penalty),
        metrics_row(spec, y_val, val_proba, tuned_bias, train_time, predict_time, "tuned", bias_penalty),
    ]
    for row in rows:
        print(
            f"  {row['bias_mode']:5s} macro={row['macro_f1_4class']:.4f} "
            f"contact={row['contact_macro_f1']:.4f} binary={row['binary_macro_f1']:.4f} "
            f"reg={row['regularized_val_score']:.4f} bias_l1={row['bias_l1']:.2f}",
            flush=True,
        )
    return rows, val_proba, (binary_model, subtype_model)


def main() -> None:
    args = parse_args()
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir, args.v1_feature_cache_dir, args.spatial_feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("Protocol  = image-only hierarchical handcrafted ML; robot/test after lock", flush=True)

    train_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_v1, train_v1_timing = img.build_or_load_image_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.v1_feature_cache_dir,
        force_rebuild=False,
    )
    train_spatial, train_spatial_timing = spatial.build_or_load_spatial_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.spatial_feature_cache_dir,
        force_rebuild=args.force_rebuild_spatial,
    )
    if not np.array_equal(train_v1["y"], train_spatial["y"]):
        raise AssertionError("Train v1/spatial label mismatch")
    y_full = train_v1["y"]
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)
    train_views, view_dims = spatial.make_views(train_v1["X"][train_idx], train_spatial["X"][train_idx])
    val_views, _ = spatial.make_views(train_v1["X"][val_idx], train_spatial["X"][val_idx])
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]

    split_summary = {
        "protocol": "image_only_hierarchical_ml_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_full_samples": int(len(train_df)),
        "train_full_label_counts": img.label_counts(y_full),
        "view_dims": view_dims,
        "train_v1_timing": train_v1_timing,
        "train_spatial_timing": train_spatial_timing,
        "bias_penalty": args.bias_penalty,
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)

    candidates = make_candidates(args.random_state)
    rows = []
    for spec in candidates.values():
        candidate_rows, _, _ = evaluate_candidate(
            spec,
            train_views,
            val_views,
            y_train,
            y_val,
            args.bias_penalty,
        )
        rows.extend(candidate_rows)

    leaderboard = pd.DataFrame(rows).sort_values(
        [args.selection_metric, "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_val_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidates[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_view": selected_spec.view,
        "selected_bias_mode": selected["bias_mode"],
        "selected_bias": selected_bias.tolist(),
        "selection_metric": args.selection_metric,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    img.write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_df = img.load_image_manifest(
        img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv"),
        test_dataset_dir,
        "robot_test",
    )
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    test_v1, test_v1_timing = img.build_or_load_image_cache(
        test_df,
        "robot_test",
        feature_dir=args.v1_feature_cache_dir,
        force_rebuild=False,
    )
    test_spatial, test_spatial_timing = spatial.build_or_load_spatial_cache(
        test_df,
        "robot_test",
        feature_dir=args.spatial_feature_cache_dir,
        force_rebuild=args.force_rebuild_spatial,
    )
    full_views, _ = spatial.make_views(train_v1["X"], train_spatial["X"])
    test_views, _ = spatial.make_views(test_v1["X"], test_spatial["X"])
    binary_model, subtype_model, train_time = fit_hierarchical(selected_spec, full_views[selected_spec.view], y_full)
    start = time.perf_counter()
    test_proba = predict_hierarchical_proba(binary_model, subtype_model, test_views[selected_spec.view])
    predict_time = time.perf_counter() - start
    final_pred = img.predict_with_bias(test_proba, selected_bias)
    y_test = test_v1["y"]
    final_row = img.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=f"{selected_spec.view}+hierarchical_binary_subtype",
        n_features=int(test_views[selected_spec.view].shape[1]),
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_hierarchical_ml",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_bias_mode": selected["bias_mode"],
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(y_test, final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    pred_frame = test_df[
        ["audio_file", "image_file", "image_path", "label", "y", "segment_group", "specimen_group", "source"]
    ].copy()
    pred_frame["pred_y"] = final_pred
    pred_frame["pred_label"] = pred_frame["pred_y"].map(img.ID2LABEL)
    for class_id, class_name in img.ID2LABEL.items():
        pred_frame[f"proba_{class_name}"] = test_proba[:, class_id]
    pred_frame.to_csv(predictions_path, index=False)

    model_path = model_dir / f"{args.run_slug}_selected_model.joblib"
    joblib.dump({"binary_model": binary_model, "subtype_model": subtype_model, "selected": selection_summary}, model_path)
    protocol_summary = {
        "protocol": "image_only_hierarchical_ml_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "final_test_report": final_row,
        "test_v1_timing": test_v1_timing,
        "test_spatial_timing": test_spatial_timing,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "model": str(model_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    img.write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)
    print("\nFinal robot/test result after frozen hierarchical image-ML selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_regularized_val_score",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Selection lock:", selection_path.resolve(), flush=True)
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
