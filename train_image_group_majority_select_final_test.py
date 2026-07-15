from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import train_image_deep_transfer_select_final_test as deep
import train_image_handcrafted_ml_select_final_test as img


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    view: str
    factory: Callable[[], object]


@dataclass
class GroupSplit:
    X_views: dict[str, np.ndarray]
    y_group: np.ndarray
    group_hashes: np.ndarray
    group_sizes: np.ndarray
    group_counts: np.ndarray
    row_y: np.ndarray
    row_to_group: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only group-majority learning. Exact duplicate image rows are "
            "collapsed inside the hand/default train/val split, candidates are "
            "selected by validation Macro F1, a lock is written, and only then "
            "robot/test is loaded."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_group_majority_deep_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-mode", choices=["specimen", "segment"], default="specimen")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbones", default="resnet18,resnet34")
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("outputs/image_deep_features"))
    parser.add_argument(
        "--image-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/multimodal_classical_cv_select/image_features"),
    )
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--force-rebuild-image", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["regularized_val_score", "macro_f1_4class", "contact_macro_f1", "binary_macro_f1"],
        default="regularized_val_score",
    )
    parser.add_argument("--bias-penalty", type=float, default=0.02)
    return parser.parse_args()


def image_hash(path: Path) -> str:
    if not path.exists():
        return f"MISSING:{path}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_image_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["image_hash"] = [image_hash(Path(path)) for path in output["image_path"]]
    return output


def duplicate_summary(frame: pd.DataFrame) -> dict[str, object]:
    grouped = frame.groupby("image_hash")
    label_nunique = grouped["y"].nunique()
    conflict_hashes = label_nunique[label_nunique > 1].index
    conflict_rows = frame[frame["image_hash"].isin(conflict_hashes)]
    return {
        "rows": int(len(frame)),
        "unique_exact_images": int(frame["image_hash"].nunique()),
        "conflicting_exact_image_groups": int(len(conflict_hashes)),
        "rows_in_conflicting_exact_image_groups": int(len(conflict_rows)),
        "label_counts": img.label_counts(frame["y"].to_numpy()),
        "conflict_label_counts": img.label_counts(conflict_rows["y"].to_numpy()) if len(conflict_rows) else {},
    }


def build_all_train_views(
    train_df: pd.DataFrame,
    backbones: list[str],
    args: argparse.Namespace,
    device,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    views: dict[str, np.ndarray] = {}
    timing: dict[str, object] = {}

    image_payload, image_timing = img.build_or_load_image_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.image_feature_cache_dir,
        force_rebuild=args.force_rebuild_image,
    )
    if not np.array_equal(image_payload["y"], train_df["y"].to_numpy()):
        raise AssertionError("Handcrafted image cache labels do not align with train manifest")
    hand_views, _ = img.build_views(image_payload["X"])
    views["hand_compact"] = hand_views["compact"]
    views["hand_full"] = hand_views["full"]
    timing["handcrafted"] = image_timing

    for backbone in backbones:
        payload, backbone_timing = deep.extract_deep_features(
            train_df,
            "hand_train_full",
            backbone,
            args.image_size,
            args.feature_cache_dir,
            args.batch_size,
            args.num_workers,
            args.force_rebuild_features,
            device,
        )
        if not np.array_equal(payload["y"], train_df["y"].to_numpy()):
            raise AssertionError(f"{backbone}: deep feature labels do not align with train manifest")
        views[backbone] = payload["X"]
        timing[backbone] = backbone_timing

    if "resnet34" in views:
        views["hand_compact_resnet34"] = np.concatenate([views["hand_compact"], views["resnet34"]], axis=1)
    if "resnet18" in views and "resnet34" in views:
        views["resnet18_resnet34"] = np.concatenate([views["resnet18"], views["resnet34"]], axis=1)
    return views, timing


def build_selected_test_views(
    test_df: pd.DataFrame,
    selected_view: str,
    args: argparse.Namespace,
    device,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    needed: set[str] = set()
    if "hand" in selected_view:
        needed.add("hand")
    for backbone in ["resnet18", "resnet34", "resnet50", "vit_b16"]:
        if backbone in selected_view:
            needed.add(backbone)

    views: dict[str, np.ndarray] = {}
    timing: dict[str, object] = {}
    if "hand" in needed:
        payload, image_timing = img.build_or_load_image_cache(
            test_df,
            "robot_test",
            feature_dir=args.image_feature_cache_dir,
            force_rebuild=args.force_rebuild_image,
        )
        hand_views, _ = img.build_views(payload["X"])
        views["hand_compact"] = hand_views["compact"]
        views["hand_full"] = hand_views["full"]
        timing["handcrafted"] = image_timing
    for backbone in sorted(needed - {"hand"}):
        payload, backbone_timing = deep.extract_deep_features(
            test_df,
            "robot_test",
            backbone,
            args.image_size,
            args.feature_cache_dir,
            args.batch_size,
            args.num_workers,
            args.force_rebuild_features,
            device,
        )
        views[backbone] = payload["X"]
        timing[backbone] = backbone_timing

    if selected_view == "hand_compact_resnet34":
        views[selected_view] = np.concatenate([views["hand_compact"], views["resnet34"]], axis=1)
    if selected_view == "resnet18_resnet34":
        views[selected_view] = np.concatenate([views["resnet18"], views["resnet34"]], axis=1)
    return {selected_view: views[selected_view]}, timing


def make_group_split(frame: pd.DataFrame, row_indices: np.ndarray, views: dict[str, np.ndarray]) -> GroupSplit:
    sub = frame.iloc[row_indices].copy()
    sub["row_position"] = row_indices
    group_items = list(sub.groupby("image_hash", sort=False))
    X_views = {name: [] for name in views}
    y_group = []
    group_hashes = []
    group_sizes = []
    group_counts = []
    row_to_group_lookup: dict[int, int] = {}

    for group_index, (hash_value, rows) in enumerate(group_items):
        positions = rows["row_position"].to_numpy(dtype=np.int64)
        counts = np.bincount(rows["y"].to_numpy(dtype=np.int64), minlength=len(img.CLASS_NAMES))
        for name, matrix in views.items():
            X_views[name].append(matrix[positions].mean(axis=0))
        y_group.append(int(counts.argmax()))
        group_hashes.append(str(hash_value))
        group_sizes.append(int(len(rows)))
        group_counts.append(counts)
        for position in positions:
            row_to_group_lookup[int(position)] = group_index

    return GroupSplit(
        X_views={name: np.vstack(rows).astype(np.float32) for name, rows in X_views.items()},
        y_group=np.asarray(y_group, dtype=np.int64),
        group_hashes=np.asarray(group_hashes),
        group_sizes=np.asarray(group_sizes, dtype=np.int64),
        group_counts=np.vstack(group_counts).astype(np.int64),
        row_y=sub["y"].to_numpy(dtype=np.int64),
        row_to_group=np.asarray([row_to_group_lookup[int(position)] for position in row_indices], dtype=np.int64),
    )


def make_candidates(random_state: int, views: dict[str, np.ndarray]) -> dict[str, CandidateSpec]:
    candidates: dict[str, CandidateSpec] = {}
    class_weights = {0: 0.85, 1: 1.4, 2: 1.2, 3: 1.2}

    def add(name: str, view: str, factory: Callable[[], object]) -> None:
        if view in views:
            candidates[name] = CandidateSpec(name=name, view=view, factory=factory)

    def logreg(c: float, class_weight: object = "balanced") -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        max_iter=5000,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def rbf_svc(c: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=c,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    for view in views:
        add(f"{view}_group_logreg_C0.1_balanced", view, lambda view=view: logreg(0.1, "balanced"))
        add(f"{view}_group_logreg_C1_balanced", view, lambda view=view: logreg(1.0, "balanced"))
        add(f"{view}_group_logreg_C3_weighted", view, lambda view=view: logreg(3.0, class_weights))
        add(f"{view}_group_rbf_svc_C1", view, lambda view=view: rbf_svc(1.0))
        add(
            f"{view}_group_knn5_distance",
            view,
            lambda view=view: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
                ]
            ),
        )
        add(
            f"{view}_group_extra_trees",
            view,
            lambda view=view: ExtraTreesClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
        add(
            f"{view}_group_random_forest",
            view,
            lambda view=view: RandomForestClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
        add(
            f"{view}_group_hgb",
            view,
            lambda view=view: HistGradientBoostingClassifier(
                max_iter=240,
                learning_rate=0.04,
                max_leaf_nodes=9,
                min_samples_leaf=4,
                l2_regularization=0.08,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
        add(
            f"{view}_group_lgbm",
            view,
            lambda view=view: LGBMClassifier(
                objective="multiclass",
                num_class=4,
                n_estimators=420,
                learning_rate=0.03,
                num_leaves=9,
                min_child_samples=4,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
        )
    return candidates


def aligned_proba(model: object, X: np.ndarray) -> np.ndarray:
    return img.proba_aligned(model, X)


def evaluate_candidate(
    spec: CandidateSpec,
    train_split: GroupSplit,
    val_split: GroupSplit,
    bias_penalty: float,
) -> tuple[list[dict[str, object]], object]:
    model = spec.factory()
    start = time.perf_counter()
    model.fit(train_split.X_views[spec.view], train_split.y_group)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    group_proba = aligned_proba(model, val_split.X_views[spec.view])
    row_proba = group_proba[val_split.row_to_group]
    predict_time = time.perf_counter() - start

    bias_options = [
        ("none", np.zeros(len(img.CLASS_NAMES), dtype=np.float64)),
        ("tuned", img.tune_class_bias(row_proba, val_split.row_y)[0]),
    ]
    rows: list[dict[str, object]] = []
    for bias_mode, bias in bias_options:
        pred = img.predict_with_bias(row_proba, bias)
        row = img.make_report_row(
            model_name=spec.name,
            split_name="hand_val",
            y_true=val_split.row_y,
            y_pred=pred,
            feature_set=spec.view,
            feature_name=f"{spec.view}_group_majority",
            n_features=train_split.X_views[spec.view].shape[1],
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
                "train_group_samples": int(len(train_split.y_group)),
                "val_group_samples": int(len(val_split.y_group)),
            }
        )
        rows.append(row)
    return rows, model


def fit_final_and_predict(
    spec: CandidateSpec,
    train_full_split: GroupSplit,
    test_split: GroupSplit,
    selected_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, object, float]:
    model = clone(spec.factory())
    start = time.perf_counter()
    model.fit(train_full_split.X_views[spec.view], train_full_split.y_group)
    train_time = time.perf_counter() - start
    group_proba = aligned_proba(model, test_split.X_views[spec.view])
    row_proba = group_proba[test_split.row_to_group]
    row_pred = img.predict_with_bias(row_proba, selected_bias)
    return row_pred, row_proba, model, train_time


def write_group_manifest(path: Path, frame: pd.DataFrame, split: GroupSplit) -> None:
    rows = []
    by_hash = frame.groupby("image_hash", sort=False)
    for group_index, hash_value in enumerate(split.group_hashes):
        group = by_hash.get_group(hash_value)
        counts = split.group_counts[group_index]
        rows.append(
            {
                "image_hash": hash_value,
                "representative_image_path": group.iloc[0]["image_path"],
                "majority_y": int(split.y_group[group_index]),
                "majority_label": img.ID2LABEL[int(split.y_group[group_index])],
                "n_rows": int(split.group_sizes[group_index]),
                "label_counts_json": json.dumps({img.ID2LABEL[i]: int(counts[i]) for i in range(len(img.CLASS_NAMES))}),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    deep.set_seed(args.random_state)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    group_dir = run_dir / "groups"
    for directory in [run_dir, report_dir, model_dir, group_dir, args.feature_cache_dir, args.image_feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("DEVICE    =", device, flush=True)
    print("Protocol  = image-only group-majority selection; robot/test after lock", flush=True)

    train_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_df = add_image_hashes(train_df)
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    train_views, train_feature_timing = build_all_train_views(train_df, backbones, args, device)
    train_split = make_group_split(train_df, train_idx, train_views)
    val_split = make_group_split(train_df, val_idx, train_views)
    train_full_split = make_group_split(train_df, np.arange(len(train_df)), train_views)

    write_group_manifest(group_dir / f"{args.run_slug}_train_inner_groups.csv", train_df.iloc[train_idx], train_split)
    write_group_manifest(group_dir / f"{args.run_slug}_val_groups.csv", train_df.iloc[val_idx], val_split)
    write_group_manifest(group_dir / f"{args.run_slug}_train_full_groups.csv", train_df, train_full_split)

    split_summary = {
        "protocol": "image_only_group_majority_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_full_rows": int(len(train_df)),
        "train_full_duplicate_summary": duplicate_summary(train_df),
        "train_inner_groups": int(len(train_split.y_group)),
        "val_groups": int(len(val_split.y_group)),
        "train_full_groups": int(len(train_full_split.y_group)),
        "train_group_majority_counts": img.label_counts(train_split.y_group),
        "val_row_label_counts": img.label_counts(val_split.row_y),
        "val_group_majority_counts": img.label_counts(val_split.y_group),
        "views": {name: int(matrix.shape[1]) for name, matrix in train_views.items()},
        "train_feature_timing": train_feature_timing,
        "bias_penalty": args.bias_penalty,
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)

    candidates = make_candidates(args.random_state, train_views)
    leaderboard_rows: list[dict[str, object]] = []
    for spec in candidates.values():
        print(f"\nVAL candidate: {spec.name}", flush=True)
        try:
            rows, _ = evaluate_candidate(spec, train_split, val_split, args.bias_penalty)
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
            continue
        for row in rows:
            print(
                f"  {row['bias_mode']:5s} macro={row['macro_f1_4class']:.4f} "
                f"reg={row['regularized_val_score']:.4f} acc={row['accuracy_4class']:.4f}",
                flush=True,
            )
        leaderboard_rows.extend(rows)

    if not leaderboard_rows:
        raise RuntimeError("No candidate completed")
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
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
    test_df = add_image_hashes(test_df)
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    test_views, test_feature_timing = build_selected_test_views(test_df, selected_spec.view, args, device)
    test_split = make_group_split(test_df, np.arange(len(test_df)), test_views)
    write_group_manifest(group_dir / f"{args.run_slug}_robot_test_groups.csv", test_df, test_split)

    final_pred, final_proba, final_model, final_fit_time = fit_final_and_predict(
        selected_spec,
        train_full_split,
        test_split,
        selected_bias,
    )
    final_row = img.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_split.row_y,
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=f"{selected_spec.view}_group_majority",
        n_features=train_full_split.X_views[selected_spec.view].shape[1],
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_group_majority",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_bias_mode": selected.get("bias_mode"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
            "train_full_groups": int(len(train_full_split.y_group)),
            "robot_test_groups": int(len(test_split.y_group)),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(test_split.row_y, final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "image_path", "label", "y", "segment_group", "specimen_group", "source", "image_hash"]
    ].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(img.ID2LABEL)
    for class_id, class_name in img.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "image_only_group_majority_no_test_until_lock",
            "selected": selection_summary,
            "model": final_model,
            "selected_bias": selected_bias,
            "label_map": img.LABEL_MAP,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "image_only_group_majority_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "test_duplicate_summary": duplicate_summary(test_df),
        "test_feature_timing": test_feature_timing,
        "final_test_report": final_row,
        "artifacts": {
            "val_leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "model_bundle": str(bundle_path.resolve()),
            "split_manifests": split_paths,
            "group_manifests": str(group_dir.resolve()),
        },
    }
    img.write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen image-only group-majority selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_regularized_val_score",
                "selected_bias_mode",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Val leaderboard:", leaderboard_path.resolve(), flush=True)
    print("Selection lock:", selection_path.resolve(), flush=True)
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
