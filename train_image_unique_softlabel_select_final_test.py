from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

import train_image_deep_transfer_select_final_test as deep_transfer
import train_image_group_majority_select_final_test as group_majority
import train_image_handcrafted_ml_select_final_test as image_base
import train_image_timm_dedup_group_select_final_test as dedup
import train_image_timm_transfer_select_final_test as timm_transfer


@dataclass(frozen=True)
class FeatureSource:
    family: str
    name: str

    @property
    def key(self) -> str:
        prefix = "tv" if self.family == "torchvision" else "timm"
        return f"{prefix}__{timm_transfer.safe_name(self.name)}"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    view: str
    family: str
    smoothing: float
    class_balance_power: float
    group_weight_power: float
    parameter: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only learning with one sample per exact image and soft label distributions. "
            "Candidate selection uses grouped hand/default OOF predictions only, writes a lock, "
            "then loads robot/test once for final evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_unique_softlabel_oof_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--feature-sources",
        default=(
            "torchvision:resnet18,"
            "timm:vit_small_patch14_dinov2.lvd142m,"
            "timm:vit_base_patch16_clip_224.openai_ft_in1k"
        ),
        help="Comma-separated FAMILY:MODEL entries; FAMILY is torchvision or timm.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--deep-feature-cache-dir", type=Path, default=Path("outputs/image_deep_features"))
    parser.add_argument("--timm-feature-cache-dir", type=Path, default=Path("outputs/image_timm_features"))
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--max-combo-views", type=int, choices=[0, 2], default=2)
    parser.add_argument("--soft-label-smoothing", type=float, default=0.5)
    parser.add_argument("--bias-top-k", type=int, default=12)
    parser.add_argument("--bias-penalty", type=float, default=0.025)
    parser.add_argument("--fold-std-penalty", type=float, default=0.10)
    parser.add_argument(
        "--selection-metric",
        choices=["robust_oof_score", "macro_f1_4class", "contact_macro_f1", "binary_macro_f1"],
        default="robust_oof_score",
    )
    return parser.parse_args()


def parse_feature_sources(raw: str) -> list[FeatureSource]:
    sources: list[FeatureSource] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Feature source must be FAMILY:MODEL, got {item!r}")
        family, name = item.split(":", 1)
        family = family.strip().lower()
        name = name.strip()
        if family not in {"torchvision", "timm"}:
            raise ValueError(f"Unsupported feature family {family!r}")
        if not name:
            raise ValueError(f"Missing model name in {item!r}")
        sources.append(FeatureSource(family=family, name=name))
    if not sources:
        raise ValueError("At least one feature source is required")
    keys = [source.key for source in sources]
    if len(keys) != len(set(keys)):
        raise ValueError("Feature source keys must be unique")
    return sources


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def build_feature_views(
    source_matrices: dict[str, np.ndarray],
    max_combo_views: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    views = dict(source_matrices)
    components = {name: [name] for name in source_matrices}
    source_names = list(source_matrices)
    if len(source_names) >= 2 and max_combo_views >= 2:
        for index, first in enumerate(source_names):
            for second in source_names[index + 1 :]:
                key = f"combo__{first}__{second}"
                views[key] = np.concatenate(
                    [l2_rows(source_matrices[first]), l2_rows(source_matrices[second])], axis=1
                ).astype(np.float32)
                components[key] = [first, second]
    if len(source_names) >= 3:
        key = "combo_all"
        views[key] = np.concatenate([l2_rows(source_matrices[name]) for name in source_names], axis=1).astype(
            np.float32
        )
        components[key] = source_names
    return views, components


def load_cached_timm_payload(
    frame: pd.DataFrame,
    split_name: str,
    source: FeatureSource,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]] | None:
    paths = timm_transfer.cache_paths(cache_dir, source.name, split_name)
    if not all(paths[key].exists() for key in ["X", "y", "paths", "missing", "metadata"]):
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    cached_y = np.load(paths["y"])
    expected_y = frame["y"].to_numpy(dtype=np.int64)
    cached_paths = np.load(paths["paths"], allow_pickle=True).astype(str)
    expected_paths = frame["image_path"].astype(str).to_numpy()
    if (
        metadata.get("feature_family") != "timm_deep_image_embedding"
        or metadata.get("backbone") != source.name
        or len(cached_y) != len(frame)
        or not np.array_equal(cached_y, expected_y)
        or not np.array_equal(cached_paths, expected_paths)
    ):
        return None
    start = time.perf_counter()
    payload = {
        "X": np.load(paths["X"]),
        "y": cached_y,
        "paths": cached_paths,
        "missing": np.load(paths["missing"]),
    }
    elapsed = time.perf_counter() - start
    print(f"Loaded timm cache directly {source.name}/{split_name}: {payload['X'].shape}", flush=True)
    return payload, {
        "cache_hit": True,
        "cache_load_time_sec": elapsed,
        "extraction_time_sec_current_run": 0.0,
        "extraction_time_sec_original": float(metadata.get("extraction_time_sec", 0.0)),
        "missing_images": int(metadata.get("missing_images", 0)),
        "cache_validation": "label_and_path_order_without_model_init",
    }


def load_feature_source(
    frame: pd.DataFrame,
    split_name: str,
    source: FeatureSource,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    if source.family == "torchvision":
        payload, timing = deep_transfer.extract_deep_features(
            frame,
            split_name,
            source.name,
            args.image_size,
            args.deep_feature_cache_dir,
            args.batch_size,
            args.num_workers,
            args.force_rebuild_features,
            device,
        )
    else:
        cached = None
        if not args.force_rebuild_features:
            cached = load_cached_timm_payload(frame, split_name, source, args.timm_feature_cache_dir)
        if cached is not None:
            payload, timing = cached
        else:
            payload, timing = timm_transfer.extract_timm_features(
                frame,
                split_name,
                source.name,
                args.timm_feature_cache_dir,
                args.batch_size,
                args.num_workers,
                args.force_rebuild_features,
                device,
            )
    if not np.array_equal(payload["y"], frame["y"].to_numpy(dtype=np.int64)):
        raise AssertionError(f"{source.key}: cached labels do not align with {split_name}")
    return np.asarray(payload["X"], dtype=np.float32), timing


def group_specimen_keys(frame: pd.DataFrame, group_hashes: np.ndarray) -> np.ndarray:
    grouped = frame.groupby("image_hash", sort=False)
    keys: list[str] = []
    for hash_value in group_hashes:
        rows = grouped.get_group(hash_value)
        specimens = sorted(rows["specimen_group"].astype(str).unique().tolist())
        keys.append("||".join(specimens))
    return np.asarray(keys)


def soft_targets(counts: np.ndarray, smoothing: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    smoothed = counts + float(smoothing)
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def class_factors(counts: np.ndarray, power: float) -> np.ndarray:
    mass = np.asarray(counts, dtype=np.float64).sum(axis=0)
    inverse = mass.sum() / (len(image_base.CLASS_NAMES) * np.maximum(mass, 1.0))
    factors = inverse ** float(power)
    return factors / factors.mean()


def group_weights(sizes: np.ndarray, power: float) -> np.ndarray:
    weights = np.asarray(sizes, dtype=np.float64) ** float(power)
    return weights / np.maximum(weights.mean(), 1e-12)


def make_model(spec: CandidateSpec, random_state: int) -> Pipeline:
    if spec.family == "soft_logreg":
        return Pipeline(
            [
                ("normalize", Normalizer()),
                (
                    "model",
                    LogisticRegression(
                        C=spec.parameter,
                        solver="lbfgs",
                        max_iter=2500,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if spec.family == "soft_ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=spec.parameter)),
            ]
        )
    if spec.family == "soft_knn":
        return Pipeline(
            [
                ("normalize", Normalizer()),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=int(spec.parameter),
                        weights="distance",
                        metric="cosine",
                        algorithm="brute",
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown candidate family {spec.family!r}")


def fit_soft_model(
    spec: CandidateSpec,
    X: np.ndarray,
    counts: np.ndarray,
    sizes: np.ndarray,
    random_state: int,
) -> Pipeline:
    model = make_model(spec, random_state)
    targets = soft_targets(counts, spec.smoothing)
    factors = class_factors(counts, spec.class_balance_power)
    weights = group_weights(sizes, spec.group_weight_power)
    if spec.family == "soft_logreg":
        n_classes = len(image_base.CLASS_NAMES)
        X_expanded = np.repeat(X, n_classes, axis=0)
        y_expanded = np.tile(np.arange(n_classes, dtype=np.int64), len(X))
        sample_weight = (targets * factors.reshape(1, -1) * weights.reshape(-1, 1)).reshape(-1)
        keep = sample_weight > 1e-12
        model.fit(
            X_expanded[keep],
            y_expanded[keep],
            model__sample_weight=sample_weight[keep],
        )
        return model
    adjusted_targets = targets * factors.reshape(1, -1)
    adjusted_targets /= adjusted_targets.sum(axis=1, keepdims=True)
    if spec.family == "soft_ridge":
        model.fit(X, adjusted_targets, model__sample_weight=weights)
    else:
        model.fit(X, adjusted_targets)
    return model


def predict_soft_model(spec: CandidateSpec, model: Pipeline, X: np.ndarray) -> np.ndarray:
    if spec.family == "soft_logreg":
        raw = model.predict_proba(X)
        classes = np.asarray(model.classes_, dtype=np.int64)
        output = np.zeros((len(X), len(image_base.CLASS_NAMES)), dtype=np.float64)
        for column, class_id in enumerate(classes):
            output[:, int(class_id)] = raw[:, column]
    else:
        output = np.asarray(model.predict(X), dtype=np.float64)
    output = np.clip(output, 1e-9, None)
    return output / output.sum(axis=1, keepdims=True)


def make_candidates(views: dict[str, np.ndarray], smoothing: float) -> dict[str, CandidateSpec]:
    candidates: dict[str, CandidateSpec] = {}
    for view in views:
        safe_view = timm_transfer.safe_name(view)
        for c_value in [0.03, 0.1, 0.3]:
            for balance_power in [0.0, 0.5, 1.0]:
                spec = CandidateSpec(
                    name=f"{safe_view}__soft_logreg_C{c_value:g}__cb{balance_power:g}",
                    view=view,
                    family="soft_logreg",
                    smoothing=smoothing,
                    class_balance_power=balance_power,
                    group_weight_power=0.0,
                    parameter=c_value,
                )
                candidates[spec.name] = spec
        for alpha in [1.0, 10.0, 100.0]:
            spec = CandidateSpec(
                name=f"{safe_view}__soft_ridge_a{alpha:g}__cb0.5",
                view=view,
                family="soft_ridge",
                smoothing=smoothing,
                class_balance_power=0.5,
                group_weight_power=0.0,
                parameter=alpha,
            )
            candidates[spec.name] = spec
        for neighbors in [3, 7, 15]:
            spec = CandidateSpec(
                name=f"{safe_view}__soft_knn_k{neighbors}__cb0.5",
                view=view,
                family="soft_knn",
                smoothing=smoothing,
                class_balance_power=0.5,
                group_weight_power=0.0,
                parameter=float(neighbors),
            )
            candidates[spec.name] = spec
    return candidates


def fold_statistics(
    row_y: np.ndarray,
    row_to_group: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    row_proba: np.ndarray,
    bias: np.ndarray,
) -> tuple[list[float], float, float]:
    scores: list[float] = []
    for _, val_groups in folds:
        mask = np.isin(row_to_group, val_groups)
        pred = image_base.predict_with_bias(row_proba[mask], bias)
        report = image_base.make_report_row(
            model_name="fold_diagnostic",
            split_name="hand_oof_fold",
            y_true=row_y[mask],
            y_pred=pred,
            feature_set="fold",
            feature_name="fold",
            n_features=0,
            train_time_sec=0.0,
            predict_time_sec=0.0,
        )
        scores.append(float(report["macro_f1_4class"]))
    return scores, float(np.mean(scores)), float(np.std(scores))


def make_oof_report_row(
    spec: CandidateSpec,
    split: dedup.ExactImageSplit,
    folds: list[tuple[np.ndarray, np.ndarray]],
    group_proba: np.ndarray,
    bias_mode: str,
    bias: np.ndarray,
    train_time: float,
    predict_time: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    row_proba = group_proba[split.row_to_group]
    pred = image_base.predict_with_bias(row_proba, bias)
    row = image_base.make_report_row(
        model_name=spec.name,
        split_name="hand_group_oof",
        y_true=split.row_y,
        y_pred=pred,
        feature_set=spec.view,
        feature_name=f"{spec.view}_unique_softlabel",
        n_features=split.X_views[spec.view].shape[1],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )
    fold_scores, fold_mean, fold_std = fold_statistics(
        split.row_y,
        split.row_to_group,
        folds,
        row_proba,
        bias,
    )
    bias_l1 = float(np.abs(bias).sum())
    row.update(
        {
            "view": spec.view,
            "family": spec.family,
            "soft_label_smoothing": spec.smoothing,
            "class_balance_power": spec.class_balance_power,
            "group_weight_power": spec.group_weight_power,
            "model_parameter": spec.parameter,
            "bias_mode": bias_mode,
            "class_bias_json": json.dumps(bias.tolist()),
            "bias_l1": bias_l1,
            "fold_macro_f1_json": json.dumps(fold_scores),
            "fold_macro_f1_mean": fold_mean,
            "fold_macro_f1_std": fold_std,
            "fold_macro_f1_worst": float(min(fold_scores)),
            "robust_oof_score": float(
                row["macro_f1_4class"]
                - args.fold_std_penalty * fold_std
                - args.bias_penalty * bias_l1
            ),
            "n_exact_image_groups": int(len(split.group_hashes)),
        }
    )
    return row


def evaluate_candidate_oof(
    spec: CandidateSpec,
    split: dedup.ExactImageSplit,
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_state: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray]:
    X = split.X_views[spec.view]
    oof_group_proba = np.zeros((len(split.group_hashes), len(image_base.CLASS_NAMES)), dtype=np.float64)
    train_time = 0.0
    predict_time = 0.0
    for fold_index, (train_groups, val_groups) in enumerate(folds, start=1):
        start = time.perf_counter()
        model = fit_soft_model(
            spec,
            X[train_groups],
            split.group_counts[train_groups],
            split.group_sizes[train_groups],
            random_state + fold_index,
        )
        train_time += time.perf_counter() - start
        start = time.perf_counter()
        oof_group_proba[val_groups] = predict_soft_model(spec, model, X[val_groups])
        predict_time += time.perf_counter() - start
    if np.any(oof_group_proba.sum(axis=1) <= 0):
        raise AssertionError(f"{spec.name}: incomplete OOF predictions")
    raw_row = make_oof_report_row(
        spec,
        split,
        folds,
        oof_group_proba,
        "none",
        np.zeros(len(image_base.CLASS_NAMES), dtype=np.float64),
        train_time,
        predict_time,
        args,
    )
    return raw_row, oof_group_proba


def write_group_manifest(path: Path, frame: pd.DataFrame, split: dedup.ExactImageSplit) -> None:
    grouped = frame.groupby("image_hash", sort=False)
    rows: list[dict[str, object]] = []
    for group_index, hash_value in enumerate(split.group_hashes):
        group = grouped.get_group(hash_value)
        counts = split.group_counts[group_index]
        distribution = soft_targets(counts.reshape(1, -1), 0.0)[0]
        rows.append(
            {
                "image_hash": hash_value,
                "representative_image_file": str(group.iloc[0]["image_file"]),
                "representative_image_path": str(group.iloc[0]["image_path"]),
                "specimen_groups_json": json.dumps(sorted(group["specimen_group"].astype(str).unique().tolist())),
                "n_rows": int(split.group_sizes[group_index]),
                "majority_label": image_base.ID2LABEL[int(counts.argmax())],
                "label_counts_json": json.dumps(
                    {image_base.ID2LABEL[index]: int(counts[index]) for index in range(len(image_base.CLASS_NAMES))}
                ),
                "label_distribution_json": json.dumps(
                    {
                        image_base.ID2LABEL[index]: float(distribution[index])
                        for index in range(len(image_base.CLASS_NAMES))
                    }
                ),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    deep_transfer.set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = parse_feature_sources(args.feature_sources)
    root_path = image_base.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    group_dir = run_dir / "groups"
    oof_dir = run_dir / "oof_proba"
    for directory in [
        run_dir,
        report_dir,
        model_dir,
        group_dir,
        oof_dir,
        args.deep_feature_cache_dir,
        args.timm_feature_cache_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("DEVICE    =", device, flush=True)
    print("Protocol  = unique-image soft-label grouped OOF; robot/test only after lock", flush=True)

    train_df = image_base.load_image_manifest(
        image_base.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_df = group_majority.add_image_hashes(train_df)

    train_source_matrices: dict[str, np.ndarray] = {}
    train_feature_timing: dict[str, dict[str, object]] = {}
    for source in sources:
        matrix, timing = load_feature_source(train_df, "hand_train_full", source, args, device)
        train_source_matrices[source.key] = matrix
        train_feature_timing[source.key] = timing
    train_views, view_components = build_feature_views(train_source_matrices, args.max_combo_views)
    train_split = dedup.build_exact_image_split(
        train_df,
        np.arange(len(train_df), dtype=np.int64),
        train_views,
        ["majority"],
    )
    specimen_keys = group_specimen_keys(train_df, train_split.group_hashes)
    majority_y = train_split.group_counts.argmax(axis=1).astype(np.int64)
    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    folds = list(splitter.split(np.zeros(len(majority_y)), majority_y, specimen_keys))
    if len(folds) != args.n_splits:
        raise AssertionError("Unexpected fold count")

    fold_rows: list[dict[str, object]] = []
    for fold_index, (train_groups, val_groups) in enumerate(folds):
        for group_index in val_groups:
            fold_rows.append(
                {
                    "image_hash": train_split.group_hashes[group_index],
                    "specimen_key": specimen_keys[group_index],
                    "majority_label": image_base.ID2LABEL[int(majority_y[group_index])],
                    "fold": fold_index,
                    "n_rows": int(train_split.group_sizes[group_index]),
                }
            )
    fold_manifest_path = run_dir / "splits" / f"{args.run_slug}_hand_unique_image_oof_folds.csv"
    fold_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(fold_manifest_path, index=False)
    write_group_manifest(group_dir / f"{args.run_slug}_hand_unique_groups.csv", train_df, train_split)

    split_summary = {
        "protocol": "image_only_unique_softlabel_grouped_oof_robot_after_lock",
        "train_rows": int(len(train_df)),
        "duplicate_summary": group_majority.duplicate_summary(train_df),
        "n_exact_image_groups": int(len(train_split.group_hashes)),
        "n_specimen_keys": int(len(np.unique(specimen_keys))),
        "n_splits": args.n_splits,
        "group_majority_counts": image_base.label_counts(majority_y),
        "row_label_counts": image_base.label_counts(train_split.row_y),
        "feature_sources": [asdict(source) | {"key": source.key} for source in sources],
        "views": {name: int(matrix.shape[1]) for name, matrix in train_views.items()},
        "view_components": view_components,
        "soft_label_smoothing": args.soft_label_smoothing,
        "train_feature_timing": train_feature_timing,
        "fold_manifest": str(fold_manifest_path.resolve()),
        "forbidden_inputs": [
            "audio features",
            "multimodal features",
            "filename class/contact tokens as predictive features",
            "robot/test images, probabilities, or labels before selection lock",
        ],
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    image_base.write_json(split_summary_path, split_summary)

    candidates = make_candidates(train_views, args.soft_label_smoothing)
    raw_rows: list[dict[str, object]] = []
    oof_by_candidate: dict[str, np.ndarray] = {}
    for candidate_index, spec in enumerate(candidates.values(), start=1):
        print(f"OOF candidate {candidate_index}/{len(candidates)}: {spec.name}", flush=True)
        try:
            row, oof_group_proba = evaluate_candidate_oof(
                spec,
                train_split,
                folds,
                args.random_state,
                args,
            )
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
            continue
        raw_rows.append(row)
        oof_by_candidate[spec.name] = oof_group_proba
        print(
            f"  macro={row['macro_f1_4class']:.4f} robust={row['robust_oof_score']:.4f} "
            f"fold_std={row['fold_macro_f1_std']:.4f}",
            flush=True,
        )
    if not raw_rows:
        raise RuntimeError("No soft-label candidate completed")

    raw_leaderboard = pd.DataFrame(raw_rows).sort_values(
        [args.selection_metric, "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    )
    tuned_rows: list[dict[str, object]] = []
    for _, raw_row in raw_leaderboard.head(max(args.bias_top_k, 0)).iterrows():
        candidate_name = str(raw_row["model"])
        spec = candidates[candidate_name]
        group_proba = oof_by_candidate[candidate_name]
        row_proba = group_proba[train_split.row_to_group]
        tuned_bias, _ = image_base.tune_class_bias(row_proba, train_split.row_y)
        tuned_rows.append(
            make_oof_report_row(
                spec,
                train_split,
                folds,
                group_proba,
                "tuned",
                tuned_bias,
                float(raw_row["model_train_time_sec"]),
                float(raw_row["model_predict_time_sec"]),
                args,
            )
        )

    leaderboard = pd.concat([pd.DataFrame(raw_rows), pd.DataFrame(tuned_rows)], ignore_index=True).sort_values(
        [args.selection_metric, "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidates[selected_name]
    selected_bias = np.asarray(json.loads(str(selected["class_bias_json"])), dtype=np.float64)
    selected_oof_path = oof_dir / f"{timm_transfer.safe_name(selected_name)}_hand_group_oof_proba.npy"
    np.save(selected_oof_path, oof_by_candidate[selected_name])
    selection_summary = {
        "selected_without_test": selected,
        "selected_candidate": asdict(selected_spec),
        "selected_bias": selected_bias.tolist(),
        "selected_view_components": view_components[selected_spec.view],
        "selection_metric": args.selection_metric,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
        "selected_oof_group_proba": str(selected_oof_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    image_base.write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection_summary, indent=2, default=str), flush=True)

    test_df = image_base.load_image_manifest(
        image_base.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv"),
        test_dataset_dir,
        "robot_test",
    )
    test_df = group_majority.add_image_hashes(test_df)
    source_by_key = {source.key: source for source in sources}
    selected_components = view_components[selected_spec.view]
    test_source_matrices: dict[str, np.ndarray] = {}
    test_feature_timing: dict[str, dict[str, object]] = {}
    for component in selected_components:
        source = source_by_key[component]
        matrix, timing = load_feature_source(test_df, "robot_test", source, args, device)
        test_source_matrices[component] = matrix
        test_feature_timing[component] = timing
    test_views, _ = build_feature_views(test_source_matrices, args.max_combo_views)
    if selected_spec.view == "combo_all" and selected_spec.view not in test_views:
        test_views[selected_spec.view] = np.concatenate(
            [l2_rows(test_source_matrices[name]) for name in selected_components], axis=1
        ).astype(np.float32)
    if selected_spec.view.startswith("combo__") and selected_spec.view not in test_views:
        test_views[selected_spec.view] = np.concatenate(
            [l2_rows(test_source_matrices[name]) for name in selected_components], axis=1
        ).astype(np.float32)
    test_split = dedup.build_exact_image_split(
        test_df,
        np.arange(len(test_df), dtype=np.int64),
        {selected_spec.view: test_views[selected_spec.view]},
        ["majority"],
    )
    write_group_manifest(group_dir / f"{args.run_slug}_robot_test_unique_groups_for_audit.csv", test_df, test_split)

    start = time.perf_counter()
    final_model = fit_soft_model(
        selected_spec,
        train_split.X_views[selected_spec.view],
        train_split.group_counts,
        train_split.group_sizes,
        args.random_state,
    )
    final_fit_time = time.perf_counter() - start
    start = time.perf_counter()
    test_group_proba = predict_soft_model(selected_spec, final_model, test_split.X_views[selected_spec.view])
    final_proba = test_group_proba[test_split.row_to_group]
    final_pred = image_base.predict_with_bias(final_proba, selected_bias)
    final_predict_time = time.perf_counter() - start

    final_row = image_base.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_split.row_y,
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=f"{selected_spec.view}_unique_softlabel",
        n_features=train_split.X_views[selected_spec.view].shape[1],
        train_time_sec=final_fit_time,
        predict_time_sec=final_predict_time,
    )
    final_row.update(
        {
            "selected_by": "hand_only_unique_image_grouped_oof_softlabel",
            "selected_oof_macro_f1": selected.get("macro_f1_4class"),
            "selected_robust_oof_score": selected.get("robust_oof_score"),
            "selected_bias_mode": selected.get("bias_mode"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
            "soft_label_smoothing": selected_spec.smoothing,
            "class_balance_power": selected_spec.class_balance_power,
            "train_exact_image_groups": int(len(train_split.group_hashes)),
            "robot_test_exact_image_groups": int(len(test_split.group_hashes)),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        image_base.confusion_matrix(test_split.row_y, final_pred, labels=image_base.LABELS),
        index=image_base.CLASS_NAMES,
        columns=image_base.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        [
            "audio_file",
            "image_file",
            "image_path",
            "label",
            "y",
            "segment_group",
            "specimen_group",
            "source",
            "image_hash",
        ]
    ].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(image_base.ID2LABEL)
    for class_id, class_name in image_base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    model_bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "image_only_unique_softlabel_grouped_oof_no_test_until_lock",
            "selected": selection_summary,
            "model": final_model,
            "selected_bias": selected_bias,
            "feature_sources": [asdict(source) | {"key": source.key} for source in sources],
            "view_components": selected_components,
            "label_map": image_base.LABEL_MAP,
        },
        model_bundle_path,
    )
    protocol_summary = {
        "protocol": "image_only_unique_softlabel_grouped_oof_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "test_feature_timing": test_feature_timing,
        "final_test_report": final_row,
        "artifacts": {
            "oof_leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "model_bundle": str(model_bundle_path.resolve()),
            "hand_group_manifest": str((group_dir / f"{args.run_slug}_hand_unique_groups.csv").resolve()),
            "test_group_manifest": str(
                (group_dir / f"{args.run_slug}_robot_test_unique_groups_for_audit.csv").resolve()
            ),
        },
    }
    protocol_summary_path = report_dir / f"{args.run_slug}_protocol_summary.json"
    image_base.write_json(protocol_summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen image-only soft-label selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "ambient_f1",
                "leaf_f1",
                "trunk_f1",
                "twig_f1",
                "selected_oof_macro_f1",
                "selected_robust_oof_score",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("OOF leaderboard:", leaderboard_path.resolve(), flush=True)
    print("Selection lock :", selection_path.resolve(), flush=True)
    print("Final report   :", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
