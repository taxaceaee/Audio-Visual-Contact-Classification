from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from lightgbm import LGBMClassifier

import train_cv_select_final_test as cv
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
IMAGE_SIZE = (128, 96)  # width, height for cv2.resize


@dataclass(frozen=True)
class ModelSpec:
    name: str
    view: str
    kind: str
    factory: Callable[[], object] | None = None
    members: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classical multimodal supervised selection. Uses hand/default audio+image "
            "features for grouped CV selection, writes a selection lock, then evaluates "
            "once on robot/test. No deep learning and no test labels are used for selection."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-set", choices=["total240"], default="total240")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--audio-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument("--force-rebuild-image", action="store_true")
    parser.add_argument(
        "--include-slow-full-image",
        action="store_true",
        help=(
            "Also try full HOG image candidates. They are slow and were kept opt-in "
            "because compact image CV is the cheaper train-only signal."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def attach_image_paths(frame: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    output = frame.copy()
    output["image_path"] = output["image_file"].map(lambda value: dataset_dir / str(value))
    return output


def image_manifest_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["image_path", "y"]].itertuples(index=False):
        digest.update(f"{row.image_path}|{int(row.y)}\n".encode("utf-8"))
    return digest.hexdigest()


def image_cache_paths(feature_dir: Path, split_name: str) -> dict[str, Path]:
    split_dir = feature_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": split_dir / "X.npy",
        "y": split_dir / "y.npy",
        "paths": split_dir / "paths.npy",
        "metadata": split_dir / "metadata.json",
    }


def normalize_hist(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32).ravel()
    total = float(values.sum())
    if total <= 0.0:
        return values
    return values / total


def channel_stats(channels: np.ndarray) -> np.ndarray:
    stats = []
    for channel in channels:
        values = channel.astype(np.float32).ravel()
        stats.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.percentile(values, 10)),
                float(np.percentile(values, 50)),
                float(np.percentile(values, 90)),
            ]
        )
    return np.asarray(stats, dtype=np.float32)


def edge_grid_features(edges: np.ndarray, grid: int = 4) -> np.ndarray:
    h, w = edges.shape
    features = [float(edges.mean())]
    for row in np.array_split(np.arange(h), grid):
        for col in np.array_split(np.arange(w), grid):
            features.append(float(edges[np.ix_(row, col)].mean()))
    return np.asarray(features, dtype=np.float32)


def extract_image_features(path: Path) -> tuple[np.ndarray, bool]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    missing = image is None
    if missing:
        image = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    else:
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    color_hist = np.concatenate(
        [
            normalize_hist(cv2.calcHist([hsv], [0], None, [24], [0, 180])),
            normalize_hist(cv2.calcHist([hsv], [1], None, [16], [0, 256])),
            normalize_hist(cv2.calcHist([hsv], [2], None, [16], [0, 256])),
        ]
    )
    stats = np.concatenate(
        [
            channel_stats(np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)),
            channel_stats(np.moveaxis(hsv.astype(np.float32) / np.asarray([180.0, 255.0, 255.0]), -1, 0)),
        ]
    )

    lbp_8 = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_16 = local_binary_pattern(gray, P=16, R=2, method="uniform")
    lbp_hist = np.concatenate(
        [
            normalize_hist(np.histogram(lbp_8, bins=np.arange(0, 11), range=(0, 10))[0]),
            normalize_hist(np.histogram(lbp_16, bins=np.arange(0, 19), range=(0, 18))[0]),
        ]
    )

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    edges = cv2.Canny(gray_u8, threshold1=60, threshold2=160)
    edge_features = edge_grid_features((edges > 0).astype(np.float32), grid=4)

    features = np.concatenate([color_hist, stats, lbp_hist, edge_features, hog_features]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Image feature contains NaN/Inf: {path}")
    return features, missing


def build_or_load_image_cache(
    frame: pd.DataFrame,
    split_name: str,
    feature_dir: Path,
    force_rebuild: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool | int]]:
    paths = image_cache_paths(feature_dir, split_name)
    signature = image_manifest_signature(frame)
    ready = all(path.exists() for path in paths.values())
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        valid_metadata = (
            metadata.get("feature_family") == "handcrafted_image_v1"
            and int(metadata.get("n_samples", -1)) == len(frame)
            and metadata.get("manifest_signature") == signature
        )
        if valid_metadata:
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": np.load(paths["y"]),
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            load_time = time.perf_counter() - start
            print(f"Loaded image cache {split_name}: {payload['X'].shape} in {load_time:.3f}s")
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
                "missing_images": int(metadata.get("missing_images", 0)),
            }
        print(f"Image cache metadata mismatch for {split_name}; rebuilding.")

    rows = []
    labels = []
    image_paths = []
    missing_count = 0
    start = time.perf_counter()
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"Extract image/{split_name}"):
        features, missing = extract_image_features(Path(row.image_path))
        rows.append(features)
        labels.append(int(row.y))
        image_paths.append(str(row.image_path))
        missing_count += int(missing)
    extraction_time = time.perf_counter() - start

    payload = {
        "X": np.stack(rows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(image_paths),
    }
    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    metadata = {
        "feature_family": "handcrafted_image_v1",
        "feature_dim": int(payload["X"].shape[1]),
        "image_size": list(IMAGE_SIZE),
        "n_samples": len(frame),
        "manifest_signature": signature,
        "missing_images": missing_count,
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Saved image cache {split_name}: {payload['X'].shape} "
        f"in {extraction_time:.2f}s missing={missing_count}"
    )
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
        "missing_images": missing_count,
    }


def build_views(X_audio: np.ndarray, X_image: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    # Image feature layout: color hist 56, channel stats 30, LBP 28, edge 17, HOG rest.
    compact_dim = 56 + 30 + 28 + 17
    X_image_compact = X_image[:, :compact_dim]
    views = {
        "audio240": X_audio,
        "image_compact": X_image_compact,
        "image_full": X_image,
        "audio_image_compact": np.hstack([X_audio, X_image_compact]).astype(np.float32),
        "audio_image_full": np.hstack([X_audio, X_image]).astype(np.float32),
    }
    dims = {name: int(matrix.shape[1]) for name, matrix in views.items()}
    return views, dims


def make_model_specs(random_state: int, include_slow_full_image: bool = False) -> dict[str, ModelSpec]:
    class_weights = base.CONFIG["class_weights"]

    def lgbm(num_leaves: int, min_child_samples: int, reg_lambda: float, n_estimators: int = 650):
        return LGBMClassifier(
            objective="multiclass",
            num_class=len(base.CLASS_NAMES),
            n_estimators=n_estimators,
            learning_rate=0.025,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=reg_lambda,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def extra_trees():
        return ExtraTreesClassifier(
            n_estimators=900,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
        )

    def hgb(max_leaf_nodes: int, min_samples_leaf: int):
        return HistGradientBoostingClassifier(
            max_iter=360,
            learning_rate=0.035,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=0.08,
            class_weight=class_weights,
            random_state=random_state,
        )

    specs = [
        ModelSpec("audio240_hgb_regularized", "audio240", "direct", lambda: hgb(15, 25)),
        ModelSpec("audio240_lgbm_regularized", "audio240", "direct", lambda: lgbm(15, 35, 2.0)),
        ModelSpec("image_compact_hgb_regularized", "image_compact", "direct", lambda: hgb(15, 25)),
        ModelSpec("image_compact_lgbm_regularized", "image_compact", "direct", lambda: lgbm(15, 30, 2.0)),
        ModelSpec("audio_image_compact_hgb_regularized", "audio_image_compact", "direct", lambda: hgb(15, 25)),
        ModelSpec("audio_image_compact_lgbm_regularized", "audio_image_compact", "direct", lambda: lgbm(15, 30, 2.0)),
        ModelSpec(
            "audio_image_compact_logreg",
            "audio_image_compact",
            "direct",
            lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.5,
                            class_weight=class_weights,
                            max_iter=3000,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]
    if include_slow_full_image:
        specs.extend(
            [
                ModelSpec(
                    "image_full_lgbm_regularized",
                    "image_full",
                    "direct",
                    lambda: lgbm(31, 30, 2.0, 700),
                ),
                ModelSpec("image_full_extra_trees", "image_full", "direct", extra_trees),
                ModelSpec(
                    "audio_image_full_lgbm_regularized",
                    "audio_image_full",
                    "direct",
                    lambda: lgbm(31, 30, 2.5, 700),
                ),
                ModelSpec("audio_image_full_extra_trees", "audio_image_full", "direct", extra_trees),
            ]
        )
    return {spec.name: spec for spec in specs}


def fit_predict_oof(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict]]:
    oof_proba = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        model = spec.factory()
        model.fit(X[train_idx], y[train_idx])
        train_time = time.perf_counter() - start
        oof_proba[val_idx] = cv.proba_aligned(model, X[val_idx])
        pred = oof_proba[val_idx].argmax(axis=1)
        macro_f1 = f1_score(y[val_idx], pred, average="macro")
        fold_rows.append(
            {
                "candidate": spec.name,
                "view": spec.view,
                "fold": fold_id,
                "macro_f1": macro_f1,
                "train_time_sec": train_time,
                "val_samples": int(len(val_idx)),
            }
        )
        print(f"  fold {fold_id}: macro_f1={macro_f1:.4f} time={train_time:.2f}s")
    return oof_proba, fold_rows


def report_row_for_prediction(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_set: str,
    feature_name: str,
    n_features: int,
    train_time_sec: float,
    predict_time_sec: float,
) -> dict:
    row = base.make_report_row(
        model_name=model_name,
        split_name=split_name,
        y_true=y_true,
        y_pred=y_pred,
        train_time_sec=train_time_sec,
        predict_time_sec=predict_time_sec,
    )
    row.update(
        {
            "feature_set": feature_set,
            "feature_name": feature_name,
            "n_features": n_features,
        }
    )
    return row


def evaluate_candidate(
    spec: ModelSpec,
    views: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, np.ndarray, list[dict]]:
    print(f"\nOOF candidate: {spec.name} view={spec.view}")
    oof_proba, fold_rows = fit_predict_oof(spec, views[spec.view], y, splits)
    default_pred = oof_proba.argmax(axis=1)
    bias, tuned_macro_f1 = cv.tune_class_bias(oof_proba, y)
    tuned_pred = cv.predict_with_bias(oof_proba, bias)
    row = report_row_for_prediction(
        model_name=spec.name,
        split_name="oof_cv",
        y_true=y,
        y_pred=tuned_pred,
        feature_set=spec.view,
        feature_name=spec.view,
        n_features=views[spec.view].shape[1],
        train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
        predict_time_sec=0.0,
    )
    row.update(
        {
            "candidate_kind": spec.kind,
            "view": spec.view,
            "default_oof_macro_f1": f1_score(y, default_pred, average="macro"),
            "tuned_oof_macro_f1": tuned_macro_f1,
            "class_bias_json": json.dumps(bias.tolist()),
            "members_json": json.dumps(list(spec.members)),
        }
    )
    print(
        f"  OOF default={row['default_oof_macro_f1']:.4f} "
        f"tuned={tuned_macro_f1:.4f} bias={np.round(bias, 3).tolist()}"
    )
    return row, oof_proba, fold_rows


def build_ensemble_rows(
    leaderboard: pd.DataFrame,
    oof_probas: dict[str, np.ndarray],
    y: np.ndarray,
) -> tuple[list[dict], dict[str, ModelSpec], dict[str, np.ndarray]]:
    rows = []
    specs = {}
    probas = {}
    ranked = leaderboard.sort_values("tuned_oof_macro_f1", ascending=False)["model"].tolist()
    recipes = {
        "multimodal_ensemble_top3": tuple(ranked[:3]),
        "multimodal_ensemble_top5": tuple(ranked[:5]),
    }
    for name, members in recipes.items():
        if len(members) < 2:
            continue
        proba = np.mean([oof_probas[member] for member in members], axis=0)
        bias, tuned_macro_f1 = cv.tune_class_bias(proba, y)
        pred = cv.predict_with_bias(proba, bias)
        row = report_row_for_prediction(
            model_name=name,
            split_name="oof_cv",
            y_true=y,
            y_pred=pred,
            feature_set="ensemble",
            feature_name="mean probability ensemble",
            n_features=0,
            train_time_sec=0.0,
            predict_time_sec=0.0,
        )
        row.update(
            {
                "candidate_kind": "ensemble",
                "view": "ensemble",
                "default_oof_macro_f1": f1_score(y, proba.argmax(axis=1), average="macro"),
                "tuned_oof_macro_f1": tuned_macro_f1,
                "class_bias_json": json.dumps(bias.tolist()),
                "members_json": json.dumps(list(members)),
            }
        )
        rows.append(row)
        specs[name] = ModelSpec(name=name, view="ensemble", kind="ensemble", members=members)
        probas[name] = proba
        print(f"\nOOF ensemble: {name} members={members} tuned={tuned_macro_f1:.4f}")
    return rows, specs, probas


def fit_predict_full(
    spec: ModelSpec,
    candidates: dict[str, ModelSpec],
    train_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    y_train: np.ndarray,
) -> tuple[np.ndarray, object, float]:
    start = time.perf_counter()
    if spec.kind == "direct":
        model = spec.factory()
        model.fit(train_views[spec.view], y_train)
        proba = cv.proba_aligned(model, test_views[spec.view])
        elapsed = time.perf_counter() - start
        return proba, {"kind": "direct", "view": spec.view, "model": model}, elapsed

    if spec.kind == "ensemble":
        artifacts = {}
        probas = []
        for member in spec.members:
            member_proba, member_artifact, _ = fit_predict_full(
                candidates[member],
                candidates,
                train_views,
                test_views,
                y_train,
            )
            probas.append(member_proba)
            artifacts[member] = member_artifact
        elapsed = time.perf_counter() - start
        return np.mean(probas, axis=0), {"kind": "ensemble", "members": artifacts}, elapsed

    raise ValueError(f"Unsupported spec kind: {spec.kind}")


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set(args.feature_set)

    root_path = base.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_slug = "multimodal_classical_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    image_feature_dir = run_dir / "image_features"
    for directory in [run_dir, report_dir, model_dir, split_dir, image_feature_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH               =", root_path.resolve())
    print("RUN_DIR                 =", run_dir.resolve())
    print("AUDIO_FEATURE_CACHE_DIR =", args.audio_feature_cache_dir.resolve())
    print("Protocol                = grouped hand/default CV selection, then final robot/test")

    train_csv = base.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv")
    train_df = attach_image_paths(base.load_manifest(train_csv, "hand_train"), train_dataset_dir)
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    train_audio, train_audio_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.audio_feature_cache_dir,
        force_rebuild=False,
    )
    train_image, train_image_timing = build_or_load_image_cache(
        train_df,
        "hand_train_full",
        feature_dir=image_feature_dir,
        force_rebuild=args.force_rebuild_image,
    )
    y_train = train_audio["y"]
    if not np.array_equal(y_train, train_image["y"]):
        raise AssertionError("Audio/image train labels are not aligned")

    train_views, view_dims = build_views(train_audio["X"], train_image["X"])
    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    train_df.assign(cv_fold=fold_assignment).to_csv(
        split_dir / "hand_train_full_multimodal_cv_folds.csv",
        index=False,
    )

    split_summary = {
        "protocol": "hand/default grouped CV only for multimodal selection; robot/test loaded after selection lock",
        "n_train_samples": len(train_df),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": args.n_folds,
        "label_counts": cv.label_counts(y_train),
        "view_dims": view_dims,
        "train_audio_timing": train_audio_timing,
        "train_image_timing": train_image_timing,
    }
    split_summary_path = report_dir / f"{run_slug}_cv_split_summary.json"
    write_json(split_summary_path, split_summary)

    candidates = make_model_specs(args.random_state, include_slow_full_image=args.include_slow_full_image)
    leaderboard_rows = []
    fold_rows = []
    oof_probas = {}
    for spec in candidates.values():
        row, proba, rows = evaluate_candidate(spec, train_views, y_train, splits)
        leaderboard_rows.append(row)
        oof_probas[spec.name] = proba
        fold_rows.extend(rows)

    initial_leaderboard = pd.DataFrame(leaderboard_rows)
    ensemble_rows, ensemble_specs, ensemble_probas = build_ensemble_rows(
        initial_leaderboard,
        oof_probas,
        y_train,
    )
    leaderboard_rows.extend(ensemble_rows)
    candidates.update(ensemble_specs)
    oof_probas.update(ensemble_probas)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["tuned_oof_macro_f1", "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_leaderboard.csv"
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
        "selected_view": selected_spec.view,
        "selected_members": list(selected_spec.members),
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float))

    test_csv = base.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv")
    test_df = attach_image_paths(base.load_manifest(test_csv, "robot_test"), test_dataset_dir)
    test_audio, test_audio_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.audio_feature_cache_dir,
        force_rebuild=False,
    )
    test_image, test_image_timing = build_or_load_image_cache(
        test_df,
        "robot_test",
        feature_dir=image_feature_dir,
        force_rebuild=args.force_rebuild_image,
    )
    if not np.array_equal(test_audio["y"], test_image["y"]):
        raise AssertionError("Audio/image test labels are not aligned")
    test_views, _ = build_views(test_audio["X"], test_image["X"])

    final_proba, final_artifact, final_fit_predict_time = fit_predict_full(
        selected_spec,
        candidates,
        train_views,
        test_views,
        y_train,
    )
    final_pred = cv.predict_with_bias(final_proba, selected_bias)
    final_row = report_row_for_prediction(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_audio["y"],
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=selected_spec.view,
        n_features=int(selected.get("n_features", view_dims.get(selected_spec.view, 0))),
        train_time_sec=final_fit_predict_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_grouped_multimodal_oof_cv_only",
            "selected_oof_macro_f1": selected["tuned_oof_macro_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )

    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "audio_path", "image_path", "label", "y", "group_key", "source"]
    ].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_audio["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "classical_multimodal_cv_select_no_test_until_final",
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "audio_feature_set": base.FEATURE_SET,
            "view_dims": view_dims,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "classical_multimodal_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "audio_feature_cache_dir": str(args.audio_feature_cache_dir.resolve()),
        "image_feature_dir": str(image_feature_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_audio_timing": test_audio_timing,
        "test_image_timing": test_image_timing,
        "artifacts": {
            "oof_leaderboard": str(leaderboard_path.resolve()),
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

    print("\nFinal robot/test result after frozen multimodal CV selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "feature_set",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
            ]
        ].to_string(index=False)
    )
    print("\nSaved artifacts:")
    print("OOF leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
