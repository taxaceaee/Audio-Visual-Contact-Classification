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
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import train_image_handcrafted_ml_select_final_test as img


FEATURE_FAMILY = "handcrafted_image_spatial_v2"
SPATIAL_IMAGE_SIZE = (128, 96)
THUMB_SIZE = (32, 24)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    view: str
    factory: Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only handcrafted spatial-v2 features + classical ML. Uses only "
            "hand/default train/val for feature/model/bias selection, writes the "
            "selection lock, then evaluates the frozen choice once on robot/test."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_spatial_v2_specimen_select")
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
    parser.add_argument(
        "--bias-penalty",
        type=float,
        default=0.025,
        help="Train-only regularizer: subtract this times L1 class-bias from validation macro F1.",
    )
    return parser.parse_args()


def spatial_cache_paths(feature_dir: Path, split_name: str) -> dict[str, Path]:
    split_dir = feature_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": split_dir / "X.npy",
        "y": split_dir / "y.npy",
        "paths": split_dir / "paths.npy",
        "metadata": split_dir / "metadata.json",
    }


def spatial_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["image_path", "y"]].itertuples(index=False):
        digest.update(f"{row.image_path}|{int(row.y)}\n".encode("utf-8"))
    return digest.hexdigest()


def hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    counts, _ = np.histogram(values.ravel(), bins=bins, range=value_range)
    counts = counts.astype(np.float32)
    total = float(counts.sum())
    if total > 0.0:
        counts /= total
    return counts


def channel_moments(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32).reshape(-1, array.shape[-1])
    pieces = []
    for col in range(values.shape[1]):
        channel = values[:, col]
        pieces.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(np.percentile(channel, 10)),
                float(np.percentile(channel, 50)),
                float(np.percentile(channel, 90)),
            ]
        )
    return np.asarray(pieces, dtype=np.float32)


def grid_stats(rgb: np.ndarray, hsv: np.ndarray, gray_u8: np.ndarray, grid: int = 4) -> np.ndarray:
    edges = (cv2.Canny(gray_u8, 60, 160) > 0).astype(np.float32)
    h, w = gray_u8.shape
    features = []
    for rows in np.array_split(np.arange(h), grid):
        for cols in np.array_split(np.arange(w), grid):
            rgb_cell = rgb[np.ix_(rows, cols)].astype(np.float32) / 255.0
            hsv_cell = hsv[np.ix_(rows, cols)].astype(np.float32) / np.asarray([180.0, 255.0, 255.0])
            edge_cell = edges[np.ix_(rows, cols)]
            features.extend(rgb_cell.reshape(-1, 3).mean(axis=0).tolist())
            features.extend(rgb_cell.reshape(-1, 3).std(axis=0).tolist())
            features.extend(hsv_cell.reshape(-1, 3).mean(axis=0).tolist())
            features.extend(hsv_cell.reshape(-1, 3).std(axis=0).tolist())
            features.append(float(edge_cell.mean()))
    return np.asarray(features, dtype=np.float32)


def crop_features(rgb: np.ndarray, hsv: np.ndarray, gray_u8: np.ndarray) -> np.ndarray:
    h, w = gray_u8.shape
    boxes = {
        "center": (h // 4, 3 * h // 4, w // 4, 3 * w // 4),
        "bottom": (h // 2, h, 0, w),
        "top": (0, h // 2, 0, w),
        "left": (0, h, 0, w // 2),
        "right": (0, h, w // 2, w),
    }
    pieces = []
    for r0, r1, c0, c1 in boxes.values():
        rgb_crop = rgb[r0:r1, c0:c1].astype(np.float32) / 255.0
        hsv_crop = hsv[r0:r1, c0:c1].astype(np.float32)
        gray_crop = gray_u8[r0:r1, c0:c1].astype(np.float32) / 255.0
        pieces.append(channel_moments(rgb_crop))
        pieces.append(channel_moments(hsv_crop / np.asarray([180.0, 255.0, 255.0])))
        pieces.append(hist(hsv_crop[:, :, 0], 12, (0.0, 180.0)))
        pieces.append(hist(hsv_crop[:, :, 1], 8, (0.0, 255.0)))
        pieces.append(hist(hsv_crop[:, :, 2], 8, (0.0, 255.0)))
        pieces.append(
            np.asarray(
                [
                    float(gray_crop.mean()),
                    float(gray_crop.std()),
                    float((gray_crop > 0.65).mean()),
                    float((gray_crop < 0.25).mean()),
                ],
                dtype=np.float32,
            )
        )
    return np.concatenate(pieces).astype(np.float32)


def extract_spatial_features(path: Path) -> tuple[np.ndarray, bool]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    missing = image is None
    if missing:
        image = np.zeros((SPATIAL_IMAGE_SIZE[1], SPATIAL_IMAGE_SIZE[0], 3), dtype=np.uint8)
    else:
        image = cv2.resize(image, SPATIAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thumb_rgb = cv2.resize(rgb, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    thumb_hsv = cv2.resize(hsv, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb_hsv = thumb_hsv / np.asarray([180.0, 255.0, 255.0])
    thumb_gray = cv2.resize(gray_u8, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    grad_x = cv2.Sobel(thumb_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(thumb_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    pieces = [
        thumb_rgb.ravel(),
        thumb_hsv.ravel(),
        thumb_gray.ravel(),
        grad_mag.ravel(),
        grid_stats(rgb, hsv, gray_u8, grid=4),
        crop_features(rgb, hsv, gray_u8),
    ]
    features = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Spatial feature contains NaN/Inf: {path}")
    return features, missing


def build_or_load_spatial_cache(
    frame: pd.DataFrame,
    split_name: str,
    feature_dir: Path,
    force_rebuild: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool | int | str]]:
    paths = spatial_cache_paths(feature_dir, split_name)
    signature = spatial_signature(frame)
    ready = all(path.exists() for path in paths.values())
    y_expected = frame["y"].to_numpy(dtype=np.int64)
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        cache_y = np.load(paths["y"])
        signature_matches = metadata.get("manifest_signature") == signature
        y_aligned = len(cache_y) == len(y_expected) and np.array_equal(cache_y, y_expected)
        if (
            metadata.get("feature_family") == FEATURE_FAMILY
            and int(metadata.get("n_samples", -1)) == len(frame)
            and (signature_matches or y_aligned)
        ):
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": cache_y,
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            load_time = time.perf_counter() - start
            print(f"Loaded spatial cache {split_name}: {payload['X'].shape} in {load_time:.3f}s")
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
                "missing_images": int(metadata.get("missing_images", 0)),
                "cache_validation": "signature" if signature_matches else "label_order",
            }
        print(f"Spatial cache metadata mismatch for {split_name}; rebuilding.")

    rows = []
    labels = []
    image_paths = []
    missing_count = 0
    start = time.perf_counter()
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"Extract spatial/{split_name}"):
        features, missing = extract_spatial_features(Path(row.image_path))
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
        "feature_family": FEATURE_FAMILY,
        "feature_dim": int(payload["X"].shape[1]),
        "n_samples": len(frame),
        "manifest_signature": signature,
        "missing_images": missing_count,
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
        "spatial_image_size": list(SPATIAL_IMAGE_SIZE),
        "thumb_size": list(THUMB_SIZE),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved spatial cache {split_name}: {payload['X'].shape} in {extraction_time:.2f}s missing={missing_count}")
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
        "missing_images": missing_count,
        "cache_validation": "rebuilt",
    }


def make_views(X_v1: np.ndarray, X_spatial: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    views = {
        "v1_compact": X_v1[:, : img.COMPACT_DIM],
        "v1_full": X_v1,
        "spatial": X_spatial,
        "v1_spatial": np.hstack([X_v1, X_spatial]).astype(np.float32),
    }
    return views, {key: int(value.shape[1]) for key, value in views.items()}


def make_candidates(random_state: int) -> dict[str, CandidateSpec]:
    class_weights = {0: 0.6, 1: 1.1, 2: 1.8, 3: 1.3}

    def logreg(c: float = 1.0, class_weight: object = "balanced") -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight=class_weight,
                        max_iter=5000,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def select_logreg(k: int, c: float = 1.5) -> Pipeline:
        return Pipeline(
            [
                ("select", SelectKBest(score_func=f_classif, k=k)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def pca_logreg(n_components: int) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_components, whiten=True, random_state=random_state)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    specs = [
        CandidateSpec("v1_compact_logreg_balanced", "v1_compact", lambda: logreg(1.0, "balanced")),
        CandidateSpec("v1_compact_logreg_weighted", "v1_compact", lambda: logreg(3.0, class_weights)),
        CandidateSpec("spatial_logreg_balanced", "spatial", lambda: logreg(1.0, "balanced")),
        CandidateSpec("spatial_logreg_weighted", "spatial", lambda: logreg(2.0, class_weights)),
        CandidateSpec("spatial_fclassif500_logreg", "spatial", lambda: select_logreg(500)),
        CandidateSpec("spatial_fclassif1000_logreg", "spatial", lambda: select_logreg(1000)),
        CandidateSpec("spatial_pca128_logreg", "spatial", lambda: pca_logreg(128)),
        CandidateSpec("v1_full_fclassif200_logreg", "v1_full", lambda: select_logreg(200)),
        CandidateSpec("v1_spatial_fclassif500_logreg", "v1_spatial", lambda: select_logreg(500)),
        CandidateSpec("v1_spatial_fclassif1000_logreg", "v1_spatial", lambda: select_logreg(1000)),
        CandidateSpec("v1_spatial_fclassif2000_logreg", "v1_spatial", lambda: select_logreg(2000)),
        CandidateSpec("v1_spatial_pca128_logreg", "v1_spatial", lambda: pca_logreg(128)),
        CandidateSpec("v1_spatial_pca256_logreg", "v1_spatial", lambda: pca_logreg(256)),
    ]
    return {spec.name: spec for spec in specs}


def evaluate_candidate(
    spec: CandidateSpec,
    train_views: dict[str, np.ndarray],
    val_views: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    bias_penalty: float,
) -> tuple[list[dict], np.ndarray]:
    print(f"\nVAL candidate: {spec.name} view={spec.view}", flush=True)
    model = spec.factory()
    start = time.perf_counter()
    model.fit(train_views[spec.view], y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    val_proba = img.proba_aligned(model, val_views[spec.view])
    predict_time = time.perf_counter() - start
    tuned_bias, _ = img.tune_class_bias(val_proba, y_val)
    rows = []
    for bias_mode, bias in [("none", np.zeros(4, dtype=np.float64)), ("tuned", tuned_bias)]:
        pred = img.predict_with_bias(val_proba, bias)
        row = img.make_report_row(
            model_name=spec.name,
            split_name="hand_val",
            y_true=y_val,
            y_pred=pred,
            feature_set=spec.view,
            feature_name=f"{spec.view}+spatial_v2",
            n_features=train_views[spec.view].shape[1],
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
                "default_window_macro_f1": f1_score(y_val, val_proba.argmax(axis=1), average="macro", zero_division=0),
            }
        )
        print(
            f"  {bias_mode:5s} macro={row['macro_f1_4class']:.4f} "
            f"reg={row['regularized_val_score']:.4f} bias_l1={bias_l1:.2f}",
            flush=True,
        )
        rows.append(row)
    return rows, val_proba


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

    print("ROOT_PATH    =", root_path.resolve(), flush=True)
    print("RUN_DIR      =", run_dir.resolve(), flush=True)
    print("Protocol     = image-only spatial-v2 specimen train/val selection; robot/test after lock", flush=True)

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
    train_spatial, train_spatial_timing = build_or_load_spatial_cache(
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
    train_views, view_dims = make_views(train_v1["X"][train_idx], train_spatial["X"][train_idx])
    val_views, _ = make_views(train_v1["X"][val_idx], train_spatial["X"][val_idx])
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]

    split_summary = {
        "protocol": "image_only_spatial_v2_specimen_train_val_selection_robot_after_lock",
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
        candidate_rows, _ = evaluate_candidate(
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
    test_spatial, test_spatial_timing = build_or_load_spatial_cache(
        test_df,
        "robot_test",
        feature_dir=args.spatial_feature_cache_dir,
        force_rebuild=args.force_rebuild_spatial,
    )
    train_full_views, _ = make_views(train_v1["X"], train_spatial["X"])
    test_views, _ = make_views(test_v1["X"], test_spatial["X"])
    start = time.perf_counter()
    final_model = clone(selected_spec.factory())
    final_model.fit(train_full_views[selected_spec.view], y_full)
    final_fit_time = time.perf_counter() - start
    final_proba = img.proba_aligned(final_model, test_views[selected_spec.view])
    final_pred = img.predict_with_bias(final_proba, selected_bias)
    final_row = img.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_v1["y"],
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=f"{selected_spec.view}+spatial_v2",
        n_features=train_full_views[selected_spec.view].shape[1],
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_spatial_v2",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_bias_mode": selected.get("bias_mode"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(test_v1["y"], final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "image_path", "label", "y", "segment_group", "specimen_group", "source"]
    ].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(img.ID2LABEL)
    for class_id, class_name in img.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "image_only_spatial_v2_no_test_until_lock",
            "selected": selection_summary,
            "model": final_model,
            "selected_bias": selected_bias,
            "label_map": img.LABEL_MAP,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "image_only_spatial_v2_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "test_v1_timing": test_v1_timing,
        "test_spatial_timing": test_spatial_timing,
        "final_test_report": final_row,
        "artifacts": {
            "val_leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "model_bundle": str(bundle_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    img.write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen image-only spatial-v2 selection:", flush=True)
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
