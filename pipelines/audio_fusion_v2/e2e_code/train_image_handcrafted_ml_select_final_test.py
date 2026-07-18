from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from lightgbm import LGBMClassifier


LABEL_MAP = {"ambient": 0, "leaf": 1, "trunk": 2, "twig": 3}
ID2LABEL = {value: key for key, value in LABEL_MAP.items()}
CLASS_NAMES = [ID2LABEL[index] for index in range(4)]
LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = [1, 2, 3]
IMAGE_SIZE = (128, 96)  # width, height for cv2.resize
FEATURE_FAMILY = "handcrafted_image_v1"
COMPACT_DIM = 56 + 30 + 28 + 17


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    view: str
    kind: str
    factory: Callable[[], object] | None = None
    members: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only handcrafted-feature + classical-ML selection. The script "
            "splits hand/default into train/val, selects only from val metrics, writes "
            "a selection lock, and only then evaluates the frozen choice on robot/test."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory containing audio_visual_dataset_default and audio_visual_dataset_robo_default.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_handcrafted_ml_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["segment", "specimen", "row"],
        default="segment",
        help="Group by segment by default so windows from the same segment cannot cross train/val.",
    )
    parser.add_argument(
        "--image-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/multimodal_classical_cv_select/image_features"),
        help=(
            "Cache directory for handcrafted image features. Reuses the existing "
            "image-only cache when labels align; robot/test is still loaded only "
            "after selected_without_test.json is written."
        ),
    )
    parser.add_argument("--force-rebuild-image", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["tuned_val_macro_f1", "default_val_macro_f1", "accuracy_4class"],
        default="tuned_val_macro_f1",
    )
    parser.add_argument(
        "--profile",
        choices=["balanced", "fast"],
        default="balanced",
        help="Balanced tries the full candidate set; fast skips the slower boosting/SVM candidates.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def require_file(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


def resolve_root(user_root: Path | None) -> Path:
    candidates = []
    if user_root is not None:
        candidates.append(user_root)
    candidates.extend(
        [
            Path("tree_structures"),
            Path("../tree_base/tree_structures"),
            Path("../tree/tree_structures"),
            Path("../Audio_Tree/tree_structures"),
            Path("/home/ttung05/Desktop/tree_base/tree_structures"),
            Path("/home/ttung05/Desktop/tree/tree_structures"),
            Path("/home/ttung05/Desktop/Audio_Tree/tree_structures"),
            Path("/home/ttung05/Desktop/tree_audio (copy)/tree_structures"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (
            (candidate / "audio_visual_dataset_default" / "dataset.csv").exists()
            and (candidate / "audio_visual_dataset_robo_default" / "dataset.csv").exists()
        ):
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find dataset root. Searched:\n{searched}")


def segment_group_key(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"_window_\d+.*$", "", stem)


def specimen_group_key(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"_segment_.*$", "", stem)


def load_image_manifest(csv_path: Path, dataset_dir: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"image_file", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} is missing columns: {sorted(required - set(frame.columns))}")

    output = pd.DataFrame(
        {
            "audio_file": frame.get("audio_file", pd.Series([""] * len(frame))).astype(str),
            "image_file": frame["image_file"].astype(str),
            "image_path": frame["image_file"].map(lambda value: dataset_dir / str(value)),
            "label": frame["category"].astype(str).str.lower(),
            "source": source,
        }
    )
    output = output[output["label"].isin(LABEL_MAP)].copy()
    output["y"] = output["label"].map(LABEL_MAP).astype(np.int64)
    output["segment_group"] = output["image_file"].map(segment_group_key)
    output["specimen_group"] = output["image_file"].map(specimen_group_key)
    output = output.reset_index(drop=True)
    if output.empty:
        raise ValueError(f"Empty manifest after label filtering: {csv_path}")

    missing_labels = set(CLASS_NAMES) - set(output["label"].unique())
    if missing_labels:
        raise ValueError(f"{source} is missing classes: {sorted(missing_labels)}")
    return output


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {ID2LABEL[index]: int((y == index).sum()) for index in LABELS}


def split_train_val(
    frame: pd.DataFrame,
    val_size: float,
    random_state: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not 0.05 <= val_size <= 0.5:
        raise ValueError("--val-size must be between 0.05 and 0.5")

    y = frame["y"].to_numpy()
    indices = np.arange(len(frame))
    if split_mode == "row":
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            stratify=y,
            random_state=random_state,
        )
        return np.sort(train_idx), np.sort(val_idx), {
            "mode": "row",
            "note": "Stratified row split; use only when grouped split is intentionally disabled.",
        }

    group_column = "segment_group" if split_mode == "segment" else "specimen_group"
    groups = frame[group_column].to_numpy()
    n_splits = max(2, int(round(1.0 / val_size)))
    full_distribution = np.bincount(y, minlength=len(CLASS_NAMES)) / len(y)
    best_candidate = None

    for seed_offset in range(50):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + seed_offset,
        )
        for train_idx, val_idx in splitter.split(indices, y, groups):
            val_distribution = np.bincount(y[val_idx], minlength=len(CLASS_NAMES)) / len(val_idx)
            class_penalty = 0 if len(set(y[train_idx])) == 4 and len(set(y[val_idx])) == 4 else 10
            size_penalty = abs((len(val_idx) / len(frame)) - val_size)
            distribution_penalty = float(np.abs(val_distribution - full_distribution).sum())
            score = class_penalty + size_penalty + distribution_penalty
            candidate = (score, train_idx, val_idx, random_state + seed_offset)
            if best_candidate is None or score < best_candidate[0]:
                best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("Could not build a grouped train/val split")

    _, train_idx, val_idx, split_seed = best_candidate
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    overlap = sorted(train_groups.intersection(val_groups))
    if overlap:
        raise AssertionError(f"Grouped split leaked {len(overlap)} groups. Example: {overlap[:5]}")

    split_info = {
        "mode": split_mode,
        "group_column": group_column,
        "split_seed": split_seed,
        "n_unique_groups": int(pd.Series(groups).nunique()),
        "train_unique_groups": len(train_groups),
        "val_unique_groups": len(val_groups),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_label_counts": label_counts(y[train_idx]),
        "val_label_counts": label_counts(y[val_idx]),
    }
    return np.sort(train_idx), np.sort(val_idx), split_info


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
            channel_stats(
                np.moveaxis(
                    hsv.astype(np.float32) / np.asarray([180.0, 255.0, 255.0]),
                    -1,
                    0,
                )
            ),
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
) -> tuple[dict[str, np.ndarray], dict[str, float | bool | int | str]]:
    paths = image_cache_paths(feature_dir, split_name)
    signature = image_manifest_signature(frame)
    ready = all(path.exists() for path in paths.values())
    y_expected = frame["y"].to_numpy(dtype=np.int64)

    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        cache_y = np.load(paths["y"])
        signature_matches = metadata.get("manifest_signature") == signature
        y_aligned = len(cache_y) == len(y_expected) and np.array_equal(cache_y, y_expected)
        valid_metadata = (
            metadata.get("feature_family") == FEATURE_FAMILY
            and int(metadata.get("n_samples", -1)) == len(frame)
            and int(metadata.get("feature_dim", -1)) == 1391
            and (signature_matches or y_aligned)
        )
        if valid_metadata:
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": cache_y,
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            load_time = time.perf_counter() - start
            print(
                f"Loaded image cache {split_name}: {payload['X'].shape} "
                f"in {load_time:.3f}s signature_match={signature_matches}"
            )
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
                "extraction_time_per_sample_sec": float(metadata["extraction_time_per_sample_sec"]),
                "missing_images": int(metadata.get("missing_images", 0)),
                "cache_validation": "signature" if signature_matches else "label_order",
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
    if payload["X"].shape != (len(frame), 1391):
        raise AssertionError(f"{split_name}: expected {(len(frame), 1391)}, got {payload['X'].shape}")

    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    metadata = {
        "feature_family": FEATURE_FAMILY,
        "feature_dim": int(payload["X"].shape[1]),
        "image_size": list(IMAGE_SIZE),
        "n_samples": len(frame),
        "manifest_signature": signature,
        "missing_images": missing_count,
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved image cache {split_name}: {payload['X'].shape} in {extraction_time:.2f}s missing={missing_count}")
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
        "missing_images": missing_count,
        "cache_validation": "rebuilt",
    }


def build_views(X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    views = {
        "compact": X[:, :COMPACT_DIM],
        "hog": X[:, COMPACT_DIM:],
        "full": X,
    }
    dims = {name: int(matrix.shape[1]) for name, matrix in views.items()}
    return views, dims


def make_candidates(random_state: int, profile: str) -> dict[str, CandidateSpec]:
    class_weights = {0: 0.6, 1: 1.1, 2: 1.8, 3: 1.3}

    def logreg(c: float = 1.0, class_weight: object = "balanced") -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        max_iter=4000,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def hgb() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=360,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.08,
            class_weight=class_weights,
            random_state=random_state,
        )

    def lgbm() -> LGBMClassifier:
        return LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=650,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    specs = [
        CandidateSpec("compact_logreg_C1_balanced", "compact", "direct", lambda: logreg(1.0, "balanced")),
        CandidateSpec("compact_logreg_C3_weighted", "compact", "direct", lambda: logreg(3.0, class_weights)),
        CandidateSpec("full_logreg_C1_balanced", "full", "direct", lambda: logreg(1.0, "balanced")),
        CandidateSpec("full_logreg_C3_weighted", "full", "direct", lambda: logreg(3.0, class_weights)),
        CandidateSpec(
            "full_fclassif200_logreg",
            "full",
            "direct",
            lambda: Pipeline(
                [
                    ("select", SelectKBest(score_func=f_classif, k=200)),
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced", random_state=random_state, n_jobs=-1)),
                ]
            ),
        ),
        CandidateSpec(
            "full_fclassif800_logreg",
            "full",
            "direct",
            lambda: Pipeline(
                [
                    ("select", SelectKBest(score_func=f_classif, k=800)),
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced", random_state=random_state, n_jobs=-1)),
                ]
            ),
        ),
        CandidateSpec(
            "full_pca128_logreg",
            "full",
            "direct",
            lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=128, whiten=True, random_state=random_state)),
                    ("model", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", random_state=random_state, n_jobs=-1)),
                ]
            ),
        ),
        CandidateSpec(
            "full_pca128_rbf_svm",
            "full",
            "direct",
            lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=128, whiten=True, random_state=random_state)),
                    (
                        "model",
                        SVC(
                            C=3.0,
                            kernel="rbf",
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]
    if profile == "balanced":
        specs.extend(
            [
                CandidateSpec("compact_hgb_regularized", "compact", "direct", hgb),
                CandidateSpec("compact_lgbm_regularized", "compact", "direct", lgbm),
                CandidateSpec(
                    "compact_extra_trees",
                    "compact",
                    "direct",
                    lambda: ExtraTreesClassifier(
                        n_estimators=800,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight=class_weights,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                CandidateSpec(
                    "compact_random_forest",
                    "compact",
                    "direct",
                    lambda: RandomForestClassifier(
                        n_estimators=600,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                CandidateSpec("full_hgb_regularized", "full", "direct", hgb),
                CandidateSpec("full_lgbm_regularized", "full", "direct", lgbm),
                CandidateSpec(
                    "full_extra_trees",
                    "full",
                    "direct",
                    lambda: ExtraTreesClassifier(
                        n_estimators=800,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight=class_weights,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                CandidateSpec(
                    "full_random_forest",
                    "full",
                    "direct",
                    lambda: RandomForestClassifier(
                        n_estimators=600,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return {spec.name: spec for spec in specs}


def proba_aligned(model: object, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not expose predict_proba")
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    output = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        target = int(np.where(LABELS == class_id)[0][0])
        output[:, target] = raw[:, column]
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    scores = np.log(np.clip(proba, 1e-12, 1.0)) + bias.reshape(1, -1)
    return scores.argmax(axis=1).astype(np.int64)


def tune_class_bias(proba: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    grid = np.linspace(-1.2, 1.2, 13)
    best_bias = np.zeros(4, dtype=np.float64)
    best_score = f1_score(y_true, predict_with_bias(proba, best_bias), average="macro", zero_division=0)

    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        score = f1_score(y_true, predict_with_bias(proba, bias), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_bias = bias

    for ambient_bias in np.linspace(-0.8, 0.8, 9):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        score = f1_score(y_true, predict_with_bias(proba, bias), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_bias = bias
    return best_bias, float(best_score)


def make_report_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_set: str,
    feature_name: str,
    n_features: int,
    train_time_sec: float,
    predict_time_sec: float,
    status: str = "ok",
) -> dict[str, object]:
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    y_true_binary = (y_true > 0).astype(np.int64)
    y_pred_binary = (y_pred > 0).astype(np.int64)
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    row = {
        "feature_set": feature_set,
        "feature_name": feature_name,
        "n_features": int(n_features),
        "split": split_name,
        "model": model_name,
        "status": status,
        "accuracy_4class": accuracy_score(y_true, y_pred),
        "macro_precision_4class": macro_precision,
        "macro_recall_4class": macro_recall,
        "macro_f1_4class": macro_f1,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "contact_macro_f1": f1_score(y_true, y_pred, labels=CONTACT_LABELS, average="macro", zero_division=0),
        "binary_accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "binary_macro_precision": binary_precision,
        "binary_macro_recall": binary_recall,
        "binary_macro_f1": binary_f1,
        "binary_contact_f1": f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0),
        "model_train_time_sec": train_time_sec,
        "model_predict_time_sec": predict_time_sec,
        "model_total_time_sec": train_time_sec + predict_time_sec,
    }
    for index, class_name in enumerate(CLASS_NAMES):
        row[f"{class_name}_precision"] = precision[index]
        row[f"{class_name}_recall"] = recall[index]
        row[f"{class_name}_f1"] = class_f1[index]
        row[f"{class_name}_support"] = support[index]
    return row


def evaluate_direct_candidate(
    spec: CandidateSpec,
    train_views: dict[str, np.ndarray],
    val_views: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict, np.ndarray, object]:
    print(f"\nVAL candidate: {spec.name} view={spec.view}")
    model = spec.factory()
    start = time.perf_counter()
    model.fit(train_views[spec.view], y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    val_proba = proba_aligned(model, val_views[spec.view])
    predict_time = time.perf_counter() - start
    default_pred = val_proba.argmax(axis=1)
    bias, tuned_macro_f1 = tune_class_bias(val_proba, y_val)
    tuned_pred = predict_with_bias(val_proba, bias)

    row = make_report_row(
        model_name=spec.name,
        split_name="hand_val",
        y_true=y_val,
        y_pred=tuned_pred,
        feature_set=spec.view,
        feature_name=spec.view,
        n_features=train_views[spec.view].shape[1],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )
    row.update(
        {
            "candidate_kind": spec.kind,
            "view": spec.view,
            "default_val_macro_f1": f1_score(y_val, default_pred, average="macro", zero_division=0),
            "tuned_val_macro_f1": tuned_macro_f1,
            "class_bias_json": json.dumps(bias.tolist()),
            "members_json": json.dumps(list(spec.members)),
        }
    )
    print(
        f"  default={row['default_val_macro_f1']:.4f} tuned={tuned_macro_f1:.4f} "
        f"acc={row['accuracy_4class']:.4f} time={train_time:.2f}s"
    )
    return row, val_proba, model


def build_ensemble_rows(
    leaderboard: pd.DataFrame,
    val_probas: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> tuple[list[dict], dict[str, CandidateSpec], dict[str, np.ndarray]]:
    rows = []
    specs = {}
    probas = {}
    ranked = leaderboard.sort_values("tuned_val_macro_f1", ascending=False)["model"].tolist()
    recipes = {
        "image_ensemble_top2": tuple(ranked[:2]),
        "image_ensemble_top3": tuple(ranked[:3]),
        "image_ensemble_top5": tuple(ranked[:5]),
    }
    for name, members in recipes.items():
        if len(members) < 2:
            continue
        proba = np.mean([val_probas[member] for member in members], axis=0)
        default_pred = proba.argmax(axis=1)
        bias, tuned_macro_f1 = tune_class_bias(proba, y_val)
        tuned_pred = predict_with_bias(proba, bias)
        row = make_report_row(
            model_name=name,
            split_name="hand_val",
            y_true=y_val,
            y_pred=tuned_pred,
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
                "default_val_macro_f1": f1_score(y_val, default_pred, average="macro", zero_division=0),
                "tuned_val_macro_f1": tuned_macro_f1,
                "class_bias_json": json.dumps(bias.tolist()),
                "members_json": json.dumps(list(members)),
            }
        )
        rows.append(row)
        specs[name] = CandidateSpec(name=name, view="ensemble", kind="ensemble", members=members)
        probas[name] = proba
        print(f"\nVAL ensemble: {name} members={members} tuned={tuned_macro_f1:.4f}")
    return rows, specs, probas


def fit_predict_full(
    spec: CandidateSpec,
    candidates: dict[str, CandidateSpec],
    train_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    y_train: np.ndarray,
) -> tuple[np.ndarray, object, float]:
    start = time.perf_counter()
    if spec.kind == "direct":
        model = clone(spec.factory())
        model.fit(train_views[spec.view], y_train)
        proba = proba_aligned(model, test_views[spec.view])
        return proba, {"kind": "direct", "view": spec.view, "model": model}, time.perf_counter() - start

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
        return np.mean(probas, axis=0), {"kind": "ensemble", "members": artifacts}, time.perf_counter() - start

    raise ValueError(f"Unsupported candidate kind: {spec.kind}")


def save_split_manifests(
    run_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, str]:
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "audio_file",
        "image_file",
        "image_path",
        "label",
        "y",
        "segment_group",
        "specimen_group",
        "source",
    ]
    paths = {
        "train_inner": split_dir / "hand_train_inner.csv",
        "val": split_dir / "hand_val.csv",
        "train_full": split_dir / "hand_train_full.csv",
    }
    train_df.iloc[train_idx][columns].to_csv(paths["train_inner"], index=False)
    train_df.iloc[val_idx][columns].to_csv(paths["val"], index=False)
    train_df[columns].to_csv(paths["train_full"], index=False)
    if test_df is not None:
        paths["test"] = split_dir / "robot_test.csv"
        test_df[columns].to_csv(paths["test"], index=False)
    return {key: str(value.resolve()) for key, value in paths.items()}


def main() -> None:
    args = parse_args()
    root_path = resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir, args.image_feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH          =", root_path.resolve())
    print("RUN_DIR            =", run_dir.resolve())
    print("IMAGE_CACHE_DIR    =", args.image_feature_cache_dir.resolve())
    print("Protocol           = image-only train/val selection; robot/test after lock")

    train_csv = require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv")
    train_df = load_image_manifest(train_csv, train_dataset_dir, "hand_train")
    train_payload, train_image_timing = build_or_load_image_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.image_feature_cache_dir,
        force_rebuild=args.force_rebuild_image,
    )
    y_train_full = train_payload["y"]
    if not np.array_equal(y_train_full, train_df["y"].to_numpy()):
        raise AssertionError("Train image cache labels do not align with manifest")

    train_idx, val_idx, split_info = split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = save_split_manifests(run_dir, train_df, None, train_idx, val_idx)

    X_train_inner = train_payload["X"][train_idx]
    y_train_inner = y_train_full[train_idx]
    X_val = train_payload["X"][val_idx]
    y_val = y_train_full[val_idx]
    train_views_inner, view_dims = build_views(X_train_inner)
    val_views, _ = build_views(X_val)

    split_summary = {
        "protocol": "image_only_handcrafted_ml_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_full_samples": int(len(train_df)),
        "train_full_label_counts": label_counts(y_train_full),
        "view_dims": view_dims,
        "train_image_timing": train_image_timing,
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    candidates = make_candidates(args.random_state, args.profile)
    leaderboard_rows = []
    val_probas = {}
    fitted_val_models = {}
    for spec in candidates.values():
        row, proba, model = evaluate_direct_candidate(
            spec,
            train_views_inner,
            val_views,
            y_train_inner,
            y_val,
        )
        leaderboard_rows.append(row)
        val_probas[spec.name] = proba
        fitted_val_models[spec.name] = model

    initial_leaderboard = pd.DataFrame(leaderboard_rows)
    ensemble_rows, ensemble_specs, ensemble_probas = build_ensemble_rows(
        initial_leaderboard,
        val_probas,
        y_val,
    )
    leaderboard_rows.extend(ensemble_rows)
    candidates.update(ensemble_specs)
    val_probas.update(ensemble_probas)

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
        "selected_kind": selected_spec.kind,
        "selected_view": selected_spec.view,
        "selected_members": list(selected_spec.members),
        "selected_bias": selected_bias.tolist(),
        "selection_metric": args.selection_metric,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float))

    test_csv = require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv")
    test_df = load_image_manifest(test_csv, test_dataset_dir, "robot_test")
    split_paths = save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    test_payload, test_image_timing = build_or_load_image_cache(
        test_df,
        "robot_test",
        feature_dir=args.image_feature_cache_dir,
        force_rebuild=args.force_rebuild_image,
    )
    if not np.array_equal(test_payload["y"], test_df["y"].to_numpy()):
        raise AssertionError("Test image cache labels do not align with manifest")

    train_full_views, _ = build_views(train_payload["X"])
    test_views, _ = build_views(test_payload["X"])
    final_proba, final_artifact, final_fit_time = fit_predict_full(
        selected_spec,
        candidates,
        train_full_views,
        test_views,
        y_train_full,
    )
    final_pred = predict_with_bias(final_proba, selected_bias)
    final_row = make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_payload["y"],
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=selected_spec.view,
        n_features=0 if selected_spec.kind == "ensemble" else train_full_views[selected_spec.view].shape[1],
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_tuned_val_macro_f1": selected.get("tuned_val_macro_f1"),
            "selected_default_val_macro_f1": selected.get("default_val_macro_f1"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )

    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_columns = [
        "audio_file",
        "image_file",
        "image_path",
        "label",
        "y",
        "segment_group",
        "specimen_group",
        "source",
    ]
    prediction_frame = test_df[prediction_columns].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(ID2LABEL)
    for class_id, class_name in ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_payload["y"], final_pred, labels=LABELS),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "image_only_handcrafted_ml_select_no_test_until_lock",
            "selected": selection_summary,
            "artifact": final_artifact,
            "selected_bias": selected_bias,
            "view_dims": view_dims,
            "label_map": LABEL_MAP,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "image_only_handcrafted_ml_select_no_test_until_lock",
        "root": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "split_summary": split_summary,
        "selection": selection_summary,
        "test_image_timing": test_image_timing,
        "final_test_report": final_row,
        "artifacts": {
            "val_leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "model_bundle": str(bundle_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    protocol_summary_path = report_dir / f"{args.run_slug}_protocol_summary.json"
    write_json(protocol_summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen image-only selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_tuned_val_macro_f1",
            ]
        ].to_string(index=False)
    )
    print("Val leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Predictions:", predictions_path.resolve())


if __name__ == "__main__":
    main()
