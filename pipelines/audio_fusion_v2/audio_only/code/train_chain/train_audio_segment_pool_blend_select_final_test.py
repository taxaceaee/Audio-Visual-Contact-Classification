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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lightgbm import LGBMClassifier

import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_highsr_temporal_tta_select_final_test as highsr
import train_cv_select_final_test as cv
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
CONSENSUS_HIGHSR_WEIGHTS = [0.65, 0.70, 0.75, 0.80, 0.85]
MODEL_BLEND_WEIGHTS = [0.25, 0.50, 0.75]


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    kind: str
    direct_factory: Callable[[], object] | None = None
    binary_factory: Callable[[], object] | None = None
    contact_factory: Callable[[], object] | None = None
    model_member: str | None = None
    source_member: str | None = None
    model_weight: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only segment-pooling experiment. It aggregates window-level audio "
            "features to segment-level features, selects models/blends with hand OOF "
            "only, writes a selection lock, then evaluates once on robot/test."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_segment_pool_blend_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--pairwise-oof-cache",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        ),
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def softmax_scores(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray = LABELS) -> float:
    scores = []
    for label in labels:
        true_mask = y_true == label
        pred_mask = pred == label
        tp = float(np.sum(true_mask & pred_mask))
        fp = float(np.sum(~true_mask & pred_mask))
        fn = float(np.sum(true_mask & ~pred_mask))
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    scores = np.log(np.clip(proba, 1e-12, 1.0)) + bias.reshape(1, -1)
    return scores.argmax(axis=1).astype(np.int64)


def tune_class_bias_small(proba: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    grid = np.asarray([-0.6, 0.0, 0.6], dtype=np.float64)
    best_bias = np.zeros(4, dtype=np.float64)
    best_score = fast_macro_f1(y_true, predict_with_bias(proba, best_bias))
    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        score = fast_macro_f1(y_true, predict_with_bias(proba, bias))
        if score > best_score:
            best_score = score
            best_bias = bias
    for ambient_bias in np.asarray([-0.3, 0.0, 0.3], dtype=np.float64):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        score = fast_macro_f1(y_true, predict_with_bias(proba, bias))
        if score > best_score:
            best_score = score
            best_bias = bias
    return best_bias, float(best_score)


def proba_aligned(model: object, X: np.ndarray, labels: np.ndarray = LABELS) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    output = np.zeros((len(X), len(labels)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        if class_id in labels:
            target = int(np.where(labels == class_id)[0][0])
            output[:, target] = raw[:, column]
    return normalize(output)


def proba_positive(model: object, X: np.ndarray, positive_label: int = 1) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if positive_label not in classes:
        return np.zeros(len(X), dtype=np.float64)
    return raw[:, int(np.where(classes == positive_label)[0][0])]


def hierarchical_proba(binary_model: object, contact_model: object, X: np.ndarray) -> np.ndarray:
    contact_probability = np.clip(proba_positive(binary_model, X, positive_label=1), 1e-12, 1.0)
    contact_class_proba = proba_aligned(contact_model, X, labels=CONTACT_LABELS)
    output = np.zeros((len(X), 4), dtype=np.float64)
    output[:, 0] = 1.0 - contact_probability
    output[:, 1:4] = contact_probability[:, None] * contact_class_proba
    return normalize(output)


def segment_pool_features(X_group: np.ndarray) -> np.ndarray:
    pooled = [
        np.mean(X_group, axis=0),
        np.std(X_group, axis=0),
        np.min(X_group, axis=0),
        np.max(X_group, axis=0),
        np.percentile(X_group, 25, axis=0),
        np.percentile(X_group, 50, axis=0),
        np.percentile(X_group, 75, axis=0),
    ]
    count = float(len(X_group))
    extras = np.asarray([count, np.log1p(count), np.sqrt(count)], dtype=np.float32)
    return np.concatenate([np.concatenate(pooled), extras]).astype(np.float32)


def make_segment_dataset(frame: pd.DataFrame, X_window: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    group_codes, group_keys = pd.factorize(frame["group_key"].astype(str), sort=False)
    rows = []
    features = []
    y_window = frame["y"].to_numpy(dtype=np.int64)
    for segment_id, group_key in enumerate(group_keys):
        indices = np.where(group_codes == segment_id)[0]
        labels = np.unique(y_window[indices])
        if len(labels) != 1:
            raise AssertionError(f"Segment group has mixed labels: {group_key} -> {labels.tolist()}")
        first = frame.iloc[int(indices[0])]
        rows.append(
            {
                "segment_id": int(segment_id),
                "group_key": str(group_key),
                "audio_file": str(first["audio_file"]),
                "audio_path": str(first["audio_path"]),
                "label": base.ID2LABEL[int(labels[0])],
                "y": int(labels[0]),
                "specimen_group": cv.specimen_group_key(str(first["audio_file"])),
                "window_count": int(len(indices)),
            }
        )
        features.append(segment_pool_features(X_window[indices]))
    segment_frame = pd.DataFrame(rows)
    X_segment = np.vstack(features).astype(np.float32)
    return segment_frame, X_segment, group_codes.astype(np.int64)


def make_segment_splits(segment_frame: pd.DataFrame, n_folds: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(len(segment_frame))
    y = segment_frame["y"].to_numpy(dtype=np.int64)
    groups = segment_frame["specimen_group"].astype(str).to_numpy()
    splits = []
    for train_idx, val_idx in splitter.split(indices, y, groups):
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise AssertionError(f"Specimen leakage across folds: {sorted(overlap)[:5]}")
        if len(set(y[val_idx].tolist())) != len(LABELS):
            raise AssertionError("A segment CV fold is missing at least one class")
        splits.append((np.asarray(train_idx), np.asarray(val_idx)))
    return splits


def expand_segment_proba(segment_proba: np.ndarray, window_to_segment: np.ndarray) -> np.ndarray:
    return segment_proba[window_to_segment]


def segment_proba_from_window(frame: pd.DataFrame, window_proba: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_codes, group_keys = pd.factorize(frame["group_key"].astype(str), sort=False)
    n_groups = int(group_codes.max()) + 1
    if rule == "sum_log_proba":
        values = np.log(np.clip(window_proba, 1e-12, 1.0))
        scores = np.vstack(
            [np.bincount(group_codes, weights=values[:, class_id], minlength=n_groups) for class_id in LABELS]
        ).T
        return softmax_scores(scores), group_codes.astype(np.int64), np.asarray(group_keys.astype(str))
    if rule == "mean_proba":
        scores = np.vstack(
            [np.bincount(group_codes, weights=window_proba[:, class_id], minlength=n_groups) for class_id in LABELS]
        ).T
        counts = np.bincount(group_codes, minlength=n_groups).astype(np.float64)
        return normalize(scores / counts[:, None]), group_codes.astype(np.int64), np.asarray(group_keys.astype(str))
    raise KeyError(f"Unknown segment proba rule: {rule}")


def load_pairwise_oof(cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return normalize(np.load(cache_path))
    fallback = Path(
        "outputs/audio_feature_benchmarks/audio_group_consistency_pair_blend_select/"
        "oof_sources/pairwise_selected_clean_oof_proba.npy"
    )
    if fallback.exists():
        return normalize(np.load(fallback))
    raise FileNotFoundError(f"Missing pairwise OOF cache: {cache_path} and fallback {fallback}")


def load_consensus_oof_sources(output: Path, train_df: pd.DataFrame, pairwise_oof_cache: Path) -> dict[str, np.ndarray]:
    highsr_oof = group_impl.load_highsr_oof(output)
    pairwise_oof = load_pairwise_oof(pairwise_oof_cache)
    if len(highsr_oof) != len(train_df) or len(pairwise_oof) != len(train_df):
        raise AssertionError("OOF source length mismatch")
    sources = {}
    for highsr_weight in CONSENSUS_HIGHSR_WEIGHTS:
        pairwise_weight = 1.0 - highsr_weight
        blended = normalize(highsr_weight * highsr_oof + pairwise_weight * pairwise_oof)
        segment_proba, _, _ = segment_proba_from_window(train_df, blended, "sum_log_proba")
        name = f"consensus_h{int(round(highsr_weight * 100)):02d}_p{int(round(pairwise_weight * 100)):02d}_log"
        sources[name] = segment_proba
    return sources


def load_final_frame_and_proba(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


def load_consensus_final_sources(output: Path, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    root = output / "audio_feature_benchmarks"
    highsr_frame, highsr_proba = load_final_frame_and_proba(
        root / "audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pairwise_frame, pairwise_proba = load_final_frame_and_proba(
        root
        / "audio_pairwise_contact_stress_cv_select/reports/audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    for frame_name, frame in [("highsr", highsr_frame), ("pairwise", pairwise_frame)]:
        if not np.array_equal(frame["audio_file"].astype(str).to_numpy(), test_df["audio_file"].astype(str).to_numpy()):
            raise AssertionError(f"{frame_name} final predictions are not aligned with robot/test manifest")
    sources = {}
    for highsr_weight in CONSENSUS_HIGHSR_WEIGHTS:
        pairwise_weight = 1.0 - highsr_weight
        blended = normalize(highsr_weight * highsr_proba + pairwise_weight * pairwise_proba)
        segment_proba, _, _ = segment_proba_from_window(test_df, blended, "sum_log_proba")
        name = f"consensus_h{int(round(highsr_weight * 100)):02d}_p{int(round(pairwise_weight * 100)):02d}_log"
        sources[name] = segment_proba
    return sources


def make_base_candidates(random_state: int) -> dict[str, SegmentSpec]:
    return {
        "seg_extra_trees_balanced": SegmentSpec(
            name="seg_extra_trees_balanced",
            kind="direct",
            direct_factory=lambda: ExtraTreesClassifier(
                n_estimators=1200,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "seg_extra_trees_leaf3": SegmentSpec(
            name="seg_extra_trees_leaf3",
            kind="direct",
            direct_factory=lambda: ExtraTreesClassifier(
                n_estimators=1200,
                max_features=0.35,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "seg_random_forest_balanced": SegmentSpec(
            name="seg_random_forest_balanced",
            kind="direct",
            direct_factory=lambda: RandomForestClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "seg_logistic_balanced": SegmentSpec(
            name="seg_logistic_balanced",
            kind="direct",
            direct_factory=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.5,
                            class_weight="balanced",
                            max_iter=4000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    }


def fit_predict_spec(spec: SegmentSpec, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    if spec.kind == "direct":
        model = spec.direct_factory()
        model.fit(X_train, y_train)
        return proba_aligned(model, X_eval)
    if spec.kind == "hierarchical":
        binary = spec.binary_factory()
        binary.fit(X_train, (y_train > 0).astype(np.int64))
        contact_idx = np.where(y_train > 0)[0]
        contact = spec.contact_factory()
        contact.fit(X_train[contact_idx], y_train[contact_idx])
        return hierarchical_proba(binary, contact, X_eval)
    raise ValueError(f"Unsupported fit spec kind: {spec.kind}")


def fit_artifact(spec: SegmentSpec, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    if spec.kind == "direct":
        model = spec.direct_factory()
        model.fit(X_train, y_train)
        return {"kind": "direct", "model": model}
    if spec.kind == "hierarchical":
        binary = spec.binary_factory()
        binary.fit(X_train, (y_train > 0).astype(np.int64))
        contact_idx = np.where(y_train > 0)[0]
        contact = spec.contact_factory()
        contact.fit(X_train[contact_idx], y_train[contact_idx])
        return {"kind": "hierarchical", "binary_model": binary, "contact_model": contact}
    raise ValueError(f"Unsupported artifact kind: {spec.kind}")


def predict_artifact(artifact: dict, X_eval: np.ndarray) -> np.ndarray:
    if artifact["kind"] == "direct":
        return proba_aligned(artifact["model"], X_eval)
    if artifact["kind"] == "hierarchical":
        return hierarchical_proba(artifact["binary_model"], artifact["contact_model"], X_eval)
    raise ValueError(f"Unsupported artifact kind: {artifact['kind']}")


def evaluate_segment_proba(
    name: str,
    kind: str,
    segment_proba: np.ndarray,
    y_window: np.ndarray,
    window_to_segment: np.ndarray,
    train_time_sec: float,
    extra: dict | None = None,
) -> dict:
    window_proba = expand_segment_proba(segment_proba, window_to_segment)
    default_pred = window_proba.argmax(axis=1)
    bias, tuned_macro = tune_class_bias_small(window_proba, y_window)
    tuned_pred = predict_with_bias(window_proba, bias)
    row = base.make_report_row(
        model_name=name,
        split_name="segment_oof_cv",
        y_true=y_window,
        y_pred=tuned_pred,
        train_time_sec=train_time_sec,
        predict_time_sec=0.0,
    )
    row.update(
        {
            "candidate_kind": kind,
            "default_oof_macro_f1": f1_score(y_window, default_pred, average="macro", zero_division=0),
            "tuned_oof_macro_f1": tuned_macro,
            "class_bias_json": json.dumps(bias.tolist()),
        }
    )
    if extra:
        row.update(extra)
    return row


def build_model_oof(
    specs: dict[str, SegmentSpec],
    X_segment: np.ndarray,
    y_segment: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    probas = {}
    rows = []
    fold_rows = []
    for spec in specs.values():
        print(f"\nSegment OOF candidate: {spec.name}", flush=True)
        oof = np.zeros((len(y_segment), len(LABELS)), dtype=np.float64)
        total_train_time = 0.0
        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            start = time.perf_counter()
            oof[val_idx] = fit_predict_spec(spec, X_segment[train_idx], y_segment[train_idx], X_segment[val_idx])
            elapsed = time.perf_counter() - start
            total_train_time += elapsed
            fold_pred = oof[val_idx].argmax(axis=1)
            fold_macro = f1_score(y_segment[val_idx], fold_pred, average="macro", zero_division=0)
            fold_rows.append(
                {
                    "candidate": spec.name,
                    "fold": fold_id,
                    "macro_f1": fold_macro,
                    "train_time_sec": elapsed,
                    "val_segments": int(len(val_idx)),
                }
            )
            print(f"  fold {fold_id}: segment_macro={fold_macro:.4f} time={elapsed:.2f}s", flush=True)
        probas[spec.name] = normalize(oof)
        rows.append({"name": spec.name, "proba": probas[spec.name], "train_time_sec": total_train_time})
    return probas, rows, fold_rows


def build_candidate_table(
    model_probas: dict[str, np.ndarray],
    model_times: dict[str, float],
    source_probas: dict[str, np.ndarray],
    y_window: np.ndarray,
    window_to_segment: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, SegmentSpec], dict[str, np.ndarray]]:
    specs: dict[str, SegmentSpec] = {}
    all_probas: dict[str, np.ndarray] = {}
    rows = []

    for name, proba in model_probas.items():
        specs[name] = SegmentSpec(name=name, kind="model_ref", model_member=name)
        all_probas[name] = proba
        rows.append(
            evaluate_segment_proba(
                name,
                "segment_model",
                proba,
                y_window,
                window_to_segment,
                model_times[name],
            )
        )

    for name, proba in source_probas.items():
        specs[name] = SegmentSpec(name=name, kind="source_ref", source_member=name)
        all_probas[name] = proba
        rows.append(
            evaluate_segment_proba(
                name,
                "consensus_source",
                proba,
                y_window,
                window_to_segment,
                0.0,
                {"source_member": name},
            )
        )

    for model_name, model_proba in model_probas.items():
        for source_name, source_proba in source_probas.items():
            for model_weight in MODEL_BLEND_WEIGHTS:
                source_weight = 1.0 - model_weight
                name = f"blend_{model_name}__{source_name}__mw{int(model_weight * 100):02d}"
                proba = normalize(model_weight * model_proba + source_weight * source_proba)
                specs[name] = SegmentSpec(
                    name=name,
                    kind="blend",
                    model_member=model_name,
                    source_member=source_name,
                    model_weight=model_weight,
                )
                all_probas[name] = proba
                rows.append(
                    evaluate_segment_proba(
                        name,
                        "model_consensus_blend",
                        proba,
                        y_window,
                        window_to_segment,
                        model_times[model_name],
                        {
                            "model_member": model_name,
                            "source_member": source_name,
                            "model_weight": model_weight,
                            "source_weight": source_weight,
                        },
                    )
                )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["tuned_oof_macro_f1", "macro_f1_4class", "contact_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    return leaderboard, specs, all_probas


def final_segment_proba_for_spec(
    selected_spec: SegmentSpec,
    base_specs: dict[str, SegmentSpec],
    final_model_probas: dict[str, np.ndarray],
    final_source_probas: dict[str, np.ndarray],
) -> np.ndarray:
    if selected_spec.kind == "model_ref":
        return final_model_probas[selected_spec.model_member]
    if selected_spec.kind == "source_ref":
        return final_source_probas[selected_spec.source_member]
    if selected_spec.kind == "blend":
        model_weight = float(selected_spec.model_weight)
        return normalize(
            model_weight * final_model_probas[selected_spec.model_member]
            + (1.0 - model_weight) * final_source_probas[selected_spec.source_member]
        )
    if selected_spec.name in base_specs:
        return final_model_probas[selected_spec.name]
    raise ValueError(f"Unsupported selected spec: {selected_spec}")


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve())
    print("RUN_DIR   =", run_dir.resolve())
    print("Protocol  = audio-only segment pooling; train OOF selection; robot/test after lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_feat, train_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.feature_cache_dir,
        force_rebuild=args.force_rebuild,
    )
    y_window = train_feat["y"]
    segment_frame, X_segment, window_to_segment = make_segment_dataset(train_df, train_feat["X"])
    y_segment = segment_frame["y"].to_numpy(dtype=np.int64)
    splits = make_segment_splits(segment_frame, args.n_folds, args.random_state)

    fold_assignment = np.full(len(segment_frame), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    segment_frame.assign(cv_fold=fold_assignment).to_csv(split_dir / "hand_train_segment_cv_folds.csv", index=False)

    split_summary = {
        "protocol": "audio-only segment pooling; hand/default OOF selection only",
        "n_window_samples": int(len(train_df)),
        "n_segment_samples": int(len(segment_frame)),
        "n_specimen_groups": int(segment_frame["specimen_group"].nunique()),
        "segment_feature_dim": int(X_segment.shape[1]),
        "label_counts_windows": {
            base.ID2LABEL[int(label)]: int(np.sum(y_window == label)) for label in LABELS
        },
        "label_counts_segments": {
            base.ID2LABEL[int(label)]: int(np.sum(y_segment == label)) for label in LABELS
        },
        "folds": [
            {
                "fold": fold_id,
                "train_segments": int(len(train_idx)),
                "val_segments": int(len(val_idx)),
                "train_specimens": int(segment_frame.iloc[train_idx]["specimen_group"].nunique()),
                "val_specimens": int(segment_frame.iloc[val_idx]["specimen_group"].nunique()),
            }
            for fold_id, (train_idx, val_idx) in enumerate(splits, start=1)
        ],
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = make_base_candidates(args.random_state)
    model_probas, model_rows, fold_rows = build_model_oof(base_specs, X_segment, y_segment, splits)
    model_times = {row["name"]: row["train_time_sec"] for row in model_rows}
    source_probas = load_consensus_oof_sources(args.output, train_df, args.pairwise_oof_cache)
    leaderboard, selected_specs, all_oof_probas = build_candidate_table(
        model_probas,
        model_times,
        source_probas,
        y_window,
        window_to_segment,
    )

    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    fold_report_path = report_dir / f"{args.run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = selected_specs[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    method_card = {
        "protocol": "audio_only_segment_pool_blend_no_test_until_lock",
        "allowed_selection_data": "hand/default audio features, hand labels, segment group_key, specimen grouped OOF, locked audio-only OOF sources",
        "forbidden_selection_data": "robot/test labels, robot/test predictions before lock, image or multimodal features",
        "checkpoint_before_this_run": "checkpoints/audio_log_consensus_pair_blend_0653825",
        "candidate_count": int(len(leaderboard)),
        "segment_feature_dim": int(X_segment.shape[1]),
        "consensus_highsr_weights": CONSENSUS_HIGHSR_WEIGHTS,
        "model_blend_weights": MODEL_BLEND_WEIGHTS,
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest hand/default window-expanded OOF macro_f1 after segment pooling/blending",
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_spec": {
            "kind": selected_spec.kind,
            "model_member": selected_spec.model_member,
            "source_member": selected_spec.source_member,
            "model_weight": selected_spec.model_weight,
        },
        "selected_bias": selected_bias.tolist(),
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
    test_df = base.load_manifest(test_csv, "robot_test")
    test_feat, test_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.feature_cache_dir,
        force_rebuild=args.force_rebuild,
    )
    test_segment_frame, X_test_segment, test_window_to_segment = make_segment_dataset(test_df, test_feat["X"])
    y_test_window = test_feat["y"]

    final_artifacts = {}
    final_model_probas = {}
    for model_name in sorted(
        {
            spec.model_member
            for spec in selected_specs.values()
            if spec.model_member is not None and spec.model_member in base_specs
        }
    ):
        start = time.perf_counter()
        artifact = fit_artifact(base_specs[model_name], X_segment, y_segment)
        final_artifacts[model_name] = artifact
        final_model_probas[model_name] = predict_artifact(artifact, X_test_segment)
        print(f"Final fit {model_name}: {time.perf_counter() - start:.2f}s", flush=True)
    final_source_probas = load_consensus_final_sources(args.output, test_df)
    final_segment_proba = final_segment_proba_for_spec(
        selected_spec,
        base_specs,
        final_model_probas,
        final_source_probas,
    )
    final_window_proba = expand_segment_proba(final_segment_proba, test_window_to_segment)
    final_pred = predict_with_bias(final_window_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=y_test_window,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_segment_pool_oof",
            "selected_oof_macro_f1": selected["tuned_oof_macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_kind": selected_spec.kind,
            "selected_model_member": selected_spec.model_member,
            "selected_source_member": selected_spec.source_member,
            "selected_model_weight": selected_spec.model_weight,
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )

    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_df]
    ].copy()
    prediction_frame["segment_id"] = test_window_to_segment
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_window_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(y_test_window, final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)
    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": method_card["protocol"],
            "method_card": method_card,
            "selection_summary": selection_summary,
            "segment_frame": segment_frame,
            "test_segment_frame": test_segment_frame,
            "final_artifacts": final_artifacts,
            "final_test_report": final_row,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": method_card["protocol"],
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "train_feature_timing": train_timing,
        "test_feature_timing": test_timing,
        "method_card": method_card,
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "method_card": str(method_card_path.resolve()),
            "split_summary": str(split_summary_path.resolve()),
            "leaderboard": str(leaderboard_path.resolve()),
            "fold_report": str(fold_report_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen segment-pool selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_kind",
                "selected_model_member",
                "selected_source_member",
                "selected_model_weight",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Method card:", method_card_path.resolve())
    print("Leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
