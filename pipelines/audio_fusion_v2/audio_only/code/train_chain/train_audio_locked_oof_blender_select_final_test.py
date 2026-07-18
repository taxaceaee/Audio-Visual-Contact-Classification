from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_audio_highsr_temporal_tta_select_final_test as highsr
import train_audio_oof_stacking_select_final_test as stack_lr
import train_audio_tta_grid_ensemble_select_final_test as ens
import train_cv_select_final_test as cv
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only locked-source OOF blender. It blends already locked audio "
            "models using only hand/default OOF predictions, writes a lock, then "
            "loads robot/test probabilities for final evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-slug", default="audio_locked_oof_blender_select")
    parser.add_argument("--weight-step", type=int, default=10)
    parser.add_argument(
        "--local-neighborhood",
        action="store_true",
        help="Search a focused OOF neighborhood around the best coarse blend instead of the full grid.",
    )
    parser.add_argument(
        "--focused-gate-only",
        action="store_true",
        help="Only evaluate contact rescue gates around the best train-only blend neighborhood.",
    )
    parser.add_argument(
        "--selection-profile",
        choices=["balanced", "binary_guard", "contact_heavy", "macro_only", "contact_only", "fold_stable"],
        default="balanced",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-12, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def calibrate_proba(proba: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-12:
        return normalize(proba)
    output = np.power(np.clip(proba, 1e-12, 1.0), gamma)
    return output / output.sum(axis=1, keepdims=True)


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(proba + bias.reshape(1, -1), axis=1).astype(np.int64)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> float:
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


def score_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = fast_macro_f1(y_true, pred, LABELS)
    contact = fast_macro_f1(y_true, pred, CONTACT_LABELS)
    binary = fast_macro_f1((y_true > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "binary_macro_f1": binary,
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def fold_scores(y_true: np.ndarray, pred: np.ndarray, fold_assignment: np.ndarray) -> dict[str, float]:
    macros = []
    contacts = []
    hybrids = []
    for fold_id in sorted(set(fold_assignment.tolist())):
        mask = fold_assignment == fold_id
        scores = score_pred(y_true[mask], pred[mask])
        macros.append(scores["macro_f1"])
        contacts.append(scores["contact_macro_f1"])
        hybrids.append(scores["hybrid_macro_contact"])
    return {
        "fold_mean_macro_f1": float(np.mean(macros)),
        "fold_mean_contact_macro_f1": float(np.mean(contacts)),
        "fold_mean_hybrid_macro_contact": float(np.mean(hybrids)),
        "fold_std_hybrid_macro_contact": float(np.std(hybrids)),
        "worst_fold_hybrid_macro_contact": float(np.min(hybrids)),
    }


def selection_score(row: dict[str, float], profile: str) -> float:
    if profile == "binary_guard":
        return float(
            0.45 * row["macro_f1"]
            + 0.20 * row["contact_macro_f1"]
            + 0.15 * row["binary_macro_f1"]
            + 0.20 * row["worst_fold_hybrid_macro_contact"]
            - 0.5 * row["fold_std_hybrid_macro_contact"]
        )
    if profile == "contact_heavy":
        return float(
            0.45 * row["macro_f1"]
            + 0.35 * row["contact_macro_f1"]
            + 0.20 * row["worst_fold_hybrid_macro_contact"]
            - 0.5 * row["fold_std_hybrid_macro_contact"]
        )
    if profile == "macro_only":
        return float(row["macro_f1"])
    if profile == "contact_only":
        return float(row["contact_macro_f1"])
    if profile == "fold_stable":
        return float(
            0.45 * row["macro_f1"]
            + 0.25 * row["contact_macro_f1"]
            + 0.30 * row["worst_fold_hybrid_macro_contact"]
            - 1.0 * row["fold_std_hybrid_macro_contact"]
        )
    return float(
        0.60 * row["macro_f1"]
        + 0.20 * row["contact_macro_f1"]
        + 0.20 * row["worst_fold_hybrid_macro_contact"]
        - 0.5 * row["fold_std_hybrid_macro_contact"]
    )


def bias_candidates() -> list[np.ndarray]:
    values = [
        [0.0, 0.0, 0.0, 0.0],
        [-0.2, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.2],
        [0.0, 0.1, 0.0, 0.1],
        [0.0, 0.15, 0.0, 0.15],
        [0.0, 0.25, 0.0, 0.25],
        [0.0, 0.3, 0.0, 0.3],
        [0.0, 0.4, 0.0, 0.4],
        [0.1, 0.2, 0.0, 0.2],
        [-0.1, 0.2, 0.0, 0.2],
        [-0.2, 0.2, 0.0, 0.2],
        [-0.3, 0.2, 0.0, 0.2],
        [-0.4, 0.2, 0.0, 0.2],
        [-0.2, 0.25, 0.0, 0.25],
        [-0.3, 0.25, 0.0, 0.25],
        [-0.2, 0.3, 0.0, 0.3],
        [0.0, 0.2, -0.1, 0.2],
        [0.0, 0.2, 0.1, 0.2],
        [0.2, 0.4, 0.0, 0.4],
        [-0.2, 0.4, 0.0, 0.4],
        [0.0, 0.0, -0.2, 0.0],
        [0.0, 0.2, -0.2, 0.2],
        [0.0, 0.4, -0.4, 0.4],
        [0.2, 0.8, 0.4, 0.8],
        [0.0, 0.8, 0.4, 0.8],
    ]
    return [np.asarray(row, dtype=np.float64) for row in values]


def load_total240_ensemble_oof(source_run: Path, ensemble_run: Path) -> np.ndarray:
    lock_path = ensemble_run / "reports" / f"{ensemble_run.name}_selected_without_test.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    members = lock["selected_members"]
    base_candidates = sorted({str(member["base_candidate"]) for member in members})
    oof_by_base = {
        base_candidate: ens.load_oof_by_base(source_run, base_candidate)
        for base_candidate in base_candidates
    }
    proba, _ = ens.ensemble_probas(oof_by_base, members)
    return normalize(proba)


def load_total240_stack_oof_from_run(
    output: Path,
    run_name: str,
    y: np.ndarray,
    fold_assignment: np.ndarray,
    random_state: int,
) -> np.ndarray:
    source_run = output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select"
    stack_run = output / "audio_feature_benchmarks" / run_name
    stack_lock_path = stack_run / "reports" / f"{run_name}_selected_without_test.json"
    stack_lock = json.loads(stack_lock_path.read_text(encoding="utf-8"))
    selected = stack_lock["selected_without_test"]
    top_k = int(selected["top_k"])
    include_views = bool(selected["include_views"])
    include_confidence = bool(selected["include_confidence"])
    meta_model_name = str(selected["meta_model"])
    gamma = float(stack_lock["selected_probability_gamma"])

    leaderboard = pd.read_csv(source_run / "reports" / f"{source_run.name}_oof_tta_grid_leaderboard.csv")
    members = stack_lr.ranked_members(leaderboard, top_k)
    base_candidates = sorted({str(member["base_candidate"]) for member in members})
    oof_by_base = {
        base_candidate: ens.load_oof_by_base(source_run, base_candidate)
        for base_candidate in base_candidates
    }
    spec = stack_lr.StackSpec(str(selected["stack_spec"]), top_k, include_views, include_confidence)
    X_stack = stack_lr.stack_matrix(oof_by_base, members, spec)
    meta_candidates = {candidate["name"]: candidate for candidate in stack_lr.meta_candidates(random_state)}
    if meta_model_name not in meta_candidates:
        raise KeyError(f"Missing stack meta model: {meta_model_name}")
    oof = stack_lr.cv_meta_proba(meta_candidates[meta_model_name]["model"], X_stack, y, fold_assignment)
    return calibrate_proba(oof, gamma)


def load_total240_stack_oof(output: Path, y: np.ndarray, fold_assignment: np.ndarray, random_state: int) -> np.ndarray:
    return load_total240_stack_oof_from_run(
        output,
        "audio_oof_stacking_select",
        y,
        fold_assignment,
        random_state,
    )


def load_highsr_oof(output: Path) -> np.ndarray:
    run = output / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    lock_path = run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    candidate = str(lock["selected_candidate"])
    weights = lock["selected_tta_weights"]
    oof_by_view = {
        view: np.load(run / "oof_proba" / candidate / f"{view}_oof_proba.npy")
        for view in highsr.STRESS_VIEWS
    }
    return normalize(highsr.weighted_proba(oof_by_view, weights))


def load_grid_tta_oof(output: Path, run_name: str) -> np.ndarray:
    run = output / "audio_feature_benchmarks" / run_name
    lock_path = run / "reports" / f"{run_name}_selected_without_test.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    candidate = str(lock["selected_base_candidate"])
    weights = lock["selected_tta_weights"]
    oof_by_view = {
        view: np.load(run / "oof_proba" / candidate / f"{view}_oof_proba.npy")
        for view in highsr.STRESS_VIEWS
    }
    return normalize(highsr.weighted_proba(oof_by_view, weights))


def weight_grid(source_names: list[str], denominator: int) -> list[dict[str, float]]:
    rows = []
    for units in itertools.product(range(denominator + 1), repeat=len(source_names)):
        if sum(units) != denominator:
            continue
        weights = {
            name: unit / denominator
            for name, unit in zip(source_names, units)
            if unit > 0
        }
        if weights:
            rows.append(weights)
    return rows


def blend_sources(source_proba: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = float(sum(weights.values()))
    output = None
    for name, weight in weights.items():
        part = float(weight) / total * source_proba[name]
        output = part if output is None else output + part
    if output is None:
        raise RuntimeError("Empty blend")
    return normalize(output)


def hierarchical_contact_blend(
    binary_source: np.ndarray,
    class_source: np.ndarray,
) -> np.ndarray:
    contact = np.clip(np.sum(binary_source[:, 1:], axis=1), 1e-12, 1.0)
    conditional = normalize(class_source[:, 1:])
    output = np.zeros_like(binary_source)
    output[:, 0] = 1.0 - contact
    output[:, 1:] = contact[:, None] * conditional
    return normalize(output)


def per_class_contact_blend(
    binary_source: np.ndarray,
    source_proba: dict[str, np.ndarray],
    class_sources: dict[str, str],
) -> np.ndarray:
    contact = np.clip(np.sum(binary_source[:, 1:], axis=1), 1e-12, 1.0)
    raw = np.zeros((len(binary_source), len(CONTACT_LABELS)), dtype=np.float64)
    for offset, class_id in enumerate(CONTACT_LABELS):
        class_name = base.ID2LABEL[int(class_id)]
        raw[:, offset] = source_proba[class_sources[class_name]][:, class_id]
    conditional = normalize(raw)
    output = np.zeros_like(binary_source)
    output[:, 0] = 1.0 - contact
    output[:, 1:] = contact[:, None] * conditional
    return normalize(output)


def local_bias_candidates() -> list[np.ndarray]:
    values = [
        [0.0, 0.2, 0.0, 0.2],
        [0.0, 0.1, 0.0, 0.1],
        [0.0, 0.15, 0.0, 0.15],
        [0.0, 0.25, 0.0, 0.25],
        [0.0, 0.3, 0.0, 0.3],
        [0.1, 0.2, 0.0, 0.2],
        [-0.1, 0.2, 0.0, 0.2],
        [0.0, 0.2, -0.1, 0.2],
        [0.0, 0.2, 0.1, 0.2],
    ]
    return [np.asarray(row, dtype=np.float64) for row in values]


def evaluate_recipe(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    proba: np.ndarray,
    recipe: dict,
    gammas: list[float] | None = None,
    biases: list[np.ndarray] | None = None,
    selection_profile: str = "balanced",
) -> dict:
    best_row = None
    best_bias = None
    best_gamma = None
    if gammas is None:
        gammas = [0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
    if biases is None:
        biases = bias_candidates()
    for gamma in gammas:
        calibrated = calibrate_proba(proba, gamma)
        for bias in biases:
            pred = predict_with_bias(calibrated, bias)
            row = score_pred(y, pred)
            row.update(fold_scores(y, pred, fold_assignment))
            row["selection_score"] = selection_score(row, selection_profile)
            if best_row is None or (
                row["selection_score"],
                row["macro_f1"],
                row["contact_macro_f1"],
                row["worst_fold_hybrid_macro_contact"],
            ) > (
                best_row["selection_score"],
                best_row["macro_f1"],
                best_row["contact_macro_f1"],
                best_row["worst_fold_hybrid_macro_contact"],
            ):
                best_row = row
                best_bias = bias
                best_gamma = gamma
    if best_row is None or best_bias is None or best_gamma is None:
        raise RuntimeError("No recipe scored")
    output = {
        **recipe,
        **best_row,
        "probability_gamma": float(best_gamma),
        "class_bias_json": json.dumps(best_bias.tolist()),
    }
    return output


def evaluate_gated_recipe(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    base_proba: np.ndarray,
    gate_proba: np.ndarray,
    class_proba: np.ndarray,
    recipe: dict,
    gammas: list[float] | None = None,
    biases: list[np.ndarray] | None = None,
    thresholds: list[float] | None = None,
) -> dict:
    best_row = None
    best_bias = None
    best_gamma = None
    best_threshold = None
    if gammas is None:
        gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
    if biases is None:
        biases = local_bias_candidates()
    if thresholds is None:
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    contact_signal = np.sum(normalize(gate_proba)[:, 1:], axis=1)
    contact_pred = 1 + np.argmax(normalize(class_proba)[:, 1:], axis=1)
    for gamma in gammas:
        calibrated = calibrate_proba(base_proba, gamma)
        for bias in biases:
            base_pred = predict_with_bias(calibrated, bias)
            for threshold in thresholds:
                pred = base_pred.copy()
                mask = (pred == 0) & (contact_signal >= threshold)
                pred[mask] = contact_pred[mask]
                row = score_pred(y, pred)
                row.update(fold_scores(y, pred, fold_assignment))
                row["selection_score"] = selection_score(row, "balanced")
                row["forced_contact_count"] = int(np.sum(mask))
                if best_row is None or (
                    row["selection_score"],
                    row["macro_f1"],
                    row["contact_macro_f1"],
                    row["worst_fold_hybrid_macro_contact"],
                ) > (
                    best_row["selection_score"],
                    best_row["macro_f1"],
                    best_row["contact_macro_f1"],
                    best_row["worst_fold_hybrid_macro_contact"],
                ):
                    best_row = row
                    best_bias = bias
                    best_gamma = gamma
                    best_threshold = threshold
    if best_row is None or best_bias is None or best_gamma is None or best_threshold is None:
        raise RuntimeError("No gated recipe scored")
    return {
        **recipe,
        **best_row,
        "probability_gamma": float(best_gamma),
        "class_bias_json": json.dumps(best_bias.tolist()),
        "force_contact_threshold": float(best_threshold),
    }


def local_weight_grid(source_names: list[str]) -> list[dict[str, float]]:
    centers = [
        {"total240_ensemble": 0.25, "total240_stack_lr": 0.10, "highsr_hgb": 0.65},
        {"total240_ensemble": 0.20, "total240_stack_lr": 0.10, "highsr_hgb": 0.70},
    ]
    rows = {}
    denom = 100
    radius = 10
    for center in centers:
        center_units = {name: int(round(center[name] * denom)) for name in source_names}
        for ens_units in range(center_units["total240_ensemble"] - radius, center_units["total240_ensemble"] + radius + 1):
            for stack_units in range(center_units["total240_stack_lr"] - radius, center_units["total240_stack_lr"] + radius + 1):
                high_units = denom - ens_units - stack_units
                units = {
                    "total240_ensemble": ens_units,
                    "total240_stack_lr": stack_units,
                    "highsr_hgb": high_units,
                }
                if any(value < 0 for value in units.values()):
                    continue
                if any(abs(units[name] - center_units[name]) > radius for name in source_names):
                    continue
                weights = {name: units[name] / denom for name in source_names if units[name] > 0}
                rows[json.dumps(weights, sort_keys=True)] = weights
    return [json.loads(key) for key in sorted(rows)]


def focused_gate_weight_grid() -> list[dict[str, float]]:
    rows = [
        {"total240_ensemble": 0.11, "total240_stack_lr": 0.19, "highsr_hgb": 0.70},
        {"total240_ensemble": 0.10, "total240_stack_lr": 0.20, "highsr_hgb": 0.70},
        {"total240_ensemble": 0.12, "total240_stack_lr": 0.18, "highsr_hgb": 0.70},
        {"total240_ensemble": 0.15, "total240_stack_lr": 0.15, "highsr_hgb": 0.70},
        {"total240_ensemble": 0.20, "total240_stack_lr": 0.14, "highsr_hgb": 0.66},
        {"total240_ensemble": 0.25, "total240_stack_lr": 0.10, "highsr_hgb": 0.65},
    ]
    return rows


def load_final_proba(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    return normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


def source_feature_matrix(source_proba: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for name in sorted(source_proba):
        proba = normalize(source_proba[name])
        parts.append(proba)
        parts.append(np.log(np.clip(proba, 1e-8, 1.0)))
        parts.append(np.sum(proba[:, 1:], axis=1, keepdims=True))
        parts.append(np.max(proba, axis=1, keepdims=True))
    return np.hstack(parts).astype(np.float64)


def meta_candidates(random_state: int) -> list[dict]:
    rows = []
    for class_weight in [None, "balanced"]:
        for c_value in [0.03, 0.1, 0.3, 1.0]:
            rows.append(
                {
                    "name": f"meta_logreg_C{c_value:g}_cw{class_weight or 'none'}",
                    "model": make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=c_value,
                            class_weight=class_weight,
                            max_iter=2000,
                            random_state=random_state,
                        ),
                    ),
                }
            )
    for leaf in [3, 8, 15]:
        rows.append(
            {
                "name": f"meta_extratrees_leaf{leaf}",
                "model": ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            }
        )
    return rows


def cv_meta_proba(model: object, X: np.ndarray, y: np.ndarray, fold_assignment: np.ndarray) -> np.ndarray:
    output = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    for fold_id in sorted(set(fold_assignment.tolist())):
        train_mask = fold_assignment != fold_id
        val_mask = fold_assignment == fold_id
        fold_model = clone(model)
        fold_model.fit(X[train_mask], y[train_mask])
        raw = fold_model.predict_proba(X[val_mask])
        classes = np.asarray(fold_model.classes_, dtype=np.int64)
        aligned = np.zeros((int(np.sum(val_mask)), len(LABELS)), dtype=np.float64)
        for column, class_id in enumerate(classes):
            aligned[:, int(np.where(LABELS == class_id)[0][0])] = raw[:, column]
        output[val_mask] = normalize(aligned)
    return output


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve())
    print("RUN_DIR   =", run_dir.resolve())
    print("Selection = train-only locked-source OOF blender")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
        force_rebuild=False,
    )
    y = clean_feat["y"]
    split_path = args.output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select" / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)

    source_proba = {
        "total240_ensemble": load_total240_ensemble_oof(
            args.output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select",
            args.output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_ensemble_select",
        ),
        "total240_stack_lr": load_total240_stack_oof(args.output, y, fold_assignment, args.random_state),
        "total240_tree_meta": load_total240_stack_oof_from_run(
            args.output,
            "audio_oof_stacking_tree_meta_select",
            y,
            fold_assignment,
            args.random_state,
        ),
        "total120_grid": load_grid_tta_oof(args.output, "audio_total120_tta_grid_hgb_select"),
        "mfcc40_grid": load_grid_tta_oof(args.output, "audio_mfcc40_tta_grid_hgb_select"),
        "highsr_hgb": load_highsr_oof(args.output),
    }
    for name, proba in source_proba.items():
        if proba.shape != (len(y), len(LABELS)):
            raise AssertionError(f"{name}: bad OOF shape {proba.shape}")
        print(f"Loaded OOF source {name}: {proba.shape}", flush=True)

    rows = []
    source_names = list(source_proba)
    local_blend_source_names = ["total240_ensemble", "total240_stack_lr", "highsr_hgb"]
    if args.focused_gate_only:
        candidate_weight_rows = focused_gate_weight_grid()
        recipe_gammas = [2.0, 2.5, 3.0]
        recipe_biases = [
            np.asarray(values, dtype=np.float64)
            for values in [
                [-0.1, 0.2, 0.0, 0.2],
                [0.0, 0.2, 0.0, 0.2],
                [-0.2, 0.2, 0.0, 0.2],
                [-0.1, 0.25, 0.0, 0.25],
            ]
        ]
    elif args.local_neighborhood:
        candidate_weight_rows = local_weight_grid(local_blend_source_names)
        recipe_gammas = [1.5, 2.0, 2.5, 3.0, 4.0]
        recipe_biases = local_bias_candidates()
    else:
        candidate_weight_rows = weight_grid(source_names, args.weight_step)
        recipe_gammas = None
        recipe_biases = None
    if not args.focused_gate_only:
        for weights in candidate_weight_rows:
            proba = blend_sources(source_proba, weights)
            rows.append(
                evaluate_recipe(
                    y,
                    fold_assignment,
                    proba,
                    {
                        "recipe_kind": "weighted_average",
                        "weights_json": json.dumps(weights),
                        "binary_source": "",
                        "class_weights_json": "",
                    },
                    gammas=recipe_gammas,
                    biases=recipe_biases,
                    selection_profile=args.selection_profile,
                )
            )
    for binary_name in ["total240_stack_lr", "total240_tree_meta", "total240_ensemble"]:
        for class_weights in candidate_weight_rows:
            class_proba = blend_sources(source_proba, class_weights)
            proba = hierarchical_contact_blend(source_proba[binary_name], class_proba)
            if not args.focused_gate_only:
                rows.append(
                    evaluate_recipe(
                        y,
                        fold_assignment,
                        proba,
                        {
                            "recipe_kind": "hierarchical_contact_blend",
                            "weights_json": "",
                            "binary_source": binary_name,
                            "class_weights_json": json.dumps(class_weights),
                        },
                        gammas=recipe_gammas,
                        biases=recipe_biases,
                        selection_profile=args.selection_profile,
                    )
                )
            for gate_source in ["highsr_hgb", "total240_stack_lr", "total240_tree_meta"]:
                rows.append(
                    evaluate_gated_recipe(
                        y,
                        fold_assignment,
                        proba,
                        source_proba[gate_source],
                        class_proba,
                        {
                            "recipe_kind": "hierarchical_contact_blend_gate",
                            "weights_json": "",
                            "binary_source": binary_name,
                            "class_weights_json": json.dumps(class_weights),
                            "gate_source": gate_source,
                        },
                        gammas=recipe_gammas,
                        biases=recipe_biases,
                    )
                )
    if not args.focused_gate_only:
        for binary_name in ["total240_stack_lr", "total240_tree_meta", "total240_ensemble"]:
            for leaf_source, trunk_source, twig_source in itertools.product(source_names, repeat=3):
                class_sources = {
                    "leaf": leaf_source,
                    "trunk": trunk_source,
                    "twig": twig_source,
                }
                proba = per_class_contact_blend(source_proba[binary_name], source_proba, class_sources)
                rows.append(
                    evaluate_recipe(
                        y,
                        fold_assignment,
                        proba,
                        {
                            "recipe_kind": "per_class_contact_blend",
                            "weights_json": "",
                            "binary_source": binary_name,
                            "class_weights_json": "",
                            "class_sources_json": json.dumps(class_sources),
                        },
                        gammas=recipe_gammas,
                        biases=recipe_biases,
                        selection_profile=args.selection_profile,
                    )
                )
        X_meta = source_feature_matrix(source_proba)
        for candidate in meta_candidates(args.random_state):
            meta_oof = cv_meta_proba(candidate["model"], X_meta, y, fold_assignment)
            rows.append(
                evaluate_recipe(
                    y,
                    fold_assignment,
                    meta_oof,
                    {
                        "recipe_kind": "meta_stack",
                        "weights_json": "",
                        "binary_source": "",
                        "class_weights_json": "",
                        "meta_model": candidate["name"],
                    },
                    gammas=recipe_gammas,
                    biases=recipe_biases,
                    selection_profile=args.selection_profile,
                )
            )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "worst_fold_hybrid_macro_contact"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_blender_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    selection_summary = {
        "selection_rule": "best train-only OOF blend of locked audio-only sources",
        "selection_profile": args.selection_profile,
        "sources": source_names,
        "selected_without_test": selected,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_path": str(split_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    final_paths = {
        "total240_ensemble": args.output
        / "audio_feature_benchmarks"
        / "audio_tta_grid_hgb_ensemble_select"
        / "reports"
        / "audio_tta_grid_hgb_ensemble_select_final_test_predictions.csv",
        "total240_stack_lr": args.output
        / "audio_feature_benchmarks"
        / "audio_oof_stacking_select"
        / "reports"
        / "audio_oof_stacking_select_final_test_predictions.csv",
        "total240_tree_meta": args.output
        / "audio_feature_benchmarks"
        / "audio_oof_stacking_tree_meta_select"
        / "reports"
        / "audio_oof_stacking_tree_meta_select_final_test_predictions.csv",
        "total120_grid": args.output
        / "audio_feature_benchmarks"
        / "audio_total120_tta_grid_hgb_select"
        / "reports"
        / "audio_total120_tta_grid_hgb_select_final_test_predictions.csv",
        "mfcc40_grid": args.output
        / "audio_feature_benchmarks"
        / "audio_mfcc40_tta_grid_hgb_select"
        / "reports"
        / "audio_mfcc40_tta_grid_hgb_select_final_test_predictions.csv",
        "highsr_hgb": args.output
        / "audio_feature_benchmarks"
        / "audio_highsr_temporal_tta_select"
        / "reports"
        / "audio_highsr_temporal_tta_select_final_test_predictions.csv",
    }
    final_source_proba = {name: load_final_proba(path) for name, path in final_paths.items()}
    if str(selected["recipe_kind"]) == "weighted_average":
        final_base_proba = blend_sources(final_source_proba, json.loads(str(selected["weights_json"])))
    elif str(selected["recipe_kind"]) == "hierarchical_contact_blend":
        class_proba = blend_sources(final_source_proba, json.loads(str(selected["class_weights_json"])))
        final_base_proba = hierarchical_contact_blend(
            final_source_proba[str(selected["binary_source"])],
            class_proba,
        )
    elif str(selected["recipe_kind"]) == "hierarchical_contact_blend_gate":
        class_proba = blend_sources(final_source_proba, json.loads(str(selected["class_weights_json"])))
        final_base_proba = hierarchical_contact_blend(
            final_source_proba[str(selected["binary_source"])],
            class_proba,
        )
    elif str(selected["recipe_kind"]) == "per_class_contact_blend":
        final_base_proba = per_class_contact_blend(
            final_source_proba[str(selected["binary_source"])],
            final_source_proba,
            json.loads(str(selected["class_sources_json"])),
        )
    elif str(selected["recipe_kind"]) == "meta_stack":
        X_meta_train = source_feature_matrix(source_proba)
        X_meta_test = source_feature_matrix(final_source_proba)
        candidates = {candidate["name"]: candidate["model"] for candidate in meta_candidates(args.random_state)}
        selected_meta = str(selected["meta_model"])
        if selected_meta not in candidates:
            raise KeyError(f"Unknown selected meta model: {selected_meta}")
        final_meta = clone(candidates[selected_meta])
        final_meta.fit(X_meta_train, y)
        final_base_proba = normalize(final_meta.predict_proba(X_meta_test))
    else:
        raise KeyError(f"Unknown recipe kind: {selected['recipe_kind']}")
    final_proba = calibrate_proba(final_base_proba, float(selected["probability_gamma"]))
    selected_bias = np.asarray(json.loads(str(selected["class_bias_json"])), dtype=np.float64)
    final_pred = predict_with_bias(final_proba, selected_bias)
    if str(selected["recipe_kind"]) == "hierarchical_contact_blend_gate":
        final_class_proba = blend_sources(final_source_proba, json.loads(str(selected["class_weights_json"])))
        gate_signal = np.sum(normalize(final_source_proba[str(selected["gate_source"])])[:, 1:], axis=1)
        contact_pred = 1 + np.argmax(normalize(final_class_proba)[:, 1:], axis=1)
        mask = (final_pred == 0) & (gate_signal >= float(selected["force_contact_threshold"]))
        final_pred[mask] = contact_pred[mask]

    test_frame = pd.read_csv(final_paths["total240_stack_lr"])
    y_test = test_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name=f"{selected['recipe_kind']}__locked_audio_blend",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_locked_oof_blender",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_recipe_kind": selected["recipe_kind"],
            "selected_weights_json": selected["weights_json"],
            "selected_binary_source": selected["binary_source"],
            "selected_class_weights_json": selected["class_weights_json"],
            "selected_class_sources_json": selected.get("class_sources_json", ""),
            "selected_meta_model": selected.get("meta_model", ""),
            "selected_gate_source": selected.get("gate_source", ""),
            "selected_force_contact_threshold": selected.get("force_contact_threshold", ""),
            "selected_probability_gamma": selected["probability_gamma"],
            "selected_class_bias_json": selected["class_bias_json"],
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_frame[[column for column in ["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"] if column in test_frame]].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(y_test, final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_locked_oof_blender_no_test_until_lock",
            "selection_summary": selection_summary,
            "final_test_report": final_row,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "audio_only_locked_oof_blender_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen locked-source OOF blender:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
                "selected_recipe_kind",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
