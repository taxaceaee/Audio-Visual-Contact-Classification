from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import train_audio_tta_contact_stress_cv_select_final_test as tta
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only HGB TTA grid search. Train/default grouped OOF predictions "
            "choose HGB train views, TTA weights, and class bias before robot/test "
            "is loaded for one final evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--feature-set", choices=sorted(base.FEATURE_SPECS), default="total240")
    parser.add_argument("--weight-step", type=int, default=10, help="Integer denominator for TTA weight grid.")
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=0.0025,
        help="Within this train-only score margin, prefer the simpler HGB training view.",
    )
    parser.add_argument(
        "--clean-feature-cache-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--train-stress-feature-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--test-stress-feature-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def make_hgb_specs() -> dict[str, stress.StressCandidate]:
    specs = [
        stress.StressCandidate("grid_hgb_default__clean", "direct_hgb_default", "single", ("clean",)),
        stress.StressCandidate("grid_hgb_default__robot_aug", "direct_hgb_default", "single", ("clean", "robot_mix")),
        stress.StressCandidate("grid_hgb_default__band_aug", "direct_hgb_default", "single", ("clean", "bandlimit")),
        stress.StressCandidate("grid_hgb_default__all_aug", "direct_hgb_default", "single", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def train_view_complexity(base_candidate: str) -> int:
    return {
        "grid_hgb_default__clean": 0,
        "grid_hgb_default__robot_aug": 1,
        "grid_hgb_default__band_aug": 1,
        "grid_hgb_default__all_aug": 2,
    }.get(base_candidate, 99)


def make_weight_grid(denominator: int) -> dict[str, dict[str, float]]:
    if denominator < 2:
        raise ValueError("--weight-step must be >= 2")
    recipes: dict[str, dict[str, float]] = {}
    for clean_units in range(denominator + 1):
        for robot_units in range(denominator - clean_units + 1):
            band_units = denominator - clean_units - robot_units
            if clean_units == robot_units == band_units == 0:
                continue
            weights = {
                "clean": clean_units / denominator,
                "robot_mix": robot_units / denominator,
                "bandlimit": band_units / denominator,
            }
            weights = {view: value for view, value in weights.items() if value > 0}
            name = "w_" + "_".join(
                f"{view[:1]}{int(round(value * denominator)):02d}" for view, value in weights.items()
            )
            recipes[name] = weights
    return recipes


def score_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)
    contact = f1_score(y_true, pred, labels=CONTACT_LABELS, average="macro", zero_division=0)
    return {
        "macro_f1": float(macro),
        "contact_macro_f1": float(contact),
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def score_proba(y_true: np.ndarray, proba: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    return score_pred(y_true, tta.predict_with_bias(proba, bias))


def weighted_by_recipe(proba_by_view: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return tta.weighted_proba(proba_by_view, weights)


def view_scores(
    y_true: np.ndarray,
    proba_by_view: dict[str, np.ndarray],
    bias: np.ndarray,
) -> dict[str, float]:
    output = {}
    for view, proba in proba_by_view.items():
        scores = score_proba(y_true, proba, bias)
        for metric, value in scores.items():
            output[f"{view}_{metric}"] = value
    output["worst_single_view_hybrid"] = min(
        output[f"{view}_hybrid_macro_contact"] for view in STRESS_VIEWS
    )
    return output


def fold_scores(
    y_true: np.ndarray,
    proba: np.ndarray,
    bias: np.ndarray,
    fold_assignment: np.ndarray,
) -> dict[str, float]:
    hybrids = []
    macros = []
    contacts = []
    for fold_id in sorted(set(fold_assignment.tolist())):
        mask = fold_assignment == fold_id
        scores = score_proba(y_true[mask], proba[mask], bias)
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


def selection_score(row: dict[str, float]) -> float:
    stability_penalty = 0.5 * row["fold_std_hybrid_macro_contact"]
    return float(
        0.55 * row["hybrid_macro_contact"]
        + 0.25 * row["worst_single_view_hybrid"]
        + 0.20 * row["worst_fold_hybrid_macro_contact"]
        - stability_penalty
    )


def tune_bias_for_selection(
    y: np.ndarray,
    proba_by_view: dict[str, np.ndarray],
    tta_proba: np.ndarray,
    fold_assignment: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    best_bias = np.zeros(4, dtype=np.float64)
    best_metrics: dict[str, float] | None = None
    del grid
    candidates = [
        np.asarray(values, dtype=np.float64)
        for values in [
            [0.0, 0.0, 0.0, 0.0],
            [-0.4, 0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.2],
            [0.0, 0.4, 0.0, 0.4],
            [0.2, 0.4, 0.0, 0.4],
            [-0.2, 0.4, 0.0, 0.4],
            [0.0, 0.0, -0.2, 0.0],
            [0.0, 0.0, -0.4, 0.0],
            [0.0, 0.2, -0.2, 0.2],
            [0.0, 0.4, -0.4, 0.4],
            [0.2, 0.8, 0.4, 0.8],
            [0.0, 0.8, 0.4, 0.8],
        ]
    ]

    seen = set()
    for bias in candidates:
        key = tuple(np.round(bias, 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        metrics = score_proba(y, tta_proba, bias)
        metrics.update(view_scores(y, proba_by_view, bias))
        metrics.update(fold_scores(y, tta_proba, bias, fold_assignment))
        metrics["selection_score"] = selection_score(metrics)
        if best_metrics is None or (
            metrics["selection_score"],
            metrics["worst_single_view_hybrid"],
            metrics["worst_fold_hybrid_macro_contact"],
            metrics["macro_f1"],
        ) > (
            best_metrics["selection_score"],
            best_metrics["worst_single_view_hybrid"],
            best_metrics["worst_fold_hybrid_macro_contact"],
            best_metrics["macro_f1"],
        ):
            best_bias = bias
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Bias grid produced no candidates")
    return best_bias, best_metrics


def evaluate_hgb_spec(
    spec: stress.StressCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    all_specs: dict[str, stress.StressCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fold_assignment: np.ndarray,
    weight_grid: dict[str, dict[str, float]],
) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    print(f"\nGrid HGB candidate: {spec.name} train_views={spec.train_views}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = stress.fit_stress_candidate(spec, base_specs, all_specs, X_by_view, y, train_idx)
        train_time = time.perf_counter() - start
        parts = []
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = stress.predict_stress_artifact(artifact, X_by_view[view][val_idx])
            scores = score_proba(y[val_idx], oof_by_view[view][val_idx], np.zeros(4, dtype=np.float64))
            parts.append(f"{view}=M{scores['macro_f1']:.4f}/C{scores['contact_macro_f1']:.4f}")
        fold_rows.append(
            {
                "candidate": spec.name,
                "fold": fold_id,
                "train_time_sec": train_time,
                "val_samples": int(len(val_idx)),
            }
        )
        print(f"  fold {fold_id}: " + " | ".join(parts) + f" time={train_time:.2f}s", flush=True)

    rows = []
    bias_grid = np.asarray([-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8], dtype=np.float64)
    for recipe_name, weights in weight_grid.items():
        tta_proba = weighted_by_recipe(oof_by_view, weights)
        bias, metrics = tune_bias_for_selection(y, oof_by_view, tta_proba, fold_assignment, bias_grid)
        pred = tta.predict_with_bias(tta_proba, bias)
        row = base.make_report_row(
            model_name=f"{spec.name}__{recipe_name}__robust_bias",
            split_name="audio_hgb_tta_grid_oof_cv",
            y_true=y,
            y_pred=pred,
            train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
            predict_time_sec=0.0,
        )
        row.update(
            {
                "base_candidate": spec.name,
                "base_name": spec.base_name,
                "train_views_json": json.dumps(list(spec.train_views)),
                "tta_recipe": recipe_name,
                "tta_weights_json": json.dumps(weights),
                "class_bias_json": json.dumps(bias.tolist()),
                "train_view_complexity": train_view_complexity(spec.name),
                "tta_macro_f1": metrics["macro_f1"],
                "tta_contact_macro_f1": metrics["contact_macro_f1"],
                "tta_hybrid_macro_contact": metrics["hybrid_macro_contact"],
                "selection_score": metrics["selection_score"],
                **{
                    key: value
                    for key, value in metrics.items()
                    if key
                    not in {
                        "macro_f1",
                        "contact_macro_f1",
                        "hybrid_macro_contact",
                        "selection_score",
                    }
                },
            }
        )
        rows.append(row)

    best_row = max(rows, key=lambda item: item["selection_score"])
    print(
        f"  best grid: {best_row['tta_recipe']} selection={best_row['selection_score']:.4f} "
        f"tta_macro={best_row['tta_macro_f1']:.4f} contact={best_row['tta_contact_macro_f1']:.4f} "
        f"worst_view={best_row['worst_single_view_hybrid']:.4f} "
        f"worst_fold={best_row['worst_fold_hybrid_macro_contact']:.4f} "
        f"bias={best_row['class_bias_json']}",
        flush=True,
    )
    return rows, oof_by_view, fold_rows


def select_with_margin(leaderboard: pd.DataFrame, margin: float) -> dict:
    top_score = float(leaderboard["selection_score"].max())
    eligible = leaderboard[leaderboard["selection_score"] >= top_score - margin].copy()
    selected = eligible.sort_values(
        [
            "train_view_complexity",
            "worst_single_view_hybrid",
            "worst_fold_hybrid_macro_contact",
            "selection_score",
        ],
        ascending=[True, False, False, False],
    ).iloc[0]
    output = selected.to_dict()
    output["top_train_only_selection_score"] = top_score
    output["eligible_count"] = int(len(eligible))
    return output


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set(args.feature_set)

    root_path = base.resolve_root(args.root)
    run_slug = (
        "audio_tta_grid_hgb_select"
        if args.feature_set == "total240"
        else f"audio_{args.feature_set}_tta_grid_hgb_select"
    )
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    oof_dir = run_dir / "oof_proba"
    for directory in [run_dir, report_dir, model_dir, split_dir, oof_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    clean_feature_cache_dir = (
        args.clean_feature_cache_dir
        if args.clean_feature_cache_dir is not None
        else (
            Path("outputs/audio_feature_benchmarks/total240_trainval_select/features")
            if args.feature_set == "total240"
            else run_dir / "features"
        )
    )
    train_stress_feature_dir = (
        args.train_stress_feature_dir
        if args.train_stress_feature_dir is not None
        else (
            Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features")
            if args.feature_set == "total240"
            else run_dir / "train_stress_features"
        )
    )
    test_stress_feature_dir = (
        args.test_stress_feature_dir
        if args.test_stress_feature_dir is not None
        else (
            Path("outputs/audio_feature_benchmarks/audio_tta_contact_stress_cv_select/test_tta_features")
            if args.feature_set == "total240"
            else run_dir / "test_tta_features"
        )
    )

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("CLEAN_FEATURE_CACHE_DIR  =", clean_feature_cache_dir.resolve())
    print("TRAIN_STRESS_FEATURE_DIR =", train_stress_feature_dir.resolve())
    print("TEST_STRESS_FEATURE_DIR  =", test_stress_feature_dir.resolve())
    print("Feature set              =", f"audio only {args.feature_set}")
    print("Selection                = train-only grouped OOF dense HGB TTA grid")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

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
            stress_feature_dir=train_stress_feature_dir,
            force_rebuild=False,
        )
        payloads[view] = payload
        timing[view] = view_timing

    X_by_view = {view: payloads[view]["X"] for view in STRESS_VIEWS}
    y = clean_feat["y"]
    for view in STRESS_VIEWS:
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_hgb_tta_grid_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    weight_grid = make_weight_grid(args.weight_step)
    split_summary = {
        "protocol": "audio-only HGB TTA grid; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "n_tta_weight_recipes": int(len(weight_grid)),
        "weight_step": int(args.weight_step),
        "selection_margin": float(args.selection_margin),
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = cv.make_candidates(args.random_state)
    hgb_specs = make_hgb_specs()
    leaderboard_rows = []
    fold_rows = []
    oof_paths = {}
    for spec in hgb_specs.values():
        rows, oof_by_view, candidate_fold_rows = evaluate_hgb_spec(
            spec,
            base_specs,
            hgb_specs,
            X_by_view,
            y,
            splits,
            fold_assignment,
            weight_grid,
        )
        leaderboard_rows.extend(rows)
        fold_rows.extend(candidate_fold_rows)
        spec_dir = oof_dir / spec.name
        spec_dir.mkdir(parents=True, exist_ok=True)
        oof_paths[spec.name] = {}
        for view, proba in oof_by_view.items():
            path = spec_dir / f"{view}_oof_proba.npy"
            np.save(path, proba)
            oof_paths[spec.name][view] = str(path.resolve())

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [
            "selection_score",
            "worst_single_view_hybrid",
            "worst_fold_hybrid_macro_contact",
            "tta_hybrid_macro_contact",
            "tta_macro_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_tta_grid_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = select_with_margin(leaderboard, args.selection_margin)
    selected_spec = hgb_specs[str(selected["base_candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selected_tta_weights = json.loads(selected["tta_weights_json"])
    selection_summary = {
        "selection_rule": (
            "highest train-only robust score, then within margin prefer simpler HGB train views "
            "with better worst-view and worst-fold hybrid"
        ),
        "selection_margin": float(args.selection_margin),
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_base_candidate": selected["base_candidate"],
        "selected_tta_recipe": selected["tta_recipe"],
        "selected_tta_weights": selected_tta_weights,
        "selected_base_name": selected_spec.base_name,
        "selected_train_views": list(selected_spec.train_views),
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
        "oof_proba_paths": oof_paths,
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
    test_df = base.load_manifest(test_csv, "robot_test")
    test_clean, test_clean_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=clean_feature_cache_dir,
        force_rebuild=False,
    )
    test_payloads = {"clean": test_clean}
    test_timing = {"clean": test_clean_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, view_timing = stress.build_or_load_stress_cache(
            test_df,
            view=view,
            stress_feature_dir=test_stress_feature_dir,
            force_rebuild=False,
        )
        test_payloads[view] = payload
        test_timing[view] = view_timing
    X_test_by_view = {view: test_payloads[view]["X"] for view in STRESS_VIEWS}

    start = time.perf_counter()
    final_artifact = stress.fit_stress_candidate(
        selected_spec,
        base_specs,
        hgb_specs,
        X_by_view,
        y,
        np.arange(len(y)),
    )
    final_fit_time = time.perf_counter() - start
    final_proba_by_view = {
        view: stress.predict_stress_artifact(final_artifact, X_test_by_view[view])
        for view in STRESS_VIEWS
    }
    final_proba = weighted_by_recipe(final_proba_by_view, selected_tta_weights)
    final_pred = tta.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=test_clean["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_hgb_tta_grid_stress_cv",
            "selected_score": selected["selection_score"],
            "selected_tta_macro_f1": selected["tta_macro_f1"],
            "selected_tta_contact_macro_f1": selected["tta_contact_macro_f1"],
            "selected_worst_single_view_hybrid": selected["worst_single_view_hybrid"],
            "selected_worst_fold_hybrid_macro_contact": selected["worst_fold_hybrid_macro_contact"],
            "selected_tta_weights_json": json.dumps(selected_tta_weights),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    audio_columns = [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_df]
    prediction_frame = test_df[audio_columns].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_clean["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_hgb_tta_grid_select_no_test_until_lock",
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
        "protocol": "audio_only_hgb_tta_grid_select_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
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

    print("\nFinal robot/test result after frozen HGB TTA grid selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
                "selected_tta_macro_f1",
                "selected_tta_contact_macro_f1",
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
