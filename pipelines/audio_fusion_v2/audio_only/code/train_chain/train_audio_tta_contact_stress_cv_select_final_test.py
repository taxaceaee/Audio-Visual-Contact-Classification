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

import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only supervised TTA selection. OOF train folds choose model, "
            "test-time augmentation weights, and class bias before robot/test is loaded."
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
        "--train-stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument("--force-rebuild-train-stress", action="store_true")
    parser.add_argument("--force-rebuild-test-stress", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def make_tta_candidates() -> dict[str, stress.StressCandidate]:
    specs = [
        stress.StressCandidate("tta_hgb_default__all_aug", "direct_hgb_default", "single", STRESS_VIEWS),
        stress.StressCandidate("tta_hgb_regularized__all_aug", "direct_hgb_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("tta_lgbm_regularized__all_aug", "direct_lightgbm_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("tta_hier_extra_hgb__all_aug", "hier_extra_hgb", "single", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def tta_recipes() -> dict[str, dict[str, float]]:
    return {
        "clean_only": {"clean": 1.0},
        "all_equal": {"clean": 1 / 3, "robot_mix": 1 / 3, "bandlimit": 1 / 3},
        "clean_half_aug_half": {"clean": 0.5, "robot_mix": 0.25, "bandlimit": 0.25},
        "robot_heavy": {"clean": 0.2, "robot_mix": 0.5, "bandlimit": 0.3},
        "band_heavy": {"clean": 0.2, "robot_mix": 0.3, "bandlimit": 0.5},
        "aug_only_equal": {"robot_mix": 0.5, "bandlimit": 0.5},
    }


def weighted_proba(proba_by_view: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    output = np.zeros_like(next(iter(proba_by_view.values())))
    for view, weight in weights.items():
        output += (weight / total) * proba_by_view[view]
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return cv.predict_with_bias(proba, bias)


def score_proba(y_true: np.ndarray, proba: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    pred = predict_with_bias(proba, bias)
    macro = f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)
    contact = f1_score(y_true, pred, labels=CONTACT_LABELS, average="macro", zero_division=0)
    return {
        "macro_f1": float(macro),
        "contact_macro_f1": float(contact),
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def tune_bias(
    proba: np.ndarray,
    y_true: np.ndarray,
    objective: str,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if grid is None:
        grid = np.asarray([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2], dtype=np.float64)

    best_bias = np.zeros(4, dtype=np.float64)
    best_scores = score_proba(y_true, proba, best_bias)
    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        scores = score_proba(y_true, proba, bias)
        if scores[objective] > best_scores[objective]:
            best_bias = bias
            best_scores = scores

    for ambient_bias in np.linspace(-0.6, 0.6, 7):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        scores = score_proba(y_true, proba, bias)
        if scores[objective] > best_scores[objective]:
            best_bias = bias
            best_scores = scores
    return best_bias, best_scores


def bias_variants(proba: np.ndarray, y_true: np.ndarray) -> list[tuple[str, np.ndarray, dict[str, float]]]:
    zero = np.zeros(4, dtype=np.float64)
    variants = [("no_bias", zero, score_proba(y_true, proba, zero))]
    for objective in ["macro_f1", "contact_macro_f1", "hybrid_macro_contact"]:
        bias, scores = tune_bias(proba, y_true, objective)
        variants.append((f"tuned_{objective}", bias, scores))
    return variants


def evaluate_candidate(
    spec: stress.StressCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    stress_specs: dict[str, stress.StressCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    print(f"\nTTA candidate: {spec.name} train_views={spec.train_views}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = stress.fit_stress_candidate(spec, base_specs, stress_specs, X_by_view, y, train_idx)
        train_time = time.perf_counter() - start
        parts = []
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = stress.predict_stress_artifact(artifact, X_by_view[view][val_idx])
            no_bias = score_proba(y[val_idx], oof_by_view[view][val_idx], np.zeros(4, dtype=np.float64))
            parts.append(f"{view}=M{no_bias['macro_f1']:.4f}/C{no_bias['contact_macro_f1']:.4f}")
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
    recipes = tta_recipes()
    for recipe_name, weights in recipes.items():
        tta_proba = weighted_proba(oof_by_view, weights)
        for bias_name, bias, scores in bias_variants(tta_proba, y):
            pred = predict_with_bias(tta_proba, bias)
            row = base.make_report_row(
                model_name=f"{spec.name}__{recipe_name}__{bias_name}",
                split_name="audio_tta_stress_oof_cv",
                y_true=y,
                y_pred=pred,
                train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
                predict_time_sec=0.0,
            )
            per_view_scores = {
                f"{view}_{metric}": score_proba(y, oof_by_view[view], bias)[metric]
                for view in STRESS_VIEWS
                for metric in ["macro_f1", "contact_macro_f1", "hybrid_macro_contact"]
            }
            worst_hybrid = min(per_view_scores[f"{view}_hybrid_macro_contact"] for view in STRESS_VIEWS)
            row.update(
                {
                    "base_candidate": spec.name,
                    "bias_variant": bias_name,
                    "tta_recipe": recipe_name,
                    "tta_weights_json": json.dumps(weights),
                    "class_bias_json": json.dumps(bias.tolist()),
                    "candidate_kind": spec.kind,
                    "base_name": spec.base_name or "",
                    "train_views_json": json.dumps(list(spec.train_views)),
                    "tta_macro_f1": scores["macro_f1"],
                    "tta_contact_macro_f1": scores["contact_macro_f1"],
                    "tta_hybrid_macro_contact": scores["hybrid_macro_contact"],
                    "worst_single_view_hybrid": worst_hybrid,
                    "selection_score": float(0.8 * scores["hybrid_macro_contact"] + 0.2 * worst_hybrid),
                    **per_view_scores,
                }
            )
            rows.append(row)
        best_for_recipe = max(
            [row for row in rows if row["base_candidate"] == spec.name and row["tta_recipe"] == recipe_name],
            key=lambda item: item["selection_score"],
        )
        print(
            f"  recipe {recipe_name}: best selection={best_for_recipe['selection_score']:.4f} "
            f"tta_macro={best_for_recipe['tta_macro_f1']:.4f} "
            f"tta_contact={best_for_recipe['tta_contact_macro_f1']:.4f} "
            f"bias={best_for_recipe['class_bias_json']}",
            flush=True,
        )
    return rows, oof_by_view, fold_rows


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_tta_contact_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    test_stress_feature_dir = run_dir / "test_tta_features"
    for directory in [run_dir, report_dir, model_dir, split_dir, test_stress_feature_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("CLEAN_FEATURE_CACHE_DIR  =", args.clean_feature_cache_dir.resolve())
    print("TRAIN_STRESS_FEATURE_DIR =", args.train_stress_feature_dir.resolve())
    print("TEST_STRESS_FEATURE_DIR  =", test_stress_feature_dir.resolve())
    print("Feature set              = audio only total240")
    print("Selection                = train-only OOF TTA weights + class bias")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    clean_feat, clean_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
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
            stress_feature_dir=args.train_stress_feature_dir,
            force_rebuild=args.force_rebuild_train_stress,
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
    split_path = split_dir / "hand_train_full_tta_contact_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only supervised TTA stress-CV; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = cv.make_candidates(args.random_state)
    tta_specs = make_tta_candidates()
    leaderboard_rows = []
    fold_rows = []
    for spec in tta_specs.values():
        rows, _, candidate_fold_rows = evaluate_candidate(
            spec,
            base_specs,
            tta_specs,
            X_by_view,
            y,
            splits,
        )
        leaderboard_rows.extend(rows)
        fold_rows.extend(candidate_fold_rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["selection_score", "tta_hybrid_macro_contact", "tta_macro_f1", "worst_single_view_hybrid"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_tta_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_spec = tta_specs[str(selected["base_candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selected_tta_weights = json.loads(selected["tta_weights_json"])
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_base_candidate": selected["base_candidate"],
        "selected_bias_variant": selected["bias_variant"],
        "selected_tta_recipe": selected["tta_recipe"],
        "selected_tta_weights": selected_tta_weights,
        "selected_base_name": selected_spec.base_name,
        "selected_train_views": list(selected_spec.train_views),
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
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
        feature_dir=args.clean_feature_cache_dir,
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
            force_rebuild=args.force_rebuild_test_stress,
        )
        test_payloads[view] = payload
        test_timing[view] = view_timing
    X_test_by_view = {view: test_payloads[view]["X"] for view in STRESS_VIEWS}

    start = time.perf_counter()
    final_artifact = stress.fit_stress_candidate(
        selected_spec,
        base_specs,
        tta_specs,
        X_by_view,
        y,
        np.arange(len(y)),
    )
    final_fit_time = time.perf_counter() - start
    final_proba_by_view = {
        view: stress.predict_stress_artifact(final_artifact, X_test_by_view[view])
        for view in STRESS_VIEWS
    }
    final_proba = weighted_proba(final_proba_by_view, selected_tta_weights)
    final_pred = predict_with_bias(final_proba, selected_bias)
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
            "selected_by": "hand_default_audio_only_supervised_tta_stress_cv",
            "selected_score": selected["selection_score"],
            "selected_tta_hybrid_macro_contact": selected["tta_hybrid_macro_contact"],
            "selected_tta_macro_f1": selected["tta_macro_f1"],
            "selected_tta_contact_macro_f1": selected["tta_contact_macro_f1"],
            "selected_worst_single_view_hybrid": selected["worst_single_view_hybrid"],
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
            "protocol": "audio_only_supervised_tta_stress_cv_select_no_test_until_final",
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
        "protocol": "audio_only_supervised_tta_stress_cv_select_no_test_until_final",
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

    print("\nFinal robot/test result after frozen audio-only TTA selection:")
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
