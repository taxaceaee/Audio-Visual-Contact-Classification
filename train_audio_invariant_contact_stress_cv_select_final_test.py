from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import train_audio_contact_stress_cv_select_final_test as contact
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS
FEATURE_SET = "total240_invariant1200"
FEATURE_NAME = "Total240 fixed invariant transforms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only invariant total240 stress-CV. Selection uses only hand/default "
            "audio and train-only perturbations; robot/test is loaded only after "
            "selected_without_test.json is written."
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
        "--stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument("--force-rebuild-stress", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def robust_row_normalize(block: np.ndarray) -> np.ndarray:
    median = np.median(block, axis=1, keepdims=True)
    q25 = np.percentile(block, 25, axis=1, keepdims=True)
    q75 = np.percentile(block, 75, axis=1, keepdims=True)
    scale = np.maximum(q75 - q25, 1e-4)
    return np.clip((block - median) / scale, -8.0, 8.0)


def invariant_total240(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    full = X[:, :120]
    top = X[:, 120:240]
    signed_log = np.sign(X) * np.log1p(np.abs(X))
    full_norm = robust_row_normalize(full)
    top_norm = robust_row_normalize(top)
    diff = top - full
    signed_ratio = np.sign(top) * np.log1p(np.abs(top) / (np.abs(full) + 1e-3))
    l2 = np.sqrt(np.mean(np.square(X), axis=1, keepdims=True) + 1e-6)
    l2_norm = np.clip(X / l2, -8.0, 8.0)
    features = np.hstack([X, signed_log, full_norm, top_norm, diff, signed_ratio, l2_norm]).astype(np.float32)
    if features.shape[1] != 1200:
        raise AssertionError(f"Expected 1200 invariant features, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("Invariant feature matrix contains NaN/Inf")
    return features


def make_candidates(random_state: int) -> dict[str, cv.CandidateSpec]:
    class_weights = base.CONFIG["class_weights"]
    return {
        "inv_hgb_default": cv.CandidateSpec(
            name="inv_hgb_default",
            kind="direct",
            direct_factory=lambda: HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.05,
                max_leaf_nodes=31,
                class_weight=class_weights,
                random_state=random_state,
            ),
        ),
        "inv_hgb_regularized": cv.CandidateSpec(
            name="inv_hgb_regularized",
            kind="direct",
            direct_factory=lambda: HistGradientBoostingClassifier(
                max_iter=320,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=28,
                l2_regularization=0.12,
                class_weight=class_weights,
                random_state=random_state + 7,
            ),
        ),
        "inv_lgbm_regularized": cv.CandidateSpec(
            name="inv_lgbm_regularized",
            kind="direct",
            direct_factory=lambda: LGBMClassifier(
                objective="multiclass",
                num_class=len(base.CLASS_NAMES),
                n_estimators=520,
                learning_rate=0.025,
                num_leaves=17,
                min_child_samples=34,
                subsample=0.82,
                colsample_bytree=0.78,
                reg_lambda=3.0,
                class_weight=class_weights,
                random_state=random_state + 11,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        "inv_extra_trees": cv.CandidateSpec(
            name="inv_extra_trees",
            kind="direct",
            direct_factory=lambda: ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight=class_weights,
                random_state=random_state + 13,
                n_jobs=-1,
            ),
        ),
    }


def make_stress_specs() -> dict[str, stress.StressCandidate]:
    specs = [
        stress.StressCandidate("inv_hgb_default__all_aug", "inv_hgb_default", "single", STRESS_VIEWS),
        stress.StressCandidate("inv_hgb_regularized__all_aug", "inv_hgb_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("inv_lgbm_regularized__all_aug", "inv_lgbm_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("inv_extra_trees__all_aug", "inv_extra_trees", "single", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def patch_feature_metadata(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["feature_set"] = FEATURE_SET
        row["feature_name"] = FEATURE_NAME
        row["n_features"] = 1200
    return rows


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_invariant_contact_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH               =", root_path.resolve())
    print("RUN_DIR                 =", run_dir.resolve())
    print("CLEAN_FEATURE_CACHE_DIR =", args.clean_feature_cache_dir.resolve())
    print("STRESS_FEATURE_DIR      =", args.stress_feature_dir.resolve())
    print("Feature set             =", FEATURE_NAME)
    print("Selection               = train-only contact-aware stress-CV")

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
            stress_feature_dir=args.stress_feature_dir,
            force_rebuild=args.force_rebuild_stress,
        )
        payloads[view] = payload
        timing[view] = view_timing

    y = clean_feat["y"]
    X_by_view = {}
    for view in STRESS_VIEWS:
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")
        X_by_view[view] = invariant_total240(payloads[view]["X"])

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_invariant_contact_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only invariant total240 contact stress-CV; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "n_features": 1200,
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = make_candidates(args.random_state)
    stress_specs = make_stress_specs()
    leaderboard_rows = []
    fold_rows = []
    for spec in stress_specs.values():
        rows, _, candidate_fold_rows = contact.evaluate_candidate(
            spec,
            base_specs,
            stress_specs,
            X_by_view,
            y,
            splits,
        )
        leaderboard_rows.extend(patch_feature_metadata(rows))
        fold_rows.extend(candidate_fold_rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [
            "selection_worst_hybrid_score",
            "selection_mean_hybrid_score",
            "worst_macro_f1",
            "worst_contact_macro_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_contact_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_spec = stress_specs[str(selected["base_candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_base_candidate": selected["base_candidate"],
        "selected_bias_variant": selected["bias_variant"],
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
    test_feat, test_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    X_test = invariant_total240(test_feat["X"])

    start = time.perf_counter()
    final_artifact = stress.fit_stress_candidate(
        selected_spec,
        base_specs,
        stress_specs,
        X_by_view,
        y,
        np.arange(len(y)),
    )
    final_fit_time = time.perf_counter() - start
    final_proba = stress.predict_stress_artifact(final_artifact, X_test)
    final_pred = contact.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=test_feat["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "feature_set": FEATURE_SET,
            "feature_name": FEATURE_NAME,
            "n_features": 1200,
            "selected_by": "hand_default_audio_only_invariant_total240_contact_stress_cv",
            "selected_worst_hybrid_score": selected["selection_worst_hybrid_score"],
            "selected_worst_macro_f1": selected["worst_macro_f1"],
            "selected_worst_contact_macro_f1": selected["worst_contact_macro_f1"],
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
        confusion_matrix(test_feat["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_invariant_total240_contact_stress_cv_select_no_test_until_final",
            "feature_set": FEATURE_SET,
            "feature_name": FEATURE_NAME,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_invariant_total240_contact_stress_cv_select_no_test_until_final",
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

    print("\nFinal robot/test result after frozen audio-only invariant selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_worst_hybrid_score",
                "selected_worst_contact_macro_f1",
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
