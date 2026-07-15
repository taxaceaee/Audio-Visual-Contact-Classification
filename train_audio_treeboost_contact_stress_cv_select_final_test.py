from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

import train_audio_contact_stress_cv_select_final_test as contact
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


class WeightedXGBClassifier:
    """Small sklearn-style wrapper so XGBoost receives supervised class weights."""

    def __init__(self, class_weights: dict[int, float], **params: object) -> None:
        self.class_weights = class_weights
        self.params = params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedXGBClassifier":
        sample_weight = np.asarray(
            [self.class_weights.get(int(label), 1.0) for label in y],
            dtype=np.float64,
        )
        self.model_ = XGBClassifier(**self.params)
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(self.model_.classes_, dtype=np.int64)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only tree/boosting stress-CV. Model and bias selection use only "
            "hand/default audio plus deterministic train-only audio perturbations. "
            "Robot/test is loaded only after selected_without_test.json is written."
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


def make_treeboost_base_candidates(random_state: int) -> dict[str, cv.CandidateSpec]:
    class_weights = base.CONFIG["class_weights"]
    base_specs = cv.make_candidates(random_state)
    base_specs.update(
        {
            "direct_extra_trees_leaf1": cv.CandidateSpec(
                name="direct_extra_trees_leaf1",
                kind="direct",
                direct_factory=lambda: ExtraTreesClassifier(
                    n_estimators=1200,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight=class_weights,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            "direct_extra_trees_log2_leaf1": cv.CandidateSpec(
                name="direct_extra_trees_log2_leaf1",
                kind="direct",
                direct_factory=lambda: ExtraTreesClassifier(
                    n_estimators=1200,
                    max_features="log2",
                    min_samples_leaf=1,
                    class_weight=class_weights,
                    random_state=random_state + 11,
                    n_jobs=-1,
                ),
            ),
            "direct_xgb_weighted_depth3": cv.CandidateSpec(
                name="direct_xgb_weighted_depth3",
                kind="direct",
                direct_factory=lambda: WeightedXGBClassifier(
                    class_weights=class_weights,
                    objective="multi:softprob",
                    num_class=len(base.CLASS_NAMES),
                    n_estimators=520,
                    learning_rate=0.025,
                    max_depth=3,
                    min_child_weight=4,
                    subsample=0.86,
                    colsample_bytree=0.82,
                    reg_lambda=3.0,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=random_state + 41,
                    n_jobs=-1,
                ),
            ),
            "direct_xgb_weighted_depth5": cv.CandidateSpec(
                name="direct_xgb_weighted_depth5",
                kind="direct",
                direct_factory=lambda: WeightedXGBClassifier(
                    class_weights=class_weights,
                    objective="multi:softprob",
                    num_class=len(base.CLASS_NAMES),
                    n_estimators=430,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=3,
                    subsample=0.82,
                    colsample_bytree=0.78,
                    reg_lambda=2.5,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=random_state + 53,
                    n_jobs=-1,
                ),
            ),
        }
    )
    return base_specs


def make_treeboost_stress_candidates() -> dict[str, stress.StressCandidate]:
    specs = [
        stress.StressCandidate("treeboost_hgb_default__all_aug", "direct_hgb_default", "single", STRESS_VIEWS),
        stress.StressCandidate("treeboost_extra_trees_repo__all_aug", "direct_extra_trees", "single", STRESS_VIEWS),
        stress.StressCandidate("treeboost_extra_trees_leaf1__clean", "direct_extra_trees_leaf1", "single", ("clean",)),
        stress.StressCandidate("treeboost_extra_trees_leaf1__all_aug", "direct_extra_trees_leaf1", "single", STRESS_VIEWS),
        stress.StressCandidate("treeboost_extra_trees_log2_leaf1__all_aug", "direct_extra_trees_log2_leaf1", "single", STRESS_VIEWS),
        stress.StressCandidate("treeboost_xgb_weighted_depth3__clean", "direct_xgb_weighted_depth3", "single", ("clean",)),
        stress.StressCandidate("treeboost_xgb_weighted_depth3__all_aug", "direct_xgb_weighted_depth3", "single", STRESS_VIEWS),
        stress.StressCandidate("treeboost_xgb_weighted_depth5__all_aug", "direct_xgb_weighted_depth5", "single", STRESS_VIEWS),
        stress.StressCandidate("treeboost_xgb_repo_regularized__all_aug", "direct_xgboost_regularized", "single", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_treeboost_contact_stress_cv_select"
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
    print("Feature set             = audio only total240")
    print("Selection               = train-only contact-aware stress-CV over tree/boosting models")

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

    X_by_view = {view: payloads[view]["X"] for view in STRESS_VIEWS}
    y = clean_feat["y"]
    for view in STRESS_VIEWS:
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_treeboost_contact_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only tree/boosting contact-aware stress-CV; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = make_treeboost_base_candidates(args.random_state)
    treeboost_specs = make_treeboost_stress_candidates()
    leaderboard_rows = []
    fold_rows = []
    for spec in treeboost_specs.values():
        rows, _, candidate_fold_rows = contact.evaluate_candidate(
            spec,
            base_specs,
            treeboost_specs,
            X_by_view,
            y,
            splits,
        )
        leaderboard_rows.extend(rows)
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
    selected_spec = treeboost_specs[str(selected["base_candidate"])]
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

    full_idx = np.arange(len(y))
    start = time.perf_counter()
    final_artifact = stress.fit_stress_candidate(
        selected_spec,
        base_specs,
        treeboost_specs,
        X_by_view,
        y,
        full_idx,
    )
    final_fit_time = time.perf_counter() - start
    final_proba = stress.predict_stress_artifact(final_artifact, test_feat["X"])
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
            "selected_by": "hand_default_audio_only_treeboost_contact_aware_stress_cv",
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
            "protocol": "audio_only_treeboost_contact_stress_cv_select_no_test_until_final",
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
        "protocol": "audio_only_treeboost_contact_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "clean_feature_cache_dir": str(args.clean_feature_cache_dir.resolve()),
        "stress_feature_dir": str(args.stress_feature_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "fold_report": str(fold_report_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions_audio_only": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen audio-only tree/boosting selection:")
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
