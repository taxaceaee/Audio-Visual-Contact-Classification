from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import train_audio_tta_grid_hgb_select_final_test as grid_hgb
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only HGB class-weight TTA selection. Class weights are selected "
            "with grouped train/default OOF only; robot/test is loaded after lock."
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
    parser.add_argument(
        "--test-stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_tta_contact_stress_cv_select/test_tta_features"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def hgb_factory(random_state: int, class_weight: dict[int, float] | str) -> object:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        class_weight=class_weight,
        random_state=random_state,
    )


def make_classweight_base_specs(random_state: int) -> dict[str, cv.CandidateSpec]:
    variants: dict[str, dict[int, float] | str] = {
        "cw_hgb_default": base.CONFIG["class_weights"],
        "cw_hgb_balanced": "balanced",
        "cw_hgb_trunk_boost": {0: 0.55, 1: 1.0, 2: 2.6, 3: 1.3},
        "cw_hgb_trunk_twig_boost": {0: 0.55, 1: 1.0, 2: 2.4, 3: 1.8},
        "cw_hgb_contact_balanced": {0: 0.50, 1: 1.4, 2: 2.2, 3: 1.6},
    }
    return {
        name: cv.CandidateSpec(
            name=name,
            kind="direct",
            direct_factory=lambda weights=weights: hgb_factory(random_state, weights),
        )
        for name, weights in variants.items()
    }


def make_classweight_stress_specs() -> dict[str, stress.StressCandidate]:
    return {
        f"{base_name}__all_aug": stress.StressCandidate(
            name=f"{base_name}__all_aug",
            base_name=base_name,
            kind="single",
            train_views=STRESS_VIEWS,
        )
        for base_name in [
            "cw_hgb_default",
            "cw_hgb_balanced",
            "cw_hgb_trunk_boost",
            "cw_hgb_trunk_twig_boost",
            "cw_hgb_contact_balanced",
        ]
    }


def focused_weight_grid() -> dict[str, dict[str, float]]:
    return {
        "clean_robot_50_50": {"clean": 0.5, "robot_mix": 0.5},
        "clean40_robot60": {"clean": 0.4, "robot_mix": 0.6},
        "clean30_robot70": {"clean": 0.3, "robot_mix": 0.7},
        "clean40_robot40_band20": {"clean": 0.4, "robot_mix": 0.4, "bandlimit": 0.2},
        "all_equal": {"clean": 1 / 3, "robot_mix": 1 / 3, "bandlimit": 1 / 3},
    }


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_hgb_classweight_tta_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("Selection                = train-only HGB class-weight TTA grid")

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
            force_rebuild=False,
        )
        payloads[view] = payload
        timing[view] = view_timing
    X_by_view = {view: payloads[view]["X"] for view in STRESS_VIEWS}
    y = clean_feat["y"]

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_hgb_classweight_tta_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only HGB class-weight TTA; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = make_classweight_base_specs(args.random_state)
    stress_specs = make_classweight_stress_specs()
    leaderboard_rows = []
    fold_rows = []
    for spec in stress_specs.values():
        rows, _, candidate_fold_rows = grid_hgb.evaluate_hgb_spec(
            spec,
            base_specs,
            stress_specs,
            X_by_view,
            y,
            splits,
            fold_assignment,
            focused_weight_grid(),
        )
        leaderboard_rows.extend(rows)
        fold_rows.extend(candidate_fold_rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [
            "selection_score",
            "tta_contact_macro_f1",
            "worst_single_view_hybrid",
            "worst_fold_hybrid_macro_contact",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_classweight_tta_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_spec = stress_specs[str(selected["base_candidate"])]
    selected_bias = np.asarray(json.loads(str(selected["class_bias_json"])), dtype=np.float64)
    selected_tta_weights = json.loads(str(selected["tta_weights_json"]))
    selection_summary = {
        "selection_rule": "best train-only HGB class-weight TTA robust score",
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_base_candidate": selected["base_candidate"],
        "selected_base_name": selected_spec.base_name,
        "selected_train_views": list(selected_spec.train_views),
        "selected_tta_recipe": selected["tta_recipe"],
        "selected_tta_weights": selected_tta_weights,
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
            stress_feature_dir=args.test_stress_feature_dir,
            force_rebuild=False,
        )
        test_payloads[view] = payload
        test_timing[view] = view_timing
    X_test_by_view = {view: test_payloads[view]["X"] for view in STRESS_VIEWS}

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
    final_proba_by_view = {
        view: stress.predict_stress_artifact(final_artifact, X_test_by_view[view])
        for view in STRESS_VIEWS
    }
    final_proba = grid_hgb.weighted_by_recipe(final_proba_by_view, selected_tta_weights)
    final_pred = grid_hgb.tta.predict_with_bias(final_proba, selected_bias)
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
            "selected_by": "hand_default_audio_only_hgb_classweight_tta",
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
            "protocol": "audio_only_hgb_classweight_tta_select_no_test_until_lock",
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_hgb_classweight_tta_select_no_test_until_lock",
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

    print("\nFinal robot/test result after frozen HGB class-weight TTA selection:")
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
