from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
            "Apply a locked train-only rule to an existing HGB TTA OOF leaderboard, "
            "write the selection lock, then evaluate once on robot/test."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_tta_grid_hgb_select"),
    )
    parser.add_argument(
        "--rule",
        choices=[
            "top_score",
            "top_contact_with_robust_floor",
            "top_worst_view_then_score",
        ],
        default="top_score",
    )
    parser.add_argument("--robust-floor", type=float, default=0.704)
    parser.add_argument("--fold-floor", type=float, default=0.712)
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


def select_row(leaderboard: pd.DataFrame, args: argparse.Namespace) -> dict:
    if args.rule == "top_score":
        row = leaderboard.sort_values(
            ["selection_score", "tta_hybrid_macro_contact", "tta_macro_f1"],
            ascending=False,
        ).iloc[0]
    elif args.rule == "top_contact_with_robust_floor":
        eligible = leaderboard[
            (leaderboard["worst_single_view_hybrid"] >= args.robust_floor)
            & (leaderboard["worst_fold_hybrid_macro_contact"] >= args.fold_floor)
        ].copy()
        if eligible.empty:
            raise RuntimeError("No leaderboard row satisfies the robust/contact floor")
        row = eligible.sort_values(
            ["tta_contact_macro_f1", "tta_macro_f1", "selection_score"],
            ascending=False,
        ).iloc[0]
    elif args.rule == "top_worst_view_then_score":
        row = leaderboard.sort_values(
            ["worst_single_view_hybrid", "selection_score", "tta_hybrid_macro_contact"],
            ascending=False,
        ).iloc[0]
    else:
        raise ValueError(f"Unknown rule: {args.rule}")
    return row.to_dict()


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    source_leaderboard = args.source_run / "reports" / "audio_tta_grid_hgb_select_oof_tta_grid_leaderboard.csv"
    if not source_leaderboard.exists():
        raise FileNotFoundError(f"Missing OOF leaderboard: {source_leaderboard}")

    run_slug = f"audio_tta_grid_hgb_rule_{args.rule}"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    leaderboard = pd.read_csv(source_leaderboard)
    selected = select_row(leaderboard, args)
    hgb_specs = grid_hgb.make_hgb_specs()
    selected_spec = hgb_specs[str(selected["base_candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selected_tta_weights = json.loads(selected["tta_weights_json"])
    selection_summary = {
        "selection_rule": args.rule,
        "robust_floor": float(args.robust_floor),
        "fold_floor": float(args.fold_floor),
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_base_candidate": selected["base_candidate"],
        "selected_tta_recipe": selected["tta_recipe"],
        "selected_tta_weights": selected_tta_weights,
        "selected_base_name": selected_spec.base_name,
        "selected_train_views": list(selected_spec.train_views),
        "selected_bias": selected_bias.tolist(),
        "source_leaderboard_path": str(source_leaderboard.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("Selection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    train_payloads = {"clean": clean_feat}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, _ = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.train_stress_feature_dir,
            force_rebuild=False,
        )
        train_payloads[view] = payload
    X_by_view = {view: train_payloads[view]["X"] for view in STRESS_VIEWS}
    y = clean_feat["y"]

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

    base_specs = cv.make_candidates(args.random_state)
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
            "selected_by": f"hand_default_audio_only_hgb_tta_grid_rule_{args.rule}",
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
            "protocol": "audio_only_hgb_tta_grid_locked_rule_no_test_until_lock",
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_hgb_tta_grid_locked_rule_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen locked-rule HGB TTA selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
