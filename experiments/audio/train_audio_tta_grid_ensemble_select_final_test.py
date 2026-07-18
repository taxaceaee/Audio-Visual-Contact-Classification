from __future__ import annotations

import argparse
import itertools
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
            "Audio-only HGB TTA ensemble selection from saved train-only OOF proba. "
            "The ensemble and bias are locked before robot/test is loaded."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-per-base", type=int, default=3)
    parser.add_argument("--feature-set", choices=sorted(base.FEATURE_SPECS), default="total240")
    parser.add_argument(
        "--source-run",
        type=Path,
        default=None,
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


def load_oof_by_base(source_run: Path, base_candidate: str) -> dict[str, np.ndarray]:
    base_dir = source_run / "oof_proba" / base_candidate
    return {
        view: np.load(base_dir / f"{view}_oof_proba.npy")
        for view in STRESS_VIEWS
    }


def tta_proba_for_row(
    oof_by_base: dict[str, dict[str, np.ndarray]],
    row: pd.Series,
) -> np.ndarray:
    return grid_hgb.weighted_by_recipe(
        oof_by_base[str(row["base_candidate"])],
        json.loads(str(row["tta_weights_json"])),
    )


def ensemble_probas(
    oof_by_base: dict[str, dict[str, np.ndarray]],
    members: list[dict],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    total = sum(float(member["member_weight"]) for member in members)
    tta_output = None
    view_output = {view: None for view in STRESS_VIEWS}
    for member in members:
        weight = float(member["member_weight"]) / total
        base_candidate = str(member["base_candidate"])
        tta_weights = json.loads(str(member["tta_weights_json"]))
        member_tta = grid_hgb.weighted_by_recipe(oof_by_base[base_candidate], tta_weights)
        tta_output = weight * member_tta if tta_output is None else tta_output + weight * member_tta
        for view in STRESS_VIEWS:
            part = weight * oof_by_base[base_candidate][view]
            view_output[view] = part if view_output[view] is None else view_output[view] + part
    return tta_output, view_output


def make_member_pool(leaderboard: pd.DataFrame, top_per_base: int) -> list[dict]:
    pool = []
    for _, sub in leaderboard.groupby("base_candidate", sort=False):
        ranked = sub.sort_values(
            ["selection_score", "tta_contact_macro_f1", "tta_macro_f1"],
            ascending=False,
        ).head(top_per_base)
        for _, row in ranked.iterrows():
            item = row.to_dict()
            item["member_id"] = str(item["model"])
            pool.append(item)
    return pool


def make_ensemble_specs(pool: list[dict]) -> list[list[dict]]:
    specs: list[list[dict]] = []
    for member in pool:
        specs.append([{**member, "member_weight": 1.0}])

    by_base = {}
    for member in pool:
        by_base.setdefault(str(member["base_candidate"]), []).append(member)
    base_best = [
        sorted(rows, key=lambda item: item["selection_score"], reverse=True)[0]
        for rows in by_base.values()
    ]

    pair_weights = [(0.5, 0.5), (0.65, 0.35), (0.35, 0.65)]
    for left, right in itertools.combinations(base_best, 2):
        for left_weight, right_weight in pair_weights:
            specs.append(
                [
                    {**left, "member_weight": left_weight},
                    {**right, "member_weight": right_weight},
                ]
            )

    if len(base_best) >= 3:
        for members in itertools.combinations(base_best, 3):
            specs.append([{**member, "member_weight": 1.0 / 3.0} for member in members])
    return specs


def score_ensemble(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    oof_by_base: dict[str, dict[str, np.ndarray]],
    members: list[dict],
) -> dict:
    tta_proba, view_proba = ensemble_probas(oof_by_base, members)
    bias, metrics = grid_hgb.tune_bias_for_selection(
        y,
        view_proba,
        tta_proba,
        fold_assignment,
        np.asarray([], dtype=np.float64),
    )
    pred = grid_hgb.tta.predict_with_bias(tta_proba, bias)
    row = base.make_report_row(
        model_name="ens__" + "__".join(str(member["member_id"]) for member in members),
        split_name="audio_hgb_tta_ensemble_oof_cv",
        y_true=y,
        y_pred=pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    row.update(
        {
            "n_members": len(members),
            "members_json": json.dumps(
                [
                    {
                        "member_id": str(member["member_id"]),
                        "base_candidate": str(member["base_candidate"]),
                        "tta_weights_json": str(member["tta_weights_json"]),
                        "member_weight": float(member["member_weight"]),
                    }
                    for member in members
                ]
            ),
            "class_bias_json": json.dumps(bias.tolist()),
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
    return row


def final_member_proba(
    final_proba_by_base: dict[str, dict[str, np.ndarray]],
    member: dict,
) -> np.ndarray:
    return grid_hgb.weighted_by_recipe(
        final_proba_by_base[str(member["base_candidate"])],
        json.loads(str(member["tta_weights_json"])),
    )


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set(args.feature_set)

    root_path = base.resolve_root(args.root)
    source_slug = (
        "audio_tta_grid_hgb_select"
        if args.feature_set == "total240"
        else f"audio_{args.feature_set}_tta_grid_hgb_select"
    )
    source_run = args.source_run
    if source_run is None:
        source_run = args.output / "audio_feature_benchmarks" / source_slug
    source_slug = source_run.name
    run_slug = (
        "audio_tta_grid_hgb_ensemble_select"
        if args.feature_set == "total240"
        else f"audio_{args.feature_set}_tta_grid_hgb_ensemble_select"
    )
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    clean_feature_cache_dir = (
        args.clean_feature_cache_dir
        if args.clean_feature_cache_dir is not None
        else (
            Path("outputs/audio_feature_benchmarks/total240_trainval_select/features")
            if args.feature_set == "total240"
            else source_run / "features"
        )
    )
    train_stress_feature_dir = (
        args.train_stress_feature_dir
        if args.train_stress_feature_dir is not None
        else (
            Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features")
            if args.feature_set == "total240"
            else source_run / "train_stress_features"
        )
    )
    test_stress_feature_dir = (
        args.test_stress_feature_dir
        if args.test_stress_feature_dir is not None
        else (
            Path("outputs/audio_feature_benchmarks/audio_tta_contact_stress_cv_select/test_tta_features")
            if args.feature_set == "total240"
            else source_run / "test_tta_features"
        )
    )

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("SOURCE_RUN               =", source_run.resolve())
    print("CLEAN_FEATURE_CACHE_DIR  =", clean_feature_cache_dir.resolve())
    print("TRAIN_STRESS_FEATURE_DIR =", train_stress_feature_dir.resolve())
    print("TEST_STRESS_FEATURE_DIR  =", test_stress_feature_dir.resolve())
    print("Feature set              =", f"audio only {args.feature_set}")
    print("Selection                = train-only saved OOF HGB TTA ensemble")

    leaderboard_path = source_run / "reports" / f"{source_slug}_oof_tta_grid_leaderboard.csv"
    split_path = source_run / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing source leaderboard: {leaderboard_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing source split assignment: {split_path}")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=clean_feature_cache_dir,
        force_rebuild=False,
    )
    y = clean_feat["y"]
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)

    source_leaderboard = pd.read_csv(leaderboard_path)
    member_pool = make_member_pool(source_leaderboard, args.top_per_base)
    base_candidates = sorted({str(member["base_candidate"]) for member in member_pool})
    oof_by_base = {
        base_candidate: load_oof_by_base(source_run, base_candidate)
        for base_candidate in base_candidates
    }

    rows = []
    for members in make_ensemble_specs(member_pool):
        rows.append(score_ensemble(y, fold_assignment, oof_by_base, members))

    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "tta_contact_macro_f1",
            "worst_single_view_hybrid",
            "worst_fold_hybrid_macro_contact",
        ],
        ascending=False,
    ).reset_index(drop=True)
    ensemble_leaderboard_path = report_dir / f"{run_slug}_oof_ensemble_leaderboard.csv"
    leaderboard.to_csv(ensemble_leaderboard_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_members = json.loads(str(selected["members_json"]))
    selected_bias = np.asarray(json.loads(str(selected["class_bias_json"])), dtype=np.float64)
    selection_summary = {
        "selection_rule": "best train-only HGB TTA ensemble score from saved OOF proba",
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_members": selected_members,
        "selected_bias": selected_bias.tolist(),
        "feature_set": args.feature_set,
        "feature_name": base.FEATURE_SPEC["display_name"],
        "source_run": str(source_run.resolve()),
        "source_leaderboard_path": str(leaderboard_path.resolve()),
        "ensemble_leaderboard_path": str(ensemble_leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("Selection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    train_payloads = {"clean": clean_feat}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, _ = stress.build_or_load_stress_cache(
        train_df,
        view=view,
            stress_feature_dir=train_stress_feature_dir,
        force_rebuild=False,
        )
        train_payloads[view] = payload
    X_by_view = {view: train_payloads[view]["X"] for view in STRESS_VIEWS}

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

    hgb_specs = grid_hgb.make_hgb_specs()
    base_specs = cv.make_candidates(args.random_state)
    final_artifacts = {}
    final_proba_by_base = {}
    start = time.perf_counter()
    for base_candidate in sorted({str(member["base_candidate"]) for member in selected_members}):
        artifact = stress.fit_stress_candidate(
            hgb_specs[base_candidate],
            base_specs,
            hgb_specs,
            X_by_view,
            y,
            np.arange(len(y)),
        )
        final_artifacts[base_candidate] = artifact
        final_proba_by_base[base_candidate] = {
            view: stress.predict_stress_artifact(artifact, X_test_by_view[view])
            for view in STRESS_VIEWS
        }
    final_fit_time = time.perf_counter() - start

    total_weight = sum(float(member["member_weight"]) for member in selected_members)
    final_proba = None
    for member in selected_members:
        member_proba = final_member_proba(final_proba_by_base, member)
        weight = float(member["member_weight"]) / total_weight
        final_proba = weight * member_proba if final_proba is None else final_proba + weight * member_proba
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
            "selected_by": "hand_default_audio_only_hgb_tta_ensemble_oof",
            "selected_score": selected["selection_score"],
            "selected_tta_macro_f1": selected["tta_macro_f1"],
            "selected_tta_contact_macro_f1": selected["tta_contact_macro_f1"],
            "selected_worst_single_view_hybrid": selected["worst_single_view_hybrid"],
            "selected_worst_fold_hybrid_macro_contact": selected["worst_fold_hybrid_macro_contact"],
            "selected_members_json": json.dumps(selected_members),
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
            "protocol": "audio_only_hgb_tta_ensemble_select_no_test_until_lock",
            "feature_set": base.FEATURE_SET,
            "feature_spec": base.FEATURE_SPEC,
            "feature_names": base.FEATURE_NAMES,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifacts": final_artifacts,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_hgb_tta_ensemble_select_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "source_run": str(source_run.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "ensemble_leaderboard": str(ensemble_leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen HGB TTA ensemble selection:")
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
    print("Ensemble leaderboard:", ensemble_leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
