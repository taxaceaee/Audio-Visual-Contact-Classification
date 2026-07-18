from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_locked_oof_blender_select_final_test as locked_blend
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]

FIXED_BINARY_SOURCE = "total240_ensemble"
FIXED_GATE_SOURCE = "total240_tree_meta"
FIXED_PROBABILITY_GAMMA = 2.0
FIXED_CLASS_BIAS = np.asarray([0.0, 0.2, 0.0, 0.2], dtype=np.float64)
GATE_THRESHOLDS = [0.85, 0.90]

CONTACT_BLEND_CANDIDATES = [
    {
        "blend_name": "highsr65_simple",
        "class_weights": {
            "total240_ensemble": 0.25,
            "total240_stack_lr": 0.10,
            "highsr_hgb": 0.65,
        },
    },
    {
        "blend_name": "highsr65_balanced",
        "class_weights": {
            "total240_ensemble": 0.20,
            "total240_stack_lr": 0.15,
            "highsr_hgb": 0.65,
        },
    },
    {
        "blend_name": "highsr70_simple",
        "class_weights": {
            "total240_ensemble": 0.10,
            "total240_stack_lr": 0.20,
            "highsr_hgb": 0.70,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report-grade audio-only gate protocol. This intentionally keeps a tiny, "
            "human-readable OOF selection grid, writes the selection lock, then loads "
            "the robot/test probabilities only for final evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-slug", default="audio_report_grade_gate_select")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def load_train_oof_sources(
    output: Path,
    y: np.ndarray,
    fold_assignment: np.ndarray,
    random_state: int,
) -> dict[str, np.ndarray]:
    return {
        "total240_ensemble": locked_blend.load_total240_ensemble_oof(
            output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select",
            output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_ensemble_select",
        ),
        "total240_stack_lr": locked_blend.load_total240_stack_oof(
            output,
            y,
            fold_assignment,
            random_state,
        ),
        "total240_tree_meta": locked_blend.load_total240_stack_oof_from_run(
            output,
            "audio_oof_stacking_tree_meta_select",
            y,
            fold_assignment,
            random_state,
        ),
        "highsr_hgb": locked_blend.load_highsr_oof(output),
    }


def final_prediction_paths(output: Path) -> dict[str, Path]:
    root = output / "audio_feature_benchmarks"
    return {
        "total240_ensemble": root
        / "audio_tta_grid_hgb_ensemble_select"
        / "reports"
        / "audio_tta_grid_hgb_ensemble_select_final_test_predictions.csv",
        "total240_stack_lr": root
        / "audio_oof_stacking_select"
        / "reports"
        / "audio_oof_stacking_select_final_test_predictions.csv",
        "total240_tree_meta": root
        / "audio_oof_stacking_tree_meta_select"
        / "reports"
        / "audio_oof_stacking_tree_meta_select_final_test_predictions.csv",
        "highsr_hgb": root
        / "audio_highsr_temporal_tta_select"
        / "reports"
        / "audio_highsr_temporal_tta_select_final_test_predictions.csv",
    }


def load_final_proba(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    return locked_blend.normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


def apply_gate_recipe(
    source_proba: dict[str, np.ndarray],
    class_weights: dict[str, float],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    class_proba = locked_blend.blend_sources(source_proba, class_weights)
    base_proba = locked_blend.hierarchical_contact_blend(
        source_proba[FIXED_BINARY_SOURCE],
        class_proba,
    )
    pred = locked_blend.predict_with_bias(
        locked_blend.calibrate_proba(base_proba, FIXED_PROBABILITY_GAMMA),
        FIXED_CLASS_BIAS,
    )
    gate_signal = np.sum(locked_blend.normalize(source_proba[FIXED_GATE_SOURCE])[:, 1:], axis=1)
    contact_pred = 1 + np.argmax(locked_blend.normalize(class_proba)[:, 1:], axis=1)
    force_mask = (pred == 0) & (gate_signal >= threshold)
    pred[force_mask] = contact_pred[force_mask]
    final_proba = locked_blend.calibrate_proba(base_proba, FIXED_PROBABILITY_GAMMA)
    return pred.astype(np.int64), final_proba, int(np.sum(force_mask))


def score_candidate(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    source_proba: dict[str, np.ndarray],
    blend_name: str,
    class_weights: dict[str, float],
    threshold: float,
) -> dict:
    pred, _, forced_count = apply_gate_recipe(source_proba, class_weights, threshold)
    row = {
        "recipe_kind": "report_grade_high_confidence_audio_gate",
        "blend_name": blend_name,
        "binary_source": FIXED_BINARY_SOURCE,
        "gate_source": FIXED_GATE_SOURCE,
        "class_weights_json": json.dumps(class_weights),
        "probability_gamma": FIXED_PROBABILITY_GAMMA,
        "class_bias_json": json.dumps(FIXED_CLASS_BIAS.tolist()),
        "force_contact_threshold": float(threshold),
        "forced_contact_count": forced_count,
    }
    row.update(locked_blend.score_pred(y, pred))
    row.update(locked_blend.fold_scores(y, pred, fold_assignment))
    row["selection_score"] = locked_blend.selection_score(row, "balanced")
    return row


def assert_final_frames_aligned(final_paths: dict[str, Path]) -> pd.DataFrame:
    reference = pd.read_csv(final_paths["total240_stack_lr"])
    reference_y = reference["y"].to_numpy(dtype=np.int64)
    reference_audio = reference["audio_file"].astype(str).to_numpy() if "audio_file" in reference else None
    for name, path in final_paths.items():
        frame = pd.read_csv(path, usecols=lambda column: column in {"audio_file", "y"})
        if not np.array_equal(frame["y"].to_numpy(dtype=np.int64), reference_y):
            raise AssertionError(f"Final prediction labels do not align for {name}: {path}")
        if reference_audio is not None and "audio_file" in frame:
            audio = frame["audio_file"].astype(str).to_numpy()
            if not np.array_equal(audio, reference_audio):
                raise AssertionError(f"Final prediction audio order does not align for {name}: {path}")
    return reference


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
    print("Protocol  = tiny train-only OOF grid, audio-only, no test until lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
        force_rebuild=False,
    )
    y = clean_feat["y"]
    split_path = (
        args.output
        / "audio_feature_benchmarks"
        / "audio_tta_grid_hgb_select"
        / "splits"
        / "hand_train_full_hgb_tta_grid_folds.csv"
    )
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)

    source_proba = load_train_oof_sources(args.output, y, fold_assignment, args.random_state)
    for name, proba in source_proba.items():
        if proba.shape != (len(y), len(LABELS)):
            raise AssertionError(f"{name}: bad OOF shape {proba.shape}")
        print(f"Loaded train OOF source {name}: {proba.shape}", flush=True)

    method_card = {
        "protocol": "report_grade_audio_only_high_confidence_gate",
        "allowed_selection_data": "hand/default train OOF predictions only",
        "forbidden_selection_data": "robot/test labels, robot/test predictions, image features, unsupervised adaptation",
        "candidate_count": len(CONTACT_BLEND_CANDIDATES) * len(GATE_THRESHOLDS),
        "fixed_binary_source": FIXED_BINARY_SOURCE,
        "fixed_gate_source": FIXED_GATE_SOURCE,
        "fixed_probability_gamma": FIXED_PROBABILITY_GAMMA,
        "fixed_class_bias": FIXED_CLASS_BIAS.tolist(),
        "contact_blend_candidates": CONTACT_BLEND_CANDIDATES,
        "gate_threshold_candidates": GATE_THRESHOLDS,
        "selection_score": "0.60*macro_f1 + 0.20*contact_macro_f1 + 0.20*worst_fold_hybrid - 0.50*fold_std_hybrid",
        "source_locks": {
            "total240_ensemble": str(
                (
                    args.output
                    / "audio_feature_benchmarks"
                    / "audio_tta_grid_hgb_ensemble_select"
                    / "reports"
                    / "audio_tta_grid_hgb_ensemble_select_selected_without_test.json"
                ).resolve()
            ),
            "total240_stack_lr": str(
                (
                    args.output
                    / "audio_feature_benchmarks"
                    / "audio_oof_stacking_select"
                    / "reports"
                    / "audio_oof_stacking_select_selected_without_test.json"
                ).resolve()
            ),
            "total240_tree_meta": str(
                (
                    args.output
                    / "audio_feature_benchmarks"
                    / "audio_oof_stacking_tree_meta_select"
                    / "reports"
                    / "audio_oof_stacking_tree_meta_select_selected_without_test.json"
                ).resolve()
            ),
            "highsr_hgb": str(
                (
                    args.output
                    / "audio_feature_benchmarks"
                    / "audio_highsr_temporal_tta_select"
                    / "reports"
                    / "audio_highsr_temporal_tta_select_selected_without_test.json"
                ).resolve()
            ),
        },
        "split_path": str(split_path.resolve()),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)

    rows = []
    for candidate in CONTACT_BLEND_CANDIDATES:
        for threshold in GATE_THRESHOLDS:
            rows.append(
                score_candidate(
                    y,
                    fold_assignment,
                    source_proba,
                    str(candidate["blend_name"]),
                    dict(candidate["class_weights"]),
                    threshold,
                )
            )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "worst_fold_hybrid_macro_contact"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_report_grade_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    selection_summary = {
        "selection_rule": "best of a predeclared 3x2 audio-only OOF grid",
        "selected_without_test": selected,
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_path": str(split_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    final_paths = final_prediction_paths(args.output)
    final_source_proba = {name: load_final_proba(path) for name, path in final_paths.items()}
    test_frame = assert_final_frames_aligned(final_paths)
    y_test = test_frame["y"].to_numpy(dtype=np.int64)

    final_pred, final_proba, forced_final_count = apply_gate_recipe(
        final_source_proba,
        json.loads(str(selected["class_weights_json"])),
        float(selected["force_contact_threshold"]),
    )
    final_row = base.make_report_row(
        model_name="report_grade_high_confidence_audio_gate",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "report_grade_audio_only_oof_grid",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_recipe_kind": selected["recipe_kind"],
            "selected_blend_name": selected["blend_name"],
            "selected_binary_source": selected["binary_source"],
            "selected_gate_source": selected["gate_source"],
            "selected_class_weights_json": selected["class_weights_json"],
            "selected_force_contact_threshold": selected["force_contact_threshold"],
            "selected_probability_gamma": selected["probability_gamma"],
            "selected_class_bias_json": selected["class_bias_json"],
            "forced_contact_count_final": forced_final_count,
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_columns = [
        column
        for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"]
        if column in test_frame
    ]
    prediction_frame = test_frame[prediction_columns].copy()
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
            "protocol": method_card["protocol"],
            "method_card": method_card,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": method_card["protocol"],
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "method_card": method_card,
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "method_card": str(method_card_path.resolve()),
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen report-grade audio protocol:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
                "selected_blend_name",
                "selected_force_contact_threshold",
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
