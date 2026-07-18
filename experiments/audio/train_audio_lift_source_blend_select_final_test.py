from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_specimen_contact_lift_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as spec
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
SOURCE_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50]
BLEND_MODES = ["raw_argmax", "segment_lift"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only anchor blend. Uses current best specimen contact-lift as "
            "anchor, blends small weights from locked audio-only sources, selects "
            "by train OOF only, then evaluates robot/test once."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_lift_source_blend_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--pairwise-oof-cache",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    return spec.normalize(proba)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray = LABELS) -> float:
    return spec.fast_macro_f1(y_true, pred, labels)


def anchor_lift_proba(frame: pd.DataFrame, highsr_proba: np.ndarray, pairwise_proba: np.ndarray) -> np.ndarray:
    window = normalize(0.80 * highsr_proba + 0.20 * pairwise_proba)
    segment, window_to_segment = spec.segment_proba_from_window(frame, window)
    specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
    segment = lift.consensus_and_lift(
        segment,
        specimen_codes,
        consensus_threshold=0.45,
        min_contact_segments=1,
        lift_min_mass=0.35,
        lift_floor=0.58,
        lift_confidence=0.45,
    )
    return normalize(segment[window_to_segment])


def postprocess(frame: pd.DataFrame, proba: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw_argmax":
        return normalize(proba)
    if mode == "segment_lift":
        segment, window_to_segment = spec.segment_proba_from_window(frame, proba)
        specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
        segment = lift.consensus_and_lift(
            segment,
            specimen_codes,
            consensus_threshold=0.45,
            min_contact_segments=1,
            lift_min_mass=0.35,
            lift_floor=0.58,
            lift_confidence=0.45,
        )
        return normalize(segment[window_to_segment])
    raise KeyError(f"Unknown blend mode: {mode}")


def evaluate_candidates(
    frame: pd.DataFrame,
    y: np.ndarray,
    anchor: np.ndarray,
    sources: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    source_items: list[tuple[str, np.ndarray | None]] = [("anchor_only", None)]
    source_items.extend(sorted(sources.items()))
    for source_name, source_proba in source_items:
        for source_weight in SOURCE_WEIGHTS:
            if source_name == "anchor_only" and source_weight > 0:
                continue
            if source_name != "anchor_only" and source_weight <= 0:
                continue
            blended = anchor if source_proba is None else normalize((1.0 - source_weight) * anchor + source_weight * source_proba)
            for mode in BLEND_MODES:
                final_proba = postprocess(frame, blended, mode)
                pred = final_proba.argmax(axis=1).astype(np.int64)
                macro = fast_macro_f1(y, pred, LABELS)
                contact = fast_macro_f1(y, pred, CONTACT_LABELS)
                binary = fast_macro_f1((y > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
                rows.append(
                    {
                        "recipe_kind": "audio_lift_source_blend",
                        "source_name": source_name,
                        "source_weight": float(source_weight),
                        "blend_mode": mode,
                        "macro_f1": macro,
                        "contact_macro_f1": contact,
                        "binary_macro_f1": binary,
                        "selection_score": float(0.60 * macro + 0.30 * contact + 0.10 * binary),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "binary_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    base.configure_feature_set("total240")
    root_path = base.resolve_root(args.root)
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve())
    print("RUN_DIR   =", run_dir.resolve())
    print("Protocol  = audio-only lift anchor/source blend; robot/test after lock")

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
    train_sources = broad.load_train_sources(args.output, y, fold_assignment, args.random_state)
    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = spec.load_pairwise_oof(args.pairwise_oof_cache)
    anchor = anchor_lift_proba(train_df, highsr_oof, pairwise_oof)
    leaderboard = evaluate_candidates(train_df, y, anchor, train_sources)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()

    method_card = {
        "protocol": "audio_only_lift_source_blend_no_test_until_lock",
        "allowed_selection_data": "hand/default train labels and locked audio-only OOF probabilities",
        "forbidden_selection_data": "robot/test labels, robot/test predictions before lock, image/multimodal features",
        "checkpoint_before_this_run": "checkpoints/audio_specimen_contact_lift_0693839",
        "source_names": sorted(train_sources),
        "candidate_count": int(len(leaderboard)),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact score over lift-anchor source blends",
        "selected_without_test": selected,
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_frame, final_sources = broad.load_final_sources(broad.final_prediction_paths(args.output))
    final_root = args.output / "audio_feature_benchmarks"
    highsr_frame = pd.read_csv(
        final_root
        / "audio_highsr_temporal_tta_select"
        / "reports"
        / "audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pair_frame = pd.read_csv(
        final_root
        / "audio_pairwise_contact_stress_cv_select"
        / "reports"
        / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    highsr_final = normalize(highsr_frame[[f"proba_{base.ID2LABEL[index]}" for index in LABELS]].to_numpy(dtype=np.float64))
    pair_final = normalize(pair_frame[[f"proba_{base.ID2LABEL[index]}" for index in LABELS]].to_numpy(dtype=np.float64))
    if not np.array_equal(highsr_frame["audio_file"].astype(str).to_numpy(), test_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("High-SR final frame is not aligned with source frame")
    final_anchor = anchor_lift_proba(test_frame, highsr_final, pair_final)
    source_name = str(selected["source_name"])
    if source_name == "anchor_only":
        final_blended = final_anchor
    else:
        final_blended = normalize((1.0 - float(selected["source_weight"])) * final_anchor + float(selected["source_weight"]) * final_sources[source_name])
    final_proba = postprocess(test_frame, final_blended, str(selected["blend_mode"]))
    final_pred = final_proba.argmax(axis=1).astype(np.int64)
    y_test = test_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_lift_source_blend",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_lift_source_blend",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_source_name": selected["source_name"],
            "selected_source_weight": selected["source_weight"],
            "selected_blend_mode": selected["blend_mode"],
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_frame[[column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_frame]].copy()
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
    print("\nFinal robot/test result after frozen lift-source blend:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_source_name",
                "selected_source_weight",
                "selected_blend_mode",
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
