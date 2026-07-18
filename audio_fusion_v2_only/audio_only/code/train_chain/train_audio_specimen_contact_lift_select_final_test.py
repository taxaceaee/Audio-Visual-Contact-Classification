from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_specimen_contact_consensus_select_final_test as spec
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
HIGHSR_WEIGHT_CANDIDATES = [0.75, 0.80, 0.85]
CONSENSUS_THRESHOLDS = [0.45, 0.55, 0.65]
MIN_CONTACT_SEGMENTS = [1, 2]
LIFT_MIN_MASS = [0.15, 0.25, 0.35, 0.45]
LIFT_FLOORS = [0.52, 0.58, 0.65]
LIFT_CONFIDENCE = [0.45, 0.55, 0.65, 0.75]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only specimen contact consensus with train-only contact-mass lift. "
            "Selection uses OOF only; robot/test is loaded after the lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_specimen_contact_lift_select")
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


def consensus_and_lift(
    segment_proba: np.ndarray,
    specimen_codes: np.ndarray,
    consensus_threshold: float,
    min_contact_segments: int,
    lift_min_mass: float | None,
    lift_floor: float,
    lift_confidence: float,
) -> np.ndarray:
    output = normalize(segment_proba.copy())
    contact_mass = output[:, 1:4].sum(axis=1)
    contact_dist = normalize(output[:, 1:4])
    for specimen_id in np.unique(specimen_codes):
        group_idx = np.where(specimen_codes == specimen_id)[0]
        contact_idx = group_idx[contact_mass[group_idx] >= consensus_threshold]
        if len(contact_idx) < min_contact_segments:
            continue
        consensus = normalize(np.mean(contact_dist[contact_idx], axis=0, keepdims=True))[0]

        # First enforce subclass consistency while preserving each segment's
        # own ambient/contact mass.
        output[group_idx, 1:4] = contact_mass[group_idx, None] * consensus.reshape(1, -1)
        output[group_idx, 0] = 1.0 - contact_mass[group_idx]

        if lift_min_mass is None or float(np.max(consensus)) < lift_confidence:
            continue
        pred_before = output[group_idx].argmax(axis=1)
        lift_mask = (pred_before == 0) & (contact_mass[group_idx] >= lift_min_mass)
        lift_idx = group_idx[lift_mask]
        if len(lift_idx) == 0:
            continue
        lifted_mass = np.maximum(contact_mass[lift_idx], lift_floor)
        lifted_mass = np.clip(lifted_mass, 1e-12, 0.98)
        output[lift_idx, 0] = 1.0 - lifted_mass
        output[lift_idx, 1:4] = lifted_mass[:, None] * consensus.reshape(1, -1)
    return normalize(output)


def candidate_specs() -> list[dict]:
    specs = []
    for highsr_weight in HIGHSR_WEIGHT_CANDIDATES:
        pairwise_weight = 1.0 - highsr_weight
        for threshold in CONSENSUS_THRESHOLDS:
            for min_segments in MIN_CONTACT_SEGMENTS:
                specs.append(
                    {
                        "highsr_weight": highsr_weight,
                        "pairwise_weight": pairwise_weight,
                        "consensus_threshold": threshold,
                        "min_contact_segments": min_segments,
                        "lift_rule": "none",
                        "lift_min_mass": None,
                        "lift_floor": 0.0,
                        "lift_confidence": 1.0,
                    }
                )
                for lift_min in LIFT_MIN_MASS:
                    for lift_floor in LIFT_FLOORS:
                        for lift_conf in LIFT_CONFIDENCE:
                            specs.append(
                                {
                                    "highsr_weight": highsr_weight,
                                    "pairwise_weight": pairwise_weight,
                                    "consensus_threshold": threshold,
                                    "min_contact_segments": min_segments,
                                    "lift_rule": "ambient_contact_mass_floor",
                                    "lift_min_mass": lift_min,
                                    "lift_floor": lift_floor,
                                    "lift_confidence": lift_conf,
                                }
                            )
    return specs


def evaluate_candidates(
    frame: pd.DataFrame,
    y: np.ndarray,
    highsr_oof: np.ndarray,
    pairwise_oof: np.ndarray,
) -> pd.DataFrame:
    rows = []
    base_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cand in candidate_specs():
        base_name = f"h{int(round(cand['highsr_weight'] * 100)):02d}_p{int(round(cand['pairwise_weight'] * 100)):02d}"
        if base_name not in base_cache:
            base_window = normalize(float(cand["highsr_weight"]) * highsr_oof + float(cand["pairwise_weight"]) * pairwise_oof)
            segment_proba, window_to_segment = spec.segment_proba_from_window(frame, base_window)
            specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
            base_cache[base_name] = (segment_proba, window_to_segment, specimen_codes)
        segment_proba, window_to_segment, specimen_codes = base_cache[base_name]
        adjusted = consensus_and_lift(
            segment_proba,
            specimen_codes,
            float(cand["consensus_threshold"]),
            int(cand["min_contact_segments"]),
            cand["lift_min_mass"],
            float(cand["lift_floor"]),
            float(cand["lift_confidence"]),
        )
        pred = adjusted[window_to_segment].argmax(axis=1).astype(np.int64)
        macro = fast_macro_f1(y, pred, LABELS)
        contact = fast_macro_f1(y, pred, CONTACT_LABELS)
        binary = fast_macro_f1((y > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
        rows.append(
            {
                "recipe_kind": "audio_specimen_contact_lift",
                "base_name": base_name,
                "segment_rule": "sum_log_proba",
                "specimen_rule": "contact_subclass_consensus",
                "agg_rule": "mean_contact_dist",
                "consensus_alpha": 1.0,
                "macro_f1": macro,
                "contact_macro_f1": contact,
                "binary_macro_f1": binary,
                "selection_score": float(0.60 * macro + 0.30 * contact + 0.10 * binary),
                **cand,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "binary_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)


def load_final_frame_and_proba(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


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
    print("Protocol  = audio-only specimen contact lift; robot/test after lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    y = train_df["y"].to_numpy(dtype=np.int64)
    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = spec.load_pairwise_oof(args.pairwise_oof_cache)
    leaderboard = evaluate_candidates(train_df, y, highsr_oof, pairwise_oof)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()

    method_card = {
        "protocol": "audio_only_specimen_contact_lift_no_test_until_lock",
        "allowed_selection_data": "hand/default audio-only OOF probabilities, hand labels, group equality metadata",
        "forbidden_selection_data": "robot/test labels, robot/test predictions before lock, image/multimodal features, parsing labels from filenames",
        "checkpoint_before_this_run": "checkpoints/audio_specimen_contact_consensus_0690682",
        "candidate_count": int(len(leaderboard)),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact score over specimen consensus + contact lift grid",
        "selected_without_test": selected,
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    final_root = args.output / "audio_feature_benchmarks"
    highsr_frame, highsr_final = load_final_frame_and_proba(
        final_root
        / "audio_highsr_temporal_tta_select"
        / "reports"
        / "audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pair_frame, pair_final = load_final_frame_and_proba(
        final_root
        / "audio_pairwise_contact_stress_cv_select"
        / "reports"
        / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    if not np.array_equal(highsr_frame["audio_file"].astype(str).to_numpy(), pair_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("Final source frames are not aligned")
    final_window = normalize(float(selected["highsr_weight"]) * highsr_final + float(selected["pairwise_weight"]) * pair_final)
    final_segment, final_window_to_segment = spec.segment_proba_from_window(highsr_frame, final_window)
    final_specimen_codes = spec.specimen_codes_for_segments(highsr_frame, final_window_to_segment)
    final_segment = consensus_and_lift(
        final_segment,
        final_specimen_codes,
        float(selected["consensus_threshold"]),
        int(selected["min_contact_segments"]),
        selected["lift_min_mass"],
        float(selected["lift_floor"]),
        float(selected["lift_confidence"]),
    )
    final_pred = final_segment[final_window_to_segment].argmax(axis=1).astype(np.int64)
    y_test = highsr_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_specimen_contact_lift",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_specimen_contact_lift",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_highsr_weight": selected["highsr_weight"],
            "selected_pairwise_weight": selected["pairwise_weight"],
            "selected_consensus_threshold": selected["consensus_threshold"],
            "selected_min_contact_segments": selected["min_contact_segments"],
            "selected_lift_rule": selected["lift_rule"],
            "selected_lift_min_mass": selected["lift_min_mass"],
            "selected_lift_floor": selected["lift_floor"],
            "selected_lift_confidence": selected["lift_confidence"],
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = highsr_frame[
        [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in highsr_frame]
    ].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    final_window_proba = final_segment[final_window_to_segment]
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_window_proba[:, class_id]
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
    print("\nFinal robot/test result after frozen specimen contact lift:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_lift_rule",
                "selected_lift_min_mass",
                "selected_lift_floor",
                "selected_lift_confidence",
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
