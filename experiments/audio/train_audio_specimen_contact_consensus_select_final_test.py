from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_highsr_temporal_tta_select_final_test as highsr
import train_cv_select_final_test as cv
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
HIGHSR_WEIGHT_CANDIDATES = [0.65, 0.70, 0.75, 0.80, 0.85]
CONTACT_THRESHOLDS = [0.45, 0.55, 0.65, 0.75]
CONSENSUS_ALPHA = [0.50, 0.75, 1.00]
MIN_CONTACT_SEGMENTS = [1, 2, 3]
AGG_RULES = ["mean_contact_dist", "sum_log_contact_dist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only specimen contact-consensus postprocessor. Selection uses "
            "hand/default OOF only, writes a lock, then evaluates once on robot/test."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_specimen_contact_consensus_select")
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
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    exp_scores = np.exp(shifted)
    return exp_scores / float(np.sum(exp_scores))


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray = LABELS) -> float:
    scores = []
    for label in labels:
        true_mask = y_true == label
        pred_mask = pred == label
        tp = float(np.sum(true_mask & pred_mask))
        fp = float(np.sum(~true_mask & pred_mask))
        fn = float(np.sum(true_mask & ~pred_mask))
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {base.ID2LABEL[int(label)]: int(np.sum(y == label)) for label in LABELS}


def segment_proba_from_window(frame: pd.DataFrame, window_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    group_codes, _ = pd.factorize(frame["group_key"].astype(str), sort=False)
    n_groups = int(group_codes.max()) + 1
    values = np.log(np.clip(window_proba, 1e-12, 1.0))
    sums = np.vstack(
        [np.bincount(group_codes, weights=values[:, class_id], minlength=n_groups) for class_id in LABELS]
    ).T
    segment_proba = np.exp(sums - sums.max(axis=1, keepdims=True))
    segment_proba = segment_proba / segment_proba.sum(axis=1, keepdims=True)
    return normalize(segment_proba), group_codes.astype(np.int64)


def specimen_codes_for_segments(frame: pd.DataFrame, window_to_segment: np.ndarray) -> np.ndarray:
    segment_frame = frame.groupby("group_key", sort=False).first().reset_index()
    specimen = segment_frame["audio_file"].map(cv.specimen_group_key).astype(str)
    specimen_codes, _ = pd.factorize(specimen, sort=False)
    if int(window_to_segment.max()) + 1 != len(segment_frame):
        raise AssertionError("Segment/frame alignment mismatch")
    return specimen_codes.astype(np.int64)


def apply_specimen_contact_consensus(
    segment_proba: np.ndarray,
    specimen_codes: np.ndarray,
    threshold: float | None,
    alpha: float,
    min_contact_segments: int,
    agg_rule: str,
) -> np.ndarray:
    if threshold is None:
        return normalize(segment_proba)
    output = normalize(segment_proba.copy())
    contact_mass = output[:, 1:4].sum(axis=1)
    contact_dist = normalize(output[:, 1:4])
    for specimen_id in np.unique(specimen_codes):
        group_idx = np.where(specimen_codes == specimen_id)[0]
        contact_idx = group_idx[contact_mass[group_idx] >= threshold]
        if len(contact_idx) < min_contact_segments:
            continue
        if agg_rule == "mean_contact_dist":
            consensus = normalize(np.mean(contact_dist[contact_idx], axis=0, keepdims=True))[0]
        elif agg_rule == "sum_log_contact_dist":
            consensus = softmax(np.sum(np.log(np.clip(contact_dist[contact_idx], 1e-12, 1.0)), axis=0))
        else:
            raise KeyError(f"Unknown agg_rule: {agg_rule}")
        blended_contact = normalize((1.0 - alpha) * contact_dist[group_idx] + alpha * consensus.reshape(1, -1))
        output[group_idx, 1:4] = contact_mass[group_idx, None] * blended_contact
        output[group_idx, 0] = 1.0 - contact_mass[group_idx]
    return normalize(output)


def window_pred_from_segment(segment_proba: np.ndarray, window_to_segment: np.ndarray) -> np.ndarray:
    return segment_proba[window_to_segment].argmax(axis=1).astype(np.int64)


def load_pairwise_oof(cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return normalize(np.load(cache_path))
    fallback = Path(
        "outputs/audio_feature_benchmarks/audio_group_consistency_pair_blend_select/"
        "oof_sources/pairwise_selected_clean_oof_proba.npy"
    )
    if fallback.exists():
        return normalize(np.load(fallback))
    raise FileNotFoundError(f"Missing pairwise OOF cache: {cache_path}")


def load_final_frame_and_proba(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


def evaluate_candidates(
    frame: pd.DataFrame,
    y: np.ndarray,
    highsr_oof: np.ndarray,
    pairwise_oof: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    segment_cache = {}
    for highsr_weight in HIGHSR_WEIGHT_CANDIDATES:
        pairwise_weight = 1.0 - highsr_weight
        blended = normalize(highsr_weight * highsr_oof + pairwise_weight * pairwise_oof)
        base_segment_proba, window_to_segment = segment_proba_from_window(frame, blended)
        specimen_codes = specimen_codes_for_segments(frame, window_to_segment)
        base_name = f"h{int(round(highsr_weight * 100)):02d}_p{int(round(pairwise_weight * 100)):02d}"
        segment_cache[base_name] = {
            "base_segment_proba": base_segment_proba,
            "window_to_segment": window_to_segment,
            "specimen_codes": specimen_codes,
        }
        candidate_specs = [
            {
                "specimen_rule": "none",
                "contact_threshold": None,
                "consensus_alpha": 0.0,
                "min_contact_segments": 0,
                "agg_rule": "none",
            }
        ]
        for threshold in CONTACT_THRESHOLDS:
            for alpha in CONSENSUS_ALPHA:
                for min_segments in MIN_CONTACT_SEGMENTS:
                    for agg_rule in AGG_RULES:
                        candidate_specs.append(
                            {
                                "specimen_rule": "contact_subclass_consensus",
                                "contact_threshold": threshold,
                                "consensus_alpha": alpha,
                                "min_contact_segments": min_segments,
                                "agg_rule": agg_rule,
                            }
                        )
        for spec in candidate_specs:
            segment_proba = apply_specimen_contact_consensus(
                base_segment_proba,
                specimen_codes,
                spec["contact_threshold"],
                spec["consensus_alpha"],
                int(spec["min_contact_segments"]),
                str(spec["agg_rule"]),
            )
            pred = window_pred_from_segment(segment_proba, window_to_segment)
            macro = fast_macro_f1(y, pred, LABELS)
            contact = fast_macro_f1(y, pred, CONTACT_LABELS)
            binary = fast_macro_f1((y > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
            rows.append(
                {
                    "recipe_kind": "audio_specimen_contact_consensus",
                    "base_name": base_name,
                    "highsr_weight": float(highsr_weight),
                    "pairwise_weight": float(pairwise_weight),
                    "segment_rule": "sum_log_proba",
                    "macro_f1": macro,
                    "contact_macro_f1": contact,
                    "binary_macro_f1": binary,
                    "selection_score": float(0.60 * macro + 0.30 * contact + 0.10 * binary),
                    **spec,
                }
            )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    return leaderboard, segment_cache


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
    print("Protocol  = audio-only specimen contact consensus; robot/test after selection lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    y = train_df["y"].to_numpy(dtype=np.int64)
    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = load_pairwise_oof(args.pairwise_oof_cache)
    if len(highsr_oof) != len(train_df) or len(pairwise_oof) != len(train_df):
        raise AssertionError("OOF source length mismatch")

    leaderboard, _ = evaluate_candidates(train_df, y, highsr_oof, pairwise_oof)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()

    method_card = {
        "protocol": "audio_only_specimen_contact_consensus_no_test_until_lock",
        "allowed_selection_data": (
            "hand/default audio-only OOF probabilities, hand labels, segment group equality, "
            "specimen group equality"
        ),
        "forbidden_selection_data": (
            "robot/test labels, robot/test predictions before selection lock, image/multimodal features, "
            "parsing class words from filenames"
        ),
        "note": (
            "Specimen ids are used only as equality groups for audio prediction consistency; "
            "the script never parses leaf/trunk/twig/ambient/contact strings as labels."
        ),
        "checkpoint_before_this_run": "checkpoints/audio_log_consensus_pair_blend_0653825",
        "candidate_count": int(len(leaderboard)),
        "label_counts": label_counts(y),
        "train_segments": int(train_df["group_key"].nunique()),
        "train_specimens": int(train_df["audio_file"].map(cv.specimen_group_key).nunique()),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact score over limited audio specimen-consensus grid",
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
    for column in ["audio_file"]:
        if not np.array_equal(highsr_frame[column].astype(str).to_numpy(), pair_frame[column].astype(str).to_numpy()):
            raise AssertionError(f"Final source frames are not aligned on {column}")

    final_blend = normalize(float(selected["highsr_weight"]) * highsr_final + float(selected["pairwise_weight"]) * pair_final)
    final_segment, final_window_to_segment = segment_proba_from_window(highsr_frame, final_blend)
    final_specimen_codes = specimen_codes_for_segments(highsr_frame, final_window_to_segment)
    final_segment = apply_specimen_contact_consensus(
        final_segment,
        final_specimen_codes,
        selected["contact_threshold"],
        float(selected["consensus_alpha"]),
        int(selected["min_contact_segments"]),
        str(selected["agg_rule"]),
    )
    final_pred = window_pred_from_segment(final_segment, final_window_to_segment)
    y_test = highsr_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_specimen_contact_consensus",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_specimen_contact_consensus",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_highsr_weight": selected["highsr_weight"],
            "selected_pairwise_weight": selected["pairwise_weight"],
            "selected_specimen_rule": selected["specimen_rule"],
            "selected_contact_threshold": selected["contact_threshold"],
            "selected_consensus_alpha": selected["consensus_alpha"],
            "selected_min_contact_segments": selected["min_contact_segments"],
            "selected_agg_rule": selected["agg_rule"],
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

    print("\nFinal robot/test result after frozen specimen-contact consensus:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_specimen_rule",
                "selected_highsr_weight",
                "selected_pairwise_weight",
                "selected_contact_threshold",
                "selected_consensus_alpha",
                "selected_min_contact_segments",
                "selected_agg_rule",
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
