from __future__ import annotations

import argparse
import itertools
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
CONTACT_THRESHOLDS = [0.45, 0.65]
CONSENSUS_ALPHA = [0.75, 1.00]
MIN_CONTACT_SEGMENTS = [1, 2]
AGG_RULES = ["mean_contact_dist", "sum_log_contact_dist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only specimen contact-consensus plus train-only class-bias "
            "selection. Robot/test is loaded only after the lock is written."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_specimen_contact_consensus_bias_select")
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


def bias_candidates() -> list[np.ndarray]:
    ambient = [-0.4, -0.2, 0.0, 0.2]
    leaf = [-0.4, 0.0, 0.4]
    trunk = [0.0, 0.4, 0.8, 1.2]
    twig = [-0.4, 0.0, 0.4]
    candidates = [np.asarray(values, dtype=np.float64) for values in itertools.product(ambient, leaf, trunk, twig)]
    candidates.append(np.zeros(4, dtype=np.float64))
    return candidates


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    scores = np.log(np.clip(proba, 1e-12, 1.0)) + bias.reshape(1, -1)
    return scores.argmax(axis=1).astype(np.int64)


def class_f1(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = []
    for label in LABELS:
        true_mask = y_true == label
        pred_mask = pred == label
        tp = float(np.sum(true_mask & pred_mask))
        fp = float(np.sum(~true_mask & pred_mask))
        fn = float(np.sum(true_mask & ~pred_mask))
        denom = 2.0 * tp + fp + fn
        out.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return np.asarray(out, dtype=np.float64)


def score_prediction(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    f1 = class_f1(y_true, pred)
    binary = spec.fast_macro_f1((y_true > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
    return {
        "macro_f1": float(np.mean(f1)),
        "contact_macro_f1": float(np.mean(f1[1:4])),
        "min_contact_f1": float(np.min(f1[1:4])),
        "ambient_f1": float(f1[0]),
        "leaf_f1": float(f1[1]),
        "trunk_f1": float(f1[2]),
        "twig_f1": float(f1[3]),
        "binary_macro_f1": binary,
    }


def selection_score(row: dict[str, float]) -> float:
    return float(
        0.55 * row["macro_f1"]
        + 0.25 * row["contact_macro_f1"]
        + 0.10 * row["min_contact_f1"]
        + 0.10 * row["binary_macro_f1"]
    )


def candidate_specs() -> list[dict]:
    output = []
    for highsr_weight in HIGHSR_WEIGHT_CANDIDATES:
        pairwise_weight = 1.0 - highsr_weight
        output.append(
            {
                "base_name": f"h{int(round(highsr_weight * 100)):02d}_p{int(round(pairwise_weight * 100)):02d}",
                "highsr_weight": highsr_weight,
                "pairwise_weight": pairwise_weight,
                "specimen_rule": "none",
                "contact_threshold": None,
                "consensus_alpha": 0.0,
                "min_contact_segments": 0,
                "agg_rule": "none",
            }
        )
        for threshold in CONTACT_THRESHOLDS:
            for alpha in CONSENSUS_ALPHA:
                for min_segments in MIN_CONTACT_SEGMENTS:
                    for agg_rule in AGG_RULES:
                        output.append(
                            {
                                "base_name": f"h{int(round(highsr_weight * 100)):02d}_p{int(round(pairwise_weight * 100)):02d}",
                                "highsr_weight": highsr_weight,
                                "pairwise_weight": pairwise_weight,
                                "specimen_rule": "contact_subclass_consensus",
                                "contact_threshold": threshold,
                                "consensus_alpha": alpha,
                                "min_contact_segments": min_segments,
                                "agg_rule": agg_rule,
                            }
                        )
    return output


def evaluate_candidates(
    frame: pd.DataFrame,
    y: np.ndarray,
    highsr_oof: np.ndarray,
    pairwise_oof: np.ndarray,
) -> pd.DataFrame:
    bias_grid = bias_candidates()
    segment_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for candidate in candidate_specs():
        base_name = str(candidate["base_name"])
        if base_name not in segment_cache:
            highsr_weight = float(candidate["highsr_weight"])
            pairwise_weight = float(candidate["pairwise_weight"])
            blended = normalize(highsr_weight * highsr_oof + pairwise_weight * pairwise_oof)
            segment_proba, window_to_segment = spec.segment_proba_from_window(frame, blended)
            specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
            segment_cache[base_name] = (segment_proba, window_to_segment, specimen_codes)
        base_segment, window_to_segment, specimen_codes = segment_cache[base_name]
        segment_proba = spec.apply_specimen_contact_consensus(
            base_segment,
            specimen_codes,
            candidate["contact_threshold"],
            float(candidate["consensus_alpha"]),
            int(candidate["min_contact_segments"]),
            str(candidate["agg_rule"]),
        )
        window_proba = segment_proba[window_to_segment]
        for bias in bias_grid:
            pred = predict_with_bias(window_proba, bias)
            scores = score_prediction(y, pred)
            rows.append(
                {
                    "recipe_kind": "audio_specimen_contact_consensus_bias",
                    "segment_rule": "sum_log_proba",
                    "class_bias_json": json.dumps(bias.tolist()),
                    "selection_score": selection_score(scores),
                    **candidate,
                    **scores,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "trunk_f1"],
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
    print("Protocol  = audio-only specimen contact consensus + bias; robot/test after lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    y = train_df["y"].to_numpy(dtype=np.int64)
    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = spec.load_pairwise_oof(args.pairwise_oof_cache)
    leaderboard = evaluate_candidates(train_df, y, highsr_oof, pairwise_oof)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)

    method_card = {
        "protocol": "audio_only_specimen_contact_consensus_bias_no_test_until_lock",
        "allowed_selection_data": "hand/default audio-only OOF probabilities, hand labels, group equality metadata",
        "forbidden_selection_data": "robot/test labels, robot/test predictions before lock, image/multimodal features, parsing labels from filenames",
        "checkpoint_before_this_run": "checkpoints/audio_specimen_contact_consensus_0690682",
        "candidate_count": int(len(leaderboard)),
        "bias_candidates": len(bias_candidates()),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact/min-contact/binary score over consensus+bias grid",
        "selected_without_test": selected,
        "selected_bias": selected_bias.tolist(),
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
    final_blend = normalize(float(selected["highsr_weight"]) * highsr_final + float(selected["pairwise_weight"]) * pair_final)
    final_segment, final_window_to_segment = spec.segment_proba_from_window(highsr_frame, final_blend)
    final_specimen_codes = spec.specimen_codes_for_segments(highsr_frame, final_window_to_segment)
    final_segment = spec.apply_specimen_contact_consensus(
        final_segment,
        final_specimen_codes,
        selected["contact_threshold"],
        float(selected["consensus_alpha"]),
        int(selected["min_contact_segments"]),
        str(selected["agg_rule"]),
    )
    final_window_proba = final_segment[final_window_to_segment]
    final_pred = predict_with_bias(final_window_proba, selected_bias)
    y_test = highsr_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_specimen_contact_consensus_bias",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_specimen_contact_consensus_bias",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_oof_min_contact_f1": selected["min_contact_f1"],
            "selected_highsr_weight": selected["highsr_weight"],
            "selected_pairwise_weight": selected["pairwise_weight"],
            "selected_specimen_rule": selected["specimen_rule"],
            "selected_contact_threshold": selected["contact_threshold"],
            "selected_consensus_alpha": selected["consensus_alpha"],
            "selected_min_contact_segments": selected["min_contact_segments"],
            "selected_agg_rule": selected["agg_rule"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
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
    print("\nFinal robot/test result after frozen specimen-contact consensus+bias:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_oof_min_contact_f1",
                "selected_highsr_weight",
                "selected_pairwise_weight",
                "selected_bias_json",
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
