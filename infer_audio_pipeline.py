#!/usr/bin/env python3
"""Full audio-to-AVR pipeline: reproduces p_contact and material_logits
from the intermediate prediction CSVs using the exact same logic as the
best model (audio_lift_source_blend, macro_f1=0.7027).

Dependencies: only numpy, pandas. No sklearn/torch needed.
Uses precomputed prediction CSVs from the locked pipeline steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]
PROBA_COLUMNS = [f"proba_{CLASS_NAMES[index]}" for index in LABELS]

# --- Fixed parameters from the selected recipe (audio_lift_source_blend) ---
ANCHOR_HIGHSR_WEIGHT = 0.80
ANCHOR_PAIRWEIGHT = 0.20
SOURCE_NAME = "report_gate_onehot"
SOURCE_WEIGHT = 0.05
BLEND_MODE = "segment_lift"
CONSENSUS_THRESHOLD = 0.45
MIN_CONTACT_SEGMENTS = 1
LIFT_MIN_MASS = 0.35
LIFT_FLOOR = 0.58
LIFT_CONFIDENCE = 0.45


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def one_hot(pred: np.ndarray) -> np.ndarray:
    output = np.zeros((len(pred), len(LABELS)), dtype=np.float64)
    output[np.arange(len(pred)), pred.astype(int)] = 1.0
    return output


def segment_proba_from_window(
    frame: pd.DataFrame, window_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """sum_log_proba group decode: identical to specimen_contact_consensus script."""
    group_codes, _ = pd.factorize(frame["group_key"].astype(str), sort=False)
    n_groups = int(group_codes.max()) + 1
    values = np.log(np.clip(window_proba, 1e-12, 1.0))
    sums = np.vstack(
        [
            np.bincount(group_codes, weights=values[:, class_id], minlength=n_groups)
            for class_id in LABELS
        ]
    ).T
    segment_proba = np.exp(sums - sums.max(axis=1, keepdims=True))
    segment_proba = segment_proba / segment_proba.sum(axis=1, keepdims=True)
    return normalize(segment_proba), group_codes.astype(np.int64)


def specimen_codes_for_segments(
    frame: pd.DataFrame, window_to_segment: np.ndarray
) -> np.ndarray:
    """Assign each segment to a specimen using the same regex as cv.specimen_group_key."""
    segment_frame = frame.groupby("group_key", sort=False).first().reset_index()
    specimen = segment_frame["audio_file"].astype(str).str.replace(
        r"_segment_.*$", "", regex=True
    )
    specimen_codes, _ = pd.factorize(specimen, sort=False)
    if int(window_to_segment.max()) + 1 != len(segment_frame):
        raise AssertionError("Segment/frame alignment mismatch")
    return specimen_codes.astype(np.int64)


def consensus_and_lift(
    segment_proba: np.ndarray,
    specimen_codes: np.ndarray,
    consensus_threshold: float = CONSENSUS_THRESHOLD,
    min_contact_segments: int = MIN_CONTACT_SEGMENTS,
    lift_min_mass: float | None = LIFT_MIN_MASS,
    lift_floor: float = LIFT_FLOOR,
    lift_confidence: float = LIFT_CONFIDENCE,
) -> np.ndarray:
    """Identical to consensus_and_lift() in train_audio_specimen_contact_lift_select_final_test.py."""
    output = normalize(segment_proba.copy())
    contact_mass = output[:, 1:4].sum(axis=1)
    contact_dist = normalize(output[:, 1:4])
    for specimen_id in np.unique(specimen_codes):
        group_idx = np.where(specimen_codes == specimen_id)[0]
        contact_idx = group_idx[contact_mass[group_idx] >= consensus_threshold]
        if len(contact_idx) < min_contact_segments:
            continue
        consensus = normalize(
            np.mean(contact_dist[contact_idx], axis=0, keepdims=True)
        )[0]
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


def postprocess(
    frame: pd.DataFrame, proba: np.ndarray, mode: str = BLEND_MODE
) -> np.ndarray:
    """Postprocess with segment_lift, identical to the lift_source_blend script."""
    if mode == "raw_argmax":
        return normalize(proba)
    if mode == "segment_lift":
        segment, window_to_segment = segment_proba_from_window(frame, proba)
        specimen_codes = specimen_codes_for_segments(frame, window_to_segment)
        segment = consensus_and_lift(
            segment, specimen_codes,
            CONSENSUS_THRESHOLD, MIN_CONTACT_SEGMENTS,
            LIFT_MIN_MASS, LIFT_FLOOR, LIFT_CONFIDENCE,
        )
        return normalize(segment[window_to_segment])
    raise KeyError(f"Unknown blend mode: {mode}")


def avr_inputs(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identical to avr_inputs() in run_avr_group_selection.py lines 33-38."""
    p = normalize(p)
    contact = 1.0 - p[:, 0]
    material = normalize(p[:, 1:4])
    logits = np.log(np.clip(material, 1e-8, 1.0))
    return contact.astype(np.float64), logits.astype(np.float64)


def load_proba_from_predictions(path: Path, onehot: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    if onehot and "pred_y" in frame.columns:
        proba = one_hot(frame["pred_y"].to_numpy(dtype=np.int64))
    else:
        proba = normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))
    return frame, proba


def build_pipeline(
    output_root: Path,
    highsr_path: Path,
    pairwise_path: Path,
    report_gate_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the full audio_lift_source_blend pipeline.

    Returns:
        frame: the aligned test DataFrame
        final_proba: (n_windows, 4) final 4-class probability
        p_contact: (n_windows,) P(contact | audio)
        material_logits: (n_windows, 3) log-probabilities for leaf/trunk/twig
    """
    # Step 1: Load High-SR and Pairwise final test predictions
    highsr_frame, highsr_proba = load_proba_from_predictions(highsr_path)
    pair_frame, pair_proba = load_proba_from_predictions(pairwise_path)
    if not np.array_equal(
        highsr_frame["audio_file"].astype(str).to_numpy(),
        pair_frame["audio_file"].astype(str).to_numpy(),
    ):
        raise AssertionError("High-SR and Pairwise frames are not aligned")

    # Step 2: Build anchor = 0.80 * highsr + 0.20 * pairwise
    anchor_window = normalize(ANCHOR_HIGHSR_WEIGHT * highsr_proba + ANCHOR_PAIRWEIGHT * pair_proba)

    # Step 3: Run anchor through segment+specimen consensus+lift
    segment_proba, window_to_segment = segment_proba_from_window(highsr_frame, anchor_window)
    specimen_codes = specimen_codes_for_segments(highsr_frame, window_to_segment)
    segment_proba = consensus_and_lift(
        segment_proba, specimen_codes,
        CONSENSUS_THRESHOLD, MIN_CONTACT_SEGMENTS,
        LIFT_MIN_MASS, LIFT_FLOOR, LIFT_CONFIDENCE,
    )
    anchor_proba = normalize(segment_proba[window_to_segment])

    # Step 4: Load report_gate_onehot source and blend
    gate_frame, gate_proba = load_proba_from_predictions(report_gate_path, onehot=True)
    if not np.array_equal(
        highsr_frame["audio_file"].astype(str).to_numpy(),
        gate_frame["audio_file"].astype(str).to_numpy(),
    ):
        raise AssertionError("Report gate frame not aligned")

    blended = normalize(
        (1.0 - SOURCE_WEIGHT) * anchor_proba + SOURCE_WEIGHT * gate_proba
    )

    # Step 5: Postprocess with segment_lift
    final_proba = postprocess(highsr_frame, blended, BLEND_MODE)

    # Step 6: Compute AVR inputs
    p_contact, material_logits = avr_inputs(final_proba)

    return highsr_frame, final_proba, p_contact, material_logits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce p_contact and material_logits from the full audio pipeline."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/avr_pipeline_export"),
        help="Directory to save output .npy files.",
    )
    args = parser.parse_args()

    root = args.output / "audio_feature_benchmarks"
    highsr_path = (
        root / "audio_highsr_temporal_tta_select" / "reports"
        / "audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pairwise_path = (
        root / "audio_pairwise_contact_stress_cv_select" / "reports"
        / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    report_gate_path = (
        root / "audio_report_grade_gate_select" / "reports"
        / "audio_report_grade_gate_select_final_test_predictions.csv"
    )

    for path in [highsr_path, pairwise_path, report_gate_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction CSV: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, final_proba, p_contact, material_logits = build_pipeline(
        args.output, highsr_path, pairwise_path, report_gate_path
    )

    np.save(args.output_dir / "p_contact.npy", p_contact)
    np.save(args.output_dir / "material_logits.npy", material_logits)
    np.save(args.output_dir / "final_4class_proba.npy", final_proba)

    # Verify against the precomputed final predictions
    final_csv_path = (
        root / "audio_lift_source_blend_select" / "reports"
        / "audio_lift_source_blend_select_final_test_predictions.csv"
    )
    if final_csv_path.exists():
        ref_frame = pd.read_csv(final_csv_path)
        ref_proba = normalize(ref_frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))
        max_diff = float(np.max(np.abs(final_proba - ref_proba)))
        print(f"Max absolute diff vs. final prediction CSV: {max_diff:.10f}")
        if max_diff > 1e-9:
            print("WARNING: pipeline differs from precomputed CSV!")
        else:
            print("PIPELINE MATCH: identical to precomputed final predictions.")

    # Compute forward predictions for verification
    final_pred = final_proba.argmax(axis=1).astype(np.int64)
    y_test = frame["y"].to_numpy(dtype=np.int64)

    def fast_macro_f1(y_true, pred, labels):
        scores = []
        for label in labels:
            tm = y_true == label
            pm = pred == label
            tp = float(np.sum(tm & pm))
            fp = float(np.sum(~tm & pm))
            fn = float(np.sum(tm & ~pm))
            denom = 2.0 * tp + fp + fn
            scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
        return float(np.mean(scores))

    macro = fast_macro_f1(y_test, final_pred, np.asarray([0, 1, 2, 3]))
    contact = fast_macro_f1(y_test, final_pred, np.asarray([1, 2, 3]))
    binary = fast_macro_f1((y_test > 0).astype(np.int64), (final_pred > 0).astype(np.int64), np.asarray([0, 1]))
    acc = float(np.mean(final_pred == y_test))

    print(f"\nRecomputed metrics (should match best model):")
    print(f"  accuracy_4class    = {acc:.6f}")
    print(f"  macro_f1_4class    = {macro:.6f}")
    print(f"  contact_macro_f1   = {contact:.6f}")
    print(f"  binary_macro_f1    = {binary:.6f}")

    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  p_contact.npy         shape={p_contact.shape} min={p_contact.min():.6f} max={p_contact.max():.6f}")
    print(f"  material_logits.npy   shape={material_logits.shape}")
    print(f"  final_4class_proba.npy shape={final_proba.shape}")

    # Summary stats
    pred_classes = final_pred
    for i, name in enumerate(CLASS_NAMES):
        subset = p_contact[pred_classes == i] if i > 0 else 1 - p_contact[pred_classes == 0]
        if len(subset) > 0:
            print(f"\n  Predicted {name}: n={len(subset)} "
                  f"p_contact mean={subset.mean():.4f} std={subset.std():.4f}")


if __name__ == "__main__":
    main()
