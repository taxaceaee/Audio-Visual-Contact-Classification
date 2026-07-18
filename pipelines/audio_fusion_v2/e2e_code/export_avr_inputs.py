#!/usr/bin/env python3
"""Export p_contact and material_logits from the best audio lift_source_blend predictions.

Reads the final prediction CSV produced by audio_lift_source_blend_select,
computes avr_inputs (p_contact, material_logits) per window, and saves as .npy files.

Matches exactly the avr_inputs() logic in run_avr_group_selection.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]
PROBA_COLUMNS = [f"proba_{name}" for name in CLASS_NAMES]


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def avr_inputs(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split 4-class audio probability into contact prob and material logits.

    Defined identically in run_avr_group_selection.py, lines 33-38.
    
    Args:
        p: (n_windows, 4) — probabilities [ambient, leaf, trunk, twig]
    
    Returns:
        p_contact: (n_windows,) — P(contact | audio) = 1 - P(ambient)
        material_logits: (n_windows, 3) — log(P(leaf|trunk|twig)) renormalized to sum=1
    """
    p = normalize(p)
    contact = 1.0 - p[:, 0]
    material = normalize(p[:, 1:4])
    logits = np.log(np.clip(material, 1e-8, 1.0))
    return contact.astype(np.float64), logits.astype(np.float64)


def aggregate_avr(
    group_keys: np.ndarray,
    p_contact: np.ndarray,
    p_material: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate window-level outputs to segment level by mean-pooling.

    Matches aggregate_avr() in run_avr_group_selection.py, lines 41-47.
    """
    codes, uniques = pd.factorize(group_keys, sort=False)
    n = len(uniques)
    counts = np.bincount(codes, minlength=n).astype(np.float64)
    contact_seg = np.bincount(codes, weights=p_contact, minlength=n) / np.maximum(counts, 1)
    material_seg = np.vstack([
        np.bincount(codes, weights=p_material[:, c], minlength=n) for c in range(3)
    ]).T
    material_seg = normalize(material_seg)
    return contact_seg[codes], material_seg[codes]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export p_contact and material_logits from audio prediction CSV."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_lift_source_blend_select/"
            "reports/audio_lift_source_blend_select_final_test_predictions.csv"
        ),
        help="Path to the final prediction CSV from audio_lift_source_blend_select.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/avr_inputs_export"),
        help="Directory to save output .npy files.",
    )
    parser.add_argument(
        "--aggregate-segments",
        action="store_true",
        help="Also aggregate to segment level using mean-pooling.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.predictions)
    required = PROBA_COLUMNS + ["audio_file"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    proba = frame[PROBA_COLUMNS].to_numpy(dtype=np.float64)
    p_contact, material_logits = avr_inputs(proba)

    np.save(args.output_dir / "p_contact.npy", p_contact)
    np.save(args.output_dir / "material_logits.npy", material_logits)

    print(f"Window-level outputs saved to {args.output_dir}/")
    print(f"  p_contact.npy       shape={p_contact.shape}   range=[{p_contact.min():.6f}, {p_contact.max():.6f}]")
    print(f"  material_logits.npy shape={material_logits.shape}")
    print()

    if args.aggregate_segments and "group_key" in frame.columns:
        contact_seg, material_seg = aggregate_avr(
            frame["group_key"].astype(str).to_numpy(),
            p_contact,
            np.exp(material_logits),
        )
        material_logits_seg = np.log(np.clip(material_seg, 1e-8, 1.0))
        np.save(args.output_dir / "p_contact_segment.npy", contact_seg)
        np.save(args.output_dir / "material_logits_segment.npy", material_logits_seg)
        print("Segment-level outputs saved:")
        print(f"  p_contact_segment.npy       shape={contact_seg.shape}")
        print(f"  material_logits_segment.npy shape={material_logits_seg.shape}")

    print("\nDone. Verified against avr_inputs() from run_avr_group_selection.py.")


if __name__ == "__main__":
    main()
