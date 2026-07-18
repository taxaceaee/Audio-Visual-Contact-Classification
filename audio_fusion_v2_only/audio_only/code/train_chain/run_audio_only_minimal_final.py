#!/usr/bin/env python3
"""Minimal audio-only paper recipe (frozen) — robot/test evaluation.

Architecture (3 sources only — selected path of audio-only SOTA):

  anchor  = specimen_lift( segment( 0.8 * highsr + 0.2 * pairwise ) )
  final   = specimen_lift( segment( 0.95 * anchor + 0.05 * report_gate_onehot ) )
  pred    = argmax(final)

This is the locked recipe from:
  checkpoints/audio_only_paper_safe_current_0702672_20260706
  (source=report_gate_onehot, weight=0.05, mode=segment_lift)

No image, no new HP search on robot. Loads precomputed final prediction CSVs
from hand-trained upstream sources (Mode A frozen replay).

Usage:
  python run_audio_only_minimal_final.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import train_audio_lift_source_blend_select_final_test as lift
import train_val_select_final_test as base

ROOT = Path("outputs/audio_feature_benchmarks")
OUT = Path("outputs/audio_only_minimal")
LABELS = np.array([0, 1, 2, 3], dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]

# Locked recipe (hand-selected, paper-safe)
W_HIGHSR = 0.80
W_PAIR = 0.20
W_GATE = 0.05  # report_gate_onehot weight
BLEND_MODE = "segment_lift"

PATHS = {
    "highsr": ROOT
    / "audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv",
    "pairwise": ROOT
    / "audio_pairwise_contact_stress_cv_select/reports/audio_pairwise_contact_stress_cv_select_final_test_predictions.csv",
    "report_gate": ROOT
    / "audio_report_grade_gate_select/reports/audio_report_grade_gate_select_final_test_predictions.csv",
    "paper_ref": ROOT
    / "audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv",
}


def load_frame(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    cols = [f"proba_{base.ID2LABEL[i]}" for i in LABELS]
    if not all(c in df.columns for c in cols):
        raise KeyError(f"missing proba cols in {path}: {df.columns.tolist()}")
    P = lift.normalize(df[cols].to_numpy(dtype=np.float64))
    return df, P


def onehot_from_pred(pred: np.ndarray, n_class: int = 4) -> np.ndarray:
    oh = np.zeros((len(pred), n_class), dtype=np.float64)
    oh[np.arange(len(pred)), pred.astype(int)] = 1.0
    return oh


def metrics_bundle(y: np.ndarray, pred: np.ndarray) -> dict:
    by = (y > 0).astype(int)
    bp = (pred > 0).astype(int)
    cm = confusion_matrix(y, pred, labels=LABELS)
    return {
        "n": int(len(y)),
        "metrics": {
            "accuracy_4class": float(accuracy_score(y, pred)),
            "macro_precision_4class": float(
                precision_score(y, pred, average="macro", zero_division=0)
            ),
            "macro_recall_4class": float(
                recall_score(y, pred, average="macro", zero_division=0)
            ),
            "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1_4class": float(
                f1_score(y, pred, average="weighted", zero_division=0)
            ),
            "binary_macro_f1": float(f1_score(by, bp, average="macro", zero_division=0)),
            "contact_macro_f1": float(
                f1_score(y, pred, labels=[1, 2, 3], average="macro", zero_division=0)
            ),
        },
        "per_class_4class": classification_report(
            y,
            pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_4class": cm.tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for k, p in PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"missing required artifact: {p}")

    high_df, highsr = load_frame(PATHS["highsr"])
    pair_df, pairwise = load_frame(PATHS["pairwise"])
    gate_df, gate_soft = load_frame(PATHS["report_gate"])
    ref_df, _ = load_frame(PATHS["paper_ref"])

    # alignment invariants
    files = ref_df["audio_file"].astype(str).to_numpy()
    y = ref_df["y"].to_numpy(dtype=np.int64)
    for name, df in [("highsr", high_df), ("pairwise", pair_df), ("report_gate", gate_df)]:
        if not np.array_equal(df["audio_file"].astype(str).to_numpy(), files):
            raise AssertionError(f"{name} audio_file order mismatch vs paper_ref")
        if not np.array_equal(df["y"].to_numpy(), y):
            raise AssertionError(f"{name} y mismatch vs paper_ref")

    # group_key for segment/specimen lift
    frame = ref_df.copy()
    if "group_key" not in frame.columns:
        frame["group_key"] = (
            frame["audio_file"]
            .astype(str)
            .str.replace(r"_window_\d+.*$", "", regex=True)
        )

    # report_gate onehot (paper selected source)
    if "pred_y" in gate_df.columns:
        gate_pred = gate_df["pred_y"].to_numpy(dtype=np.int64)
    else:
        gate_pred = gate_soft.argmax(1).astype(np.int64)
    gate_onehot = onehot_from_pred(gate_pred)

    # --- minimal recipe ---
    anchor = lift.anchor_lift_proba(frame, highsr, pairwise)
    blended = lift.normalize((1.0 - W_GATE) * anchor + W_GATE * gate_onehot)
    final_proba = lift.postprocess(frame, blended, BLEND_MODE)
    pred = final_proba.argmax(1).astype(np.int64)

    # paper reference preds
    paper_pred = ref_df["pred_y"].to_numpy(dtype=np.int64)
    match_rate = float(np.mean(pred == paper_pred))

    result = {
        "protocol": "audio_only_minimal_3source_frozen_recipe",
        "recipe": {
            "anchor": f"{W_HIGHSR}*highsr + {W_PAIR}*pairwise → specimen_lift",
            "blend": f"{1 - W_GATE}*anchor + {W_GATE}*report_gate_onehot",
            "postprocess": BLEND_MODE,
            "sources": ["highsr_default", "pairwise_contact", "report_gate_onehot"],
        },
        "invariants": {
            "image_features": False,
            "robot_labels_in_selection": False,
            "filename_class_features": False,
            "hp_search_on_robot": False,
            "frozen_replay_of_paper_safe_recipe": True,
            "match_paper_pred_rate": match_rate,
        },
        "split": "robot_test_final",
        **metrics_bundle(y, pred),
    }

    # save
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    pd.DataFrame(
        result["confusion_matrix_4class"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(OUT / "confusion_matrix.csv")
    pred_out = frame[
        [c for c in ["audio_file", "label", "y", "group_key"] if c in frame.columns]
    ].copy()
    pred_out["pred_y"] = pred
    pred_out["pred_label"] = pred_out["pred_y"].map(base.ID2LABEL)
    for i, name in enumerate(CLASS_NAMES):
        pred_out[f"proba_{name}"] = final_proba[:, i]
    pred_out.to_csv(OUT / "predictions.csv", index=False)

    # human report
    cm = np.array(result["confusion_matrix_4class"])
    m = result["metrics"]
    lines = [
        "# Audio-only minimal pipeline — robot/test results",
        "",
        "## Recipe (frozen, 3 sources)",
        "",
        "```text",
        f"anchor = specimen_lift(segment({W_HIGHSR}*highsr + {W_PAIR}*pairwise))",
        f"final  = specimen_lift(segment({1-W_GATE}*anchor + {W_GATE}*report_gate_onehot))",
        "pred   = argmax(final)",
        "```",
        "",
        f"- Match paper-safe preds: **{match_rate:.4f}**",
        f"- n = {result['n']}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| **Macro F1** | **{m['macro_f1_4class']:.6f}** |",
        f"| Accuracy | {m['accuracy_4class']:.6f} |",
        f"| Macro precision | {m['macro_precision_4class']:.6f} |",
        f"| Macro recall | {m['macro_recall_4class']:.6f} |",
        f"| Weighted F1 | {m['weighted_f1_4class']:.6f} |",
        f"| Contact macro F1 | {m['contact_macro_f1']:.6f} |",
        f"| Binary macro F1 | {m['binary_macro_f1']:.6f} |",
        "",
        "## Per-class",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CLASS_NAMES:
        r = result["per_class_4class"][c]
        lines.append(
            f"| {c} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "## Confusion matrix (rows=true, cols=pred)",
        "",
        "| true \\ pred | " + " | ".join(CLASS_NAMES) + " |",
        "|---|" + "---:|" * 4,
    ]
    for i, c in enumerate(CLASS_NAMES):
        cells = " | ".join(str(int(cm[i, j])) for j in range(4))
        lines.append(f"| **{c}** | {cells} |")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `{OUT / 'metrics.json'}`",
        f"- `{OUT / 'confusion_matrix.csv'}`",
        f"- `{OUT / 'predictions.csv'}`",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))

    print(json.dumps(result, indent=2, default=float))
    print("\n" + "\n".join(lines[:40]))
    print(f"\nSaved → {OUT.resolve()}")


if __name__ == "__main__":
    main()
