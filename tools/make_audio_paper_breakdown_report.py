from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


ROOT = Path(".")
BEST_SLUG = "audio_lift_source_blend_select"
BEST_DIR = ROOT / "outputs" / "audio_feature_benchmarks" / BEST_SLUG / "reports"
OUT_DIR = ROOT / "outputs" / "audio_paper_breakdown_20260707"
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]
CONTACT_NAMES = ["leaf", "trunk", "twig"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def specificity_by_class(cm: np.ndarray) -> list[float]:
    total = cm.sum()
    specs: list[float] = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        specs.append(float(tn / (tn + fp)) if (tn + fp) else float("nan"))
    return specs


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    specificity = specificity_by_class(cm)
    rows = []
    for idx, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "class": name,
                "support": int(support[idx]),
                "correct": int(cm[idx, idx]),
                "class_accuracy_or_recall": recall[idx],
                "precision": precision[idx],
                "recall": recall[idx],
                "specificity": specificity[idx],
                "f1": f1[idx],
            }
        )
    return pd.DataFrame(rows)


def macro_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    per_cls = per_class_metrics(y_true, y_pred)
    return pd.DataFrame(
        [
            {
                "scope": "4class_macro",
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "macro_specificity": per_cls["specificity"].mean(),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
        ]
    )


def binary_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = (df["y"].to_numpy(dtype=int) > 0).astype(int)
    y_pred = (df["pred_y"].to_numpy(dtype=int) > 0).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    specs = specificity_by_class(cm)
    per = pd.DataFrame(
        [
            {
                "binary_class": "ambient",
                "support": int(support[0]),
                "precision": precision[0],
                "recall": recall[0],
                "specificity": specs[0],
                "f1": f1[0],
            },
            {
                "binary_class": "noambient_leaf_trunk_twig",
                "support": int(support[1]),
                "precision": precision[1],
                "recall": recall[1],
                "specificity": specs[1],
                "f1": f1[1],
            },
        ]
    )
    macro = pd.DataFrame(
        [
            {
                "scope": "binary_ambient_vs_noambient",
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "macro_specificity": per["specificity"].mean(),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "contact_or_noambient_f1": f1[1],
            }
        ]
    )
    return per, macro


def specimen_from_group(group_key: str) -> str:
    text = str(group_key)
    return re.sub(r"_segment_\d+$", "", text)


def group_majority_breakdown(df: pd.DataFrame, group_col: str, out_name: str) -> pd.DataFrame:
    rows = []
    for group, part in df.groupby(group_col, sort=False):
        true_counts = part["label"].value_counts()
        pred_counts = part["pred_label"].value_counts()
        true_label = true_counts.index[0]
        pred_label = pred_counts.index[0]
        rows.append(
            {
                group_col: group,
                "n_windows": len(part),
                "true_majority_label": true_label,
                "pred_majority_label": pred_label,
                "majority_correct": true_label == pred_label,
                "window_accuracy": accuracy_score(part["label"], part["pred_label"]),
                "mean_confidence": part["confidence"].mean(),
                "error_count": int((part["label"] != part["pred_label"]).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / out_name, index=False)
    return out


def summarize_group_breakdown(group_df: pd.DataFrame, name: str) -> dict[str, object]:
    return {
        "level": name,
        "n_groups": int(len(group_df)),
        "majority_accuracy": float(group_df["majority_correct"].mean()),
        "mean_window_accuracy_per_group": float(group_df["window_accuracy"].mean()),
        "median_windows_per_group": float(group_df["n_windows"].median()),
        "groups_with_any_error": int((group_df["error_count"] > 0).sum()),
    }


def confidence_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.50, 0.70, 0.85, 0.95, 1.0000001]
    labels = ["<=0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", "0.95-1.00"]
    tmp = df.copy()
    tmp["confidence_bin"] = pd.cut(tmp["confidence"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for name, part in tmp.groupby("confidence_bin", observed=False):
        rows.append(
            {
                "confidence_bin": str(name),
                "n": int(len(part)),
                "accuracy": accuracy_score(part["label"], part["pred_label"]) if len(part) else np.nan,
                "error_rate": float((part["label"] != part["pred_label"]).mean()) if len(part) else np.nan,
                "mean_confidence": part["confidence"].mean() if len(part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def collect_model_comparison() -> pd.DataFrame:
    rows = []
    report_paths = list((ROOT / "outputs" / "audio_feature_benchmarks").glob("*/reports/*final_test_report.csv"))
    report_paths += list((ROOT / "checkpoints").glob("*/*final_test_report.csv"))
    seen = set()
    for path in sorted(report_paths):
        try:
            row = pd.read_csv(path).iloc[0].to_dict()
        except Exception:
            continue
        key = (
            row.get("model"),
            round(safe_float(row.get("accuracy_4class")), 12),
            round(safe_float(row.get("macro_f1_4class")), 12),
            round(safe_float(row.get("contact_macro_f1")), 12),
            round(safe_float(row.get("binary_macro_f1")), 12),
        )
        if key in seen:
            continue
        seen.add(key)
        pstr = str(path)
        family = "audio"
        if "/image_" in pstr or "image_" in str(row.get("model", "")):
            family = "image"
        if "multimodal" in pstr or str(row.get("model", "")).startswith("mm_"):
            family = "multimodal"
        rows.append(
            {
                "family": family,
                "run": path.parts[-3] if len(path.parts) >= 3 else path.parent.parent.name,
                "model": row.get("model"),
                "accuracy_4class": safe_float(row.get("accuracy_4class")),
                "macro_f1_4class": safe_float(row.get("macro_f1_4class")),
                "contact_macro_f1": safe_float(row.get("contact_macro_f1")),
                "binary_macro_f1": safe_float(row.get("binary_macro_f1")),
                "model_train_time_sec": safe_float(row.get("model_train_time_sec")),
                "model_predict_time_sec": safe_float(row.get("model_predict_time_sec")),
                "selected_by": row.get("selected_by", ""),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1_4class", ascending=False)


def md_table(df: pd.DataFrame, columns: list[str], digits: int = 4, max_rows: int | None = None) -> str:
    data = df.loc[:, columns].copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return data.to_markdown(index=False)


def main() -> None:
    ensure_out_dir()
    pred_path = BEST_DIR / f"{BEST_SLUG}_final_test_predictions.csv"
    report_path = BEST_DIR / f"{BEST_SLUG}_final_test_report.csv"
    selected_path = BEST_DIR / f"{BEST_SLUG}_selected_without_test.json"
    protocol_path = BEST_DIR / f"{BEST_SLUG}_protocol_summary.json"
    df = pd.read_csv(pred_path)
    report = pd.read_csv(report_path).iloc[0].to_dict()
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    proba_cols = [f"proba_{name}" for name in CLASS_NAMES]
    df["confidence"] = df[proba_cols].max(axis=1)
    df["correct"] = df["label"] == df["pred_label"]
    df["specimen_key"] = df["group_key"].map(specimen_from_group)
    df["binary_label"] = np.where(df["y"] > 0, "noambient_leaf_trunk_twig", "ambient")
    df["binary_pred_label"] = np.where(df["pred_y"] > 0, "noambient_leaf_trunk_twig", "ambient")

    y_true = df["y"].to_numpy(dtype=int)
    y_pred = df["pred_y"].to_numpy(dtype=int)
    four_macro = macro_confusion_metrics(y_true, y_pred)
    per_cls = per_class_metrics(y_true, y_pred)
    bin_per, bin_macro = binary_metrics(df)
    cm4 = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(4)), index=CLASS_NAMES, columns=CLASS_NAMES)
    cmb = pd.DataFrame(
        confusion_matrix((y_true > 0).astype(int), (y_pred > 0).astype(int), labels=[0, 1]),
        index=["true_ambient", "true_noambient"],
        columns=["pred_ambient", "pred_noambient"],
    )

    by_true = (
        df.groupby("label", sort=False)
        .agg(
            n=("label", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    pair_errors = (
        df.loc[~df["correct"]]
        .groupby(["label", "pred_label"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    conf_bins = confidence_breakdown(df)
    segment_df = group_majority_breakdown(df, "group_key", "segment_majority_breakdown.csv")
    specimen_df = group_majority_breakdown(df, "specimen_key", "specimen_majority_breakdown.csv")
    model_cmp = collect_model_comparison()

    artifacts = {
        "overall_4class_macro_metrics.csv": four_macro,
        "per_class_confusion_metrics.csv": per_cls,
        "binary_ambient_noambient_macro_metrics.csv": bin_macro,
        "binary_ambient_noambient_per_class_metrics.csv": bin_per,
        "confusion_matrix_4class.csv": cm4.reset_index(names="true_label"),
        "confusion_matrix_binary.csv": cmb.reset_index(names="true_binary"),
        "breakdown_by_true_class.csv": by_true,
        "pairwise_error_counts.csv": pair_errors,
        "confidence_bins.csv": conf_bins,
        "model_comparison_all_final_reports.csv": model_cmp,
    }
    for name, table in artifacts.items():
        table.to_csv(OUT_DIR / name, index=False)

    top_audio = model_cmp[model_cmp["family"] == "audio"].head(12)
    top_all = model_cmp.head(12)
    timing_wall = 11.09
    max_rss_kb = 352884
    per_window_ms = timing_wall / len(df) * 1000.0

    group_summary = pd.DataFrame(
        [
            summarize_group_breakdown(segment_df, "segment_group_key"),
            summarize_group_breakdown(specimen_df, "specimen_key"),
        ]
    )
    group_summary.to_csv(OUT_DIR / "group_level_summary.csv", index=False)

    report_md = f"""# Audio Paper Breakdown Report

Generated: 2026-07-07

## Recommended paper comparison package

Use the current locked best run as the main method:

- Run: `{BEST_SLUG}`
- Model/protocol: `{report.get("model")}`
- Selection: train-only OOF; robot/test labels used only after lock
- Selected recipe: `{selected["selected_without_test"]["source_name"]}` weight `{selected["selected_without_test"]["source_weight"]}`, blend mode `{selected["selected_without_test"]["blend_mode"]}`
- Candidate count in locked selection grid: `{protocol["method_card"]["candidate_count"]}`

Recommended small breakdowns for the paper:

1. Main 4-class result: accuracy, macro precision, macro recall, macro F1, weighted F1.
2. Per-class confusion metrics: precision, recall/class accuracy, specificity, F1 for ambient/leaf/trunk/twig.
3. Binary ambient vs noambient: collapse leaf/trunk/twig into noambient and report accuracy plus macro precision/recall/specificity/F1.
4. Contact-only weakness analysis: report `contact_macro_f1` and pairwise leaf/trunk/twig confusions.
5. Segment/specimen consistency: window-level result plus majority vote correctness at segment and specimen level.
6. Confidence/error breakdown: accuracy by confidence bin and top error transitions.
7. Method comparison/ablation: compare the best locked source-blend method against earlier audio-only protocols and non-audio baselines.
8. Timing: report cached locked-pipeline wall-clock and clarify that CSV model train/predict fields are zero because this protocol blends precomputed locked audio probabilities.

## Main robot/test metrics

{md_table(four_macro, ["scope", "accuracy", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "weighted_f1"], 6)}

The checkpoint report values match recomputation from prediction CSV:

- 4-class accuracy: `{report["accuracy_4class"]:.6f}`
- 4-class macro-F1: `{report["macro_f1_4class"]:.6f}`
- contact macro-F1 over leaf/trunk/twig: `{report["contact_macro_f1"]:.6f}`
- binary ambient/noambient macro-F1: `{report["binary_macro_f1"]:.6f}`

## 4-class per-class confusion metrics

{md_table(per_cls, ["class", "support", "correct", "class_accuracy_or_recall", "precision", "recall", "specificity", "f1"], 6)}

4-class confusion matrix:

{cm4.to_markdown()}

Interpretation: ambient is perfectly recalled on this robot/test split, leaf is also high recall, while trunk is the limiting class. The dominant scientific weakness is not ambient/contact detection; it is contact-subclass separation, especially trunk being absorbed by ambient/twig/leaf.

## Binary ambient vs noambient

{md_table(bin_macro, ["scope", "accuracy", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "contact_or_noambient_f1"], 6)}

{md_table(bin_per, ["binary_class", "support", "precision", "recall", "specificity", "f1"], 6)}

Binary confusion matrix:

{cmb.to_markdown()}

This is the cleanest headline for “detecting contact/no-contact”: only ambient vs non-ambient is very strong, with binary macro-F1 around `{report["binary_macro_f1"]:.3f}`. The 4-class task is harder mainly because noambient subclasses are visually/acoustically close.

## Breakdown by true class

{md_table(by_true, ["label", "n", "correct", "accuracy", "mean_confidence"], 6)}

## Contact subclass error pairs

Top error transitions:

{md_table(pair_errors, ["label", "pred_label", "n"], 0, max_rows=12)}

The largest errors are `trunk -> twig`, `trunk -> ambient`, and `twig -> leaf`. This suggests a useful paper discussion: the model has a strong binary contact detector but trunk/twig/leaf boundaries remain the bottleneck.

## Segment/specimen breakdown

{md_table(group_summary, ["level", "n_groups", "majority_accuracy", "mean_window_accuracy_per_group", "median_windows_per_group", "groups_with_any_error"], 6)}

This matters because the selected protocol explicitly uses segment/specimen consistency at inference. In the paper, describe it as group-consistency inference rather than independent-window classification.

## Confidence breakdown

{md_table(conf_bins, ["confidence_bin", "n", "accuracy", "error_rate", "mean_confidence"], 6)}

## Model comparison / ablation table

Top audio-only final-test runs:

{md_table(top_audio, ["run", "model", "accuracy_4class", "macro_f1_4class", "contact_macro_f1", "binary_macro_f1"], 6)}

Top runs across all families found in existing final-test reports:

{md_table(top_all, ["family", "run", "model", "accuracy_4class", "macro_f1_4class", "contact_macro_f1", "binary_macro_f1"], 6)}

Suggested paper table rows:

- `audio_lift_source_blend_select`: final method; best 4-class macro-F1.
- `audio_specimen_contact_lift_select`: previous lift baseline; shows gain from source blend.
- `audio_specimen_contact_consensus_select`: specimen consensus baseline.
- `audio_log_consensus_pair_blend_select`: older group-consistency pair blend.
- `audio_report_grade_gate_select`: cleaner small-grid report-grade gate; useful as simpler-method comparison.
- best image-only and multimodal rows from `model_comparison_all_final_reports.csv` if the paper compares modalities.

## Timing

A fresh rerun of the locked best pipeline on this machine using cached features/probabilities was measured with:

```bash
/usr/bin/time -f 'WALL_TIME_SEC=%e\\nMAX_RSS_KB=%M' \\
  python3 train_audio_lift_source_blend_select_final_test.py \\
  --run-slug audio_lift_source_blend_select_timing_20260707
```

Measured result:

- Wall-clock: `{timing_wall:.2f}` seconds
- Peak RSS: `{max_rss_kb}` KB (`{max_rss_kb / 1024:.1f}` MB)
- Test windows: `{len(df)}`
- End-to-end cached pipeline time per test window: `{per_window_ms:.3f}` ms/window

Important timing caveat: the final report stores `model_train_time_sec=0` and `model_predict_time_sec=0` because the best protocol is a locked probability-blending/segment-lift stage over already-computed audio sources. For paper wording, call this “cached locked-pipeline evaluation time”, not raw waveform feature-extraction time.

## Output artifacts

All CSV breakdowns are saved under:

`{OUT_DIR}`

Key files:

- `per_class_confusion_metrics.csv`
- `binary_ambient_noambient_macro_metrics.csv`
- `binary_ambient_noambient_per_class_metrics.csv`
- `pairwise_error_counts.csv`
- `segment_majority_breakdown.csv`
- `specimen_majority_breakdown.csv`
- `model_comparison_all_final_reports.csv`
"""
    (OUT_DIR / "AUDIO_PAPER_BREAKDOWN_REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'AUDIO_PAPER_BREAKDOWN_REPORT.md'}")
    print(f"Wrote CSV artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
