from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_lift_source_blend_select_final_test as best
import train_audio_specimen_contact_consensus_select_final_test as spec
import train_audio_specimen_contact_lift_select_final_test as lift
import train_val_select_final_test as base


OUTPUT = Path("outputs")
RUN_SLUG = "audio_contribution_ablation_20260707"
OUT_DIR = OUTPUT / "audio_paper_breakdown_20260707" / "contribution_ablation"
LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]


def normalize(proba: np.ndarray) -> np.ndarray:
    return best.normalize(proba)


def metric_row(scope: str, stage: str, description: str, y_true: np.ndarray, proba: np.ndarray, elapsed: float) -> dict:
    pred = proba.argmax(axis=1).astype(np.int64)
    binary_true = (y_true > 0).astype(np.int64)
    binary_pred = (pred > 0).astype(np.int64)
    return {
        "scope": scope,
        "stage": stage,
        "description": description,
        "n": int(len(y_true)),
        "accuracy_4class": float(accuracy_score(y_true, pred)),
        "macro_precision_4class": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "contact_macro_f1": float(f1_score(y_true, pred, labels=CONTACT_LABELS, average="macro", zero_division=0)),
        "binary_accuracy": float(accuracy_score(binary_true, binary_pred)),
        "binary_macro_precision": float(precision_score(binary_true, binary_pred, average="macro", zero_division=0)),
        "binary_macro_recall": float(recall_score(binary_true, binary_pred, average="macro", zero_division=0)),
        "binary_macro_f1": float(f1_score(binary_true, binary_pred, average="macro", zero_division=0)),
        "ambient_f1": float(f1_score(y_true, pred, labels=[0], average="macro", zero_division=0)),
        "leaf_f1": float(f1_score(y_true, pred, labels=[1], average="macro", zero_division=0)),
        "trunk_f1": float(f1_score(y_true, pred, labels=[2], average="macro", zero_division=0)),
        "twig_f1": float(f1_score(y_true, pred, labels=[3], average="macro", zero_division=0)),
        "eval_time_sec": float(elapsed),
    }


def timed_stage(scope: str, stage: str, description: str, y: np.ndarray, fn) -> tuple[dict, np.ndarray]:
    start = time.perf_counter()
    proba = normalize(fn())
    elapsed = time.perf_counter() - start
    return metric_row(scope, stage, description, y, proba, elapsed), proba


def segment_pool(frame: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    segment, window_to_segment = spec.segment_proba_from_window(frame, proba)
    return normalize(segment[window_to_segment])


def consensus_no_lift(frame: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    segment, window_to_segment = spec.segment_proba_from_window(frame, proba)
    specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
    adjusted = lift.consensus_and_lift(
        segment,
        specimen_codes,
        consensus_threshold=0.45,
        min_contact_segments=1,
        lift_min_mass=None,
        lift_floor=0.0,
        lift_confidence=1.0,
    )
    return normalize(adjusted[window_to_segment])


def anchor_lift(frame: pd.DataFrame, highsr: np.ndarray, pairwise: np.ndarray) -> np.ndarray:
    return best.anchor_lift_proba(frame, highsr, pairwise)


def final_best(frame: pd.DataFrame, anchor: np.ndarray, source: np.ndarray) -> np.ndarray:
    blended = normalize(0.95 * anchor + 0.05 * source)
    return best.postprocess(frame, blended, "segment_lift")


def load_train_context() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    base.configure_feature_set("total240")
    root = base.resolve_root(None)
    train_df = base.load_manifest(root / "audio_visual_dataset_default" / "dataset.csv", "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
        force_rebuild=False,
    )
    y = clean_feat["y"].astype(np.int64)
    split_path = OUTPUT / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select" / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)
    sources = broad.load_train_sources(OUTPUT, y, fold_assignment, 42)
    highsr = group_impl.load_highsr_oof(OUTPUT)
    pairwise = spec.load_pairwise_oof(
        Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        )
    )
    sources["highsr_anchor_source"] = highsr
    sources["pairwise_anchor_source"] = pairwise
    return train_df, y, sources


def load_test_context() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    frame, sources = broad.load_final_sources(broad.final_prediction_paths(OUTPUT))
    pair_frame = pd.read_csv(
        OUTPUT
        / "audio_feature_benchmarks"
        / "audio_pairwise_contact_stress_cv_select"
        / "reports"
        / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    if not np.array_equal(pair_frame["audio_file"].astype(str).to_numpy(), frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("Pairwise final source is not aligned with final source frame")
    sources["highsr_anchor_source"] = sources["highsr_default"]
    sources["pairwise_anchor_source"] = normalize(pair_frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))
    return frame, frame["y"].to_numpy(dtype=np.int64), sources


def build_ladder(scope: str, frame: pd.DataFrame, y: np.ndarray, sources: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    highsr = sources["highsr_anchor_source"]
    pairwise = sources["pairwise_anchor_source"]

    row, _ = timed_stage(
        scope,
        "01_highsr_raw",
        "High-SR selected source only, window argmax",
        y,
        lambda: highsr,
    )
    rows.append(row)
    row, _ = timed_stage(
        scope,
        "02_pairwise_raw",
        "Pairwise contact-stress source only, window argmax",
        y,
        lambda: pairwise,
    )
    rows.append(row)
    row, base_blend = timed_stage(
        scope,
        "03_highsr80_pairwise20_raw",
        "Raw window blend: 0.80 High-SR + 0.20 pairwise",
        y,
        lambda: 0.80 * highsr + 0.20 * pairwise,
    )
    rows.append(row)
    row, _ = timed_stage(
        scope,
        "04_segment_pooling",
        "Add segment-level log-probability pooling",
        y,
        lambda: segment_pool(frame, base_blend),
    )
    rows.append(row)
    row, _ = timed_stage(
        scope,
        "05_specimen_consensus_no_lift",
        "Add specimen-level contact subclass consensus without ambient lift",
        y,
        lambda: consensus_no_lift(frame, base_blend),
    )
    rows.append(row)
    row, anchor = timed_stage(
        scope,
        "06_anchor_consensus_lift",
        "Add ambient-to-contact lift; this is the pre-source anchor",
        y,
        lambda: anchor_lift(frame, highsr, pairwise),
    )
    rows.append(row)
    row, _ = timed_stage(
        scope,
        "07_anchor_plus_report_gate_raw",
        "Blend anchor with 0.05 report_gate_onehot, no second segment lift",
        y,
        lambda: normalize(0.95 * anchor + 0.05 * sources["report_gate_onehot"]),
    )
    rows.append(row)
    row, _ = timed_stage(
        scope,
        "08_final_anchor_source_segment_lift",
        "Final best: anchor + 0.05 report_gate_onehot + post-blend segment lift",
        y,
        lambda: final_best(frame, anchor, sources["report_gate_onehot"]),
    )
    rows.append(row)

    table = pd.DataFrame(rows)
    metric_cols = [
        "accuracy_4class",
        "macro_precision_4class",
        "macro_recall_4class",
        "macro_f1_4class",
        "contact_macro_f1",
        "binary_macro_f1",
        "ambient_f1",
        "leaf_f1",
        "trunk_f1",
        "twig_f1",
    ]
    for col in metric_cols:
        table[f"delta_prev_{col}"] = table[col].diff()
        table[f"delta_final_{col}"] = table[col] - float(table.iloc[-1][col])
    return table


def build_source_sweep(scope: str, frame: pd.DataFrame, y: np.ndarray, sources: dict[str, np.ndarray]) -> pd.DataFrame:
    highsr = sources["highsr_anchor_source"]
    pairwise = sources["pairwise_anchor_source"]
    anchor = anchor_lift(frame, highsr, pairwise)
    rows = []
    source_names = [
        "anchor_only",
        "highsr_default",
        "highsr_regularized",
        "highsr_extratrees",
        "mfcc40_grid",
        "report_gate_proba",
        "report_gate_onehot",
        "total120_grid",
        "total240_grid",
        "total240_ensemble",
        "total240_stack_lr",
        "total240_tree_meta",
    ]
    weights = [0.05, 0.10, 0.20, 0.33, 0.50]
    for source_name in source_names:
        source_weights = [0.0] if source_name == "anchor_only" else weights
        for weight in source_weights:
            if source_name == "anchor_only":
                blended = anchor
            else:
                blended = normalize((1.0 - weight) * anchor + weight * sources[source_name])
            for mode in ["raw_argmax", "segment_lift"]:
                start = time.perf_counter()
                proba = best.postprocess(frame, blended, mode)
                elapsed = time.perf_counter() - start
                row = metric_row(
                    scope,
                    f"{source_name}_w{weight:g}_{mode}",
                    f"Anchor blended with {source_name} at weight {weight:g}; mode={mode}",
                    y,
                    proba,
                    elapsed,
                )
                row["source_name"] = source_name
                row["source_weight"] = weight
                row["blend_mode"] = mode
                rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["macro_f1_4class", "contact_macro_f1", "binary_macro_f1"],
        ascending=False,
    )
    best_macro = float(table.iloc[0]["macro_f1_4class"])
    anchor_macro = float(
        table[
            (table["source_name"] == "anchor_only")
            & (table["source_weight"] == 0.0)
            & (table["blend_mode"] == "segment_lift")
        ].iloc[0]["macro_f1_4class"]
    )
    table["delta_best_macro_f1"] = table["macro_f1_4class"] - best_macro
    table["delta_anchor_segment_lift_macro_f1"] = table["macro_f1_4class"] - anchor_macro
    return table


def md_table(df: pd.DataFrame, cols: list[str], rows: int | None = None) -> str:
    data = df[cols].copy()
    if rows is not None:
        data = data.head(rows)
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    return data.to_markdown(index=False)


def write_report(oof_ladder: pd.DataFrame, test_ladder: pd.DataFrame, oof_sweep: pd.DataFrame, test_sweep: pd.DataFrame) -> None:
    cross = build_source_cross_eval(oof_sweep, test_sweep)
    ladder_cross = build_ladder_cross_eval(oof_ladder, test_ladder)
    final_test = test_ladder.iloc[-1]
    final_oof = oof_ladder.iloc[-1]
    key_cols = [
        "stage",
        "accuracy_4class",
        "macro_f1_4class",
        "delta_prev_macro_f1_4class",
        "contact_macro_f1",
        "delta_prev_contact_macro_f1",
        "binary_macro_f1",
        "delta_prev_binary_macro_f1",
        "trunk_f1",
        "twig_f1",
    ]
    source_cols = [
        "source_name",
        "source_weight",
        "blend_mode",
        "accuracy_4class",
        "macro_f1_4class",
        "contact_macro_f1",
        "binary_macro_f1",
        "delta_anchor_segment_lift_macro_f1",
    ]
    cross_cols = [
        "source_name",
        "source_weight",
        "blend_mode",
        "oof_rank",
        "test_rank",
        "oof_macro_f1_4class",
        "test_macro_f1_4class",
        "test_minus_oof_macro_f1",
        "test_contact_macro_f1",
        "test_binary_macro_f1",
    ]
    ladder_cross_cols = [
        "stage",
        "oof_macro_f1_4class",
        "test_macro_f1_4class",
        "test_minus_oof_macro_f1",
        "oof_delta_prev_macro_f1_4class",
        "test_delta_prev_macro_f1_4class",
    ]
    report = f"""# Audio Model Contribution Ablation Report

Generated: 2026-07-07

## Purpose

This report decomposes the current best audio-only model into additive components. OOF results are the selection-safe evidence because they are computed on hand/train out-of-fold predictions. Robot/test results are reported as post-lock diagnostic ablations to explain what each component contributes on the final distribution.

Final best reference:

- OOF macro-F1: `{final_oof["macro_f1_4class"]:.6f}`
- Robot/test accuracy: `{final_test["accuracy_4class"]:.6f}`
- Robot/test macro-F1: `{final_test["macro_f1_4class"]:.6f}`
- Robot/test contact macro-F1: `{final_test["contact_macro_f1"]:.6f}`
- Robot/test binary macro-F1: `{final_test["binary_macro_f1"]:.6f}`

## Sequential Contribution Ladder: OOF

{md_table(oof_ladder, key_cols)}

## Sequential Contribution Ladder: Robot/Test

{md_table(test_ladder, key_cols)}

Main paper reading: the largest robot/test jump comes from the group/specimen machinery around the High-SR + pairwise anchor. The final `report_gate_onehot` source blend is selected safely on OOF, but on robot/test its main value is preserving the best locked configuration rather than producing a large standalone gain over the already-strong anchor.

## OOF vs Robot/Test Component Transfer

{md_table(ladder_cross, ladder_cross_cols)}

## Source/Weight Contribution Sweep: OOF Top 15

{md_table(oof_sweep, source_cols, rows=15)}

## Source/Weight Contribution Sweep: Robot/Test Top 15

{md_table(test_sweep, source_cols, rows=15)}

## Selection-Safety Cross Evaluation

Top OOF-selected source variants and their robot/test diagnostics:

{md_table(cross.sort_values("oof_rank"), cross_cols, rows=12)}

Top robot/test source variants and their OOF ranks:

{md_table(cross.sort_values("test_rank"), cross_cols, rows=12)}

Important: robot/test has a few post-hoc variants above the locked best, for example some High-SR/ExtraTrees blends. They are not the main paper model because the protocol is OOF-selected. They can be mentioned only as diagnostic evidence that source choice is distribution-sensitive and should be validated on a fresh external test before claiming a new best.

## What To Put In The Paper

Use two ablation tables:

1. **Component ablation ladder**: rows 01-08 above, with macro-F1, contact macro-F1, binary macro-F1, and delta from previous row. This explains how each mechanism changes the task.
2. **Selection-safe source contribution**: top rows from OOF, then robot/test diagnostic values for the same rows. This justifies why `report_gate_onehot` at low weight is the locked final recipe.
3. **Diagnostic-only robot/test source sweep**: optional appendix table; do not use it to redefine the best model unless rerun on a fresh unseen split.

Recommended wording:

- The proposed method first builds a High-SR/pairwise contact anchor, then applies segment/specimen consistency and ambient-contact lift.
- OOF selection chooses a small report-gate source blend plus post-blend segment lift.
- Binary ambient/noambient detection is already strong; remaining error is mostly contact subclass separation, especially trunk/twig/leaf.
- Robot/test ablation is diagnostic only; final method selection should be described as OOF-selected.

## Artifacts

- `oof_component_ladder.csv`
- `test_component_ladder.csv`
- `oof_source_weight_sweep.csv`
- `test_source_weight_sweep.csv`
- `source_selection_cross_eval.csv`
- `component_ladder_oof_test_cross_eval.csv`
"""
    (OUT_DIR / "AUDIO_MODEL_CONTRIBUTION_ABLATION_REPORT.md").write_text(report, encoding="utf-8")


def build_source_cross_eval(oof_sweep: pd.DataFrame, test_sweep: pd.DataFrame) -> pd.DataFrame:
    key = ["source_name", "source_weight", "blend_mode"]
    oof = oof_sweep.copy().reset_index(drop=True)
    test = test_sweep.copy().reset_index(drop=True)
    oof["oof_rank"] = np.arange(1, len(oof) + 1)
    test["test_rank"] = np.arange(1, len(test) + 1)
    keep = key + [
        "oof_rank",
        "accuracy_4class",
        "macro_f1_4class",
        "contact_macro_f1",
        "binary_macro_f1",
        "delta_anchor_segment_lift_macro_f1",
    ]
    oof = oof[keep].rename(
        columns={
            "accuracy_4class": "oof_accuracy_4class",
            "macro_f1_4class": "oof_macro_f1_4class",
            "contact_macro_f1": "oof_contact_macro_f1",
            "binary_macro_f1": "oof_binary_macro_f1",
            "delta_anchor_segment_lift_macro_f1": "oof_delta_anchor_segment_lift_macro_f1",
        }
    )
    test = test[
        key
        + [
            "test_rank",
            "accuracy_4class",
            "macro_f1_4class",
            "contact_macro_f1",
            "binary_macro_f1",
            "delta_anchor_segment_lift_macro_f1",
        ]
    ].rename(
        columns={
            "accuracy_4class": "test_accuracy_4class",
            "macro_f1_4class": "test_macro_f1_4class",
            "contact_macro_f1": "test_contact_macro_f1",
            "binary_macro_f1": "test_binary_macro_f1",
            "delta_anchor_segment_lift_macro_f1": "test_delta_anchor_segment_lift_macro_f1",
        }
    )
    merged = oof.merge(test, on=key, how="inner")
    merged["test_minus_oof_macro_f1"] = merged["test_macro_f1_4class"] - merged["oof_macro_f1_4class"]
    return merged


def build_ladder_cross_eval(oof_ladder: pd.DataFrame, test_ladder: pd.DataFrame) -> pd.DataFrame:
    oof = oof_ladder[
        [
            "stage",
            "accuracy_4class",
            "macro_f1_4class",
            "contact_macro_f1",
            "binary_macro_f1",
            "delta_prev_macro_f1_4class",
        ]
    ].rename(
        columns={
            "accuracy_4class": "oof_accuracy_4class",
            "macro_f1_4class": "oof_macro_f1_4class",
            "contact_macro_f1": "oof_contact_macro_f1",
            "binary_macro_f1": "oof_binary_macro_f1",
            "delta_prev_macro_f1_4class": "oof_delta_prev_macro_f1_4class",
        }
    )
    test = test_ladder[
        [
            "stage",
            "accuracy_4class",
            "macro_f1_4class",
            "contact_macro_f1",
            "binary_macro_f1",
            "delta_prev_macro_f1_4class",
        ]
    ].rename(
        columns={
            "accuracy_4class": "test_accuracy_4class",
            "macro_f1_4class": "test_macro_f1_4class",
            "contact_macro_f1": "test_contact_macro_f1",
            "binary_macro_f1": "test_binary_macro_f1",
            "delta_prev_macro_f1_4class": "test_delta_prev_macro_f1_4class",
        }
    )
    merged = oof.merge(test, on="stage", how="inner")
    merged["test_minus_oof_macro_f1"] = merged["test_macro_f1_4class"] - merged["oof_macro_f1_4class"]
    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_frame, y_train, train_sources = load_train_context()
    test_frame, y_test, test_sources = load_test_context()

    oof_ladder = build_ladder("oof_train", train_frame, y_train, train_sources)
    test_ladder = build_ladder("robot_test", test_frame, y_test, test_sources)
    oof_sweep = build_source_sweep("oof_train", train_frame, y_train, train_sources)
    test_sweep = build_source_sweep("robot_test", test_frame, y_test, test_sources)

    oof_ladder.to_csv(OUT_DIR / "oof_component_ladder.csv", index=False)
    test_ladder.to_csv(OUT_DIR / "test_component_ladder.csv", index=False)
    oof_sweep.to_csv(OUT_DIR / "oof_source_weight_sweep.csv", index=False)
    test_sweep.to_csv(OUT_DIR / "test_source_weight_sweep.csv", index=False)
    build_source_cross_eval(oof_sweep, test_sweep).to_csv(OUT_DIR / "source_selection_cross_eval.csv", index=False)
    build_ladder_cross_eval(oof_ladder, test_ladder).to_csv(
        OUT_DIR / "component_ladder_oof_test_cross_eval.csv",
        index=False,
    )
    write_report(oof_ladder, test_ladder, oof_sweep, test_sweep)

    print(f"Wrote {OUT_DIR / 'AUDIO_MODEL_CONTRIBUTION_ABLATION_REPORT.md'}")
    print(f"Wrote CSV artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
