from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_highsr_temporal_tta_select_final_test as highsr
import train_audio_group_consistency_pair_blend_select_final_test as group_blend
import train_audio_lift_source_blend_select_final_test as best
import train_audio_oof_stacking_select_final_test as stack
import train_audio_pairwise_contact_stress_cv_select_final_test as pairwise
import train_audio_report_grade_gate_select_final_test as report_gate
import train_audio_tta_grid_ensemble_select_final_test as grid_ensemble
import train_audio_tta_grid_hgb_select_final_test as grid
import train_val_select_final_test as base


SAMPLE_RATE = 16000
RUN_SLUG = "audio_lift_source_blend_all16_select"
OUTPUT_ROOT = Path("outputs/audio_sample_rate_ablation/all16")
FEATURE_SETS = ("total240", "total120", "mfcc40")
HIGHSR_RUNS = {
    "highsr_default": ("audio_highsr_temporal_tta_select", "highsr_hgb_default__all_aug"),
    "highsr_regularized": ("audio_highsr_temporal_hgb_regularized_select", "highsr_hgb_regularized__all_aug"),
    "highsr_extratrees": ("audio_highsr_temporal_extratrees_select", "highsr_extratrees__all_aug"),
}
LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample-rate ablation of the current best audio lift/source blend. "
            "This variant forces regenerated audio branches to 16 kHz and writes "
            "all artifacts to a new directory, leaving the original best run untouched."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-upstream", action="store_true", help="Use existing ablation upstream artifacts if present.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def configure_sample_rate(sample_rate: int) -> None:
    base.CONFIG["audio"]["sr"] = sample_rate
    base.CONFIG["audio"]["target_len"] = sample_rate
    highsr.TARGET_SR = sample_rate
    highsr.TARGET_LEN = sample_rate
    highsr.FEATURE_FAMILY = f"audio_highsr_temporal_texture_v1_sr{sample_rate}"


def call_main(module: object, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [getattr(module, "__file__", "module")] + argv
        module.main()
    finally:
        sys.argv = old_argv


def grid_run_name(feature_set: str) -> str:
    return "audio_tta_grid_hgb_select" if feature_set == "total240" else f"audio_{feature_set}_tta_grid_hgb_select"


def highsr_shared_feature_cache_dir(args: argparse.Namespace) -> Path:
    legacy = args.output / "audio_feature_benchmarks" / f"audio_highsr_temporal_tta_select_all{16 if SAMPLE_RATE == 16000 else 44100}" / "features"
    if legacy.exists() and not args.force_rebuild:
        return legacy
    default_run_name = HIGHSR_RUNS["highsr_default"][0]
    return args.output / "audio_feature_benchmarks" / default_run_name / "features"


def feature_cache_root(args: argparse.Namespace, feature_set: str) -> Path:
    return args.output / "audio_feature_benchmarks" / f"{feature_set}_sr{SAMPLE_RATE}_caches"


def total240_cache_root(args: argparse.Namespace) -> Path:
    return feature_cache_root(args, "total240")


def ensure_grid_source(args: argparse.Namespace, feature_set: str) -> None:
    run_name = grid_run_name(feature_set)
    report = args.output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    if report.exists() and not args.force_rebuild:
        print(f"Reuse {feature_set} grid source: {report}", flush=True)
        return
    cache_root = feature_cache_root(args, feature_set)
    argv = [
        "--output",
        str(args.output),
        "--random-state",
        str(args.random_state),
        "--n-folds",
        str(args.n_folds),
        "--feature-set",
        feature_set,
        "--clean-feature-cache-dir",
        str(cache_root / "features"),
        "--train-stress-feature-dir",
        str(cache_root / "train_stress_features"),
        "--test-stress-feature-dir",
        str(cache_root / "test_stress_features"),
    ]
    if args.root is not None:
        argv.extend(["--root", str(args.root)])
    call_main(grid, argv)


def ensure_highsr_source(args: argparse.Namespace, run_name: str, candidate_name: str) -> None:
    report = args.output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    if report.exists() and not args.force_rebuild:
        print(f"Reuse high-SR source: {report}", flush=True)
        return
    argv = [
        "--output",
        str(args.output),
        "--random-state",
        str(args.random_state),
        "--n-folds",
        str(args.n_folds),
        "--n-jobs",
        str(args.n_jobs),
        "--run-slug",
        run_name,
        "--feature-cache-dir",
        str(highsr_shared_feature_cache_dir(args)),
        "--candidate-names",
        candidate_name,
    ]
    if args.force_rebuild:
        argv.append("--force-rebuild")
    if args.root is not None:
        argv.extend(["--root", str(args.root)])
    call_main(highsr, argv)


def ensure_total240_ensemble(args: argparse.Namespace) -> None:
    run_name = "audio_tta_grid_hgb_ensemble_select"
    report = args.output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    if report.exists() and not args.force_rebuild:
        print(f"Reuse total240 ensemble source: {report}", flush=True)
        return
    cache_root = total240_cache_root(args)
    argv = [
        "--output",
        str(args.output),
        "--random-state",
        str(args.random_state),
        "--feature-set",
        "total240",
        "--source-run",
        str(args.output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select"),
        "--clean-feature-cache-dir",
        str(cache_root / "features"),
        "--train-stress-feature-dir",
        str(cache_root / "train_stress_features"),
        "--test-stress-feature-dir",
        str(cache_root / "test_stress_features"),
    ]
    if args.root is not None:
        argv.extend(["--root", str(args.root)])
    call_main(grid_ensemble, argv)


def ensure_stack_source(args: argparse.Namespace, run_name: str) -> None:
    report = args.output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    if report.exists() and not args.force_rebuild:
        print(f"Reuse stack source {run_name}: {report}", flush=True)
        return
    cache_root = total240_cache_root(args)
    argv = [
        "--output",
        str(args.output),
        "--random-state",
        str(args.random_state),
        "--run-slug",
        run_name,
        "--source-run",
        str(args.output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select"),
        "--clean-feature-cache-dir",
        str(cache_root / "features"),
        "--train-stress-feature-dir",
        str(cache_root / "train_stress_features"),
        "--test-stress-feature-dir",
        str(cache_root / "test_stress_features"),
    ]
    if args.root is not None:
        argv.extend(["--root", str(args.root)])
    call_main(stack, argv)


def ensure_pairwise_source(args: argparse.Namespace) -> None:
    run_name = "audio_pairwise_contact_stress_cv_select"
    report = args.output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    if report.exists() and not args.force_rebuild:
        print(f"Reuse pairwise source: {report}", flush=True)
        return
    cache_root = total240_cache_root(args)
    argv = [
        "--output",
        str(args.output),
        "--random-state",
        str(args.random_state),
        "--n-folds",
        str(args.n_folds),
        "--clean-feature-cache-dir",
        str(cache_root / "features"),
        "--stress-feature-dir",
        str(cache_root / "train_stress_features"),
    ]
    if args.root is not None:
        argv.extend(["--root", str(args.root)])
    call_main(pairwise, argv)


def ensure_group_pairwise_oof(args: argparse.Namespace) -> Path:
    run_name = "audio_group_consistency_pair_blend_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_name
    cache = run_dir / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"
    report = run_dir / "reports" / f"{run_name}_final_test_predictions.csv"
    if cache.exists() and report.exists() and not args.force_rebuild:
        print(f"Reuse group pairwise OOF: {cache}", flush=True)
        return cache
    cache_root = total240_cache_root(args)
    argv = [
        "--output",
        str(args.output),
        "--random-state",
        str(args.random_state),
        "--run-slug",
        run_name,
        "--clean-feature-cache-dir",
        str(cache_root / "features"),
        "--stress-feature-dir",
        str(cache_root / "train_stress_features"),
    ]
    if args.force_rebuild:
        argv.append("--force-rebuild-pairwise-oof")
    if args.root is not None:
        argv.extend(["--root", str(args.root)])
    call_main(group_blend, argv)
    return cache


def ensure_report_gate(args: argparse.Namespace) -> None:
    run_name = "audio_report_grade_gate_select"
    report = args.output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    if report.exists() and not args.force_rebuild:
        print(f"Reuse report gate source: {report}", flush=True)
        return
    argv = [
        "--output",
        str(args.output),
        "--random-state",
        str(args.random_state),
        "--run-slug",
        run_name,
    ]
    if args.root is not None:
        argv.extend(["--root", str(args.root)])
    call_main(report_gate, argv)


def ensure_full_upstream(args: argparse.Namespace) -> Path:
    for feature_set in FEATURE_SETS:
        ensure_grid_source(args, feature_set)
    ensure_total240_ensemble(args)
    ensure_stack_source(args, "audio_oof_stacking_select")
    ensure_stack_source(args, "audio_oof_stacking_tree_meta_select")
    for run_name, candidate_name in HIGHSR_RUNS.values():
        ensure_highsr_source(args, run_name, candidate_name)
    ensure_pairwise_source(args)
    pairwise_oof_cache = ensure_group_pairwise_oof(args)
    ensure_report_gate(args)
    return pairwise_oof_cache


def load_grid_oof(output: Path, feature_set: str) -> np.ndarray:
    run_name = grid_run_name(feature_set)
    run = output / "audio_feature_benchmarks" / run_name
    lock = json.loads((run / "reports" / f"{run_name}_selected_without_test.json").read_text(encoding="utf-8"))
    candidate = str(lock["selected_base_candidate"])
    weights = lock["selected_tta_weights"]
    oof_by_view = {
        view: np.load(run / "oof_proba" / candidate / f"{view}_oof_proba.npy")
        for view in grid.STRESS_VIEWS
    }
    return best.normalize(grid.weighted_by_recipe(oof_by_view, weights))


def load_highsr_oof(output: Path, run_name: str) -> np.ndarray:
    run = output / "audio_feature_benchmarks" / run_name
    lock = json.loads((run / "reports" / f"{run_name}_selected_without_test.json").read_text(encoding="utf-8"))
    candidate = str(lock["selected_candidate"])
    weights = lock["selected_tta_weights"]
    oof_by_view = {
        view: np.load(run / "oof_proba" / candidate / f"{view}_oof_proba.npy")
        for view in highsr.STRESS_VIEWS
    }
    return best.normalize(highsr.weighted_proba(oof_by_view, weights))


def load_final_proba(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, best.normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


def load_grid_final(output: Path, feature_set: str) -> tuple[pd.DataFrame, np.ndarray]:
    run_name = grid_run_name(feature_set)
    return load_final_proba(
        output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    )


def load_highsr_final(output: Path, run_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    return load_final_proba(
        output / "audio_feature_benchmarks" / run_name / "reports" / f"{run_name}_final_test_predictions.csv"
    )


def evaluate_candidates(frame: pd.DataFrame, y: np.ndarray, anchor: np.ndarray, sources: dict[str, np.ndarray]) -> pd.DataFrame:
    return best.evaluate_candidates(frame, y, anchor, sources)


def run_experiment(args: argparse.Namespace, sample_rate: int = SAMPLE_RATE, run_slug: str = RUN_SLUG) -> Path:
    configure_sample_rate(sample_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    pairwise_oof_cache = (
        args.output
        / "audio_feature_benchmarks"
        / "audio_group_consistency_pair_blend_select"
        / "oof_sources"
        / "pairwise_selected_clean_oof_proba.npy"
    )
    if not args.skip_upstream:
        pairwise_oof_cache = ensure_full_upstream(args)

    root_path = base.resolve_root(args.root)
    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    base.configure_feature_set("total240")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.output / "audio_feature_benchmarks" / f"total240_sr{sample_rate}_caches" / "features",
        force_rebuild=False,
    )
    y = clean_feat["y"]

    split_path = args.output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select" / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)
    train_sources = best.broad.load_train_sources(args.output, y, fold_assignment, args.random_state)
    highsr_oof = best.group_impl.load_highsr_oof(args.output)
    pairwise_oof = best.spec.load_pairwise_oof(pairwise_oof_cache)
    anchor = best.anchor_lift_proba(train_df, highsr_oof, pairwise_oof)
    leaderboard = evaluate_candidates(train_df, y, anchor, train_sources)

    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = report_dir / f"{run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    method_card = {
        "protocol": f"audio_only_lift_source_blend_sample_rate_ablation_sr{sample_rate}",
        "sample_rate_hz": sample_rate,
        "allowed_selection_data": "hand/default train labels and regenerated audio-only OOF probabilities",
        "forbidden_selection_data": "robot/test labels, robot/test predictions before lock, image/multimodal features",
        "reference_best_checkpoint": "checkpoints/audio_only_paper_safe_current_0702672_20260706",
        "changed_variable": "audio sample rate only within regenerated source families",
        "source_names": sorted(train_sources),
        "candidate_count": int(len(leaderboard)),
    }
    method_card_path = report_dir / f"{run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact score over lift-anchor source blends",
        "selected_without_test": selected,
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    reference_frame, final_sources = best.broad.load_final_sources(best.broad.final_prediction_paths(args.output))
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
    highsr_final = best.normalize(highsr_frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))
    pair_final = best.normalize(pair_frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))
    if not np.array_equal(highsr_frame["audio_file"].astype(str).to_numpy(), reference_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("High-SR final frame is not aligned with source frame")
    if not np.array_equal(pair_frame["audio_file"].astype(str).to_numpy(), reference_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("Pairwise final frame is not aligned with source frame")

    final_anchor = best.anchor_lift_proba(reference_frame, highsr_final, pair_final)
    source_name = str(selected["source_name"])
    if source_name == "anchor_only":
        final_blended = final_anchor
    else:
        final_blended = best.normalize(
            (1.0 - float(selected["source_weight"])) * final_anchor
            + float(selected["source_weight"]) * final_sources[source_name]
        )
    final_proba = best.postprocess(reference_frame, final_blended, str(selected["blend_mode"]))
    final_pred = final_proba.argmax(axis=1).astype(np.int64)
    y_test = reference_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name=f"audio_lift_source_blend_sr{sample_rate}",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": f"train_only_oof_audio_lift_source_blend_sr{sample_rate}",
            "sample_rate_hz": sample_rate,
            "reference_best_macro_f1_4class": 0.7026719927789447,
            "delta_vs_reference_best_macro_f1_4class": float(final_row["macro_f1_4class"] - 0.7026719927789447),
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_source_name": selected["source_name"],
            "selected_source_weight": selected["source_weight"],
            "selected_blend_mode": selected["blend_mode"],
        }
    )
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = reference_frame[[column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in reference_frame]].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)
    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(y_test, final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)
    protocol_summary = {
        "protocol": method_card["protocol"],
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "method_card": method_card,
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)
    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump({"protocol": method_card["protocol"], "selection_summary": selection_summary, "final_test_report": final_row}, bundle_path)

    print("\nFinal robot/test report:")
    print(pd.DataFrame([final_row]).to_string(index=False), flush=True)
    print(f"\nWrote: {final_report_path.resolve()}", flush=True)
    return final_report_path


def main() -> None:
    args = parse_args()
    run_experiment(args, SAMPLE_RATE, RUN_SLUG)


if __name__ == "__main__":
    main()
