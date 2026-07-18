from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_contact_stress_cv_select_final_test as contact
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only feature-slice stress-CV selection from cached total240 features. "
            "Selection uses hand/default only; robot/test is loaded after lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--clean-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def feature_slices() -> dict[str, np.ndarray]:
    full = np.arange(0, 120)
    top = np.arange(120, 240)
    blocks = {
        "mfcc40": np.arange(0, 40),
        "stft28": np.arange(40, 68),
        "mel28": np.arange(68, 96),
        "fft24": np.arange(96, 120),
    }
    slices = {
        "total240": np.arange(0, 240),
        "full_total120": full,
        "top_total120": top,
        "full_mfcc40": blocks["mfcc40"],
        "full_stft28": blocks["stft28"],
        "full_mel28": blocks["mel28"],
        "full_fft24": blocks["fft24"],
        "top_mfcc40": blocks["mfcc40"] + 120,
        "top_stft28": blocks["stft28"] + 120,
        "top_mel28": blocks["mel28"] + 120,
        "top_fft24": blocks["fft24"] + 120,
        "full_stft_mel_fft80": np.r_[blocks["stft28"], blocks["mel28"], blocks["fft24"]],
        "top_stft_mel_fft80": np.r_[blocks["stft28"] + 120, blocks["mel28"] + 120, blocks["fft24"] + 120],
        "full_top_fft48": np.r_[blocks["fft24"], blocks["fft24"] + 120],
        "full_top_stft56": np.r_[blocks["stft28"], blocks["stft28"] + 120],
        "full_top_mel56": np.r_[blocks["mel28"], blocks["mel28"] + 120],
        "full_top_mfcc80": np.r_[blocks["mfcc40"], blocks["mfcc40"] + 120],
    }
    return {name: columns.astype(np.int64) for name, columns in slices.items()}


def fit_direct_artifact(
    base_spec: cv.CandidateSpec,
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    train_views: tuple[str, ...],
) -> object:
    X_train, y_train = stress.build_training_matrix(X_by_view, y, train_idx, train_views)
    return stress.fit_single_candidate(base_spec, X_train, y_train)


def evaluate_slice_candidate(
    slice_name: str,
    columns: np.ndarray,
    base_name: str,
    base_spec: cv.CandidateSpec,
    X240_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    X_by_view = {view: X240_by_view[view][:, columns] for view in STRESS_VIEWS}
    train_views = STRESS_VIEWS
    print(f"\nSlice candidate: {slice_name} base={base_name} dim={len(columns)}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = fit_direct_artifact(base_spec, X_by_view, y, train_idx, train_views)
        train_time = time.perf_counter() - start
        view_scores = {}
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = stress.predict_stress_artifact(artifact, X_by_view[view][val_idx])
            no_bias = contact.score_view(y[val_idx], oof_by_view[view][val_idx], np.zeros(4, dtype=np.float64))
            view_scores[f"{view}_macro_f1"] = no_bias["macro_f1"]
            view_scores[f"{view}_contact_macro_f1"] = no_bias["contact_macro_f1"]
        fold_row = {
            "slice": slice_name,
            "base_name": base_name,
            "fold": fold_id,
            "train_time_sec": train_time,
            "val_samples": int(len(val_idx)),
            **view_scores,
        }
        fold_rows.append(fold_row)
        print(
            f"  fold {fold_id}: "
            + " | ".join(
                f"{view}=M{view_scores[f'{view}_macro_f1']:.4f}/C{view_scores[f'{view}_contact_macro_f1']:.4f}"
                for view in STRESS_VIEWS
            )
            + f" time={train_time:.2f}s",
            flush=True,
        )

    rows = []
    for variant_name, bias, scores in contact.bias_variants(oof_by_view, y):
        clean_pred = contact.predict_with_bias(oof_by_view["clean"], bias)
        row = base.make_report_row(
            model_name=f"{slice_name}__{base_name}__{variant_name}",
            split_name="slice_contact_stress_oof_cv",
            y_true=y,
            y_pred=clean_pred,
            train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
            predict_time_sec=0.0,
        )
        row.update(
            {
                "slice_name": slice_name,
                "slice_dim": int(len(columns)),
                "slice_columns_json": json.dumps(columns.tolist()),
                "base_name": base_name,
                "bias_variant": variant_name,
                "train_views_json": json.dumps(list(train_views)),
                "class_bias_json": json.dumps(bias.tolist()),
                **scores,
                "selection_worst_hybrid_score": scores["worst_hybrid_macro_contact"],
                "selection_mean_hybrid_score": scores["mean_hybrid_macro_contact"],
            }
        )
        rows.append(row)
        print(
            f"  {variant_name}: worst hybrid={scores['worst_hybrid_macro_contact']:.4f} "
            f"worst macro={scores['worst_macro_f1']:.4f} "
            f"worst contact={scores['worst_contact_macro_f1']:.4f} "
            f"bias={np.round(bias, 3).tolist()}",
            flush=True,
        )
    return rows, oof_by_view, fold_rows


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_slice_contact_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH               =", root_path.resolve())
    print("RUN_DIR                 =", run_dir.resolve())
    print("CLEAN_FEATURE_CACHE_DIR =", args.clean_feature_cache_dir.resolve())
    print("STRESS_FEATURE_DIR      =", args.stress_feature_dir.resolve())
    print("Selection               = train-only feature slice + contact-aware stress-CV")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    clean_feat, clean_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    payloads = {"clean": clean_feat}
    timing = {"clean": clean_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, view_timing = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.stress_feature_dir,
            force_rebuild=False,
        )
        payloads[view] = payload
        timing[view] = view_timing

    X240_by_view = {view: payloads[view]["X"] for view in STRESS_VIEWS}
    y = clean_feat["y"]
    for view in STRESS_VIEWS:
        if X240_by_view[view].shape[1] != 240:
            raise AssertionError(f"Expected total240 cache for {view}, got {X240_by_view[view].shape}")
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_slice_contact_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    base_specs = cv.make_candidates(args.random_state)
    model_specs = {
        "direct_hgb_default": base_specs["direct_hgb_default"],
        "direct_hgb_regularized": base_specs["direct_hgb_regularized"],
    }
    slices = feature_slices()

    split_summary = {
        "protocol": "audio-only feature-slice contact-aware stress-CV; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "feature_slices": {name: columns.tolist() for name, columns in slices.items()},
        "base_models": list(model_specs),
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    leaderboard_rows = []
    fold_rows = []
    for slice_name, columns in slices.items():
        for base_name, base_spec in model_specs.items():
            rows, _, candidate_fold_rows = evaluate_slice_candidate(
                slice_name,
                columns,
                base_name,
                base_spec,
                X240_by_view,
                y,
                splits,
            )
            leaderboard_rows.extend(rows)
            fold_rows.extend(candidate_fold_rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [
            "selection_worst_hybrid_score",
            "selection_mean_hybrid_score",
            "worst_macro_f1",
            "worst_contact_macro_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_slice_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_slice = str(selected["slice_name"])
    selected_base_name = str(selected["base_name"])
    selected_columns = np.asarray(json.loads(selected["slice_columns_json"]), dtype=np.int64)
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_slice": selected_slice,
        "selected_slice_dim": int(len(selected_columns)),
        "selected_base_name": selected_base_name,
        "selected_bias_variant": selected["bias_variant"],
        "selected_train_views": list(STRESS_VIEWS),
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
    test_df = base.load_manifest(test_csv, "robot_test")
    test_feat, test_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    X_by_view = {view: X240_by_view[view][:, selected_columns] for view in STRESS_VIEWS}
    X_test = test_feat["X"][:, selected_columns]
    full_idx = np.arange(len(y))
    start = time.perf_counter()
    final_artifact = fit_direct_artifact(
        model_specs[selected_base_name],
        X_by_view,
        y,
        full_idx,
        STRESS_VIEWS,
    )
    final_fit_time = time.perf_counter() - start
    final_proba = stress.predict_stress_artifact(final_artifact, X_test)
    final_pred = contact.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=test_feat["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_slice_contact_stress_cv",
            "selected_slice": selected_slice,
            "selected_slice_dim": int(len(selected_columns)),
            "selected_worst_hybrid_score": selected["selection_worst_hybrid_score"],
            "selected_worst_macro_f1": selected["worst_macro_f1"],
            "selected_worst_contact_macro_f1": selected["worst_contact_macro_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_feat["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_slice_contact_stress_cv_select_no_test_until_final",
            "feature_set": base.FEATURE_SET,
            "feature_spec": base.FEATURE_SPEC,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_slice_contact_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "fold_report": str(fold_report_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen feature-slice audio selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "selected_slice",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_worst_hybrid_score",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
