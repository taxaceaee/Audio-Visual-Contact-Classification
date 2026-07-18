from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from xgboost import XGBClassifier

import train_audio_contact_stress_cv_select_final_test as contact
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS
LOGMEL_FEATURE_SET = "total240_logmel16x16_compact"
LOGMEL_FEATURE_NAME = "Total240 + compact fixed log-mel temporal grid"
RAW_LOGMEL_CACHE_SET = "total240_logmel32x32"


class WeightedXGBClassifier:
    def __init__(self, class_weights: dict[int, float], **params: object) -> None:
        self.class_weights = class_weights
        self.params = params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedXGBClassifier":
        sample_weight = np.asarray(
            [self.class_weights.get(int(label), 1.0) for label in y],
            dtype=np.float64,
        )
        self.model_ = XGBClassifier(**self.params)
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(self.model_.classes_, dtype=np.int64)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only total240 + fixed log-mel temporal-grid stress-CV. Selection "
            "uses only hand/default audio and train-only perturbations; robot/test is "
            "loaded only after selected_without_test.json is written."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--total-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--total-stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument("--force-rebuild-logmel", action="store_true")
    parser.add_argument("--force-rebuild-total-stress", action="store_true")
    parser.add_argument(
        "--reuse-selection-lock",
        action="store_true",
        help="Reuse an existing selected_without_test.json lock and only run the final locked evaluation.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def pool_axis(values: np.ndarray, bins: int, axis: int) -> np.ndarray:
    values = np.moveaxis(values, axis, -1)
    edges = np.linspace(0, values.shape[-1], bins + 1).round().astype(int)
    pooled = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            stop = min(start + 1, values.shape[-1])
        pooled.append(values[..., start:stop].mean(axis=-1))
    output = np.stack(pooled, axis=-1)
    return np.moveaxis(output, -1, axis)


def extract_logmel_grid_features(signal: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=signal.astype(np.float32),
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=32,
        fmin=25.0,
        fmax=min(9000.0, sr / 2.0),
        power=2.0,
    )
    log_abs = librosa.power_to_db(mel, ref=1.0, top_db=96.0)
    log_rel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    log_abs = pool_axis(log_abs, bins=32, axis=1)
    log_rel = pool_axis(log_rel, bins=32, axis=1)

    abs_grid = np.clip((log_abs + 96.0) / 96.0, -1.0, 2.0)
    rel_grid = np.clip((log_rel + 80.0) / 80.0, 0.0, 1.0)
    mel_mean = rel_grid.mean(axis=1)
    mel_std = rel_grid.std(axis=1)
    time_mean = rel_grid.mean(axis=0)
    time_std = rel_grid.std(axis=0)
    diff_time = np.diff(rel_grid, axis=1)
    diff_stats = np.concatenate(
        [
            diff_time.mean(axis=1),
            diff_time.std(axis=1),
            np.percentile(diff_time, [10, 50, 90], axis=1).reshape(-1),
        ]
    )
    energy = np.square(signal.astype(np.float32))
    envelope = pool_axis(energy.reshape(1, -1), bins=32, axis=1).reshape(-1)
    envelope = np.log1p(envelope * 1e4)
    scalar_stats = np.asarray(
        [
            float(np.mean(signal)),
            float(np.std(signal)),
            float(np.max(np.abs(signal))),
            float(np.sqrt(np.mean(energy) + 1e-12)),
            float(np.percentile(np.abs(signal), 50)),
            float(np.percentile(np.abs(signal), 90)),
            float(np.percentile(np.abs(signal), 99)),
            float((np.abs(signal) > 0.05).mean()),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            abs_grid.reshape(-1),
            rel_grid.reshape(-1),
            mel_mean,
            mel_std,
            time_mean,
            time_std,
            diff_stats,
            envelope,
            scalar_stats,
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Log-mel feature contains NaN/Inf")
    return features


def logmel_cache_paths(feature_dir: Path, split_name: str, view: str) -> dict[str, Path]:
    view_dir = feature_dir / split_name / view
    view_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": view_dir / "X.npy",
        "y": view_dir / "y.npy",
        "paths": view_dir / "paths.npy",
        "metadata": view_dir / "metadata.json",
    }


def build_or_load_logmel_cache(
    frame: pd.DataFrame,
    split_name: str,
    view: str,
    feature_dir: Path,
    force_rebuild: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
    paths = logmel_cache_paths(feature_dir, split_name, view)
    signature = base.manifest_signature(frame)
    ready = all(path.exists() for path in paths.values())
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        valid = (
            metadata.get("feature_set") == RAW_LOGMEL_CACHE_SET
            and int(metadata.get("logmel_dim", -1)) == int(metadata.get("feature_dim", -2))
            and int(metadata.get("n_samples", -1)) == len(frame)
            and metadata.get("manifest_signature") == signature
            and metadata.get("stress_view") == view
        )
        if valid:
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": np.load(paths["y"]),
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            load_time = time.perf_counter() - start
            print(f"Loaded logmel cache {split_name}/{view}: {payload['X'].shape} in {load_time:.3f}s")
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
            }
        print(f"Logmel cache metadata mismatch for {split_name}/{view}; rebuilding.")

    sr = int(base.CONFIG["audio"]["sr"])
    rows = []
    labels = []
    audio_paths = []
    start = time.perf_counter()
    for row in tqdm(
        frame.itertuples(index=False),
        total=len(frame),
        desc=f"Extract logmel/{split_name}/{view}",
    ):
        signal = base.load_audio(row.audio_path, target_sr=sr, duration=base.CONFIG["audio"]["duration"])
        if view != "clean":
            signal = stress.apply_stress_view(signal, view=view, key=str(row.audio_path), sr=sr)
        rows.append(extract_logmel_grid_features(signal, sr=sr))
        labels.append(int(row.y))
        audio_paths.append(str(row.audio_path))

    extraction_time = time.perf_counter() - start
    X = np.stack(rows).astype(np.float32)
    payload = {
        "X": X,
        "y": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(audio_paths),
    }
    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    metadata = {
        "feature_set": RAW_LOGMEL_CACHE_SET,
        "feature_name": "Fixed log-mel 32x32 raw cache",
        "feature_dim": int(X.shape[1]),
        "logmel_dim": int(X.shape[1]),
        "n_samples": len(frame),
        "manifest_signature": signature,
        "stress_view": view,
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved logmel cache {split_name}/{view}: {payload['X'].shape} in {extraction_time:.2f}s")
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
    }


def combine_total_logmel(
    total_by_view: dict[str, dict[str, np.ndarray]],
    logmel_by_view: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    combined = {}
    for view in STRESS_VIEWS:
        if not np.array_equal(total_by_view[view]["y"], logmel_by_view[view]["y"]):
            raise AssertionError(f"Label mismatch for {view}")
        if not np.array_equal(total_by_view[view]["paths"].astype(str), logmel_by_view[view]["paths"].astype(str)):
            raise AssertionError(f"Path mismatch for {view}")
        combined[view] = np.hstack(
            [total_by_view[view]["X"], compact_logmel_matrix(logmel_by_view[view]["X"])]
        ).astype(np.float32)
    return combined


def pool_32_grid_to_16(flat_grid: np.ndarray) -> np.ndarray:
    grid = flat_grid.reshape(-1, 32, 32)
    pooled = grid.reshape(grid.shape[0], 16, 2, 16, 2).mean(axis=(2, 4))
    return pooled.reshape(grid.shape[0], -1)


def compact_logmel_matrix(X: np.ndarray) -> np.ndarray:
    abs16 = pool_32_grid_to_16(X[:, :1024])
    rel16 = pool_32_grid_to_16(X[:, 1024:2048])
    summaries = X[:, 2048:]
    return np.hstack([abs16, rel16, summaries]).astype(np.float32)


def make_logmel_candidates(random_state: int) -> dict[str, cv.CandidateSpec]:
    class_weights = base.CONFIG["class_weights"]
    return {
        "logmel_lgbm_regularized": cv.CandidateSpec(
            name="logmel_lgbm_regularized",
            kind="direct",
            direct_factory=lambda: LGBMClassifier(
                objective="multiclass",
                num_class=len(base.CLASS_NAMES),
                n_estimators=360,
                learning_rate=0.025,
                num_leaves=17,
                min_child_samples=34,
                subsample=0.82,
                colsample_bytree=0.72,
                reg_lambda=3.0,
                class_weight=class_weights,
                random_state=random_state + 11,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        "logmel_extra_trees": cv.CandidateSpec(
            name="logmel_extra_trees",
            kind="direct",
            direct_factory=lambda: ExtraTreesClassifier(
                n_estimators=520,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight=class_weights,
                random_state=random_state + 13,
                n_jobs=-1,
            ),
        ),
        "logmel_xgb_weighted": cv.CandidateSpec(
            name="logmel_xgb_weighted",
            kind="direct",
            direct_factory=lambda: WeightedXGBClassifier(
                class_weights=class_weights,
                objective="multi:softprob",
                num_class=len(base.CLASS_NAMES),
                n_estimators=260,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=4,
                subsample=0.84,
                colsample_bytree=0.72,
                reg_lambda=3.0,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state + 17,
                n_jobs=-1,
            ),
        ),
    }


def make_logmel_stress_specs() -> dict[str, stress.StressCandidate]:
    specs = [
        stress.StressCandidate("logmel_lgbm_regularized__all_aug", "logmel_lgbm_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("logmel_extra_trees__all_aug", "logmel_extra_trees", "single", STRESS_VIEWS),
        stress.StressCandidate("logmel_xgb_weighted__all_aug", "logmel_xgb_weighted", "single", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def patch_feature_metadata(rows: list[dict], n_features: int) -> list[dict]:
    for row in rows:
        row["feature_set"] = LOGMEL_FEATURE_SET
        row["feature_name"] = LOGMEL_FEATURE_NAME
        row["n_features"] = n_features
    return rows


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_logmel_contact_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    logmel_feature_dir = run_dir / "logmel_features"
    for directory in [run_dir, report_dir, model_dir, split_dir, logmel_feature_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("TOTAL_FEATURE_CACHE_DIR  =", args.total_feature_cache_dir.resolve())
    print("TOTAL_STRESS_FEATURE_DIR =", args.total_stress_feature_dir.resolve())
    print("LOGMEL_FEATURE_DIR       =", logmel_feature_dir.resolve())
    print("Feature set              =", LOGMEL_FEATURE_NAME)
    print("Selection                = train-only contact-aware stress-CV")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    total_clean, total_clean_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.total_feature_cache_dir,
        force_rebuild=False,
    )
    total_by_view = {"clean": total_clean}
    total_timing = {"clean": total_clean_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, view_timing = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.total_stress_feature_dir,
            force_rebuild=args.force_rebuild_total_stress,
        )
        total_by_view[view] = payload
        total_timing[view] = view_timing

    logmel_by_view = {}
    logmel_timing = {}
    for view in STRESS_VIEWS:
        payload, view_timing = build_or_load_logmel_cache(
            train_df,
            split_name="hand_train_full",
            view=view,
            feature_dir=logmel_feature_dir,
            force_rebuild=args.force_rebuild_logmel,
        )
        logmel_by_view[view] = payload
        logmel_timing[view] = view_timing

    X_by_view = combine_total_logmel(total_by_view, logmel_by_view)
    y = total_clean["y"]
    n_features = int(X_by_view["clean"].shape[1])

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_logmel_contact_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only total240 + fixed logmel-grid contact stress-CV; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "n_features": n_features,
        "total_feature_timing": total_timing,
        "logmel_feature_timing": logmel_timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = make_logmel_candidates(args.random_state)
    stress_specs = make_logmel_stress_specs()
    leaderboard_path = report_dir / f"{run_slug}_oof_contact_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    if args.reuse_selection_lock:
        if not selection_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing selection lock: {selection_path}")
        selection_summary = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selection_summary["selected_without_test"]
        selected_spec = stress_specs[str(selection_summary["selected_base_candidate"])]
        selected_bias = np.asarray(selection_summary["selected_bias"], dtype=np.float64)
        print("\nReusing existing selection lock before loading robot/test:")
        print(json.dumps(selection_summary, indent=2, default=float), flush=True)
    else:
        leaderboard_rows = []
        fold_rows = []
        for spec in stress_specs.values():
            rows, _, candidate_fold_rows = contact.evaluate_candidate(
                spec,
                base_specs,
                stress_specs,
                X_by_view,
                y,
                splits,
            )
            leaderboard_rows.extend(patch_feature_metadata(rows, n_features))
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
        leaderboard.to_csv(leaderboard_path, index=False)
        pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

        selected = leaderboard.iloc[0].to_dict()
        selected_spec = stress_specs[str(selected["base_candidate"])]
        selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
        selection_summary = {
            "selected_without_test": selected,
            "selected_model": selected["model"],
            "selected_base_candidate": selected["base_candidate"],
            "selected_bias_variant": selected["bias_variant"],
            "selected_base_name": selected_spec.base_name,
            "selected_train_views": list(selected_spec.train_views),
            "selected_bias": selected_bias.tolist(),
            "leaderboard_path": str(leaderboard_path.resolve()),
            "fold_report_path": str(fold_report_path.resolve()),
            "split_summary_path": str(split_summary_path.resolve()),
        }
        write_json(selection_path, selection_summary)
        print("\nSelection lock written before loading robot/test:")
        print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
    test_df = base.load_manifest(test_csv, "robot_test")
    total_test, total_test_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.total_feature_cache_dir,
        force_rebuild=False,
    )
    logmel_test, logmel_test_timing = build_or_load_logmel_cache(
        test_df,
        split_name="robot_test",
        view="clean",
        feature_dir=logmel_feature_dir,
        force_rebuild=args.force_rebuild_logmel,
    )
    if not np.array_equal(total_test["y"], logmel_test["y"]):
        raise AssertionError("Test label mismatch between total240 and logmel caches")
    X_test = np.hstack([total_test["X"], compact_logmel_matrix(logmel_test["X"])]).astype(np.float32)

    start = time.perf_counter()
    final_artifact = stress.fit_stress_candidate(
        selected_spec,
        base_specs,
        stress_specs,
        X_by_view,
        y,
        np.arange(len(y)),
    )
    final_fit_time = time.perf_counter() - start
    final_proba = stress.predict_stress_artifact(final_artifact, X_test)
    final_pred = contact.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=total_test["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "feature_set": LOGMEL_FEATURE_SET,
            "feature_name": LOGMEL_FEATURE_NAME,
            "n_features": n_features,
            "selected_by": "hand_default_audio_only_total240_logmel_contact_stress_cv",
            "selected_worst_hybrid_score": selected["selection_worst_hybrid_score"],
            "selected_worst_macro_f1": selected["worst_macro_f1"],
            "selected_worst_contact_macro_f1": selected["worst_contact_macro_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    audio_columns = [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_df]
    prediction_frame = test_df[audio_columns].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(total_test["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_total240_logmel_contact_stress_cv_select_no_test_until_final",
            "feature_set": LOGMEL_FEATURE_SET,
            "feature_name": LOGMEL_FEATURE_NAME,
            "n_features": n_features,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_total240_logmel_contact_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": {
            "total240": total_test_timing,
            "logmel": logmel_test_timing,
        },
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

    print("\nFinal robot/test result after frozen audio-only logmel selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_worst_hybrid_score",
                "selected_worst_contact_macro_f1",
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
