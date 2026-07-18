from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.metrics import confusion_matrix

import train_audio_contact_stress_cv_select_final_test as contact
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")
FEATURE_FAMILY = "audio_filterbank_env_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only filterbank/envelope handcrafted features with contact-aware "
            "stress-CV. Robot/test is loaded only after selection lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def summary_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return np.zeros(8, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(
        [
            np.mean(values),
            np.std(values),
            np.min(values),
            np.percentile(values, 10),
            np.percentile(values, 50),
            np.percentile(values, 90),
            np.max(values),
            np.sqrt(np.mean(values**2)),
        ],
        dtype=np.float32,
    )


def frame_rms(signal: np.ndarray, frame_len: int = 320, hop: int = 160) -> np.ndarray:
    if len(signal) < frame_len:
        return np.asarray([float(np.sqrt(np.mean(signal**2)))], dtype=np.float32)
    starts = np.arange(0, len(signal) - frame_len + 1, hop)
    frames = np.stack([signal[start : start + frame_len] for start in starts])
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12).astype(np.float32)


def band_specs(sr: int) -> list[tuple[float, float]]:
    high_cap = sr / 2 - 80
    bands = [
        (25, 80),
        (80, 160),
        (160, 315),
        (315, 630),
        (630, 1000),
        (1000, 1600),
        (1600, 2500),
        (2500, 3600),
        (3600, 5000),
        (5000, 6500),
        (6500, min(7800, high_cap)),
    ]
    return [(low, high) for low, high in bands if high > low and high < sr / 2]


def bandpass(signal: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, signal).astype(np.float32)


def modulation_features(envelope: np.ndarray, hop_sec: float) -> np.ndarray:
    envelope = np.asarray(envelope, dtype=np.float64)
    envelope = envelope - np.mean(envelope)
    spectrum = np.abs(np.fft.rfft(envelope)) ** 2
    freqs = np.fft.rfftfreq(len(envelope), d=hop_sec)
    total = float(np.sum(spectrum)) + 1e-12
    bands = [(0.1, 4), (4, 12), (12, 30), (30, 80)]
    out = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        value = float(np.sum(spectrum[mask]) / total)
        out.extend([value, np.log1p(value)])
    if len(spectrum):
        out.extend([float(freqs[int(np.argmax(spectrum))]), float(np.max(spectrum) / total)])
    else:
        out.extend([0.0, 0.0])
    return np.asarray(out, dtype=np.float32)


def extract_filterbank_features(signal: np.ndarray, sr: int) -> np.ndarray:
    signal = signal.astype(np.float32)
    signal = signal - np.mean(signal)
    peak = float(np.max(np.abs(signal))) + 1e-8
    signal = signal / peak
    total_energy = float(np.mean(signal**2)) + 1e-12

    features = []
    features.extend(summary_stats(signal))
    features.extend(summary_stats(np.abs(signal)))
    full_rms = frame_rms(signal)
    features.extend(summary_stats(full_rms))
    features.extend(modulation_features(full_rms, hop_sec=160 / sr))

    band_energies = []
    band_centers = []
    for low, high in band_specs(sr):
        filtered = bandpass(signal, sr, low, high)
        abs_filtered = np.abs(filtered)
        rms = frame_rms(filtered)
        energy = float(np.mean(filtered**2))
        ratio = energy / total_energy
        band_energies.append(ratio)
        band_centers.append((low + high) / 2.0)

        features.extend([ratio, np.log1p(ratio)])
        features.extend(summary_stats(filtered))
        features.extend(summary_stats(abs_filtered))
        features.extend(summary_stats(rms))
        features.extend(modulation_features(rms, hop_sec=160 / sr))

        chunk_rms = [np.sqrt(np.mean(chunk**2) + 1e-12) for chunk in np.array_split(filtered, 4)]
        features.extend(chunk_rms)
        max_pos = float(np.argmax(rms) / max(1, len(rms) - 1))
        centroid = float(np.sum(np.arange(len(rms)) * rms) / (np.sum(rms) + 1e-12) / max(1, len(rms) - 1))
        features.extend([max_pos, centroid])

    band_energies_arr = np.asarray(band_energies, dtype=np.float64)
    band_centers_arr = np.asarray(band_centers, dtype=np.float64)
    band_total = float(np.sum(band_energies_arr)) + 1e-12
    norm_energy = band_energies_arr / band_total
    features.extend(summary_stats(norm_energy))
    features.extend(
        [
            float(np.sum(norm_energy * band_centers_arr)),
            float(band_centers_arr[int(np.argmax(norm_energy))]) if len(norm_energy) else 0.0,
            float(-np.sum(norm_energy * np.log(norm_energy + 1e-12)) / np.log(max(2, len(norm_energy)))),
        ]
    )
    if len(norm_energy) >= 3:
        low = float(norm_energy[:3].sum())
        mid = float(norm_energy[3:7].sum())
        high = float(norm_energy[7:].sum())
        features.extend([low / (mid + 1e-8), low / (high + 1e-8), mid / (high + 1e-8)])
    else:
        features.extend([0.0, 0.0, 0.0])

    output = np.asarray(features, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Filterbank feature contains NaN/Inf")
    return output


def cache_paths(feature_dir: Path, split_name: str, view: str) -> dict[str, Path]:
    view_dir = feature_dir / split_name / view
    view_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": view_dir / "X.npy",
        "y": view_dir / "y.npy",
        "paths": view_dir / "paths.npy",
        "metadata": view_dir / "metadata.json",
    }


def manifest_signature(frame: pd.DataFrame, view: str) -> str:
    digest = hashlib.sha256()
    for row in frame[["audio_path", "y"]].itertuples(index=False):
        digest.update(f"{view}|{row.audio_path}|{int(row.y)}\n".encode("utf-8"))
    return digest.hexdigest()


def build_or_load_feature_cache(
    frame: pd.DataFrame,
    split_name: str,
    view: str,
    feature_dir: Path,
    force_rebuild: bool,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
    paths = cache_paths(feature_dir, split_name, view)
    signature = manifest_signature(frame, view)
    ready = all(path.exists() for path in paths.values())
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        valid = (
            metadata.get("feature_family") == FEATURE_FAMILY
            and metadata.get("view") == view
            and int(metadata.get("n_samples", -1)) == len(frame)
            and metadata.get("manifest_signature") == signature
        )
        if valid:
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": np.load(paths["y"]),
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            load_time = time.perf_counter() - start
            print(f"Loaded {FEATURE_FAMILY}/{split_name}/{view}: {payload['X'].shape} in {load_time:.3f}s")
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
            }
        print(f"Cache metadata mismatch for {split_name}/{view}; rebuilding.")

    sr = int(base.CONFIG["audio"]["sr"])

    def extract_one(audio_path: str, label: int) -> tuple[np.ndarray, int, str]:
        signal = base.load_audio(Path(audio_path), target_sr=sr, duration=base.CONFIG["audio"]["duration"])
        if view != "clean":
            signal = stress.apply_stress_view(signal, view=view, key=str(audio_path), sr=sr)
        return extract_filterbank_features(signal, sr=sr), int(label), str(audio_path)

    jobs = [(str(row.audio_path), int(row.y)) for row in frame.itertuples(index=False)]
    start = time.perf_counter()
    if n_jobs == 1:
        results = [
            extract_one(audio_path, label)
            for audio_path, label in jobs
        ]
    else:
        print(f"Parallel extract {FEATURE_FAMILY}/{split_name}/{view}: n_jobs={n_jobs} samples={len(jobs)}", flush=True)
        results = joblib.Parallel(n_jobs=n_jobs, batch_size=32, verbose=5)(
            joblib.delayed(extract_one)(audio_path, label)
            for audio_path, label in jobs
        )
    extraction_time = time.perf_counter() - start
    rows, labels, audio_paths = zip(*results)
    payload = {
        "X": np.stack(rows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(audio_paths),
    }
    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    metadata = {
        "feature_family": FEATURE_FAMILY,
        "view": view,
        "feature_dim": int(payload["X"].shape[1]),
        "n_samples": len(frame),
        "manifest_signature": signature,
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {FEATURE_FAMILY}/{split_name}/{view}: {payload['X'].shape} in {extraction_time:.2f}s")
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
    }


def make_candidates() -> dict[str, stress.StressCandidate]:
    specs = [
        stress.StressCandidate("filterbank_hgb_default__all_aug", "direct_hgb_default", "single", STRESS_VIEWS),
        stress.StressCandidate("filterbank_hgb_regularized__all_aug", "direct_hgb_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("filterbank_lgbm__all_aug", "direct_lightgbm_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("filterbank_hier_extra_hgb__all_aug", "hier_extra_hgb", "single", STRESS_VIEWS),
        stress.StressCandidate("filterbank_hier_hgb_lgbm__all_aug", "hier_hgb_lightgbm", "single", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def evaluate_candidate(
    spec: stress.StressCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    filter_specs: dict[str, stress.StressCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], list[dict]]:
    print(f"\nFilterbank candidate: {spec.name} train_views={spec.train_views}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = stress.fit_stress_candidate(spec, base_specs, filter_specs, X_by_view, y, train_idx)
        train_time = time.perf_counter() - start
        view_scores = {}
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = stress.predict_stress_artifact(artifact, X_by_view[view][val_idx])
            no_bias = contact.score_view(y[val_idx], oof_by_view[view][val_idx], np.zeros(4, dtype=np.float64))
            view_scores[f"{view}_macro_f1"] = no_bias["macro_f1"]
            view_scores[f"{view}_contact_macro_f1"] = no_bias["contact_macro_f1"]
        fold_row = {
            "candidate": spec.name,
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
            model_name=f"{spec.name}__{variant_name}",
            split_name="filterbank_contact_stress_oof_cv",
            y_true=y,
            y_pred=clean_pred,
            train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
            predict_time_sec=0.0,
        )
        row.update(
            {
                "feature_family": FEATURE_FAMILY,
                "candidate": spec.name,
                "base_name": spec.base_name or "",
                "bias_variant": variant_name,
                "train_views_json": json.dumps(list(spec.train_views)),
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
    return rows, fold_rows


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_filterbank_contact_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    feature_dir = run_dir / "features"
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, feature_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH      =", root_path.resolve())
    print("RUN_DIR        =", run_dir.resolve())
    print("FEATURE_FAMILY =", FEATURE_FAMILY)
    print("Selection      = train-only filterbank contact-aware stress-CV")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    payloads = {}
    timings = {}
    for view in STRESS_VIEWS:
        payload, timing = build_or_load_feature_cache(
            train_df,
            split_name="hand_train_full",
            view=view,
            feature_dir=feature_dir,
            force_rebuild=args.force_rebuild,
            n_jobs=args.n_jobs,
        )
        payloads[view] = payload
        timings[view] = timing
    X_by_view = {view: payloads[view]["X"] for view in STRESS_VIEWS}
    y = payloads["clean"]["y"]
    for view in STRESS_VIEWS:
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_filterbank_contact_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only filterbank contact-aware stress-CV; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "feature_family": FEATURE_FAMILY,
        "feature_dim": int(X_by_view["clean"].shape[1]),
        "feature_timing": timings,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = cv.make_candidates(args.random_state)
    filter_specs = make_candidates()
    leaderboard_rows = []
    fold_rows = []
    for spec in filter_specs.values():
        rows, candidate_fold_rows = evaluate_candidate(
            spec,
            base_specs,
            filter_specs,
            X_by_view,
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
    leaderboard_path = report_dir / f"{run_slug}_oof_filterbank_contact_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_spec = filter_specs[str(selected["candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_candidate": selected["candidate"],
        "selected_base_name": selected_spec.base_name,
        "selected_bias_variant": selected["bias_variant"],
        "selected_train_views": list(selected_spec.train_views),
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
    test_payload, test_timing = build_or_load_feature_cache(
        test_df,
        split_name="robot_test",
        view="clean",
        feature_dir=feature_dir,
        force_rebuild=args.force_rebuild,
        n_jobs=args.n_jobs,
    )

    full_idx = np.arange(len(y))
    start = time.perf_counter()
    final_artifact = stress.fit_stress_candidate(
        selected_spec,
        base_specs,
        filter_specs,
        X_by_view,
        y,
        full_idx,
    )
    final_fit_time = time.perf_counter() - start
    final_proba = stress.predict_stress_artifact(final_artifact, test_payload["X"])
    final_pred = contact.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=test_payload["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_filterbank_contact_stress_cv",
            "feature_family": FEATURE_FAMILY,
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
        confusion_matrix(test_payload["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_filterbank_contact_stress_cv_select_no_test_until_final",
            "feature_family": FEATURE_FAMILY,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_filterbank_contact_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "feature_dir": str(feature_dir.resolve()),
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

    print("\nFinal robot/test result after frozen filterbank contact-aware audio selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
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
