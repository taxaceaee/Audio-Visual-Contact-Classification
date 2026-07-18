from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm

import train_cv_select_final_test as cv
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")


@dataclass(frozen=True)
class StressCandidate:
    name: str
    base_name: str | None
    kind: str
    train_views: tuple[str, ...] = ("clean",)
    members: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supervised stress-CV for classical ML. Selection uses only hand/default: "
            "clean grouped CV plus deterministic train-only stress perturbations. "
            "Robot/test is loaded only after a selection lock is written."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-set", choices=sorted(base.FEATURE_SPECS), default="total240")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--clean-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument("--force-rebuild-stress", action="store_true")
    return parser.parse_args()


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def normalize_peak(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) + 1e-8
    return (signal / peak).astype(np.float32)


def fft_bandlimit(
    signal: np.ndarray,
    sr: int,
    low_hz: float | None,
    high_hz: float | None,
) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    mask = np.ones_like(freqs, dtype=bool)
    if low_hz is not None:
        mask &= freqs >= low_hz
    if high_hz is not None:
        mask &= freqs <= high_hz
    filtered = np.fft.irfft(spectrum * mask, n=len(signal))
    return filtered.astype(np.float32)


def add_noise_at_snr(signal: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(signal**2))) + 1e-8
    noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=signal.shape)
    return (signal + noise).astype(np.float32)


def apply_stress_view(signal: np.ndarray, view: str, key: str, sr: int) -> np.ndarray:
    if view == "clean":
        return signal.astype(np.float32)

    rng = np.random.default_rng(stable_seed(f"{view}|{key}"))
    output = signal.astype(np.float32).copy()

    if view == "robot_mix":
        output = fft_bandlimit(output, sr, low_hz=90.0, high_hz=5200.0)
        output = output * float(rng.uniform(0.65, 1.25))
        output = add_noise_at_snr(output, rng, snr_db=float(rng.uniform(18.0, 28.0)))
        drive = float(rng.uniform(1.15, 1.65))
        output = np.tanh(drive * output) / np.tanh(drive)
        shift = int(rng.integers(-1200, 1201))
        output = np.roll(output, shift)
        return normalize_peak(output)

    if view == "bandlimit":
        output = fft_bandlimit(output, sr, low_hz=180.0, high_hz=3600.0)
        output = output * float(rng.uniform(0.75, 1.15))
        output = np.clip(output, -0.78, 0.78)
        output = add_noise_at_snr(output, rng, snr_db=float(rng.uniform(24.0, 34.0)))
        return normalize_peak(output)

    raise ValueError(f"Unknown stress view: {view}")


def extract_features_from_signal(signal: np.ndarray, sr: int) -> np.ndarray:
    if base.FEATURE_SET == "mfcc40":
        features = base.extract_mfcc_compact(signal, sr)
    elif base.FEATURE_SET == "stft28":
        features = base.extract_stft_compact(signal, sr)
    elif base.FEATURE_SET == "mel28":
        features = base.extract_mel_compact(signal, sr)
    elif base.FEATURE_SET == "fft24":
        features = base.extract_fft_compact(signal, sr)
    elif base.FEATURE_SET == "total120":
        features = base.extract_total_120(signal, sr)
    elif base.FEATURE_SET == "total240":
        crop_cfg = base.CONFIG["event_crop"]
        top_signal = base.get_top_energy_window(
            signal,
            sr=sr,
            window_sec=crop_cfg["top_energy_sec"],
            hop_sec=crop_cfg["top_energy_hop_sec"],
        )
        features = np.concatenate(
            [base.extract_total_120(signal, sr), base.extract_total_120(top_signal, sr)]
        ).astype(np.float32)
    else:
        raise KeyError(f"Unknown feature set: {base.FEATURE_SET}")

    if features.shape != (base.EXPECTED_DIM,):
        raise AssertionError(f"{base.FEATURE_SET}: expected {base.EXPECTED_DIM}, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Augmented feature contains NaN/Inf")
    return features.astype(np.float32)


def stress_cache_paths(stress_feature_dir: Path, view: str) -> dict[str, Path]:
    view_dir = stress_feature_dir / view
    view_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": view_dir / "X.npy",
        "y": view_dir / "y.npy",
        "paths": view_dir / "paths.npy",
        "metadata": view_dir / "metadata.json",
    }


def build_or_load_stress_cache(
    frame: pd.DataFrame,
    view: str,
    stress_feature_dir: Path,
    force_rebuild: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
    if view == "clean":
        raise ValueError("Use the clean feature cache for view='clean'")

    paths = stress_cache_paths(stress_feature_dir, view)
    signature = base.manifest_signature(frame)
    ready = all(path.exists() for path in paths.values())
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        valid_metadata = (
            metadata.get("feature_set") == base.FEATURE_SET
            and int(metadata.get("feature_dim", -1)) == base.EXPECTED_DIM
            and int(metadata.get("n_samples", -1)) == len(frame)
            and metadata.get("manifest_signature") == signature
            and metadata.get("stress_view") == view
        )
        if valid_metadata:
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": np.load(paths["y"]),
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            load_time = time.perf_counter() - start
            print(f"Loaded stress cache {view}: {payload['X'].shape} in {load_time:.3f}s")
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
            }
        print(f"Stress cache metadata mismatch for {view}; rebuilding.")

    sr = int(base.CONFIG["audio"]["sr"])
    rows = []
    labels = []
    audio_paths = []
    start = time.perf_counter()
    for row in tqdm(
        frame.itertuples(index=False),
        total=len(frame),
        desc=f"Extract stress/{view}",
    ):
        signal = base.load_audio(row.audio_path, target_sr=sr, duration=base.CONFIG["audio"]["duration"])
        stressed = apply_stress_view(signal, view=view, key=str(row.audio_path), sr=sr)
        rows.append(extract_features_from_signal(stressed, sr=sr))
        labels.append(int(row.y))
        audio_paths.append(str(row.audio_path))
    extraction_time = time.perf_counter() - start

    payload = {
        "X": np.stack(rows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(audio_paths),
    }
    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    metadata = {
        "feature_set": base.FEATURE_SET,
        "feature_dim": base.EXPECTED_DIM,
        "n_samples": len(frame),
        "manifest_signature": signature,
        "stress_view": view,
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved stress cache {view}: {payload['X'].shape} in {extraction_time:.2f}s")
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
    }


def make_stress_candidates() -> dict[str, StressCandidate]:
    specs = [
        StressCandidate("hier_extra_hgb__clean", "hier_extra_hgb", "single", ("clean",)),
        StressCandidate("hier_extra_hgb__robot_aug", "hier_extra_hgb", "single", ("clean", "robot_mix")),
        StressCandidate("hier_extra_hgb__all_aug", "hier_extra_hgb", "single", STRESS_VIEWS),
        StressCandidate("hier_hgb_hgb__clean", "hier_hgb_hgb", "single", ("clean",)),
        StressCandidate("hier_hgb_hgb__robot_aug", "hier_hgb_hgb", "single", ("clean", "robot_mix")),
        StressCandidate("hier_hgb_lightgbm__robot_aug", "hier_hgb_lightgbm", "single", ("clean", "robot_mix")),
        StressCandidate("direct_hgb_default__clean", "direct_hgb_default", "single", ("clean",)),
        StressCandidate("direct_hgb_default__robot_aug", "direct_hgb_default", "single", ("clean", "robot_mix")),
        StressCandidate("direct_hgb_regularized__robot_aug", "direct_hgb_regularized", "single", ("clean", "robot_mix")),
        StressCandidate("direct_hgb_regularized__all_aug", "direct_hgb_regularized", "single", STRESS_VIEWS),
        StressCandidate("direct_lightgbm_regularized__robot_aug", "direct_lightgbm_regularized", "single", ("clean", "robot_mix")),
    ]
    return {spec.name: spec for spec in specs}


def build_training_matrix(
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    train_views: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    X_parts = [X_by_view[view][train_idx] for view in train_views]
    y_parts = [y[train_idx] for _ in train_views]
    return np.vstack(X_parts), np.concatenate(y_parts)


def fit_single_candidate(base_spec: cv.CandidateSpec, X_train: np.ndarray, y_train: np.ndarray) -> object:
    if base_spec.kind == "direct":
        model = base_spec.direct_factory()
        model.fit(X_train, y_train)
        return {"kind": "direct", "model": model}

    if base_spec.kind == "hierarchical":
        binary_model = base_spec.binary_factory()
        binary_model.fit(X_train, (y_train > 0).astype(np.int64))
        contact_idx = np.where(y_train > 0)[0]
        contact_model = base_spec.contact_factory()
        contact_model.fit(X_train[contact_idx], y_train[contact_idx])
        return {
            "kind": "hierarchical",
            "binary_model": binary_model,
            "contact_model": contact_model,
        }

    raise ValueError(f"Unsupported base candidate kind: {base_spec.kind}")


def predict_single_artifact(artifact: dict, X: np.ndarray) -> np.ndarray:
    if artifact["kind"] == "direct":
        return cv.proba_aligned(artifact["model"], X)
    if artifact["kind"] == "hierarchical":
        return cv.hierarchical_proba(artifact["binary_model"], artifact["contact_model"], X)
    raise ValueError(f"Unsupported artifact kind: {artifact['kind']}")


def fit_stress_candidate(
    spec: StressCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    stress_specs: dict[str, StressCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
) -> object:
    if spec.kind == "single":
        X_train, y_train = build_training_matrix(X_by_view, y, train_idx, spec.train_views)
        return fit_single_candidate(base_specs[spec.base_name], X_train, y_train)

    if spec.kind == "ensemble":
        return {
            "kind": "ensemble",
            "members": {
                member: fit_stress_candidate(
                    stress_specs[member],
                    base_specs,
                    stress_specs,
                    X_by_view,
                    y,
                    train_idx,
                )
                for member in spec.members
            },
        }

    raise ValueError(f"Unsupported stress candidate kind: {spec.kind}")


def predict_stress_artifact(artifact: object, X: np.ndarray) -> np.ndarray:
    if artifact["kind"] == "ensemble":
        return np.mean(
            [predict_stress_artifact(member_artifact, X) for member_artifact in artifact["members"].values()],
            axis=0,
        )
    return predict_single_artifact(artifact, X)


def tune_bias_multiview(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if grid is None:
        grid = np.linspace(-1.2, 1.2, 13)

    def score_bias(bias: np.ndarray) -> dict[str, float]:
        scores = {
            view: f1_score(y_true, cv.predict_with_bias(proba, bias), average="macro")
            for view, proba in proba_by_view.items()
        }
        scores["stress_worst_macro_f1"] = min(scores.values())
        scores["stress_mean_macro_f1"] = float(np.mean(list(scores.values())[:-1]))
        return scores

    best_bias = np.zeros(4, dtype=np.float64)
    best_scores = score_bias(best_bias)

    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        scores = score_bias(bias)
        if (
            scores["stress_worst_macro_f1"] > best_scores["stress_worst_macro_f1"]
            or (
                scores["stress_worst_macro_f1"] == best_scores["stress_worst_macro_f1"]
                and scores["stress_mean_macro_f1"] > best_scores["stress_mean_macro_f1"]
            )
        ):
            best_bias = bias
            best_scores = scores

    for ambient_bias in np.linspace(-0.8, 0.8, 9):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        scores = score_bias(bias)
        if (
            scores["stress_worst_macro_f1"] > best_scores["stress_worst_macro_f1"]
            or (
                scores["stress_worst_macro_f1"] == best_scores["stress_worst_macro_f1"]
                and scores["stress_mean_macro_f1"] > best_scores["stress_mean_macro_f1"]
            )
        ):
            best_bias = bias
            best_scores = scores

    return best_bias, best_scores


def evaluate_stress_candidate(
    spec: StressCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    stress_specs: dict[str, StressCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    print(f"\nStress candidate: {spec.name} train_views={spec.train_views or spec.members}")
    oof_by_view = {view: np.zeros((len(y), 4), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = fit_stress_candidate(spec, base_specs, stress_specs, X_by_view, y, train_idx)
        train_time = time.perf_counter() - start
        view_scores = {}
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = predict_stress_artifact(artifact, X_by_view[view][val_idx])
            pred = oof_by_view[view][val_idx].argmax(axis=1)
            view_scores[f"{view}_macro_f1"] = f1_score(y[val_idx], pred, average="macro")
        fold_row = {
            "candidate": spec.name,
            "fold": fold_id,
            "train_time_sec": train_time,
            "val_samples": len(val_idx),
            **view_scores,
            "fold_worst_macro_f1": min(view_scores.values()),
        }
        fold_rows.append(fold_row)
        print(
            f"  fold {fold_id}: "
            + " | ".join(f"{view}={view_scores[f'{view}_macro_f1']:.4f}" for view in STRESS_VIEWS)
            + f" | worst={fold_row['fold_worst_macro_f1']:.4f} time={train_time:.2f}s"
        )

    bias, tuned_scores = tune_bias_multiview(oof_by_view, y)
    clean_pred = cv.predict_with_bias(oof_by_view["clean"], bias)
    row = base.make_report_row(
        model_name=spec.name,
        split_name="stress_oof_cv",
        y_true=y,
        y_pred=clean_pred,
        train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
        predict_time_sec=0.0,
    )
    row.update(
        {
            "stress_candidate_kind": spec.kind,
            "base_name": spec.base_name or "",
            "train_views_json": json.dumps(list(spec.train_views)),
            "members_json": json.dumps(list(spec.members)),
            "class_bias_json": json.dumps(bias.tolist()),
            **{f"tuned_{view}_macro_f1": tuned_scores[view] for view in STRESS_VIEWS},
            "stress_worst_macro_f1": tuned_scores["stress_worst_macro_f1"],
            "stress_mean_macro_f1": tuned_scores["stress_mean_macro_f1"],
        }
    )
    print(
        "  tuned: "
        + " | ".join(f"{view}={tuned_scores[view]:.4f}" for view in STRESS_VIEWS)
        + f" | worst={tuned_scores['stress_worst_macro_f1']:.4f} "
        + f"| bias={np.round(bias, 3).tolist()}"
    )
    return row, oof_by_view, fold_rows


def build_ensemble_candidates(
    leaderboard: pd.DataFrame,
    stress_specs: dict[str, StressCandidate],
) -> dict[str, StressCandidate]:
    ranked = leaderboard.sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    )["model"].tolist()
    recipes = {
        "stress_ensemble_top3": tuple(ranked[:3]),
        "stress_ensemble_top5": tuple(ranked[:5]),
    }
    return {
        name: StressCandidate(name=name, base_name=None, kind="ensemble", members=members)
        for name, members in recipes.items()
        if len(members) >= 2 and all(member in stress_specs for member in members)
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set(args.feature_set)

    root_path = base.resolve_root(args.root)
    run_slug = f"{args.feature_set}_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    stress_feature_dir = run_dir / "stress_features"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, stress_feature_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH               =", root_path.resolve())
    print("RUN_DIR                 =", run_dir.resolve())
    print("CLEAN_FEATURE_CACHE_DIR =", args.clean_feature_cache_dir.resolve())
    print("Feature set             =", base.FEATURE_SPEC["display_name"])
    print("Selection               = maximize worst-case grouped stress-CV macro-F1")

    train_csv = base.require_file(
        root_path / "audio_visual_dataset_default" / "dataset.csv",
        "hand/default dataset.csv",
    )
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    clean_feat, clean_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    feature_payloads = {"clean": clean_feat}
    timing = {"clean": clean_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, view_timing = build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=stress_feature_dir,
            force_rebuild=args.force_rebuild_stress,
        )
        feature_payloads[view] = payload
        timing[view] = view_timing

    X_by_view = {view: feature_payloads[view]["X"] for view in STRESS_VIEWS}
    y = clean_feat["y"]
    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    train_df.assign(cv_fold=fold_assignment).to_csv(
        split_dir / "hand_train_full_stress_cv_folds.csv",
        index=False,
    )

    split_summary = {
        "protocol": "hand/default grouped stress-CV only for selection; robot/test loaded after selection lock",
        "n_train_samples": len(train_df),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": args.n_folds,
        "stress_views": list(STRESS_VIEWS),
        "label_counts": cv.label_counts(y),
        "feature_timing": timing,
    }
    split_summary_path = report_dir / f"{run_slug}_stress_split_summary.json"
    write_json(split_summary_path, split_summary)

    base_specs = cv.make_candidates(args.random_state)
    stress_specs = make_stress_candidates()
    leaderboard_rows = []
    fold_rows = []
    oof_by_candidate = {}
    for spec in list(stress_specs.values()):
        row, oof_by_view, candidate_fold_rows = evaluate_stress_candidate(
            spec,
            base_specs,
            stress_specs,
            X_by_view,
            y,
            splits,
        )
        leaderboard_rows.append(row)
        fold_rows.extend(candidate_fold_rows)
        oof_by_candidate[spec.name] = oof_by_view

    initial_leaderboard = pd.DataFrame(leaderboard_rows)
    ensemble_specs = build_ensemble_candidates(initial_leaderboard, stress_specs)
    stress_specs.update(ensemble_specs)
    for spec in ensemble_specs.values():
        row, oof_by_view, candidate_fold_rows = evaluate_stress_candidate(
            spec,
            base_specs,
            stress_specs,
            X_by_view,
            y,
            splits,
        )
        leaderboard_rows.append(row)
        fold_rows.extend(candidate_fold_rows)
        oof_by_candidate[spec.name] = oof_by_view

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    ).reset_index(drop=True)
    fold_report = pd.DataFrame(fold_rows)
    leaderboard_path = report_dir / f"{run_slug}_oof_stress_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    fold_report.to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = stress_specs[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_kind": selected_spec.kind,
        "selected_base_name": selected_spec.base_name,
        "selected_train_views": list(selected_spec.train_views),
        "selected_members": list(selected_spec.members),
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float))

    test_csv = base.require_file(
        root_path / "audio_visual_dataset_robo_default" / "dataset.csv",
        "robot dataset.csv",
    )
    test_df = base.load_manifest(test_csv, "robot_test")
    test_feat, test_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )

    start = time.perf_counter()
    full_idx = np.arange(len(y))
    final_artifact = fit_stress_candidate(
        selected_spec,
        base_specs,
        stress_specs,
        X_by_view,
        y,
        full_idx,
    )
    final_fit_time = time.perf_counter() - start
    final_proba = predict_stress_artifact(final_artifact, test_feat["X"])
    final_pred = cv.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_feat["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_grouped_stress_cv_only",
            "selected_stress_worst_macro_f1": selected["stress_worst_macro_f1"],
            "selected_stress_mean_macro_f1": selected["stress_mean_macro_f1"],
            "selected_clean_oof_macro_f1": selected["tuned_clean_macro_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )

    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]
    ].copy()
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
            "config": base.CONFIG,
            "feature_set": base.FEATURE_SET,
            "feature_spec": base.FEATURE_SPEC,
            "feature_names": base.FEATURE_NAMES,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "clean_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "clean_feature_cache_dir": str(args.clean_feature_cache_dir.resolve()),
        "stress_feature_dir": str(stress_feature_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "stress_leaderboard": str(leaderboard_path.resolve()),
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

    print("\nFinal robot/test result after frozen stress-CV selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_stress_worst_macro_f1",
                "selected_clean_oof_macro_f1",
            ]
        ].to_string(index=False)
    )
    print("\nSaved artifacts:")
    print("Stress leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
