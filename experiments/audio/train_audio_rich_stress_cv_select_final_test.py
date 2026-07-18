from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import librosa
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit", "robot_hard")
FEATURE_FAMILY = "audio_rich_fast_v1"


@dataclass(frozen=True)
class RichCandidate:
    name: str
    kind: str
    train_views: tuple[str, ...] = ("clean",)
    factory: Callable[[], object] | None = None
    members: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only rich handcrafted features with train-only stress-CV selection. "
            "Robot/test is loaded only after the selected model and bias are locked."
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


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def normalize_peak(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) + 1e-8
    return (signal / peak).astype(np.float32)


def add_noise_at_snr(signal: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(signal**2))) + 1e-8
    noise_rms = rms / (10 ** (snr_db / 20.0))
    return (signal + rng.normal(0.0, noise_rms, size=signal.shape)).astype(np.float32)


def apply_audio_view(signal: np.ndarray, view: str, key: str, sr: int) -> np.ndarray:
    if view in ("clean", "robot_mix", "bandlimit"):
        return stress.apply_stress_view(signal, view=view, key=key, sr=sr)

    rng = np.random.default_rng(stable_seed(f"{view}|{key}"))
    output = signal.astype(np.float32).copy()

    if view == "robot_hard":
        output = stress.fft_bandlimit(output, sr, low_hz=140.0, high_hz=4400.0)
        output = output * float(rng.uniform(0.55, 1.45))
        output = add_noise_at_snr(output, rng, snr_db=float(rng.uniform(14.0, 24.0)))
        drive = float(rng.uniform(1.25, 2.1))
        output = np.tanh(drive * output) / np.tanh(drive)
        output = scipy.signal.lfilter([1.0, -0.93], [1.0], output).astype(np.float32)
        shift = int(rng.integers(-1800, 1801))
        return normalize_peak(np.roll(output, shift))

    if view == "thin_band":
        low = float(rng.uniform(220.0, 420.0))
        high = float(rng.uniform(2600.0, 4600.0))
        output = stress.fft_bandlimit(output, sr, low_hz=low, high_hz=high)
        output = add_noise_at_snr(output, rng, snr_db=float(rng.uniform(20.0, 32.0)))
        return normalize_peak(output)

    raise ValueError(f"Unknown audio view: {view}")


def summary_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return np.zeros(9, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(
        [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 75)),
            float(np.percentile(values, 90)),
            float(np.max(values)),
        ],
        dtype=np.float32,
    )


def distribution_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 3:
        return np.zeros(4, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.asarray(
        [
            float(scipy.stats.skew(values, bias=False)),
            float(scipy.stats.kurtosis(values, fisher=True, bias=False)),
            float(np.mean(np.abs(values))),
            float(np.sqrt(np.mean(values**2))),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def bandpower_features(signal: np.ndarray, sr: int) -> np.ndarray:
    freqs, power = scipy.signal.welch(
        signal,
        fs=sr,
        nperseg=512,
        noverlap=256,
        scaling="spectrum",
    )
    total = float(np.sum(power)) + 1e-12
    bands = [
        (20, 80),
        (80, 160),
        (160, 320),
        (320, 640),
        (640, 1250),
        (1250, 2500),
        (2500, 4000),
        (4000, 6000),
        (6000, 8000),
    ]
    values = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band = float(np.sum(power[mask]) / total)
        values.append(band)
        values.append(float(np.log1p(band)))
    arr = np.asarray(values, dtype=np.float32)
    low = float(arr[0] + arr[2] + arr[4])
    mid = float(arr[6] + arr[8] + arr[10])
    high = float(arr[12] + arr[14] + arr[16])
    ratios = np.asarray(
        [
            low / (mid + 1e-8),
            low / (high + 1e-8),
            mid / (high + 1e-8),
            (low + mid) / (high + 1e-8),
        ],
        dtype=np.float32,
    )
    dominant = freqs[int(np.argmax(power))]
    centroid = float(np.sum(freqs * power) / total)
    entropy = float(-np.sum((power / total) * np.log(power / total + 1e-12)) / np.log(len(power)))
    return np.concatenate([arr, ratios, np.asarray([dominant, centroid, entropy], dtype=np.float32)])


def temporal_chunk_features(signal: np.ndarray, sr: int) -> np.ndarray:
    features = []
    for chunk in np.array_split(signal, 4):
        envelope = np.abs(scipy.signal.hilbert(chunk))
        features.extend(summary_stats(chunk))
        features.extend(distribution_stats(chunk))
        features.extend(summary_stats(envelope))
        features.extend(bandpower_features(chunk, sr))
    return np.asarray(features, dtype=np.float32)


def librosa_stream_features(signal: np.ndarray, sr: int) -> np.ndarray:
    stft_cfg = base.CONFIG["stft_base"]
    hop = stft_cfg["hop_length"]
    n_fft = stft_cfg["n_fft"]
    magnitude = np.abs(
        librosa.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop,
            win_length=stft_cfg["win_length"],
            window=stft_cfg["window"],
        )
    )
    power = magnitude**2

    streams = [
        librosa.feature.rms(S=magnitude, frame_length=n_fft, hop_length=hop)[0],
        librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0],
        librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0],
        librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.50)[0],
        librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.85)[0],
        librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.95)[0],
        librosa.feature.spectral_flatness(S=magnitude)[0],
    ]
    normalized = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-8)
    flux = np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0))
    streams.append(np.pad(flux, (1, 0)))

    features = []
    for stream in streams:
        features.extend(summary_stats(stream))
        features.extend(distribution_stats(stream))

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=80, n_fft=n_fft, hop_length=hop), ref=np.max),
        n_mfcc=30,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for matrix in (mfcc, delta, delta2):
        features.extend(np.mean(matrix, axis=1))
        features.extend(np.std(matrix, axis=1))
        features.extend(np.percentile(matrix, 10, axis=1))
        features.extend(np.percentile(matrix, 50, axis=1))
        features.extend(np.percentile(matrix, 90, axis=1))

    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop),
        ref=np.max,
    )
    for group in np.array_split(mel, 8, axis=0):
        stream = np.mean(group, axis=0)
        features.extend(summary_stats(stream))

    contrast = librosa.feature.spectral_contrast(S=magnitude + 1e-8, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))
    features.extend(np.percentile(contrast, 10, axis=1))
    features.extend(np.percentile(contrast, 90, axis=1))

    return np.asarray(features, dtype=np.float32)


def extract_rich_audio_features(signal: np.ndarray, sr: int) -> np.ndarray:
    crop_cfg = base.CONFIG["event_crop"]
    top_signal = base.get_top_energy_window(
        signal,
        sr=sr,
        window_sec=crop_cfg["top_energy_sec"],
        hop_sec=crop_cfg["top_energy_hop_sec"],
    )
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    tempo_features = np.asarray(
        [
            float(np.mean(onset_env)),
            float(np.std(onset_env)),
            float(np.max(onset_env) if onset_env.size else 0.0),
            float(np.argmax(onset_env) / max(1, onset_env.size)),
        ],
        dtype=np.float32,
    )
    autocorr = scipy.signal.correlate(signal, signal, mode="full", method="fft")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr = autocorr[: min(len(autocorr), sr // 2)]
    autocorr = autocorr / (np.max(np.abs(autocorr)) + 1e-8)
    features = np.concatenate(
        [
            base.extract_total_120(signal, sr),
            base.extract_total_120(top_signal, sr),
            temporal_chunk_features(signal, sr),
            bandpower_features(top_signal, sr),
            summary_stats(np.abs(scipy.signal.hilbert(signal))),
            summary_stats(autocorr),
            tempo_features,
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Rich audio feature contains NaN/Inf")
    return features


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
        signal = apply_audio_view(signal, view=view, key=str(audio_path), sr=sr)
        return extract_rich_audio_features(signal, sr=sr), int(label), str(audio_path)

    start = time.perf_counter()
    jobs = [(str(row.audio_path), int(row.y)) for row in frame.itertuples(index=False)]
    if n_jobs == 1:
        results = [
            extract_one(audio_path, label)
            for audio_path, label in tqdm(jobs, total=len(jobs), desc=f"Extract {FEATURE_FAMILY}/{split_name}/{view}")
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


def build_training_matrix(
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    train_views: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([X_by_view[view][train_idx] for view in train_views]),
        np.concatenate([y[train_idx] for _ in train_views]),
    )


def make_candidates(random_state: int) -> dict[str, RichCandidate]:
    class_weights = base.CONFIG["class_weights"]

    def lgbm(num_leaves: int = 31, min_child_samples: int = 24, reg_lambda: float = 2.0):
        return LGBMClassifier(
            objective="multiclass",
            num_class=len(base.CLASS_NAMES),
            n_estimators=760,
            learning_rate=0.025,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.86,
            colsample_bytree=0.78,
            reg_lambda=reg_lambda,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def hgb(max_leaf_nodes: int = 15, min_samples_leaf: int = 24):
        return HistGradientBoostingClassifier(
            max_iter=420,
            learning_rate=0.03,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=0.12,
            class_weight=class_weights,
            random_state=random_state,
        )

    specs = [
        RichCandidate("rich_lgbm_all_aug", "single", STRESS_VIEWS, lambda: lgbm(31, 24, 2.0)),
    ]
    return {spec.name: spec for spec in specs}


def fit_candidate(
    spec: RichCandidate,
    candidates: dict[str, RichCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
) -> object:
    if spec.kind == "single":
        X_train, y_train = build_training_matrix(X_by_view, y, train_idx, spec.train_views)
        model = spec.factory()
        model.fit(X_train, y_train)
        return {"kind": "single", "model": model}
    if spec.kind == "ensemble":
        return {
            "kind": "ensemble",
            "members": {
                member: fit_candidate(candidates[member], candidates, X_by_view, y, train_idx)
                for member in spec.members
            },
        }
    raise ValueError(f"Unsupported kind: {spec.kind}")


def predict_artifact(artifact: object, X: np.ndarray) -> np.ndarray:
    if artifact["kind"] == "single":
        return cv.proba_aligned(artifact["model"], X)
    if artifact["kind"] == "ensemble":
        return np.mean([predict_artifact(member, X) for member in artifact["members"].values()], axis=0)
    raise ValueError(f"Unsupported artifact kind: {artifact['kind']}")


def tune_bias_multiview(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if grid is None:
        grid = np.linspace(-1.4, 1.4, 15)

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


def evaluate_candidate(
    spec: RichCandidate,
    candidates: dict[str, RichCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, list[dict]]:
    print(f"\nRich audio candidate: {spec.name} train_views={spec.train_views or spec.members}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = fit_candidate(spec, candidates, X_by_view, y, train_idx)
        train_time = time.perf_counter() - start
        view_scores = {}
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = predict_artifact(artifact, X_by_view[view][val_idx])
            pred = oof_by_view[view][val_idx].argmax(axis=1)
            view_scores[f"{view}_macro_f1"] = f1_score(y[val_idx], pred, average="macro")
        fold_row = {
            "candidate": spec.name,
            "fold": fold_id,
            "train_time_sec": train_time,
            "val_samples": int(len(val_idx)),
            **view_scores,
            "fold_worst_macro_f1": min(view_scores.values()),
        }
        fold_rows.append(fold_row)
        print(
            f"  fold {fold_id}: "
            + " | ".join(f"{view}={view_scores[f'{view}_macro_f1']:.4f}" for view in STRESS_VIEWS)
            + f" | worst={fold_row['fold_worst_macro_f1']:.4f} time={train_time:.2f}s",
            flush=True,
        )

    bias, tuned_scores = tune_bias_multiview(oof_by_view, y)
    clean_pred = cv.predict_with_bias(oof_by_view["clean"], bias)
    row = base.make_report_row(
        model_name=spec.name,
        split_name="rich_audio_stress_oof_cv",
        y_true=y,
        y_pred=clean_pred,
        train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
        predict_time_sec=0.0,
    )
    row.update(
        {
            "feature_set": FEATURE_FAMILY,
            "feature_name": FEATURE_FAMILY,
            "n_features": int(X_by_view["clean"].shape[1]),
            "candidate_kind": spec.kind,
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
        + f" | worst={tuned_scores['stress_worst_macro_f1']:.4f} bias={np.round(bias, 3).tolist()}",
        flush=True,
    )
    return row, fold_rows


def build_ensemble_candidates(leaderboard: pd.DataFrame) -> dict[str, RichCandidate]:
    return {}
    ranked = leaderboard.sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    )["model"].tolist()
    recipes = {
        "rich_ensemble_top3": tuple(ranked[:3]),
        "rich_ensemble_top5": tuple(ranked[:5]),
    }
    return {
        name: RichCandidate(name=name, kind="ensemble", members=members)
        for name, members in recipes.items()
        if len(members) >= 2
    }


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_rich_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    feature_dir = run_dir / "features"
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, feature_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH   =", root_path.resolve())
    print("RUN_DIR     =", run_dir.resolve())
    print("FEATURES    =", FEATURE_FAMILY)
    print("Selection   = train-only grouped stress-CV; robot/test loaded after selection lock")

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
    train_df.assign(cv_fold=fold_assignment).to_csv(split_dir / "hand_train_full_rich_stress_cv_folds.csv", index=False)

    split_summary = {
        "protocol": "audio-only rich features; train/default grouped stress-CV selection; robot/test after lock",
        "n_train_samples": len(train_df),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": args.n_folds,
        "stress_views": list(STRESS_VIEWS),
        "label_counts": cv.label_counts(y),
        "feature_dim": int(X_by_view["clean"].shape[1]),
        "feature_timing": timings,
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    candidates = make_candidates(args.random_state)
    leaderboard_rows = []
    fold_rows = []
    for spec in list(candidates.values()):
        row, rows = evaluate_candidate(spec, candidates, X_by_view, y, splits)
        leaderboard_rows.append(row)
        fold_rows.extend(rows)

    initial_leaderboard = pd.DataFrame(leaderboard_rows)
    candidates.update(build_ensemble_candidates(initial_leaderboard))
    for spec in [spec for spec in candidates.values() if spec.kind == "ensemble"]:
        row, rows = evaluate_candidate(spec, candidates, X_by_view, y, splits)
        leaderboard_rows.append(row)
        fold_rows.extend(rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_stress_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidates[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_kind": selected_spec.kind,
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

    start = time.perf_counter()
    full_idx = np.arange(len(y))
    final_artifact = fit_candidate(selected_spec, candidates, X_by_view, y, full_idx)
    final_fit_time = time.perf_counter() - start
    final_proba = predict_artifact(final_artifact, test_payload["X"])
    final_pred = cv.predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_payload["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "feature_set": FEATURE_FAMILY,
            "feature_name": FEATURE_FAMILY,
            "n_features": int(X_by_view["clean"].shape[1]),
            "selected_by": "hand_default_grouped_rich_audio_stress_cv_only",
            "selected_stress_worst_macro_f1": selected["stress_worst_macro_f1"],
            "selected_stress_mean_macro_f1": selected["stress_mean_macro_f1"],
            "selected_clean_oof_macro_f1": selected["tuned_clean_macro_f1"],
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
            "protocol": "audio_only_rich_stress_cv_select_no_test_until_final",
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
        "protocol": "audio_only_rich_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "feature_dir": str(feature_dir.resolve()),
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

    print("\nFinal robot/test result after frozen rich audio stress-CV selection:")
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
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Stress leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
