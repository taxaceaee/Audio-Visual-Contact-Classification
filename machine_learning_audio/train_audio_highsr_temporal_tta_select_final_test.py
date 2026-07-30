from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier

import train_audio_multifeature_tta_grid_ensemble_select_final_test as mf_ens
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")
FEATURE_FAMILY = "audio_highsr_temporal_texture_v1"
TARGET_SR = 44100
TARGET_LEN = TARGET_SR
EXPECTED_DIM: int | None = None


@dataclass(frozen=True)
class Candidate:
    name: str
    train_views: tuple[str, ...]
    model_factory: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only high-sample-rate temporal texture features with train-only "
            "OOF TTA selection. Robot/test is loaded only after the selection lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--run-slug", default="audio_highsr_temporal_tta_select")
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Optional shared high-SR feature cache directory.",
    )
    parser.add_argument("--weight-step", type=int, default=5)
    parser.add_argument(
        "--candidate-names",
        nargs="*",
        default=None,
        help="Optional subset of high-SR candidates to evaluate, chosen before robot/test is loaded.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def manifest_signature(frame: pd.DataFrame, view: str) -> str:
    digest = hashlib.sha256()
    for row in frame[["audio_path", "y"]].itertuples(index=False):
        digest.update(f"{FEATURE_FAMILY}|{view}|{row.audio_path}|{int(row.y)}\n".encode("utf-8"))
    return digest.hexdigest()


def load_audio_native(path: Path) -> tuple[np.ndarray, int]:
    signal, sr = sf.read(path, always_2d=False)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        sr = TARGET_SR
    if len(signal) < TARGET_LEN:
        signal = np.pad(signal, (0, TARGET_LEN - len(signal)))
    else:
        signal = signal[:TARGET_LEN]
    return signal.astype(np.float32), sr


def normalize_peak(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) + 1e-8
    return (signal / peak).astype(np.float32)


def top_energy_window(signal: np.ndarray, sr: int, window_sec: float = 0.45, hop_sec: float = 0.025) -> np.ndarray:
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    if len(signal) <= window_len:
        return signal
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(signal) - window_len + 1, hop_len):
        chunk = signal[start : start + window_len]
        energy = float(np.mean(chunk * chunk))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return signal[best_start : best_start + window_len]


def stats(values: np.ndarray, quantiles: tuple[float, ...] = (10, 25, 50, 75, 90)) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return np.zeros(4 + len(quantiles), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    out = [float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values))]
    out.extend(float(np.percentile(values, q)) for q in quantiles)
    return np.asarray(out, dtype=np.float32)


def frame_rms(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(signal) < frame_len:
        return np.asarray([float(np.sqrt(np.mean(signal**2) + 1e-12))], dtype=np.float32)
    starts = np.arange(0, len(signal) - frame_len + 1, hop)
    return np.asarray(
        [float(np.sqrt(np.mean(signal[start : start + frame_len] ** 2) + 1e-12)) for start in starts],
        dtype=np.float32,
    )


def temporal_pool(matrix: np.ndarray, n_freq_groups: int, n_time_bins: int) -> np.ndarray:
    freq_groups = np.array_split(matrix, n_freq_groups, axis=0)
    parts = []
    for group in freq_groups:
        series = np.mean(group, axis=0)
        parts.extend(stats(series, quantiles=(10, 50, 90)))
    for time_part in np.array_split(matrix, n_time_bins, axis=1):
        grouped = [float(np.mean(freq_group)) for freq_group in np.array_split(time_part, n_freq_groups, axis=0)]
        parts.extend(grouped)
    return np.asarray(parts, dtype=np.float32)


def modulation_summary(series: np.ndarray, frame_hop_sec: float) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    if series.size < 4:
        return np.zeros(14, dtype=np.float32)
    series = series - np.mean(series)
    spectrum = np.abs(np.fft.rfft(series)) ** 2
    freqs = np.fft.rfftfreq(len(series), d=frame_hop_sec)
    total = float(np.sum(spectrum)) + 1e-12
    output = []
    for low, high in [(0.5, 4), (4, 10), (10, 20), (20, 40), (40, 80), (80, 160)]:
        mask = (freqs >= low) & (freqs < high)
        value = float(np.sum(spectrum[mask]) / total)
        output.extend([value, np.log1p(value)])
    output.extend([float(freqs[int(np.argmax(spectrum))]), float(np.max(spectrum) / total)])
    return np.asarray(output, dtype=np.float32)


def spectral_band_features(signal: np.ndarray, sr: int) -> np.ndarray:
    epsilon = 1e-12
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    total = float(np.sum(spectrum)) + epsilon
    bands = [
        (20, 80),
        (80, 160),
        (160, 315),
        (315, 630),
        (630, 1000),
        (1000, 1600),
        (1600, 2500),
        (2500, 4000),
        (4000, 6300),
        (6300, 9000),
        (9000, 12500),
        (12500, 17000),
        (17000, 22000),
    ]
    powers = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        powers.append(float(np.sum(spectrum[mask]) / total))
    powers = np.asarray(powers, dtype=np.float32)
    low = float(np.sum(powers[:4]))
    mid = float(np.sum(powers[4:9]))
    high = float(np.sum(powers[9:]))
    centroid = float(np.sum(freqs * spectrum) / total)
    dominant = float(freqs[int(np.argmax(spectrum))])
    entropy = float(-np.sum((spectrum / total) * np.log((spectrum / total) + epsilon)) / np.log(len(spectrum)))
    return np.concatenate(
        [
            powers,
            np.log1p(powers),
            np.asarray(
                [
                    low / (mid + 1e-8),
                    low / (high + 1e-8),
                    mid / (high + 1e-8),
                    (low + mid) / (high + 1e-8),
                    centroid,
                    dominant,
                    entropy,
                ],
                dtype=np.float32,
            ),
        ]
    )


def extract_texture_block(signal: np.ndarray, sr: int) -> np.ndarray:
    signal = signal.astype(np.float32)
    signal = signal - float(np.mean(signal))
    peak = float(np.max(np.abs(signal))) + 1e-8
    raw_rms = float(np.sqrt(np.mean(signal**2) + 1e-12))
    signal = signal / peak

    n_fft = 2048
    hop = 441
    win = 1764
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=96,
        fmin=20,
        fmax=sr / 2,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, sr=sr, n_mfcc=32)
    delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=n_fft, hop_length=hop, fmin=80.0)
    rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    rolloff85 = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)[0]
    rolloff95 = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(S=logmel, sr=sr, hop_length=hop)

    parts = [
        np.asarray(
            [
                peak,
                raw_rms,
                raw_rms / (peak + 1e-8),
                float(np.mean(np.abs(signal))),
                float(np.percentile(np.abs(signal), 95)),
            ],
            dtype=np.float32,
        ),
        spectral_band_features(signal, sr),
        temporal_pool(logmel, n_freq_groups=12, n_time_bins=10),
        temporal_pool(np.maximum(logmel, -80.0), n_freq_groups=8, n_time_bins=6),
        np.concatenate([stats(row, quantiles=(10, 50, 90)) for row in mfcc]).astype(np.float32),
        np.concatenate([stats(row, quantiles=(50,)) for row in delta]).astype(np.float32),
        np.concatenate([stats(row, quantiles=(10, 50, 90)) for row in contrast]).astype(np.float32),
        np.concatenate(
            [
                stats(rms, quantiles=(10, 50, 90)),
                stats(zcr, quantiles=(10, 50, 90)),
                stats(centroid, quantiles=(10, 50, 90)),
                stats(bandwidth, quantiles=(10, 50, 90)),
                stats(rolloff85, quantiles=(10, 50, 90)),
                stats(rolloff95, quantiles=(10, 50, 90)),
                stats(flatness, quantiles=(10, 50, 90)),
                stats(onset, quantiles=(10, 50, 90)),
                modulation_summary(rms, frame_hop_sec=hop / sr),
                modulation_summary(onset, frame_hop_sec=hop / sr),
            ]
        ).astype(np.float32),
    ]
    output = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Texture block contains NaN/Inf")
    return output


def extract_highsr_features(signal: np.ndarray, sr: int) -> np.ndarray:
    full = extract_texture_block(signal, sr)
    top = extract_texture_block(top_energy_window(signal, sr), sr)
    coarse = librosa.resample(normalize_peak(signal), orig_sr=sr, target_sr=16000).astype(np.float32)
    coarse_features = np.concatenate(
        [
            base.extract_total_120(coarse, 16000),
            base.extract_total_120(base.get_top_energy_window(coarse, sr=16000), 16000),
        ]
    ).astype(np.float32)
    output = np.concatenate([full, top, coarse_features]).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("High-SR feature contains NaN/Inf")
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
            print(f"Loaded {FEATURE_FAMILY}/{split_name}/{view}: {payload['X'].shape} in {load_time:.3f}s", flush=True)
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
            }
        print(f"Cache metadata mismatch for {split_name}/{view}; rebuilding.", flush=True)

    def extract_one(audio_path: str, label: int) -> tuple[np.ndarray, int, str]:
        signal, sr = load_audio_native(Path(audio_path))
        if view != "clean":
            signal = stress.apply_stress_view(signal, view=view, key=str(audio_path), sr=sr)
        features = extract_highsr_features(signal, sr)
        return features, int(label), str(audio_path)

    jobs = [(str(row.audio_path), int(row.y)) for row in frame.itertuples(index=False)]
    start = time.perf_counter()
    if n_jobs == 1:
        results = [extract_one(audio_path, label) for audio_path, label in jobs]
    else:
        print(f"Parallel extract {FEATURE_FAMILY}/{split_name}/{view}: n_jobs={n_jobs} samples={len(jobs)}", flush=True)
        results = joblib.Parallel(n_jobs=n_jobs, batch_size=16, verbose=5)(
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
    if not np.isfinite(payload["X"]).all():
        raise ValueError(f"{split_name}/{view}: feature matrix contains NaN/Inf")

    global EXPECTED_DIM
    if EXPECTED_DIM is None:
        EXPECTED_DIM = int(payload["X"].shape[1])
    elif int(payload["X"].shape[1]) != EXPECTED_DIM:
        raise AssertionError(f"Feature dim mismatch: expected {EXPECTED_DIM}, got {payload['X'].shape[1]}")

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
    print(f"Saved {FEATURE_FAMILY}/{split_name}/{view}: {payload['X'].shape} in {extraction_time:.2f}s", flush=True)
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
    }


def make_weight_grid(denominator: int) -> dict[str, dict[str, float]]:
    if denominator < 2:
        raise ValueError("--weight-step must be >= 2")
    recipes = {}
    for units in itertools.product(range(denominator + 1), repeat=len(STRESS_VIEWS)):
        if sum(units) != denominator or sum(units) == 0:
            continue
        weights = {
            view: unit / denominator
            for view, unit in zip(STRESS_VIEWS, units)
            if unit > 0
        }
        name = "w_" + "_".join(f"{view[:1]}{unit:02d}" for view, unit in zip(STRESS_VIEWS, units) if unit > 0)
        recipes[name] = weights
    return recipes


def weighted_proba(proba_by_view: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = float(sum(weights.values()))
    output = None
    for view, weight in weights.items():
        part = (float(weight) / total) * proba_by_view[view]
        output = part if output is None else output + part
    if output is None:
        raise RuntimeError("Empty TTA weights")
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(proba + bias.reshape(1, -1), axis=1).astype(np.int64)


def score_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = mf_ens.fast_macro_f1(y_true, pred, LABELS)
    contact = mf_ens.fast_macro_f1(y_true, pred, CONTACT_LABELS)
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def fold_scores(y_true: np.ndarray, proba: np.ndarray, bias: np.ndarray, fold_assignment: np.ndarray) -> dict[str, float]:
    hybrids = []
    macros = []
    contacts = []
    for fold_id in sorted(set(fold_assignment.tolist())):
        mask = fold_assignment == fold_id
        scores = score_pred(y_true[mask], predict_with_bias(proba[mask], bias))
        macros.append(scores["macro_f1"])
        contacts.append(scores["contact_macro_f1"])
        hybrids.append(scores["hybrid_macro_contact"])
    return {
        "fold_mean_macro_f1": float(np.mean(macros)),
        "fold_mean_contact_macro_f1": float(np.mean(contacts)),
        "fold_mean_hybrid_macro_contact": float(np.mean(hybrids)),
        "fold_std_hybrid_macro_contact": float(np.std(hybrids)),
        "worst_fold_hybrid_macro_contact": float(np.min(hybrids)),
    }


def selection_score(row: dict[str, float]) -> float:
    return float(
        0.50 * row["hybrid_macro_contact"]
        + 0.25 * row["worst_single_view_hybrid"]
        + 0.25 * row["worst_fold_hybrid_macro_contact"]
        - 0.6 * row["fold_std_hybrid_macro_contact"]
    )


def bias_candidates() -> list[np.ndarray]:
    candidates = [
        [0.0, 0.0, 0.0, 0.0],
        [-0.2, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.2],
        [0.0, 0.4, 0.0, 0.4],
        [0.2, 0.4, 0.0, 0.4],
        [-0.2, 0.4, 0.0, 0.4],
        [0.0, 0.0, -0.2, 0.0],
        [0.0, 0.2, -0.2, 0.2],
        [0.0, 0.4, -0.4, 0.4],
        [0.2, 0.8, 0.4, 0.8],
        [0.0, 0.8, 0.4, 0.8],
        [0.3, 0.7, 0.2, 0.7],
        [0.4, 0.6, 0.0, 0.6],
    ]
    return [np.asarray(values, dtype=np.float64) for values in candidates]


def evaluate_proba(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    proba_by_view: dict[str, np.ndarray],
    tta_proba: np.ndarray,
    bias: np.ndarray,
) -> dict[str, float]:
    row = score_pred(y, predict_with_bias(tta_proba, bias))
    for view in STRESS_VIEWS:
        scores = score_pred(y, predict_with_bias(proba_by_view[view], bias))
        for metric, value in scores.items():
            row[f"{view}_{metric}"] = value
    row["worst_single_view_hybrid"] = min(row[f"{view}_hybrid_macro_contact"] for view in STRESS_VIEWS)
    row.update(fold_scores(y, tta_proba, bias, fold_assignment))
    row["selection_score"] = selection_score(row)
    return row


def tune_bias_for_selection(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    proba_by_view: dict[str, np.ndarray],
    tta_proba: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    best_bias = np.zeros(4, dtype=np.float64)
    best_row: dict[str, float] | None = None
    for bias in bias_candidates():
        row = evaluate_proba(y, fold_assignment, proba_by_view, tta_proba, bias)
        if best_row is None or (
            row["selection_score"],
            row["worst_single_view_hybrid"],
            row["worst_fold_hybrid_macro_contact"],
            row["macro_f1"],
        ) > (
            best_row["selection_score"],
            best_row["worst_single_view_hybrid"],
            best_row["worst_fold_hybrid_macro_contact"],
            best_row["macro_f1"],
        ):
            best_row = row
            best_bias = bias
    if best_row is None:
        raise RuntimeError("No bias candidate scored")
    return best_bias, best_row


def make_candidates(random_state: int) -> dict[str, Candidate]:
    class_weights = {0: 0.5, 1: 1.2, 2: 1.8, 3: 1.4}
    return {
        "highsr_hgb_default__clean": Candidate(
            "highsr_hgb_default__clean",
            ("clean",),
            lambda: HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.045,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.02,
                class_weight=class_weights,
                random_state=random_state,
            ),
        ),
        "highsr_hgb_default__all_aug": Candidate(
            "highsr_hgb_default__all_aug",
            STRESS_VIEWS,
            lambda: HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.045,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.02,
                class_weight=class_weights,
                random_state=random_state,
            ),
        ),
        "highsr_hgb_regularized__all_aug": Candidate(
            "highsr_hgb_regularized__all_aug",
            STRESS_VIEWS,
            lambda: HistGradientBoostingClassifier(
                max_iter=340,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=35,
                l2_regularization=0.18,
                class_weight=class_weights,
                random_state=random_state,
            ),
        ),
        "highsr_lgbm__all_aug": Candidate(
            "highsr_lgbm__all_aug",
            STRESS_VIEWS,
            lambda: LGBMClassifier(
                objective="multiclass",
                num_class=4,
                n_estimators=520,
                learning_rate=0.025,
                num_leaves=21,
                min_child_samples=28,
                subsample=0.85,
                colsample_bytree=0.70,
                reg_lambda=2.5,
                class_weight=class_weights,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        "highsr_extratrees__all_aug": Candidate(
            "highsr_extratrees__all_aug",
            STRESS_VIEWS,
            lambda: ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight=class_weights,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    }


def training_matrix(
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    train_views: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([X_by_view[view][train_idx] for view in train_views]),
        np.concatenate([y[train_idx] for _ in train_views]),
    )


def fit_candidate(candidate: Candidate, X_by_view: dict[str, np.ndarray], y: np.ndarray, train_idx: np.ndarray) -> object:
    X_train, y_train = training_matrix(X_by_view, y, train_idx, candidate.train_views)
    model = candidate.model_factory()
    model.fit(X_train, y_train)
    return model


def proba_aligned(model: object, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    output = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        output[:, int(np.where(LABELS == class_id)[0][0])] = raw[:, column]
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def evaluate_candidate(
    candidate: Candidate,
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fold_assignment: np.ndarray,
    weight_grid: dict[str, dict[str, float]],
    existing_oof_by_view: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    print(f"\nHigh-SR candidate: {candidate.name} train_views={candidate.train_views}", flush=True)
    fold_rows = []
    if existing_oof_by_view is not None:
        oof_by_view = existing_oof_by_view
        print("  loaded complete saved OOF proba; skipping CV refit", flush=True)
    else:
        oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            start = time.perf_counter()
            model = fit_candidate(candidate, X_by_view, y, train_idx)
            train_time = time.perf_counter() - start
            parts = []
            for view in STRESS_VIEWS:
                oof_by_view[view][val_idx] = proba_aligned(model, X_by_view[view][val_idx])
                scores = score_pred(y[val_idx], np.argmax(oof_by_view[view][val_idx], axis=1))
                parts.append(f"{view}=M{scores['macro_f1']:.4f}/C{scores['contact_macro_f1']:.4f}")
            fold_rows.append(
                {
                    "candidate": candidate.name,
                    "fold": fold_id,
                    "train_time_sec": train_time,
                    "val_samples": int(len(val_idx)),
                }
            )
            print(f"  fold {fold_id}: " + " | ".join(parts) + f" time={train_time:.2f}s", flush=True)

    rows = []
    for recipe_name, weights in weight_grid.items():
        tta_proba = weighted_proba(oof_by_view, weights)
        bias, metrics = tune_bias_for_selection(y, fold_assignment, oof_by_view, tta_proba)
        pred = predict_with_bias(tta_proba, bias)
        row = base.make_report_row(
            model_name=f"{candidate.name}__{recipe_name}__bias",
            split_name="highsr_temporal_tta_oof_cv",
            y_true=y,
            y_pred=pred,
            train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
            predict_time_sec=0.0,
        )
        row.update(
            {
                "feature_family": FEATURE_FAMILY,
                "candidate": candidate.name,
                "train_views_json": json.dumps(list(candidate.train_views)),
                "tta_recipe": recipe_name,
                "tta_weights_json": json.dumps(weights),
                "class_bias_json": json.dumps(bias.tolist()),
                "tta_macro_f1": metrics["macro_f1"],
                "tta_contact_macro_f1": metrics["contact_macro_f1"],
                "tta_hybrid_macro_contact": metrics["hybrid_macro_contact"],
                "selection_score": metrics["selection_score"],
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"macro_f1", "contact_macro_f1", "hybrid_macro_contact", "selection_score"}
                },
            }
        )
        rows.append(row)
    best = max(rows, key=lambda item: item["selection_score"])
    print(
        f"  best grid: {best['tta_recipe']} selection={best['selection_score']:.4f} "
        f"macro={best['tta_macro_f1']:.4f} contact={best['tta_contact_macro_f1']:.4f} "
        f"worst_view={best['worst_single_view_hybrid']:.4f} bias={best['class_bias_json']}",
        flush=True,
    )
    return rows, oof_by_view, fold_rows


def load_saved_oof(candidate_dir: Path, n_samples: int) -> dict[str, np.ndarray] | None:
    paths = {view: candidate_dir / f"{view}_oof_proba.npy" for view in STRESS_VIEWS}
    if not all(path.exists() for path in paths.values()):
        return None
    output = {view: np.load(path) for view, path in paths.items()}
    if any(proba.shape != (n_samples, len(LABELS)) for proba in output.values()):
        return None
    if any(not np.isfinite(proba).all() for proba in output.values()):
        return None
    return output


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = args.run_slug
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    feature_dir = args.feature_cache_dir if args.feature_cache_dir is not None else run_dir / "features"
    oof_dir = run_dir / "oof_proba"
    for directory in [run_dir, report_dir, model_dir, split_dir, feature_dir, oof_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH      =", root_path.resolve())
    print("RUN_DIR        =", run_dir.resolve())
    print("FEATURE_FAMILY =", FEATURE_FAMILY)
    print("TARGET_SR      =", TARGET_SR)
    print("Selection      = train-only grouped OOF high-SR temporal TTA")

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

    feature_dim = int(X_by_view["clean"].shape[1])
    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_highsr_temporal_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    weight_grid = make_weight_grid(args.weight_step)
    split_summary = {
        "protocol": "audio-only high-SR temporal TTA; robot/test after selection lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "feature_family": FEATURE_FAMILY,
        "target_sr": TARGET_SR,
        "feature_dim": feature_dim,
        "n_tta_weight_recipes": len(weight_grid),
        "feature_timing": timings,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    leaderboard_rows = []
    fold_rows = []
    candidates = make_candidates(args.random_state)
    if args.candidate_names:
        missing = sorted(set(args.candidate_names) - set(candidates))
        if missing:
            raise KeyError(f"Unknown candidate(s): {missing}. Available: {sorted(candidates)}")
        candidates = {name: candidates[name] for name in args.candidate_names}
    for candidate in candidates.values():
        candidate_dir = oof_dir / candidate.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        existing_oof = None if args.force_rebuild else load_saved_oof(candidate_dir, len(y))
        rows, oof_by_view, candidate_fold_rows = evaluate_candidate(
            candidate,
            X_by_view,
            y,
            splits,
            fold_assignment,
            weight_grid,
            existing_oof_by_view=existing_oof,
        )
        leaderboard_rows.extend(rows)
        fold_rows.extend(candidate_fold_rows)
        for view, proba in oof_by_view.items():
            np.save(candidate_dir / f"{view}_oof_proba.npy", proba)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["selection_score", "worst_single_view_hybrid", "worst_fold_hybrid_macro_contact", "tta_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_tta_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_candidate = candidates[str(selected["candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selected_tta_weights = json.loads(selected["tta_weights_json"])
    selection_summary = {
        "selection_rule": "highest train-only robust high-SR temporal OOF TTA score",
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_candidate": selected["candidate"],
        "selected_train_views": list(selected_candidate.train_views),
        "selected_tta_recipe": selected["tta_recipe"],
        "selected_tta_weights": selected_tta_weights,
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
    test_payloads = {}
    test_timings = {}
    for view in STRESS_VIEWS:
        payload, timing = build_or_load_feature_cache(
            test_df,
            split_name="robot_test",
            view=view,
            feature_dir=feature_dir,
            force_rebuild=args.force_rebuild,
            n_jobs=args.n_jobs,
        )
        test_payloads[view] = payload
        test_timings[view] = timing
    X_test_by_view = {view: test_payloads[view]["X"] for view in STRESS_VIEWS}

    start = time.perf_counter()
    final_model = fit_candidate(selected_candidate, X_by_view, y, np.arange(len(y)))
    final_fit_time = time.perf_counter() - start
    final_proba_by_view = {view: proba_aligned(final_model, X_test_by_view[view]) for view in STRESS_VIEWS}
    final_proba = weighted_proba(final_proba_by_view, selected_tta_weights)
    final_pred = predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=test_payloads["clean"]["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_highsr_temporal_oof_tta",
            "feature_family": FEATURE_FAMILY,
            "target_sr": TARGET_SR,
            "feature_dim": feature_dim,
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["tta_macro_f1"],
            "selected_oof_contact_macro_f1": selected["tta_contact_macro_f1"],
            "selected_tta_weights_json": json.dumps(selected_tta_weights),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    audio_columns = ["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]
    prediction_frame = test_df[audio_columns].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_payloads["clean"]["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_highsr_temporal_tta_no_test_until_lock",
            "feature_family": FEATURE_FAMILY,
            "target_sr": TARGET_SR,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_model": final_model,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "audio_only_highsr_temporal_tta_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timings,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen high-SR temporal TTA selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
                "selected_oof_macro_f1",
                "selected_oof_contact_macro_f1",
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
