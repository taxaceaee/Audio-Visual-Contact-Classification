from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

LABEL_MAP = {"ambient": 0, "leaf": 1, "trunk": 2, "twig": 3}
ID2LABEL = {value: key for key, value in LABEL_MAP.items()}
CLASS_NAMES = [ID2LABEL[index] for index in range(4)]
CONTACT_LABELS = [1, 2, 3]

CONFIG = {
    "audio": {
        "sr": 16000,
        "duration": 1.0,
        "target_len": 16000,
        "normalize": "peak",
    },
    "event_crop": {
        "top_energy_sec": 0.4,
        "top_energy_hop_sec": 0.05,
    },
    "stft_base": {
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "window": "hann",
    },
    "mfcc": {
        "n_mfcc": 20,
        "n_mels": 64,
        "fmin": 20,
        "fmax": 8000,
    },
    "mel": {
        "n_mels": 64,
        "n_groups": 7,
    },
    "fft": {
        "bands": [
            (20, 100),
            (100, 250),
            (250, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 4000),
            (4000, 6000),
            (6000, 8000),
        ],
    },
    "class_weights": {0: 0.6, 1: 1.1, 2: 1.8, 3: 1.3},
    "random_state": 42,
}

FEATURE_SPECS = {
    "mfcc40": {
        "display_name": "MFCC 40D",
        "dim": 40,
    },
    "stft28": {
        "display_name": "STFT 28D",
        "dim": 28,
    },
    "mel28": {
        "display_name": "Mel groups 28D",
        "dim": 28,
    },
    "fft24": {
        "display_name": "FFT 24D",
        "dim": 24,
    },
    "total120": {
        "display_name": "Total 120D",
        "dim": 120,
    },
    "total240": {
        "display_name": "Total 240D",
        "dim": 240,
    },
}

EXPECTED_MODEL_NAMES = [
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting",
    "LogisticRegression",
    "KNN",
    "DecisionTree",
    "RBF_SVM",
    "LightGBM",
    "XGBoost",
]

FEATURE_SET = "total240"
FEATURE_SPEC = FEATURE_SPECS[FEATURE_SET]
EXPECTED_DIM = int(FEATURE_SPEC["dim"])
FEATURE_NAMES: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the hand/default training manifest into train/val, select the "
            "best model on val, retrain it on all hand data, and evaluate once on "
            "the robot/test manifest."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Path that contains audio_visual_dataset_default and audio_visual_dataset_robo_default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory where features, reports, and models are written.",
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SPECS),
        default="total240",
        help="Feature group to extract.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Approximate validation fraction from the hand/default manifest.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/val split and models.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["segment", "row"],
        default="segment",
        help="Use segment-level groups by default to reduce train/val leakage.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=[
            "macro_f1_4class",
            "accuracy_4class",
            "weighted_f1",
            "contact_macro_f1",
            "binary_macro_f1",
        ],
        default="macro_f1_4class",
        help="Validation metric used to select the final model.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore feature caches and rebuild features.",
    )
    return parser.parse_args()


def configure_feature_set(feature_set: str) -> None:
    global FEATURE_SET, FEATURE_SPEC, EXPECTED_DIM, FEATURE_NAMES
    FEATURE_SET = feature_set
    FEATURE_SPEC = FEATURE_SPECS[FEATURE_SET]
    EXPECTED_DIM = int(FEATURE_SPEC["dim"])
    FEATURE_NAMES = build_feature_names()


def require_file(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


def resolve_root(user_root: Path | None) -> Path:
    candidates = []
    if user_root is not None:
        candidates.append(user_root)
    candidates.extend(
        [
            Path("tree_structures"),
            Path("data/tree_structures"),
            Path("../data/tree_structures"),
            Path("/kaggle/input/datasets/thanhtung1511/raw-dataset/raw_data"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (
            (candidate / "audio_visual_dataset_default" / "dataset.csv").exists()
            and (candidate / "audio_visual_dataset_robo_default" / "dataset.csv").exists()
        ):
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find dataset root. Searched:\n{searched}")


def load_manifest(csv_path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {"audio_file", "category"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"{csv_path} is missing columns: {sorted(required_columns)}")

    base = csv_path.parent
    output = pd.DataFrame(
        {
            "audio_file": frame["audio_file"].astype(str),
            "image_file": frame.get("image_file", pd.Series([""] * len(frame))).astype(str),
            "audio_path": frame["audio_file"].map(lambda value: base / str(value)),
            "label": frame["category"].astype(str).str.lower(),
            "source": source,
        }
    )
    output = output[output["label"].isin(LABEL_MAP)].copy()
    output["y"] = output["label"].map(LABEL_MAP).astype(np.int64)
    output["group_key"] = output["audio_file"].map(segment_group_key)
    output = output.reset_index(drop=True)

    if output.empty:
        raise ValueError(f"Empty manifest after label filtering: {csv_path}")

    missing_labels = set(CLASS_NAMES) - set(output["label"].unique())
    if missing_labels:
        raise ValueError(f"{source} is missing classes: {sorted(missing_labels)}")

    missing_mask = ~output["audio_path"].map(lambda path: Path(path).exists())
    if missing_mask.any():
        examples = output.loc[missing_mask, "audio_path"].head(5).tolist()
        raise FileNotFoundError(
            f"{source} has {int(missing_mask.sum())} missing audio paths. Examples: {examples}"
        )
    return output


def segment_group_key(audio_file: str) -> str:
    stem = Path(audio_file).stem
    return re.sub(r"_window_\d+.*$", "", stem)


def split_train_val(
    frame: pd.DataFrame,
    val_size: float,
    random_state: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not 0.05 <= val_size <= 0.5:
        raise ValueError("--val-size must be between 0.05 and 0.5")

    y = frame["y"].to_numpy()
    indices = np.arange(len(frame))

    if split_mode == "row":
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            stratify=y,
            random_state=random_state,
        )
        split_info = {
            "mode": "row",
            "note": "Stratified row-level split. Windows from the same segment may cross splits.",
        }
        return np.sort(train_idx), np.sort(val_idx), split_info

    groups = frame["group_key"].to_numpy()
    n_splits = max(2, int(round(1.0 / val_size)))
    full_distribution = np.bincount(y, minlength=len(CLASS_NAMES)) / len(y)
    best_candidate = None

    for seed_offset in range(50):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + seed_offset,
        )
        for train_idx, val_idx in splitter.split(indices, y, groups):
            val_distribution = np.bincount(y[val_idx], minlength=len(CLASS_NAMES)) / len(val_idx)
            train_classes = set(y[train_idx].tolist())
            val_classes = set(y[val_idx].tolist())
            class_penalty = 0 if len(train_classes) == 4 and len(val_classes) == 4 else 10
            size_penalty = abs((len(val_idx) / len(frame)) - val_size)
            distribution_penalty = float(np.abs(val_distribution - full_distribution).sum())
            score = class_penalty + size_penalty + distribution_penalty
            candidate = (score, train_idx, val_idx, random_state + seed_offset)
            if best_candidate is None or candidate[0] < best_candidate[0]:
                best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("Could not build a grouped train/val split")

    _, train_idx, val_idx, split_seed = best_candidate
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    overlap = sorted(train_groups.intersection(val_groups))
    if overlap:
        raise AssertionError(f"Grouped split leaked {len(overlap)} groups. Example: {overlap[:5]}")

    split_info = {
        "mode": "segment",
        "group_column": "group_key",
        "group_rule": "audio filename stem with trailing _window_<n>... removed",
        "split_seed": split_seed,
        "n_unique_groups": int(pd.Series(groups).nunique()),
        "train_unique_groups": len(train_groups),
        "val_unique_groups": len(val_groups),
    }
    return np.sort(train_idx), np.sort(val_idx), split_info


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        class_name: int((frame["label"] == class_name).sum())
        for class_name in CLASS_NAMES
    }


def save_split_manifests(
    run_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, str]:
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    columns = ["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]
    paths = {
        "train_inner": split_dir / "hand_train_inner.csv",
        "val": split_dir / "hand_val.csv",
        "train_full": split_dir / "hand_train_full.csv",
        "test": split_dir / "robot_test.csv",
    }
    train_df.iloc[train_idx][columns].to_csv(paths["train_inner"], index=False)
    train_df.iloc[val_idx][columns].to_csv(paths["val"], index=False)
    train_df[columns].to_csv(paths["train_full"], index=False)
    test_df[columns].to_csv(paths["test"], index=False)
    return {key: str(value.resolve()) for key, value in paths.items()}


def load_audio(path: Path, target_sr: int = 16000, duration: float = 1.0) -> np.ndarray:
    signal, _ = librosa.load(path, sr=target_sr, mono=True)
    target_len = int(target_sr * duration)
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]
    peak = np.max(np.abs(signal)) + 1e-8
    return (signal / peak).astype(np.float32)


def get_top_energy_window(
    signal: np.ndarray,
    sr: int = 16000,
    window_sec: float = 0.4,
    hop_sec: float = 0.05,
) -> np.ndarray:
    window_length = int(window_sec * sr)
    hop_length = int(hop_sec * sr)
    if len(signal) <= window_length:
        return signal

    best_start = 0
    best_energy = -1.0
    for start in range(0, len(signal) - window_length + 1, hop_length):
        segment = signal[start : start + window_length]
        energy = float(np.mean(segment**2))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return signal[best_start : best_start + window_length]


def summarize_4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.asarray(
        [
            np.mean(values),
            np.std(values),
            np.max(values),
            np.percentile(values, 90),
        ],
        dtype=np.float32,
    )


def extract_mfcc_compact(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    stft_cfg = CONFIG["stft_base"]
    mfcc_cfg = CONFIG["mfcc"]
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=mfcc_cfg["n_mfcc"],
        n_mels=mfcc_cfg["n_mels"],
        n_fft=stft_cfg["n_fft"],
        win_length=stft_cfg["win_length"],
        hop_length=stft_cfg["hop_length"],
        fmin=mfcc_cfg["fmin"],
        fmax=mfcc_cfg["fmax"],
    )
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).astype(np.float32)


def extract_stft_compact(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    stft_cfg = CONFIG["stft_base"]
    n_fft = stft_cfg["n_fft"]
    hop_length = stft_cfg["hop_length"]
    win_length = stft_cfg["win_length"]
    magnitude = np.abs(
        librosa.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=stft_cfg["window"],
        )
    )
    rms = librosa.feature.rms(S=magnitude, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sr,
        roll_percent=0.85,
    )[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    normalized = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-8)
    flux = np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0))
    flux = np.pad(flux, (1, 0))
    streams = [rms, zcr, centroid, bandwidth, rolloff, flatness, flux]
    return np.concatenate([summarize_4(stream) for stream in streams]).astype(np.float32)


def extract_mel_compact(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    stft_cfg = CONFIG["stft_base"]
    mel_cfg = CONFIG["mel"]
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=mel_cfg["n_mels"],
        n_fft=stft_cfg["n_fft"],
        win_length=stft_cfg["win_length"],
        hop_length=stft_cfg["hop_length"],
        fmin=CONFIG["mfcc"]["fmin"],
        fmax=CONFIG["mfcc"]["fmax"],
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    groups = np.array_split(log_mel, mel_cfg["n_groups"], axis=0)
    features = []
    for group in groups:
        features.extend(summarize_4(np.mean(group, axis=0)))
    return np.asarray(features, dtype=np.float32)


def extract_fft_compact(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    epsilon = 1e-8
    fft = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    power = fft**2
    total_power = np.sum(power) + epsilon

    band_powers = []
    for low, high in CONFIG["fft"]["bands"]:
        indices = np.where((frequencies >= low) & (frequencies < high))[0]
        band_powers.append(np.sum(power[indices]) / total_power)
    band_powers = np.asarray(band_powers, dtype=np.float32)
    log_band_powers = np.log1p(band_powers)

    low = np.sum(band_powers[0:3])
    mid = np.sum(band_powers[3:6])
    high = np.sum(band_powers[6:8])
    ratios = np.asarray(
        [
            low / (mid + epsilon),
            low / (high + epsilon),
            mid / (high + epsilon),
            (low + mid) / (high + epsilon),
        ],
        dtype=np.float32,
    )

    probability = power / total_power
    entropy = -np.sum(probability * np.log(probability + epsilon)) / np.log(len(probability))
    centroid = np.sum(frequencies * power) / total_power
    dominant_index = int(np.argmax(power))
    extras = np.asarray(
        [
            entropy,
            centroid,
            frequencies[dominant_index],
            fft[dominant_index],
        ],
        dtype=np.float32,
    )
    return np.concatenate([band_powers, log_band_powers, ratios, extras]).astype(np.float32)


def extract_total_120(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    return np.concatenate(
        [
            extract_mfcc_compact(signal, sr),
            extract_stft_compact(signal, sr),
            extract_mel_compact(signal, sr),
            extract_fft_compact(signal, sr),
        ]
    ).astype(np.float32)


def extract_features_for_file(path: Path, sr: int = 16000) -> np.ndarray:
    signal = load_audio(path, target_sr=sr, duration=CONFIG["audio"]["duration"])

    if FEATURE_SET == "mfcc40":
        features = extract_mfcc_compact(signal, sr)
    elif FEATURE_SET == "stft28":
        features = extract_stft_compact(signal, sr)
    elif FEATURE_SET == "mel28":
        features = extract_mel_compact(signal, sr)
    elif FEATURE_SET == "fft24":
        features = extract_fft_compact(signal, sr)
    elif FEATURE_SET == "total120":
        features = extract_total_120(signal, sr)
    elif FEATURE_SET == "total240":
        crop_cfg = CONFIG["event_crop"]
        top_signal = get_top_energy_window(
            signal,
            sr=sr,
            window_sec=crop_cfg["top_energy_sec"],
            hop_sec=crop_cfg["top_energy_hop_sec"],
        )
        features = np.concatenate(
            [
                extract_total_120(signal, sr),
                extract_total_120(top_signal, sr),
            ]
        ).astype(np.float32)
    else:
        raise KeyError(f"Unknown FEATURE_SET: {FEATURE_SET}")

    if features.shape != (EXPECTED_DIM,):
        raise AssertionError(f"{FEATURE_SET}: expected {(EXPECTED_DIM,)}, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Feature contains NaN/Inf: {path}")
    return features


def base_feature_names() -> dict[str, list[str]]:
    mfcc_names = (
        [f"mfcc_mean_{index:02d}" for index in range(20)]
        + [f"mfcc_std_{index:02d}" for index in range(20)]
    )
    stft_streams = ["rms", "zcr", "centroid", "bandwidth", "rolloff", "flatness", "flux"]
    summary_stats = ["mean", "std", "max", "p90"]
    stft_names = [
        f"stft_{stream}_{stat}"
        for stream in stft_streams
        for stat in summary_stats
    ]
    mel_names = [
        f"mel_group{group}_{stat}"
        for group in range(7)
        for stat in summary_stats
    ]
    ratio_names = ["low_mid", "low_high", "mid_high", "lowmid_high"]
    extra_names = ["entropy", "centroid", "dominant_freq", "dominant_magnitude"]
    fft_names = (
        [f"fft_band_{index}" for index in range(8)]
        + [f"fft_log_band_{index}" for index in range(8)]
        + [f"fft_ratio_{name}" for name in ratio_names]
        + [f"fft_{name}" for name in extra_names]
    )
    return {
        "mfcc40": mfcc_names,
        "stft28": stft_names,
        "mel28": mel_names,
        "fft24": fft_names,
        "total120": mfcc_names + stft_names + mel_names + fft_names,
    }


def build_feature_names() -> list[str]:
    names_by_set = base_feature_names()
    if FEATURE_SET == "total240":
        total_names = names_by_set["total120"]
        names = [f"full_{name}" for name in total_names] + [
            f"top_{name}" for name in total_names
        ]
    else:
        names = names_by_set[FEATURE_SET]
    if len(names) != EXPECTED_DIM:
        raise AssertionError(f"Expected {EXPECTED_DIM} names, got {len(names)}")
    return names


def manifest_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["audio_path", "y"]].itertuples(index=False):
        digest.update(f"{row.audio_path}|{int(row.y)}\n".encode("utf-8"))
    return digest.hexdigest()


def cache_paths(feature_dir: Path, split_name: str) -> dict[str, Path]:
    split_dir = feature_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": split_dir / "X.npy",
        "y": split_dir / "y.npy",
        "paths": split_dir / "paths.npy",
        "metadata": split_dir / "metadata.json",
    }


def build_or_load_feature_cache(
    frame: pd.DataFrame,
    split_name: str,
    feature_dir: Path,
    force_rebuild: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
    paths = cache_paths(feature_dir, split_name)
    signature = manifest_signature(frame)
    ready = all(path.exists() for path in paths.values())

    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        valid_metadata = (
            metadata.get("feature_set") == FEATURE_SET
            and int(metadata.get("feature_dim", -1)) == EXPECTED_DIM
            and int(metadata.get("n_samples", -1)) == len(frame)
            and metadata.get("manifest_signature") == signature
        )
        if valid_metadata:
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": np.load(paths["y"]),
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            load_time = time.perf_counter() - start
            timing = {
                "cache_hit": True,
                "cache_load_time_sec": load_time,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
                "extraction_time_per_sample_sec": float(
                    metadata["extraction_time_per_sample_sec"]
                ),
            }
            print(f"Loaded cache {split_name}: {payload['X'].shape} in {load_time:.3f}s")
            return payload, timing
        print(f"Cache metadata mismatch for {split_name}; rebuilding.")

    feature_rows = []
    labels = []
    audio_paths = []
    start = time.perf_counter()
    for row in tqdm(
        frame.itertuples(index=False),
        total=len(frame),
        desc=f"Extract {FEATURE_SET}/{split_name}",
    ):
        feature_rows.append(extract_features_for_file(row.audio_path, sr=CONFIG["audio"]["sr"]))
        labels.append(int(row.y))
        audio_paths.append(str(row.audio_path))
    extraction_time = time.perf_counter() - start

    payload = {
        "X": np.stack(feature_rows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(audio_paths),
    }
    if payload["X"].shape != (len(frame), EXPECTED_DIM):
        raise AssertionError(
            f"{split_name}: expected {(len(frame), EXPECTED_DIM)}, got {payload['X'].shape}"
        )
    if not np.isfinite(payload["X"]).all():
        raise ValueError(f"{split_name}: feature matrix contains NaN/Inf")

    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    metadata = {
        "feature_set": FEATURE_SET,
        "feature_display_name": FEATURE_SPEC["display_name"],
        "feature_dim": EXPECTED_DIM,
        "n_samples": len(frame),
        "manifest_signature": signature,
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    timing = {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
    }
    print(f"Saved cache {split_name}: {payload['X'].shape} in {extraction_time:.2f}s")
    return payload, timing


def make_models(random_state: int) -> dict[str, object]:
    class_weights = CONFIG["class_weights"]
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            class_weight=class_weights,
            random_state=random_state,
        ),
        "LogisticRegression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight=class_weights,
                        max_iter=3000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "DecisionTree": DecisionTreeClassifier(
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=random_state,
        ),
        "RBF_SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=10,
                        gamma="scale",
                        class_weight=class_weights,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "LightGBM": LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASS_NAMES),
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_NAMES),
            n_estimators=600,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if list(models) != EXPECTED_MODEL_NAMES:
        raise AssertionError(f"Unexpected model order: {list(models)}")
    return models


def make_report_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time_sec: float,
    predict_time_sec: float,
    status: str = "ok",
) -> dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )
    macro_precision_4class, macro_recall_4class, macro_f1_4class, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
    )
    y_true_binary = (y_true > 0).astype(np.int64)
    y_pred_binary = (y_pred > 0).astype(np.int64)
    binary_macro_precision, binary_macro_recall, binary_macro_f1, _ = (
        precision_recall_fscore_support(
            y_true_binary,
            y_pred_binary,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )

    row = {
        "feature_set": FEATURE_SET,
        "feature_name": FEATURE_SPEC["display_name"],
        "n_features": EXPECTED_DIM,
        "split": split_name,
        "model": model_name,
        "status": status,
        "accuracy_4class": accuracy_score(y_true, y_pred),
        "macro_precision_4class": macro_precision_4class,
        "macro_recall_4class": macro_recall_4class,
        "macro_f1_4class": macro_f1_4class,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "contact_macro_f1": f1_score(
            y_true,
            y_pred,
            labels=CONTACT_LABELS,
            average="macro",
            zero_division=0,
        ),
        "binary_accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "binary_macro_precision": binary_macro_precision,
        "binary_macro_recall": binary_macro_recall,
        "binary_macro_f1": binary_macro_f1,
        "binary_contact_f1": f1_score(
            y_true_binary,
            y_pred_binary,
            pos_label=1,
            zero_division=0,
        ),
        "model_train_time_sec": train_time_sec,
        "model_predict_time_sec": predict_time_sec,
        "model_total_time_sec": train_time_sec + predict_time_sec,
    }
    for index, class_name in enumerate(CLASS_NAMES):
        row[f"{class_name}_precision"] = precision[index]
        row[f"{class_name}_recall"] = recall[index]
        row[f"{class_name}_f1"] = class_f1[index]
        row[f"{class_name}_support"] = support[index]
    return row


def failed_report_row(model_name: str, split_name: str, exception: Exception) -> dict[str, object]:
    row = {
        "feature_set": FEATURE_SET,
        "feature_name": FEATURE_SPEC["display_name"],
        "n_features": EXPECTED_DIM,
        "split": split_name,
        "model": model_name,
        "status": f"failed: {exception}",
        "accuracy_4class": np.nan,
        "macro_precision_4class": np.nan,
        "macro_recall_4class": np.nan,
        "macro_f1_4class": np.nan,
        "weighted_f1": np.nan,
        "contact_macro_f1": np.nan,
        "binary_accuracy": np.nan,
        "binary_macro_precision": np.nan,
        "binary_macro_recall": np.nan,
        "binary_macro_f1": np.nan,
        "binary_contact_f1": np.nan,
        "model_train_time_sec": np.nan,
        "model_predict_time_sec": np.nan,
        "model_total_time_sec": np.nan,
    }
    for class_name in CLASS_NAMES:
        row[f"{class_name}_precision"] = np.nan
        row[f"{class_name}_recall"] = np.nan
        row[f"{class_name}_f1"] = np.nan
        row[f"{class_name}_support"] = np.nan
    return row


def evaluate_candidates_on_val(
    models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    fitted_models = {}
    for model_name, base_model in models.items():
        print(f"\nTraining {model_name} on train_inner; evaluating on val ...")
        try:
            model = clone(base_model)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start

            start = time.perf_counter()
            prediction = model.predict(X_val)
            predict_time = time.perf_counter() - start

            row = make_report_row(
                model_name=model_name,
                split_name="val",
                y_true=y_val,
                y_pred=prediction,
                train_time_sec=train_time,
                predict_time_sec=predict_time,
            )
            rows.append(row)
            fitted_models[model_name] = model
            print(
                "val | "
                f"accuracy={row['accuracy_4class']:.4f} | "
                f"macro_f1={row['macro_f1_4class']:.4f} | "
                f"binary_macro_f1={row['binary_macro_f1']:.4f} | "
                f"time={row['model_total_time_sec']:.2f}s"
            )
        except Exception as exception:
            print(f"{model_name} failed on val: {exception}")
            rows.append(failed_report_row(model_name, "val", exception))

    report_df = pd.DataFrame(rows)
    if set(report_df["model"]) != set(EXPECTED_MODEL_NAMES):
        raise AssertionError("Validation report does not contain all expected models")
    return report_df, fitted_models


def select_best_model(report_df: pd.DataFrame, selection_metric: str) -> pd.Series:
    ok_rows = report_df[report_df["status"].eq("ok")].copy()
    ok_rows = ok_rows[ok_rows[selection_metric].notna()]
    if ok_rows.empty:
        raise RuntimeError(f"No successful validation rows for metric {selection_metric}")
    sort_columns = [selection_metric]
    if selection_metric != "macro_f1_4class":
        sort_columns.append("macro_f1_4class")
    sort_columns.append("accuracy_4class")
    ok_rows = ok_rows.sort_values(sort_columns, ascending=False).reset_index(drop=True)
    return ok_rows.iloc[0]


def fit_and_test_final_model(
    model_name: str,
    base_model: object,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, object], object, np.ndarray]:
    print(f"\nRetraining selected model {model_name} on full hand train; evaluating on robot test ...")
    model = clone(base_model)
    start = time.perf_counter()
    model.fit(X_train_full, y_train_full)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    prediction = model.predict(X_test)
    predict_time = time.perf_counter() - start

    row = make_report_row(
        model_name=model_name,
        split_name="test",
        y_true=y_test,
        y_pred=prediction,
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )
    row["trained_on"] = "hand_train_full_after_val_selection"
    print(
        "test | "
        f"accuracy={row['accuracy_4class']:.4f} | "
        f"macro_f1={row['macro_f1_4class']:.4f} | "
        f"binary_macro_f1={row['binary_macro_f1']:.4f} | "
        f"time={row['model_total_time_sec']:.2f}s"
    )
    return row, model, prediction


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def main() -> None:
    args = parse_args()
    CONFIG["random_state"] = args.random_state
    configure_feature_set(args.feature_set)

    root_path = resolve_root(args.root)
    run_slug = f"{FEATURE_SET}_trainval_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    feature_dir = run_dir / "features"
    model_dir = run_dir / "models"
    report_dir = run_dir / "reports"
    for directory in [run_dir, feature_dir, model_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve())
    print("RUN_DIR   =", run_dir.resolve())
    print("Feature set:", FEATURE_SPEC["display_name"])
    print("Selection metric:", args.selection_metric)

    train_csv = require_file(
        root_path / "audio_visual_dataset_default" / "dataset.csv",
        "hand/default dataset.csv",
    )
    test_csv = require_file(
        root_path / "audio_visual_dataset_robo_default" / "dataset.csv",
        "robot dataset.csv",
    )

    train_df = load_manifest(train_csv, "hand_train")
    test_df = load_manifest(test_csv, "robot_test")
    train_idx, val_idx, split_info = split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)

    split_summary = {
        "train_full_samples": len(train_df),
        "train_inner_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": len(test_df),
        "requested_val_size": args.val_size,
        "actual_val_size": len(val_idx) / len(train_df),
        "train_full_label_counts": label_counts(train_df),
        "train_inner_label_counts": label_counts(train_df.iloc[train_idx]),
        "val_label_counts": label_counts(train_df.iloc[val_idx]),
        "test_label_counts": label_counts(test_df),
        "split_info": split_info,
        "split_manifest_paths": split_paths,
    }
    write_json(report_dir / f"{run_slug}_split_summary.json", split_summary)
    print("Split summary:")
    print(json.dumps(split_summary, indent=2))

    sample_feature = extract_features_for_file(
        train_df["audio_path"].iloc[0],
        sr=CONFIG["audio"]["sr"],
    )
    if sample_feature.shape != (EXPECTED_DIM,):
        raise AssertionError(f"Sample feature shape mismatch: {sample_feature.shape}")
    print("Sample feature shape:", sample_feature.shape)

    train_feat, train_feature_timing = build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=feature_dir,
        force_rebuild=args.force_rebuild,
    )
    test_feat, test_feature_timing = build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=feature_dir,
        force_rebuild=args.force_rebuild,
    )

    X_train_full = train_feat["X"]
    y_train_full = train_feat["y"]
    X_train_inner = X_train_full[train_idx]
    y_train_inner = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    X_test = test_feat["X"]
    y_test = test_feat["y"]

    models = make_models(args.random_state)
    val_report_df, fitted_val_models = evaluate_candidates_on_val(
        models,
        X_train_inner,
        y_train_inner,
        X_val,
        y_val,
    )
    val_report_df = val_report_df.sort_values(
        [args.selection_metric, "macro_f1_4class", "accuracy_4class"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    val_report_path = report_dir / f"{run_slug}_val_leaderboard.csv"
    val_report_df.to_csv(val_report_path, index=False)

    best_row = select_best_model(val_report_df, args.selection_metric)
    best_model_name = str(best_row["model"])
    print(
        f"\nSelected best model from val: {best_model_name} "
        f"({args.selection_metric}={float(best_row[args.selection_metric]):.4f})"
    )

    final_row, final_model, test_prediction = fit_and_test_final_model(
        best_model_name,
        models[best_model_name],
        X_train_full,
        y_train_full,
        X_test,
        y_test,
    )
    final_report_df = pd.DataFrame([final_row])
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    final_report_df.to_csv(final_report_path, index=False)

    test_predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]
    ].copy()
    prediction_frame["pred_y"] = test_prediction.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(ID2LABEL)
    prediction_frame.to_csv(test_predictions_path, index=False)

    cm = confusion_matrix(y_test, test_prediction, labels=[0, 1, 2, 3])
    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_path)

    timing_summary = {
        "feature_extraction": {
            "hand_train_full": train_feature_timing,
            "robot_test": test_feature_timing,
        },
        "validation_models": val_report_df[
            [
                "model",
                "status",
                "model_train_time_sec",
                "model_predict_time_sec",
                "model_total_time_sec",
            ]
        ].to_dict(orient="records"),
        "final_model": {
            "model": best_model_name,
            "model_train_time_sec": final_row["model_train_time_sec"],
            "model_predict_time_sec": final_row["model_predict_time_sec"],
            "model_total_time_sec": final_row["model_total_time_sec"],
        },
    }

    protocol_summary = {
        "protocol": "split_hand_train_to_train_val_select_on_val_retrain_full_train_test_robot",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "feature_set": FEATURE_SET,
        "feature_spec": FEATURE_SPEC,
        "selection_metric": args.selection_metric,
        "selected_model": best_model_name,
        "selected_val_row": best_row.to_dict(),
        "final_test_row": final_row,
        "split_summary": split_summary,
        "timing_summary": timing_summary,
        "artifacts": {
            "val_leaderboard": str(val_report_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(test_predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    bundle_path = model_dir / f"{run_slug}_best_model_bundle.joblib"
    joblib.dump(
        {
            "config": CONFIG,
            "feature_set": FEATURE_SET,
            "feature_spec": FEATURE_SPEC,
            "feature_names": FEATURE_NAMES,
            "label_map": LABEL_MAP,
            "id2label": ID2LABEL,
            "protocol_summary": protocol_summary,
            "selected_model_name": best_model_name,
            "selected_model": final_model,
            "validation_fitted_models": fitted_val_models,
            "validation_report": val_report_df,
            "final_test_report": final_report_df,
        },
        bundle_path,
    )
    protocol_summary["artifacts"]["best_model_bundle"] = str(bundle_path.resolve())
    write_json(summary_path, protocol_summary)

    print("\nSaved artifacts:")
    print("val leaderboard:", val_report_path.resolve())
    print("final test report:", final_report_path.resolve())
    print("protocol summary:", summary_path.resolve())
    print("best model bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
