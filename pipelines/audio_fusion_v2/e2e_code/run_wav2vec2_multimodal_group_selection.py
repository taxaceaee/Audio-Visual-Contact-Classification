from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "wav2vec2_multimodal_group_selection"
MODEL_KIND = os.environ.get("AUDIO_SSL_MODEL", "wav2vec2")
MODEL_BUNDLE = torchaudio.pipelines.HUBERT_BASE if MODEL_KIND == "hubert" else torchaudio.pipelines.WAV2VEC2_BASE
CACHE = OUT / "audio_ssl_features" / f"{MODEL_KIND}_hand_train_full_X.npy"
REPORT = OUT / "audio_feature_benchmarks" / f"{MODEL_KIND}_multimodal_group_selection"


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    sr, data = wavfile.read(path)
    was_integer = np.issubdtype(data.dtype, np.integer)
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if was_integer:
        data = data / 32768.0
    if sr != target_sr:
        data = resample_poly(data, target_sr, sr).astype(np.float32)
    target_len = target_sr
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    else:
        data = data[:target_len]
    peak = np.max(np.abs(data)) + 1e-8
    return (data / peak).astype(np.float32)


def extract_hand(frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        return np.load(CACHE)
    bundle = MODEL_BUNDLE
    model = bundle.get_model().to(device).eval()
    sr = bundle.sample_rate
    rows = []
    batch = []
    for idx, audio_file in enumerate(frame.audio_file.astype(str)):
        batch.append(load_audio(ROOT / "audio_visual_dataset_default" / audio_file, sr))
        if len(batch) == 32 or idx == len(frame) - 1:
            x = torch.from_numpy(np.stack(batch)).to(device)
            with torch.inference_mode():
                features, _ = model.extract_features(x)
                h = features[-1]
                pooled = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=1)
            rows.append(pooled.cpu().numpy().astype(np.float32))
            batch = []
            print(f"wav2vec2 hand {idx + 1}/{len(frame)}", flush=True)
    X = np.concatenate(rows, axis=0)
    np.save(CACHE, X)
    return X


def aligned_proba(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, col]
    return suite.normalize(p)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    frame, train_idx, val_idx = suite.build_group_split(ROOT, REPORT)
    y = frame.y.to_numpy(dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio = extract_hand(frame, device)
    image = np.load(OUT / "image_deep_features" / "resnet18_224" / "hand_train_full" / "X.npy")
    if not np.array_equal(np.load(OUT / "image_deep_features" / "resnet18_224" / "hand_train_full" / "y.npy"), y):
        raise AssertionError("Image labels are not aligned")
    run = OUT / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    audio_lock = json.loads((run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    pair = suite.normalize(np.load(OUT / "audio_feature_benchmarks" / "audio_group_consistency_pair_blend_select" / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"))[val_idx]
    high = np.load(run / "oof_proba" / audio_lock["selected_candidate"] / "clean_oof_proba.npy")[val_idx]
    audio_old = suite.segment_lift(frame.audio_file.iloc[val_idx], suite.normalize(0.8 * high + 0.2 * pair))
    rows = []
    for feature_name, X in {
        "wav2vec2_image": np.hstack([audio, image]),
        "wav2vec2_only": audio,
        "wav2vec2_image_oldaudio": np.hstack([audio, image, np.log(audio_old)]) if False else None,
    }.items():
        if X is None:
            continue
        for alpha in [1e-5, 1e-4, 1e-3]:
            model = Pipeline([("scale", StandardScaler()), ("model", SGDClassifier(loss="log_loss", alpha=alpha, class_weight="balanced", max_iter=120, tol=1e-3, random_state=42, n_jobs=-1))]).fit(X[train_idx], y[train_idx])
            p = aligned_proba(model, X[val_idx])
            pred = p.argmax(axis=1)
            rows.append({"feature_set": feature_name, "model": f"sgd_logloss_alpha{alpha:g}", "accuracy_4class": float(accuracy_score(y[val_idx], pred)), "macro_precision_4class": float(precision_score(y[val_idx], pred, average="macro", zero_division=0)), "macro_recall_4class": float(recall_score(y[val_idx], pred, average="macro", zero_division=0)), "macro_f1_4class": float(f1_score(y[val_idx], pred, average="macro", zero_division=0)), "binary_macro_f1": float(f1_score(y[val_idx] > 0, pred > 0, average="macro", zero_division=0))})
    leaderboard = pd.DataFrame(rows).sort_values(["macro_f1_4class", "binary_macro_f1"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(REPORT / "hand_group_wav2vec2_leaderboard.csv", index=False)
    lock = {"protocol": "wav2vec2_multimodal_specimen_group_val_only", "group_overlap": 0, "test_loaded": False, "cache": str(CACHE.resolve()), "best_val_candidate": leaderboard.iloc[0].to_dict()}
    (REPORT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
