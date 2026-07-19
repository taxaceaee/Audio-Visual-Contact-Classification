"""Clear multimodal claim pipeline (equiv. sealed paper_claim_v2).

All data roots resolve relative to the repository root (or TREE_BUNDLE_ROOT).
No hard-coded host paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]


def _repo_root() -> Path:
    # .../fusion_base/code/paper_claim/clear_pipeline.py -> fusion_base
    return Path(__file__).resolve().parents[2]


def bundle_paths(root: Path | None = None) -> dict[str, Path]:
    import os

    root = Path(os.environ.get("TREE_BUNDLE_ROOT", root or _repo_root())).resolve()
    data = root / "data"
    return {
        "root": root,
        "manifest_root": data / "manifests",
        "hand_csv": data / "manifests" / "audio_visual_dataset_default" / "dataset.csv",
        "robot_csv": data / "manifests" / "audio_visual_dataset_robo_default" / "dataset.csv",
        "hand_dir": data / "manifests" / "audio_visual_dataset_default",
        "robot_dir": data / "manifests" / "audio_visual_dataset_robo_default",
        "v2_base": data / "features" / "v2_base" / "segment_rule_stack_v2_base_segment_outputs.npz",
        "clip_hand": data / "features" / "clip" / "hand_train_full" / "X.npy",
        "clip_robot": data / "features" / "clip" / "robot_test" / "X.npy",
        "audio_mix": data / "features" / "total240" / "robot_mix" / "X.npy",
        "audio_robot": data / "features" / "total240" / "robot_test" / "X.npy",
        "w2v_hand": data / "features" / "wav2vec2" / "hand_train_full_X.npy",
        "w2v_robot": data / "features" / "wav2vec2" / "robot_test_X.npy",
    }


def load_manifest(csv_path: Path, dataset_dir: Path, source: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    labels = {name: i for i, name in enumerate(CLASS_NAMES)}
    frame = pd.DataFrame(
        {
            "audio_file": raw["audio_file"].astype(str),
            "image_file": raw["image_file"].astype(str),
            "image_path": raw["image_file"].map(lambda x: str(dataset_dir / x)),
            "label": raw["category"].astype(str).str.lower(),
            "source": source,
        }
    )
    frame = frame[frame.label.isin(labels)].copy()
    frame["y"] = frame["label"].map(labels).astype(np.int64)
    return frame.reset_index(drop=True)


def seg(files: pd.Series) -> pd.Series:
    return files.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)


@dataclass(frozen=True)
class HParams:
    """Frozen claim HPs (hand-selected / prior hand locks)."""

    ts_th: float = 0.5
    cs_th: float = 0.1
    img_th: float = 0.625
    img_cs: float = 0.75
    b_leaf: float = -1.3  # paper_claim_v2 sealed selection


def pool_segment(files, x: np.ndarray, y: np.ndarray | None = None):
    keys = seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))]).astype(np.float32)
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))], np.int64)
    return u, c, px, py


def _bin(x, y, C=0.05):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=42),
    )
    m.fit(x, y)
    return m


def _pos(m, x) -> np.ndarray:
    p = m.predict_proba(x)
    cl = list(m.classes_)
    return p[:, cl.index(1)] if 1 in cl else p[:, -1]


def fit_detectors_hand(paths: dict[str, Path] | None = None):
    """Fit amb + secondary detectors on hand only (no robot)."""
    paths = paths or bundle_paths()
    fr = load_manifest(paths["hand_csv"], paths["hand_dir"], "hand_train")
    y = fr.y.to_numpy(np.int64)
    clip = np.load(paths["clip_hand"]).astype(np.float32)
    u, c, clip_s, ys = pool_segment(fr.audio_file, clip, y)
    mix = np.load(paths["audio_mix"]).astype(np.float32)
    mix_s = np.vstack([mix[c == i].mean(0) for i in range(len(u))])
    w2v = np.load(paths["w2v_hand"]).astype(np.float32)
    w2v_s = np.vstack([w2v[c == i].mean(0) for i in range(len(u))])

    X_amb = np.hstack([mix_s, w2v_s, clip_s])
    m_ts = _bin(X_amb, (ys == 2).astype(int), 0.05)
    m_cs = _bin(X_amb, (ys > 0).astype(int), 0.05)
    m_bin = _bin(clip_s, (ys > 0).astype(int), 0.03)
    m_tr = _bin(clip_s, (ys == 2).astype(int), 0.1)
    return {"ts": m_ts, "cs": m_cs, "bin": m_bin, "tr": m_tr}


def predict_robot(hp: HParams | None = None, paths: dict[str, Path] | None = None) -> dict:
    """Run clear pipeline on robot/test. Returns y, pred (window rows), extras."""
    hp = hp or HParams()
    paths = paths or bundle_paths()
    det = fit_detectors_hand(paths)

    z = np.load(paths["v2_base"], allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"]).astype(np.int64)
    p_meta = z["p_meta"].astype(np.float64)
    order = {k: i for i, k in enumerate(ids)}

    tfr = load_manifest(paths["robot_csv"], paths["robot_dir"], "robot_test")
    yt = tfr.y.to_numpy(np.int64)
    clip_r = np.load(paths["clip_robot"]).astype(np.float32)
    tu, tc, clip_s, _ = pool_segment(tfr.audio_file, clip_r, yt)
    ta = np.load(paths["audio_robot"]).astype(np.float32)
    ta_s = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(paths["w2v_robot"]).astype(np.float32)
    tw_s = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])
    X_amb = np.hstack([ta_s, tw_s, clip_s])

    pred = np.array([pred[order[str(k)]] for k in tu], dtype=np.int64)
    soft = np.array([p_meta[order[str(k)]] for k in tu])

    bins = _pos(det["bin"], clip_s)
    tr = _pos(det["tr"], clip_s)
    ts = _pos(det["ts"], X_amb)
    cs = _pos(det["cs"], X_amb)

    amb = pred == 0
    pred[amb & (ts >= hp.ts_th) & (np.maximum(cs, bins) >= hp.cs_th)] = 2
    amb = pred == 0
    pred[amb & (tr >= hp.img_th) & (bins >= hp.img_cs)] = 2
    m = (pred == 1) | (pred == 3)
    if m.any():
        logits = np.log(np.clip(soft[m][:, 1:4], 1e-12, None))
        logits = logits + np.array([hp.b_leaf, 0.0, 0.0])
        pred[m] = 1 + logits.argmax(1)

    sk = seg(tfr.audio_file).to_numpy()
    ou = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred[ou[k]] for k in sk], dtype=np.int64)
    return {
        "y": yt,
        "pred": pred_rows,
        "pred_seg": pred,
        "segment_ids": tu,
        "hp": hp,
        "architecture": "clear_v2_equiv_clip_mix_w2v",
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    binary_y = (y > 0).astype(np.int64)
    binary_p = (pred > 0).astype(np.int64)
    return {
        "accuracy_4class": float(accuracy_score(y, pred)),
        "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
        "binary_macro_f1": float(
            f1_score(binary_y, binary_p, average="macro", zero_division=0)
        ),
        "confusion_matrix_4class": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
        "per_class": classification_report(
            y,
            pred,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


__all__ = [
    "HParams",
    "bundle_paths",
    "fit_detectors_hand",
    "predict_robot",
    "metrics",
    "load_manifest",
    "seg",
    "pool_segment",
]
