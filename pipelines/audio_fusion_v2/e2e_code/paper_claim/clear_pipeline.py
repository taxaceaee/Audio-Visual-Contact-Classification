"""Clear minimal multimodal claim pipeline (equiv. to sealed paper_claim_v2).

Architecture (4 steps only)
---------------------------
  1) v2 base:  hier_trunk_meta_else
  2) amb-lift: ambient→trunk via CLIP ‖ wav2vec2 ‖ robot_mix
  3) secondary: ambient→trunk via CLIP trunk + CLIP binary
  4) material:  protect-trunk leaf bias on meta soft

Dropped vs historical full stack (prediction-equivalent on sealed HPs):
  - Eff / DINO / ConvNeXt in amb detector
  - bandlimit stress view (robot_mix alone)
  - do_tt, multi material biases, v2_rule_soft, hier material soft

Protocol
--------
  - Detectors fit on hand only
  - HPs from hand selection lock (never tuned on robot here)
  - No filename class features, no robot UDA
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
V2_BASE = Path(
    "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
    "segment_rule_stack_v2_base_segment_outputs.npz"
)
CLIP = Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k")
AUDIO_CLEAN = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"
)
AUDIO_MIX = Path(
    "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/robot_mix/X.npy"
)
AUDIO_ROBOT = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/X.npy"
)
W2V_HAND = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")
W2V_ROBOT = Path("outputs/audio_wav2vec2_features/robot_test_X.npy")


@dataclass(frozen=True)
class HParams:
    """Frozen claim HPs (hand-selected / prior hand locks)."""

    ts_th: float = 0.5
    cs_th: float = 0.1
    img_th: float = 0.625
    img_cs: float = 0.75
    b_leaf: float = -1.3  # paper_claim_v2 sealed selection


def pool_segment(files, x: np.ndarray, y: np.ndarray | None = None):
    keys = sel.seg(files).to_numpy()
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


def fit_detectors_hand():
    """Fit amb + secondary detectors on hand only (no robot)."""
    fr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y = fr.y.to_numpy(np.int64)
    clip = np.load(CLIP / "hand_train_full/X.npy").astype(np.float32)
    u, c, clip_s, ys = pool_segment(fr.audio_file, clip, y)
    mix = np.load(AUDIO_MIX).astype(np.float32)
    mix_s = np.vstack([mix[c == i].mean(0) for i in range(len(u))])
    w2v = np.load(W2V_HAND).astype(np.float32)
    w2v_s = np.vstack([w2v[c == i].mean(0) for i in range(len(u))])

    # clear amb features: CLIP ‖ w2v ‖ robot_mix
    X_amb = np.hstack([mix_s, w2v_s, clip_s])
    m_ts = _bin(X_amb, (ys == 2).astype(int), 0.05)
    m_cs = _bin(X_amb, (ys > 0).astype(int), 0.05)
    m_bin = _bin(clip_s, (ys > 0).astype(int), 0.03)
    m_tr = _bin(clip_s, (ys == 2).astype(int), 0.1)
    return {"ts": m_ts, "cs": m_cs, "bin": m_bin, "tr": m_tr}


def predict_robot(hp: HParams | None = None) -> dict:
    """Run clear pipeline on robot/test. Returns y, pred (window rows), extras."""
    hp = hp or HParams()
    det = fit_detectors_hand()

    z = np.load(V2_BASE, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    # step 1 — v2 base
    pred = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"]).astype(np.int64)
    p_meta = z["p_meta"].astype(np.float64)
    order = {k: i for i, k in enumerate(ids)}

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    clip_r = np.load(CLIP / "robot_test/X.npy").astype(np.float32)
    tu, tc, clip_s, _ = pool_segment(tfr.audio_file, clip_r, yt)
    ta = np.load(AUDIO_ROBOT).astype(np.float32)
    ta_s = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_ROBOT).astype(np.float32)
    tw_s = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])
    # robot has no robot_mix transform cache; amb det trained on mix, applied on clean robot
    # (same as full claim path — stress views only at train-time for detector)
    X_amb = np.hstack([ta_s, tw_s, clip_s])

    pred = np.array([pred[order[str(k)]] for k in tu], dtype=np.int64)
    soft = np.array([p_meta[order[str(k)]] for k in tu])

    bins = _pos(det["bin"], clip_s)
    tr = _pos(det["tr"], clip_s)
    ts = _pos(det["ts"], X_amb)
    cs = _pos(det["cs"], X_amb)

    # step 2 — amb-lift
    amb = pred == 0
    pred[amb & (ts >= hp.ts_th) & (np.maximum(cs, bins) >= hp.cs_th)] = 2
    # step 3 — secondary CLIP trunk
    amb = pred == 0
    pred[amb & (tr >= hp.img_th) & (bins >= hp.img_cs)] = 2
    # step 4 — protect-trunk material (leaf/twig only)
    m = (pred == 1) | (pred == 3)
    if m.any():
        logits = np.log(np.clip(soft[m][:, 1:4], 1e-12, None))
        logits = logits + np.array([hp.b_leaf, 0.0, 0.0])
        pred[m] = 1 + logits.argmax(1)

    # expand segment → window rows
    sk = sel.seg(tfr.audio_file).to_numpy()
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
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    return {
        "macro_f1_4class": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix_4class": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
        "per_class": classification_report(
            y,
            pred,
            labels=[0, 1, 2, 3],
            target_names=["ambient", "leaf", "trunk", "twig"],
            output_dict=True,
            zero_division=0,
        ),
    }
