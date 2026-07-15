#!/usr/bin/env python3
"""Simplified multimodal pipeline (diagnostic + reproducible).

Complexity reduction vs full claim stack
----------------------------------------
DROPPED (ablation-safe for robot Macro F1 >= 0.85, often *higher*):
  - twig->trunk lift (do_tt / m_tt detector)
  - multi-param material bias (b_trunk, b_twig, T!=1)
  - v2_rule_soft hybrid soft  → use meta soft only
  - cascade grid over 4 HPs   → single frozen leaf_bias

KEPT (needed for >= 0.85 with margin):
  - v2 base (hier_trunk_meta_else)
  - stress amb-lift (ts/cs on multi-img + w2v + robot_mix/bandlimit)
  - secondary CLIP trunk on residual ambient
  - protect-trunk material: redecode leaf/twig only with b_leaf bias on meta

Default frozen HPs (from hand prior locks + simplified material):
  amb: ts_th=0.5, cs_th=0.1
  sec: img_th=0.625, img_cs=0.75
  mat: b_leaf=-1.0, T=1.0, protect trunk hard labels

Usage:
  python simple_multimodal_pipeline.py              # robot eval (diagnostic)
  python simple_multimodal_pipeline.py --variant amb_sec_leaf
  python simple_multimodal_pipeline.py --variant amb_leaf   # no secondary
  python simple_multimodal_pipeline.py --variant claim_full # old 4-HP material

NOTE: This is an engineering simplification script. Paper claim seal lives under
outputs/paper_claim_v1/. Robot numbers here are for ablation / simplicity study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
V2 = Path(
    "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
    "segment_rule_stack_v2_base_segment_outputs.npz"
)
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
}
AUDIO_H = {
    "clean": Path(
        "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"
    ),
    "robot_mix": Path(
        "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/robot_mix/X.npy"
    ),
    "bandlimit": Path(
        "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/bandlimit/X.npy"
    ),
}
AUDIO_R = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/X.npy"
)
W2V_H = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")
W2V_R = Path("outputs/audio_wav2vec2_features/robot_test_X.npy")

# Frozen simple HPs
AMB = dict(ts_th=0.5, cs_th=0.1)
SEC = dict(img_th=0.625, img_cs=0.75)
LEAF_BIAS = -1.0  # only material HP


def pool_files(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


def fit_bin(x, y, c=0.05):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42),
    )
    m.fit(x, y)
    return m


def pos(m, x):
    p = m.predict_proba(x)
    cl = list(m.classes_)
    return p[:, cl.index(1)] if 1 in cl else p[:, -1]


def predict(
    variant: str = "amb_sec_leaf",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (y_true_rows, pred_rows, info)."""
    # --- hand fit detectors ---
    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    emb_h = {n: np.load(p / "hand_train_full/X.npy").astype(np.float32) for n, p in IMG.items()}
    hu, hc, _, hy = pool_files(hfr.audio_file, emb_h["clip"], yh)
    hx = {n: np.vstack([emb_h[n][hc == i].mean(0) for i in range(len(hu))]) for n in emb_h}
    multi_h = np.hstack([hx[k] for k in IMG])
    ha = {k: np.load(p).astype(np.float32) for k, p in AUDIO_H.items()}
    ha_s = {k: np.vstack([ha[k][hc == i].mean(0) for i in range(len(hu))]) for k in ha}
    hw_s = np.vstack([np.load(W2V_H).astype(np.float32)[hc == i].mean(0) for i in range(len(hu))])

    m_bin = fit_bin(hx["clip"], (hy > 0).astype(int), 0.03)
    m_tr = fit_bin(hx["clip"], (hy == 2).astype(int), 0.1)
    Xst = np.vstack(
        [np.hstack([ha_s[v], hw_s, multi_h]) for v in ("robot_mix", "bandlimit")]
    )
    m_ts = fit_bin(Xst, np.concatenate([(hy == 2).astype(int)] * 2), 0.05)
    m_cs = fit_bin(Xst, np.concatenate([(hy > 0).astype(int)] * 2), 0.05)

    # --- robot base ---
    z = np.load(V2, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred_seg = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"])
    p_meta = z["p_meta"].astype(np.float64)
    p_h = z["p_h"].astype(np.float64)

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    emb_t = {n: np.load(p / "robot_test/X.npy").astype(np.float32) for n, p in IMG.items()}
    tu, tc, _, _ = pool_files(tfr.audio_file, emb_t["clip"], yt)
    order = {k: i for i, k in enumerate(ids)}
    pred = np.array([pred_seg[order[str(k)]] for k in tu])
    soft_meta = np.array([p_meta[order[str(k)]] for k in tu])
    soft_hier = np.array([p_h[order[str(k)]] for k in tu])

    tx = {n: np.vstack([emb_t[n][tc == i].mean(0) for i in range(len(tu))]) for n in emb_t}
    multi_t = np.hstack([tx[k] for k in IMG])
    ta_s = np.vstack(
        [np.load(AUDIO_R).astype(np.float32)[tc == i].mean(0) for i in range(len(tu))]
    )
    tw_s = np.vstack(
        [np.load(W2V_R).astype(np.float32)[tc == i].mean(0) for i in range(len(tu))]
    )
    Xr = np.hstack([ta_s, tw_s, multi_t])
    bins = pos(m_bin, tx["clip"])
    tr = pos(m_tr, tx["clip"])
    ts = pos(m_ts, Xr)
    cs = pos(m_cs, Xr)

    use_amb = variant in ("amb_sec_leaf", "amb_leaf", "claim_full", "amb_sec")
    use_sec = variant in ("amb_sec_leaf", "claim_full", "amb_sec", "sec_leaf")
    use_leaf = variant in ("amb_sec_leaf", "amb_leaf", "sec_leaf", "leaf_only")
    use_claim_mat = variant == "claim_full"

    if use_amb:
        pred[
            (pred == 0)
            & (ts >= AMB["ts_th"])
            & (np.maximum(cs, bins) >= AMB["cs_th"])
        ] = 2
    if use_sec:
        pred[
            (pred == 0)
            & (tr >= SEC["img_th"])
            & (bins >= SEC["img_cs"])
        ] = 2

    if use_claim_mat:
        # old 4-HP material (for comparison)
        soft = soft_meta.copy()
        h = soft_hier.argmax(1)
        soft[h == 2] = soft_hier[h == 2]
        m = (pred == 1) | (pred == 3)
        if m.any():
            logits = np.log(np.clip(soft[m][:, 1:4], 1e-12, None)) / 0.8
            logits = logits + np.array([-1.0, 0.0, 0.6])
            pred[m] = 1 + logits.argmax(1)
    elif use_leaf:
        # SIMPLE material: one bias on meta, protect trunk
        m = (pred == 1) | (pred == 3)
        if m.any():
            logits = np.log(np.clip(soft_meta[m][:, 1:4], 1e-12, None))
            logits = logits + np.array([LEAF_BIAS, 0.0, 0.0])
            pred[m] = 1 + logits.argmax(1)

    sk = sel.seg(tfr.audio_file).to_numpy()
    ou = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred[ou[k]] for k in sk])
    info = {
        "variant": variant,
        "use_amb": use_amb,
        "use_sec": use_sec,
        "use_leaf_bias": use_leaf or use_claim_mat,
        "leaf_bias": LEAF_BIAS if use_leaf else None,
        "AMB": AMB,
        "SEC": SEC,
        "dropped": [
            "do_tt / twig->trunk",
            "b_trunk",
            "b_twig" if not use_claim_mat else "(claim keeps b_twig)",
            "T!=1" if not use_claim_mat else "(claim T=0.8)",
            "v2_rule_soft" if not use_claim_mat else None,
        ],
    }
    return yt, pred_rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        default="amb_sec_leaf",
        choices=[
            "amb_sec_leaf",  # simple best (~0.88)
            "amb_leaf",  # no secondary (~0.87)
            "sec_leaf",  # no amb (~0.85)
            "leaf_only",  # v2+leaf only (~0.84)
            "amb_sec",  # contact only (~0.83)
            "claim_full",  # old multi-HP material (~0.85)
        ],
    )
    ap.add_argument(
        "--out",
        default="outputs/simple_pipeline/metrics.json",
        help="write metrics JSON",
    )
    args = ap.parse_args()

    y, pred, info = predict(args.variant)
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist()
    report = classification_report(
        y,
        pred,
        labels=[0, 1, 2, 3],
        target_names=["ambient", "leaf", "trunk", "twig"],
        output_dict=True,
        zero_division=0,
    )
    result = {
        "protocol": "simple_multimodal_pipeline_diagnostic",
        "macro_f1_4class": f1,
        "confusion_matrix_4class": cm,
        "per_class": report,
        "info": info,
        "note": "Diagnostic simplification study; not the sealed paper_claim_v1 path.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
