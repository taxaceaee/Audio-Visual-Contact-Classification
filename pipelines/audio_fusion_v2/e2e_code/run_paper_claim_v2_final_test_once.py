#!/usr/bin/env python3
"""PAPER CLAIM v2 (simple) — pure single-shot robot open + seal.

Requirements:
  - hand selection_lock exists
  - claim not sealed
  - recipe hash matches RECIPE_v2_simple.json
  - robot opened exactly once

Usage:
  python run_paper_claim_v2_final_test_once.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import paper_claim_protocol as claim
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
RECIPE_PATH = claim.RECIPE_V2_PATH
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


def main():
    recipe = claim.load_recipe(RECIPE_PATH)
    out_dir = claim.claim_out_dir(recipe)
    claim.assert_not_sealed(out_dir)
    lock = claim.load_hand_selection_lock(out_dir, recipe_path=RECIPE_PATH)

    primary = recipe["frozen_contact_stack"]["primary"]
    secondary = recipe["frozen_contact_stack"]["secondary"]
    sc = lock["selected_candidate"]
    mode = sc.get("mode", "protect_trunk_leaf_bias_only")
    b_leaf = float(sc.get("b_leaf", 0.0))

    v2_path = Path(recipe["artifacts"]["v2_robot_base"])

    # Fit detectors on hand only (no robot labels)
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
    # no m_tt — simplified v2

    # --- first and only robot load for this claim ---
    z = np.load(v2_path, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred_seg = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"])
    p_meta = z["p_meta"].astype(np.float64)

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

    # frozen contact (no do_tt)
    pred[
        (pred == 0)
        & (ts >= primary["ts_th"])
        & (np.maximum(cs, bins) >= primary["cs_th"])
    ] = 2
    pred[
        (pred == 0)
        & (tr >= secondary["img_th"])
        & (bins >= secondary["img_cs"])
    ] = 2

    if mode not in ("none", "identity"):
        m = (pred == 1) | (pred == 3)
        if m.any():
            logits = np.log(np.clip(soft_meta[m][:, 1:4], 1e-12, None))
            logits = logits + np.array([b_leaf, 0.0, 0.0], dtype=np.float64)
            pred[m] = 1 + logits.argmax(1)

    sk = sel.seg(tfr.audio_file).to_numpy()
    ou = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred[ou[k]] for k in sk])

    result = {
        "split": "robot_test_final",
        "protocol": "paper_claim_v2_simple_pure_single_shot",
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": lock["recipe_sha256"],
        "locked_candidate": sc,
        "frozen_contact_stack": recipe["frozen_contact_stack"],
        "simplification": recipe.get("simplification_vs_v1"),
        **claim.metrics_bundle(yt, pred_rows),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "no_test_hp_tuning": True,
            "filename_class_features": False,
            "robot_uda": False,
            "pure_single_shot_architecture": True,
            "robot_opened_once_for_claim": True,
            "simplified_v2": True,
            "free_material_params": ["b_leaf"],
            "no_do_tt": True,
            "soft_source": "meta",
        },
    }
    (out_dir / "final_test_metrics.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    seal = claim.seal_claim(out_dir, lock, result)
    print(json.dumps({"result": result, "seal": seal}, indent=2, default=float))


if __name__ == "__main__":
    main()
