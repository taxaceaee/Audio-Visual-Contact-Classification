"""One-shot robot test: v2 base + stress-trained trunk rescue (strict)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import multimodal_085_protocol as proto
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_v2_trunk_stress_group_selection",
    )
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
META_FEAT = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_features.npy")
META_Y = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_y.npy")


def pool_files(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


def fit_bin(x, y, c=0.1):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42),
    )
    m.fit(x, y)
    return m


def pos_proba(m, x):
    p = m.predict_proba(x)
    classes = list(m.classes_)
    return p[:, classes.index(1)] if 1 in classes else p[:, -1]


def main():
    lock = proto.load_selection_lock(OUT / "selection_lock.json")
    sc = lock["selected_candidate"]
    v2p = lock["v2_params"]
    alpha, tb, th = float(v2p["alpha"]), float(v2p["tb"]), float(v2p["th"])
    mw = np.array(v2p["mw"], dtype=np.float64)
    t_th = float(sc["t_th"])
    c_th = float(sc["c_th"])
    mode = sc["mode"]
    require_meta_not_leaf = bool(sc["require_meta_not_leaf"])

    audio_base.configure_feature_set("total240")
    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    emb_h = {n: np.load(p / "hand_train_full/X.npy").astype(np.float32) for n, p in IMG.items()}
    hu, hc, _, hy = pool_files(hfr.audio_file, emb_h["clip"], yh)
    hx = {n: np.vstack([emb_h[n][hc == i].mean(0) for i in range(len(hu))]) for n in emb_h}
    hx_c, hx_cat = hx["clip"], np.hstack([hx["clip"], hx["eff"]])
    hx_multi = np.hstack([hx[k] for k in ("clip", "eff", "dino", "convnext")])
    ha = {k: np.load(p).astype(np.float32) for k, p in AUDIO_H.items()}
    ha_seg = {k: np.vstack([ha[k][hc == i].mean(0) for i in range(len(hu))]) for k in ha}
    hw = np.load(W2V_H).astype(np.float32)
    hw_seg = np.vstack([hw[hc == i].mean(0) for i in range(len(hu))])

    # fit heads
    m_bin = fit_bin(hx_c, (hy > 0).astype(int), 0.03)
    m_tr = fit_bin(hx_c, (hy == 2).astype(int), 0.1)
    m_mat = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.03,
            max_iter=1500,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42,
        ),
    )
    m_mat.fit(hx_cat[hy > 0], hy[hy > 0] - 1)
    m_img4 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.03,
            max_iter=1500,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42,
        ),
    )
    m_img4.fit(hx_c, hy)
    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=4000,
            class_weight={0: 1, 1: 1, 2: 1.2, 3: 1.2},
            multi_class="multinomial",
            random_state=42,
        ),
    )
    meta.fit(np.load(META_FEAT), np.load(META_Y))

    Xstress = np.vstack(
        [
            np.hstack([ha_seg["robot_mix"], hw_seg, hx_multi]),
            np.hstack([ha_seg["bandlimit"], hw_seg, hx_multi]),
        ]
    )
    y_tr = (hy == 2).astype(int)
    y_c = (hy > 0).astype(int)
    m_ts = fit_bin(Xstress, np.concatenate([y_tr, y_tr]), 0.05)
    m_cs = fit_bin(Xstress, np.concatenate([y_c, y_c]), 0.05)
    m_tc = fit_bin(np.hstack([ha_seg["clean"], hw_seg, hx_multi]), y_tr, 0.05)

    # robot
    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    emb_t = {n: np.load(p / "robot_test/X.npy").astype(np.float32) for n, p in IMG.items()}
    tu, tc, _, ty = pool_files(tfr.audio_file, emb_t["clip"], yt)
    tx = {n: np.vstack([emb_t[n][tc == i].mean(0) for i in range(len(tu))]) for n in emb_t}
    tx_c = tx["clip"]
    tx_cat = np.hstack([tx["clip"], tx["eff"]])
    tx_multi = np.hstack([tx[k] for k in ("clip", "eff", "dino", "convnext")])
    ta = np.load(AUDIO_R).astype(np.float32)
    ta_seg = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_R).astype(np.float32)
    tw_seg = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])
    Xr = np.hstack([ta_seg, tw_seg, tx_multi])

    audio = pd.read_csv(
        "outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/"
        "audio_lift_source_blend_select_final_test_predictions.csv"
    )
    raw = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    assert np.array_equal(
        audio.audio_file.astype(str).to_numpy(), raw.audio_file.astype(str).to_numpy()
    )
    proba = suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy(np.float64))
    pc, lg = avr.avr_inputs(proba)
    sk = sel.seg(tfr.audio_file).to_numpy()
    sa_mat = np.vstack(
        [suite.normalize(np.exp(lg[sk == k].mean(0, keepdims=True)))[0] for k in tu]
    )
    spc = np.array([pc[sk == k].mean() for k in tu], np.float64)

    bin_s = pos_proba(m_bin, tx_c)
    trunk_img = pos_proba(m_tr, tx_c)
    mat = suite.normalize(m_mat.predict_proba(tx_cat))
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1))
        logits[:, 1] += tb * trunk_img
        mat = proto.normalize(np.exp(logits))
    mat = proto.normalize(sa_mat * (1 - mw) + mat * mw)
    contact = alpha * spc + (1 - alpha) * bin_s
    pred_h = np.zeros(len(tu), np.int64)
    hit = contact >= th
    pred_h[hit] = 1 + mat[hit].argmax(1)
    img_mat3 = suite.normalize(m_img4.predict_proba(tx_c)[:, 1:])
    z = sel.feat_meta(sa_mat, img_mat3, spc)
    p_meta = meta.predict_proba(z)
    # align classes
    p_meta_full = np.zeros((len(tu), 4), np.float64)
    for j, c in enumerate(meta.classes_):
        p_meta_full[:, int(c)] = p_meta[:, j]
    p_meta_full = proto.normalize(p_meta_full)
    pred_m = p_meta_full.argmax(1)
    pred_v2 = np.where(pred_h == 2, pred_h, pred_m)

    trunk_stress = pos_proba(m_ts, Xr)
    trunk_clean = pos_proba(m_tc, Xr)
    contact_stress = pos_proba(m_cs, Xr)

    if mode == "v2_only" or t_th < 0:
        pred_seg = pred_v2
    else:
        if mode == "stress_trunk":
            tscore = trunk_stress
        elif mode == "max_trunk":
            tscore = np.maximum(trunk_stress, trunk_img)
        elif mode == "stress_trunk_and_contact":
            tscore = trunk_stress
        else:
            tscore = np.maximum(trunk_stress, trunk_clean)
        pred_seg = pred_v2.copy()
        mask = pred_seg == 0
        mask &= tscore >= t_th
        if mode == "stress_trunk_and_contact":
            mask &= contact_stress >= c_th
        else:
            mask &= np.maximum(bin_s, contact_stress) >= c_th
        if require_meta_not_leaf:
            mask &= pred_m != 1
        pred_seg[mask] = 2

    order = {k: i for i, k in enumerate(tu)}
    pred = np.array([pred_seg[order[k]] for k in sk])
    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "locked_candidate": sc,
        **proto.metrics_bundle(yt, pred),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "no_test_hp_tuning": True,
            "filename_class_features": False,
            "robot_uda": False,
        },
    }
    proto.write_final_metrics(OUT / "multimodal_085_v2_trunk_stress_final_test_metrics.json", result)
    lock_after = dict(lock)
    lock_after["test_loaded"] = True
    lock_after["final_test_macro_f1"] = result["metrics"]["macro_f1_4class"]
    (OUT / "selection_lock_after_test.json").write_text(
        json.dumps(lock_after, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
