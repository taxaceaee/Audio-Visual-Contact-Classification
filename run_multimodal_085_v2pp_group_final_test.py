"""One-shot robot test for v2++ contact-lift + material/tt override."""
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
        "SEG_OUT", "outputs/audio_feature_benchmarks/multimodal_085_v2pp_group_selection"
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


def fit_multi(x, y, c=0.03):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42,
        ),
    )
    m.fit(x, y)
    return m


def pos(m, x):
    p = m.predict_proba(x)
    cl = list(m.classes_)
    return p[:, cl.index(1)] if 1 in cl else p[:, -1]


def main():
    lock = proto.load_selection_lock(OUT / "selection_lock.json")
    sc = lock["selected_candidate"]
    v2p = lock["v2_params"]
    alpha, tb, th = float(v2p["alpha"]), float(v2p["tb"]), float(v2p["th"])
    mw = np.array(v2p["mw"], dtype=np.float64)

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
    hx_c = hx["clip"]
    hx_cat = np.hstack([hx["clip"], hx["eff"]])
    hx_multi = np.hstack([hx[k] for k in ("clip", "eff", "dino", "convnext")])
    ha = {k: np.load(p).astype(np.float32) for k, p in AUDIO_H.items()}
    ha_seg = {k: np.vstack([ha[k][hc == i].mean(0) for i in range(len(hu))]) for k in ha}
    hw = np.load(W2V_H).astype(np.float32)
    hw_seg = np.vstack([hw[hc == i].mean(0) for i in range(len(hu))])

    m_bin = fit_bin(hx_c, (hy > 0).astype(int), 0.03)
    m_tr = fit_bin(hx_c, (hy == 2).astype(int), 0.1)
    m_mat = fit_multi(hx_cat[hy > 0], hy[hy > 0] - 1)
    m_mat_m = fit_multi(hx_multi[hy > 0], hy[hy > 0] - 1)
    m_img4 = fit_multi(hx_c, hy, 0.03)
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
    Xst = np.vstack(
        [
            np.hstack([ha_seg["robot_mix"], hw_seg, hx_multi]),
            np.hstack([ha_seg["bandlimit"], hw_seg, hx_multi]),
        ]
    )
    m_ts = fit_bin(Xst, np.concatenate([(hy == 2).astype(int)] * 2), 0.05)
    m_cs = fit_bin(Xst, np.concatenate([(hy > 0).astype(int)] * 2), 0.05)
    wood = (hy == 2) | (hy == 3)
    Xw = np.vstack(
        [
            np.hstack([ha_seg["clean"][wood], hw_seg[wood], hx_multi[wood]]),
            np.hstack([ha_seg["robot_mix"][wood], hw_seg[wood], hx_multi[wood]]),
            np.hstack([ha_seg["bandlimit"][wood], hw_seg[wood], hx_multi[wood]]),
        ]
    )
    yw = np.concatenate([(hy[wood] == 2).astype(int)] * 3)
    m_tt = fit_bin(Xw, yw, 0.1)

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

    bin_s = pos(m_bin, tx_c)
    trunk = pos(m_tr, tx_c)
    mat = suite.normalize(m_mat.predict_proba(tx_cat))
    mat_m = suite.normalize(m_mat_m.predict_proba(tx_multi))
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1))
        logits[:, 1] += tb * trunk
        mat = proto.normalize(np.exp(logits))
    mat_v2 = proto.normalize(sa_mat * (1 - mw) + mat * mw)
    contact = alpha * spc + (1 - alpha) * bin_s
    pred_h = np.zeros(len(tu), np.int64)
    hit = contact >= th
    pred_h[hit] = 1 + mat_v2[hit].argmax(1)
    img_mat3 = suite.normalize(m_img4.predict_proba(tx_c)[:, 1:])
    z = sel.feat_meta(sa_mat, img_mat3, spc)
    pm = meta.predict_proba(z)
    p_meta = np.zeros((len(tu), 4))
    for j, c in enumerate(meta.classes_):
        p_meta[:, int(c)] = pm[:, j]
    pred_m = proto.normalize(p_meta).argmax(1)
    pred = np.where(pred_h == 2, pred_h, pred_m)

    ts = pos(m_ts, Xr)
    cs = pos(m_cs, Xr)
    tt = pos(m_tt, Xr)
    mw_img = np.array([0.35, float(sc["mw_trunk"]), 0.35])
    mat_img = mat_m.copy()
    if float(sc["tb_img"]):
        logits = np.log(np.clip(mat_img, 1e-8, 1))
        logits[:, 1] += float(sc["tb_img"]) * trunk
        mat_img = proto.normalize(np.exp(logits))
    # map material weights
    mat_img = proto.normalize(sa_mat * (1 - mw_img) + mat_img * mw_img)
    img_w = float(sc["img_mat_w"])
    mat_blend = proto.normalize((1 - img_w) * mat_v2 + img_w * mat_img)

    lift_mask = (pred == 0) & (ts >= float(sc["ts_th"])) & (
        np.maximum(cs, bin_s) >= float(sc["cs_th"])
    )
    pred = pred.copy()
    if sc["lift_mode"] == "force_trunk":
        pred[lift_mask] = 2
    else:
        pred[lift_mask] = 1 + mat_blend[lift_mask].argmax(1)

    tt_mode = sc["tt_mode"]
    tt_th = float(sc["tt_th"])
    wood_mask = (pred == 2) | (pred == 3)
    if tt_mode == "tt_head":
        pred[wood_mask & (tt >= tt_th)] = 2
        if sc["allow_trunk_to_twig"]:
            lean_twig = mat_blend[:, 2] > mat_blend[:, 1]
            pred[wood_mask & (tt <= 1 - tt_th) & (pred == 2) & lean_twig] = 3
    elif tt_mode == "mat_override":
        conf = mat_blend.max(1)
        arg = mat_blend.argmax(1) + 1
        pred[wood_mask & (conf >= tt_th)] = arg[wood_mask & (conf >= tt_th)]
    if sc["twig_to_trunk"]:
        pred[(pred == 3) & (tt >= tt_th) & (trunk >= 0.45)] = 2

    order = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred[order[k]] for k in sk])
    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "locked_candidate": sc,
        **proto.metrics_bundle(yt, pred_rows),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "no_test_hp_tuning": True,
            "filename_class_features": False,
            "robot_uda": False,
        },
    }
    proto.write_final_metrics(OUT / "multimodal_085_v2pp_final_test_metrics.json", result)
    lock_after = dict(lock)
    lock_after["test_loaded"] = True
    lock_after["final_test_macro_f1"] = result["metrics"]["macro_f1_4class"]
    (OUT / "selection_lock_after_test.json").write_text(
        json.dumps(lock_after, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
