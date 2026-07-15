"""One-shot robot/test for multimodal_085_drive after hand selection lock."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import multimodal_085_protocol as proto
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_drive_group_selection",
    )
)
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
}
AUDIO_CLEAN_H = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"
)
AUDIO_CLEAN_R = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/X.npy"
)
AUDIO_MIX_H = Path(
    "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/robot_mix/X.npy"
)
AUDIO_BAND_H = Path(
    "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/bandlimit/X.npy"
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


def fit_lr(x, y, c=0.1, multi=True):
    if multi and len(np.unique(y)) > 2:
        clf = LogisticRegression(
            C=c,
            max_iter=2500,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42,
        )
    else:
        clf = LogisticRegression(C=c, max_iter=2500, class_weight="balanced", random_state=42)
    m = make_pipeline(StandardScaler(), clf)
    m.fit(x, y)
    return m


def proba4(model, x):
    p = model.predict_proba(x)
    out = np.zeros((len(x), 4), np.float64)
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, j]
    return proto.normalize(out)


def proba1(model, x):
    p = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 in classes:
        return p[:, classes.index(1)]
    return p[:, -1]


def make_mat(sa_mat, imat, w, tb, trunk_s):
    mat = proto.normalize(sa_mat * (1.0 - w) + imat * w)
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1.0))
        logits[:, 1] += tb * trunk_s
        mat = proto.normalize(np.exp(logits))
    return mat


def decode_hier(contact, mat3, th):
    pred = np.zeros(len(contact), np.int64)
    hit = contact >= th
    pred[hit] = 1 + mat3[hit].argmax(1)
    return pred


def main():
    lock = proto.load_selection_lock(OUT / "selection_lock.json")
    sc = lock["selected_candidate"]
    alpha = float(sc["alpha"])
    th = float(sc["th"])
    tb = float(sc["tb"])
    mw = np.array(sc["mw"], dtype=np.float64)
    bin_src = sc["bin_src"]
    mat_src = sc["mat_src"]
    rule = sc["rule"]
    jw = float(sc["jw"])
    trunk_force = float(sc["trunk_force"])

    audio_base.configure_feature_set("total240")

    # ---- fit on full hand ----
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
    ha = {
        "clean": np.load(AUDIO_CLEAN_H).astype(np.float32),
        "robot_mix": np.load(AUDIO_MIX_H).astype(np.float32),
        "bandlimit": np.load(AUDIO_BAND_H).astype(np.float32),
    }
    ha_seg = {
        v: np.vstack([ha[v][hc == i].mean(0) for i in range(len(hu))]) for v in ha
    }
    hw = np.load(W2V_H).astype(np.float32)
    hw_seg = np.vstack([hw[hc == i].mean(0) for i in range(len(hu))])

    # locked audio hand OOF for segment sa
    af = audio_base.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train"
    )
    high = group_audio.load_highsr_oof(Path("outputs"))
    pair = specimen.load_pairwise_oof(
        Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        )
    )
    anchor = lift.anchor_lift_proba(af, high, pair)
    sg = sel.spec(hfr.audio_file).to_numpy()
    assignment = np.full(len(yh), -1, np.int64)
    for k, (_, va) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(yh)), yh, sg
        )
    ):
        assignment[va] = k
    src = broad.load_train_sources(Path("outputs"), yh, assignment, 42)
    audio_h = lift.postprocess(
        af, lift.normalize(0.95 * anchor + 0.05 * src["report_gate_onehot"]), "segment_lift"
    )
    pc_h, lg_h = avr.avr_inputs(audio_h)
    sk_h = sel.seg(hfr.audio_file).to_numpy()
    h_sa_mat = np.vstack(
        [suite.normalize(np.exp(lg_h[sk_h == k].mean(0, keepdims=True)))[0] for k in hu]
    )
    h_spc = np.array([pc_h[sk_h == k].mean() for k in hu], np.float64)

    m_clip4 = fit_lr(hx_c, hy, 0.03)
    m_w2v = fit_lr(hw_seg, hy, 0.1)
    m_bin = fit_lr(hx_c, (hy > 0).astype(int), 0.03, multi=False)
    m_bin_m = fit_lr(hx_multi, (hy > 0).astype(int), 0.03, multi=False)
    m_trunk = fit_lr(hx_c, (hy == 2).astype(int), 0.1, multi=False)
    contact_h = hy > 0
    m_mat = fit_lr(hx_cat[contact_h], hy[contact_h] - 1, 0.03)
    m_mat_m = fit_lr(hx_multi[contact_h], hy[contact_h] - 1, 0.03)

    # joint bag multi-view
    Xbag = np.vstack(
        [
            np.hstack([ha_seg[v], hw_seg, hx_multi])
            for v in ("clean", "robot_mix", "bandlimit")
        ]
    )
    ybag = np.concatenate([hy, hy, hy])
    m_joint = fit_lr(Xbag, ybag, 0.05)
    m_trunk_j = fit_lr(Xbag, (ybag == 2).astype(int), 0.1, multi=False)

    meta = fit_lr(np.load(META_FEAT), np.load(META_Y), 0.1)

    # ---- robot AFTER lock ----
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
    ta = np.load(AUDIO_CLEAN_R).astype(np.float32)
    ta_seg = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_R).astype(np.float32)
    tw_seg = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])

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

    bin_clip = proba1(m_bin, tx_c)
    bin_multi = proba1(m_bin_m, tx_multi)
    trunk_clip = proba1(m_trunk, tx_c)
    if bin_src == "clip":
        b = bin_clip
    elif bin_src == "multi":
        b = bin_multi
    elif bin_src == "max":
        b = np.maximum(bin_clip, bin_multi)
    else:
        b = 0.5 * (bin_clip + bin_multi)
    contact = alpha * spc + (1 - alpha) * b

    mat_ce = suite.normalize(m_mat.predict_proba(tx_cat))
    mat_m = suite.normalize(m_mat_m.predict_proba(tx_multi))
    if mat_src == "clip_eff":
        imat = mat_ce
    elif mat_src == "multi":
        imat = mat_m
    else:
        imat = proto.normalize(0.5 * mat_ce + 0.5 * mat_m)
    trunk_j = proba1(m_trunk_j, np.hstack([ta_seg, tw_seg, tx_multi]))
    trunk_s = np.maximum(trunk_clip, trunk_j)
    mat = make_mat(sa_mat, imat, mw, tb, trunk_s)
    pred_h = decode_hier(contact, mat, th)

    img_mat3 = suite.normalize(m_clip4.predict_proba(tx_c)[:, 1:])
    z_meta = sel.feat_meta(sa_mat, img_mat3, spc)
    p_meta = proba4(meta, z_meta)
    meta_arg = p_meta.argmax(1)
    pj = proba4(m_joint, np.hstack([ta_seg, tw_seg, tx_multi]))

    if rule == "hier_only":
        pred_seg = pred_h.copy()
    elif rule == "hier_trunk_meta_else":
        pred_seg = np.where(pred_h == 2, pred_h, meta_arg)
    elif rule == "max_contact_then_mat":
        contact2 = np.maximum(spc, b)
        pred_seg = np.where(contact2 >= th, 1 + mat.argmax(1), 0)
    elif rule == "joint_if_conf":
        pred_seg = np.where(pj.max(1) >= p_meta.max(1), pj.argmax(1), meta_arg)
        pred_seg = np.where(pred_h == 2, 2, pred_seg)
    elif rule == "soft_v2_joint":
        p_v2 = p_meta.copy()
        p_h = np.zeros_like(p_v2)
        p_h[np.arange(len(p_h)), pred_h] = 1.0
        p_v2[pred_h == 2] = p_h[pred_h == 2]
        pred_seg = proto.normalize((1 - jw) * p_v2 + jw * pj).argmax(1)
    elif rule == "joint_only":
        pred_seg = pj.argmax(1)
    elif rule == "hier_trunk_joint_else":
        pred_seg = np.where(pred_h == 2, 2, pj.argmax(1))
    else:
        raise ValueError(rule)

    if trunk_force > 0:
        tscore = np.maximum(trunk_clip, trunk_j)
        bscore = np.maximum(bin_clip, bin_multi)
        force = (pred_seg == 0) & (tscore >= trunk_force) & (bscore >= 0.35)
        pred_seg = pred_seg.copy()
        pred_seg[force] = 2

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
    proto.write_final_metrics(OUT / "multimodal_085_drive_final_test_metrics.json", result)
    lock_after = dict(lock)
    lock_after["test_loaded"] = True
    lock_after["final_test_macro_f1"] = result["metrics"]["macro_f1_4class"]
    (OUT / "selection_lock_after_test.json").write_text(
        json.dumps(lock_after, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
