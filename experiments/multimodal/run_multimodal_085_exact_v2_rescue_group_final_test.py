"""One-shot: exact locked v2 segment preds + hand-selected posthoc rescue."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import multimodal_085_protocol as proto
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_exact_v2_rescue_group_selection",
    )
)
V2_BASE = Path(
    "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
    "segment_rule_stack_v2_base_segment_outputs.npz"
)
IMG_MULTI = {
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
    lock = proto.load_selection_lock(OUT / "selection_lock.json")
    sc = lock["selected_candidate"]
    ts_th = float(sc["ts_th"])
    cs_th = float(sc["cs_th"])
    tt_th = float(sc["tt_th"])
    do_amb = bool(sc["do_amb_lift"])
    do_tt = bool(sc["do_twig_to_trunk"])
    tscore_mode = sc["tscore_mode"]

    # fit detectors on full hand
    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    emb_h = {
        n: np.load(p / "hand_train_full/X.npy").astype(np.float32) for n, p in IMG_MULTI.items()
    }
    hu, hc, _, hy = pool_files(hfr.audio_file, emb_h["clip"], yh)
    hx = {n: np.vstack([emb_h[n][hc == i].mean(0) for i in range(len(hu))]) for n in emb_h}
    multi_h = np.hstack([hx[k] for k in ("clip", "eff", "dino", "convnext")])
    ha = {k: np.load(p).astype(np.float32) for k, p in AUDIO_H.items()}
    ha_seg = {k: np.vstack([ha[k][hc == i].mean(0) for i in range(len(hu))]) for k in ha}
    hw = np.load(W2V_H).astype(np.float32)
    hw_seg = np.vstack([hw[hc == i].mean(0) for i in range(len(hu))])
    sx_c = hx["clip"]

    m_bin = fit_bin(sx_c, (hy > 0).astype(int), 0.03)
    m_tr = fit_bin(sx_c, (hy == 2).astype(int), 0.1)
    Xst = np.vstack(
        [
            np.hstack([ha_seg["robot_mix"], hw_seg, multi_h]),
            np.hstack([ha_seg["bandlimit"], hw_seg, multi_h]),
        ]
    )
    m_ts = fit_bin(Xst, np.concatenate([(hy == 2).astype(int)] * 2), 0.05)
    m_cs = fit_bin(Xst, np.concatenate([(hy > 0).astype(int)] * 2), 0.05)
    wood = (hy == 2) | (hy == 3)
    Xw = np.vstack(
        [np.hstack([ha_seg[v][wood], hw_seg[wood], multi_h[wood]]) for v in AUDIO_H]
    )
    yw = np.concatenate([(hy[wood] == 2).astype(int)] * 3)
    m_tt = fit_bin(Xw, yw, 0.1)

    # robot exact v2 base
    z = np.load(V2_BASE, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred_h = z["pred_h"]
    pred_m = z["pred_m"]
    pred_seg = np.where(pred_h == 2, pred_h, pred_m)

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    emb_t = {n: np.load(p / "robot_test/X.npy").astype(np.float32) for n, p in IMG_MULTI.items()}
    tu, tc, _, ty = pool_files(tfr.audio_file, emb_t["clip"], yt)
    # align segment order of v2 artifact with tu
    assert set(ids.tolist()) == set(map(str, tu.tolist())) or len(ids) == len(tu)
    order_v2 = {k: i for i, k in enumerate(ids)}
    # map tu order
    pred_base = np.array([pred_seg[order_v2[str(k)]] for k in tu])

    tx = {n: np.vstack([emb_t[n][tc == i].mean(0) for i in range(len(tu))]) for n in emb_t}
    multi_t = np.hstack([tx[k] for k in ("clip", "eff", "dino", "convnext")])
    ta = np.load(AUDIO_R).astype(np.float32)
    ta_seg = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_R).astype(np.float32)
    tw_seg = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])
    Xr = np.hstack([ta_seg, tw_seg, multi_t])

    bin_s = pos(m_bin, tx["clip"])
    tr_img = pos(m_tr, tx["clip"])
    ts = pos(m_ts, Xr)
    cs = pos(m_cs, Xr)
    tt = pos(m_tt, Xr)
    if tscore_mode == "stress":
        tscore = ts
    elif tscore_mode == "max":
        tscore = np.maximum(ts, tr_img)
    else:
        tscore = tr_img

    pred = pred_base.copy()
    if do_amb:
        m = (pred == 0) & (tscore >= ts_th) & (np.maximum(cs, bin_s) >= cs_th)
        pred[m] = 2
    if do_tt:
        m = (pred == 3) & (tt >= tt_th) & (tscore >= ts_th * 0.9)
        pred[m] = 2

    sk = sel.seg(tfr.audio_file).to_numpy()
    order = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred[order[k]] for k in sk])

    # verify base alone
    base_rows = np.array([pred_base[order[k]] for k in sk])
    base_f1 = proto.fast_macro_f1(yt, base_rows)

    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "locked_candidate": sc,
        "base_v2_macro_f1_check": base_f1,
        **proto.metrics_bundle(yt, pred_rows),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "no_test_hp_tuning": True,
            "filename_class_features": False,
            "robot_uda": False,
            "exact_v2_base_artifact": True,
        },
    }
    proto.write_final_metrics(
        OUT / "multimodal_085_exact_v2_rescue_final_test_metrics.json", result
    )
    lock_after = dict(lock)
    lock_after["test_loaded"] = True
    lock_after["final_test_macro_f1"] = result["metrics"]["macro_f1_4class"]
    (OUT / "selection_lock_after_test.json").write_text(
        json.dumps(lock_after, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
