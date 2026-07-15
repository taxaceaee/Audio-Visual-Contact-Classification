"""One-shot robot: locked 0.837 contact + hand-selected leaf-contaminant material decode."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
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
        "outputs/audio_feature_benchmarks/multimodal_085_leaf_contaminant_mat_group_selection",
    )
)
V2_BASE = Path(
    "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
    "segment_rule_stack_v2_base_segment_outputs.npz"
)
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
    "swin": Path("outputs/image_timm_features/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"),
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
CLAP_H = Path("outputs/audio_clap_features/hand_train_full_X.npy")
CLAP_R = Path("outputs/audio_clap_features/robot_test_X.npy")

PRIMARY = dict(ts_th=0.5, cs_th=0.1, tt_th=0.45, do_tt=True)
SECONDARY = dict(th=0.625, cth=0.75)


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


def fit_multi(x, y, c=0.1):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=2500,
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


def proba_aligned(m, x, n_cls=3):
    p = m.predict_proba(x)
    out = np.zeros((len(x), n_cls), dtype=np.float64)
    for j, c in enumerate(m.classes_):
        if 0 <= int(c) < n_cls:
            out[:, int(c)] = p[:, j]
    s = out.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    return out / s


def material_decode(contact, soft4, T, b_leaf, b_trunk, b_twig):
    out = contact.copy()
    m = out > 0
    if not m.any():
        return out
    logits = np.log(np.clip(soft4[m][:, 1:4], 1e-12, None)) / float(T)
    logits = logits + np.array([b_leaf, b_trunk, b_twig], dtype=np.float64)
    out[m] = 1 + logits.argmax(1)
    return out


def soft4_from_mat3(mat3, contact_mask):
    p = np.zeros((len(mat3), 4), dtype=np.float64)
    p[:, 0] = 1.0
    p[contact_mask, 0] = 0.0
    p[contact_mask, 1:4] = mat3[contact_mask]
    s = p[contact_mask].sum(1, keepdims=True)
    s[s <= 0] = 1.0
    p[contact_mask] = p[contact_mask] / s
    return p


def main():
    lock = proto.load_selection_lock(OUT / "selection_lock.json")
    sc = lock["selected_candidate"]
    src = sc["source"]
    T = float(sc["T"])
    bl, bt, bg = float(sc["b_leaf"]), float(sc["b_trunk"]), float(sc["b_twig"])

    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    emb_h = {
        n: np.load(p / "hand_train_full/X.npy").astype(np.float32)
        for n, p in IMG.items()
        if (p / "hand_train_full/X.npy").exists()
    }
    hu, hc, _, hy = pool_files(hfr.audio_file, emb_h["clip"], yh)
    hx = {n: np.vstack([emb_h[n][hc == i].mean(0) for i in range(len(hu))]) for n in emb_h}
    multi_h = np.hstack([hx[k] for k in ("clip", "eff", "dino", "convnext")])
    ha = {k: np.load(p).astype(np.float32) for k, p in AUDIO_H.items()}
    ha_s = {k: np.vstack([ha[k][hc == i].mean(0) for i in range(len(hu))]) for k in ha}
    hw = np.load(W2V_H).astype(np.float32)
    hw_s = np.vstack([hw[hc == i].mean(0) for i in range(len(hu))])
    clap_h = np.load(CLAP_H).astype(np.float32)
    clap_hs = np.vstack([clap_h[hc == i].mean(0) for i in range(len(hu))])

    m_bin = fit_bin(hx["clip"], (hy > 0).astype(int), 0.03)
    m_tr = fit_bin(hx["clip"], (hy == 2).astype(int), 0.1)
    Xst = np.vstack(
        [
            np.hstack([ha_s["robot_mix"], hw_s, multi_h]),
            np.hstack([ha_s["bandlimit"], hw_s, multi_h]),
        ]
    )
    m_ts = fit_bin(Xst, np.concatenate([(hy == 2).astype(int)] * 2), 0.05)
    m_cs = fit_bin(Xst, np.concatenate([(hy > 0).astype(int)] * 2), 0.05)
    wood = (hy == 2) | (hy == 3)
    Xw = np.vstack([np.hstack([ha_s[v][wood], hw_s[wood], multi_h[wood]]) for v in AUDIO_H])
    yw = np.concatenate([(hy[wood] == 2).astype(int)] * 3)
    m_tt = fit_bin(Xw, yw, 0.1)

    # material experts on full hand contact
    contact_tr = hy > 0
    ymat = hy[contact_tr] - 1
    experts = {}
    experts["multi"] = fit_multi(multi_h[contact_tr], ymat, 0.05)
    for name in ("clip", "eff", "dino", "convnext", "swin"):
        if name in hx:
            experts[name] = fit_multi(hx[name][contact_tr], ymat, 0.08)
    experts["clap"] = fit_multi(clap_hs[contact_tr], ymat, 0.05)
    Xav_h = np.hstack([ha_s["clean"], hw_s, multi_h, clap_hs])
    experts["av"] = fit_multi(Xav_h[contact_tr], ymat, 0.03)

    z = np.load(V2_BASE, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred_seg = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"])
    p_h = z["p_h"].astype(np.float64)
    p_meta = z["p_meta"].astype(np.float64)

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    emb_t = {
        n: np.load(p / "robot_test/X.npy").astype(np.float32)
        for n, p in IMG.items()
        if (p / "robot_test/X.npy").exists()
    }
    tu, tc, _, ty = pool_files(tfr.audio_file, emb_t["clip"], yt)
    order = {k: i for i, k in enumerate(ids)}
    pred = np.array([pred_seg[order[str(k)]] for k in tu])
    p_h_t = np.array([p_h[order[str(k)]] for k in tu])
    p_meta_t = np.array([p_meta[order[str(k)]] for k in tu])

    tx = {n: np.vstack([emb_t[n][tc == i].mean(0) for i in range(len(tu))]) for n in emb_t}
    multi_t = np.hstack([tx[k] for k in ("clip", "eff", "dino", "convnext")])
    ta = np.load(AUDIO_R).astype(np.float32)
    ta_s = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_R).astype(np.float32)
    tw_s = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])
    clap_t = np.load(CLAP_R).astype(np.float32)
    clap_ts = np.vstack([clap_t[tc == i].mean(0) for i in range(len(tu))])
    Xr = np.hstack([ta_s, tw_s, multi_t])

    bins = pos(m_bin, tx["clip"])
    tr = pos(m_tr, tx["clip"])
    ts = pos(m_ts, Xr)
    cs = pos(m_cs, Xr)
    tt = pos(m_tt, Xr)

    # locked contact stack
    pred[(pred == 0) & (ts >= PRIMARY["ts_th"]) & (np.maximum(cs, bins) >= PRIMARY["cs_th"])] = 2
    if PRIMARY["do_tt"]:
        pred[(pred == 3) & (tt >= PRIMARY["tt_th"]) & (ts >= PRIMARY["ts_th"] * 0.9)] = 2
    pred[(pred == 0) & (tr >= SECONDARY["th"]) & (bins >= SECONDARY["cth"])] = 2

    if src == "identity_contact":
        pred_final = pred
    else:
        contact_mask = pred > 0
        mat = {}
        mat["multi"] = proba_aligned(experts["multi"], multi_t)
        for name in ("clip", "eff", "dino", "convnext", "swin"):
            if name in experts and name in tx:
                mat[name] = proba_aligned(experts[name], tx[name])
        mat["clap"] = proba_aligned(experts["clap"], clap_ts)
        Xav_t = np.hstack([ta_s, tw_s, multi_t, clap_ts])
        mat["av"] = proba_aligned(experts["av"], Xav_t)
        mat["blend_img_av"] = proto.normalize(0.5 * mat["multi"] + 0.5 * mat["av"])
        mat["blend_meta_multi"] = proto.normalize(
            0.5 * proto.normalize(p_meta_t[:, 1:4]) + 0.5 * mat["multi"]
        )
        mat["blend_hier_av"] = proto.normalize(
            0.5 * proto.normalize(p_h_t[:, 1:4]) + 0.5 * mat["av"]
        )
        mat["blend_all"] = proto.normalize(
            (mat["multi"] + mat["av"] + proto.normalize(p_meta_t[:, 1:4]) + mat["clap"]) / 4.0
        )
        mat["meta3"] = proto.normalize(p_meta_t[:, 1:4])
        mat["hier3"] = proto.normalize(p_h_t[:, 1:4])

        if src in ("meta4",):
            soft4 = p_meta_t
        elif src in ("hier4",):
            soft4 = p_h_t
        elif src in ("blend_mh4",):
            soft4 = proto.normalize(0.5 * p_meta_t + 0.5 * p_h_t)
        elif src in mat:
            soft4 = soft4_from_mat3(mat[src], contact_mask)
        else:
            soft4 = soft4_from_mat3(mat["multi"], contact_mask)
        pred_final = material_decode(pred, soft4, T, bl, bt, bg)

    sk = sel.seg(tfr.audio_file).to_numpy()
    ou = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred_final[ou[k]] for k in sk])
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
            "leaf_contaminant_surrogate_selection": True,
            "locked_0837_contact": True,
        },
    }
    proto.write_final_metrics(
        OUT / "multimodal_085_leaf_contaminant_mat_final_test_metrics.json", result
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
