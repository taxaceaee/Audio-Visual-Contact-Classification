"""One-shot robot eval for hand-selected joint multi-task multimodal model."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import multimodal_085_protocol as proto
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
from run_multimodal_085_joint_multitask_group_selection import JointMultiTask

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_joint_multitask_group_selection",
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
}
AUDIO_R = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/X.npy"
)
W2V_R = Path("outputs/audio_wav2vec2_features/robot_test_X.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pool_files(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


@torch.no_grad()
def predict(model, X):
    model.eval()
    xv = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits4, logitsb, logitsm = model(xv, mod_drop_p=0.0)
    p4 = torch.softmax(logits4, 1).cpu().numpy()
    pb = torch.softmax(logitsb, 1).cpu().numpy()[:, 1]
    pm = torch.softmax(logitsm, 1).cpu().numpy()
    return p4, pb, pm


def main():
    lock = proto.load_selection_lock(OUT / "selection_lock.json")
    sc = lock["selected_candidate"]
    rule = sc["rule"]
    wj = float(sc.get("wj", 0.0))

    ckpt = torch.load(OUT / "final_joint_multitask.pt", map_location="cpu", weights_only=False)
    model = JointMultiTask(ckpt["d_audio"], ckpt["d_w2v"], ckpt["d_img"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sc_a = StandardScaler()
    sc_a.mean_ = ckpt["scaler_audio_mean"]
    sc_a.scale_ = ckpt["scaler_audio_scale"]
    sc_a.n_features_in_ = len(sc_a.mean_)
    sc_w = StandardScaler()
    sc_w.mean_ = ckpt["scaler_w2v_mean"]
    sc_w.scale_ = ckpt["scaler_w2v_scale"]
    sc_w.n_features_in_ = len(sc_w.mean_)
    sc_i = StandardScaler()
    sc_i.mean_ = ckpt["scaler_img_mean"]
    sc_i.scale_ = ckpt["scaler_img_scale"]
    sc_i.n_features_in_ = len(sc_i.mean_)

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    emb = {n: np.load(p / "robot_test/X.npy").astype(np.float32) for n, p in IMG.items()}
    tu, tc, _, ty = pool_files(tfr.audio_file, emb["clip"], yt)
    multi = np.hstack(
        [np.vstack([emb[n][tc == i].mean(0) for i in range(len(tu))]) for n in IMG]
    )
    ta = np.load(AUDIO_R).astype(np.float32)
    ta_s = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_R).astype(np.float32)
    tw_s = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])
    X = np.hstack(
        [sc_a.transform(ta_s), sc_w.transform(tw_s), sc_i.transform(multi)]
    ).astype(np.float32)

    p4, pb, pm = predict(model, X)
    joint_pred = p4.argmax(1)

    z = np.load(V2_BASE, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred_v2 = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"])
    p_meta = z["p_meta"].astype(np.float64)
    order = {k: i for i, k in enumerate(ids)}
    pred_v2 = np.array([pred_v2[order[str(k)]] for k in tu])
    p_meta = np.array([p_meta[order[str(k)]] for k in tu])

    if rule == "joint_only":
        pred = joint_pred
    elif rule == "v2_contact_joint_mat":
        pred = pred_v2.copy()
        c = pred > 0
        pred[c] = 1 + pm[c].argmax(1)
    elif rule == "soft_blend":
        p = proto.normalize((1 - wj) * p_meta + wj * p4)
        pred = p.argmax(1)
    elif rule == "hier_trunk_joint_else":
        pred = joint_pred.copy()
        pred[pred_v2 == 2] = 2
    elif rule == "max_trunk":
        pred = joint_pred.copy()
        m = ((joint_pred == 2) | (pred_v2 == 2)) & (pb >= 0.4)
        pred[m] = 2
    else:
        pred = joint_pred

    sk = sel.seg(tfr.audio_file).to_numpy()
    ou = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred[ou[k]] for k in sk])

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
            "joint_multitask_moddrop": True,
        },
    }
    proto.write_final_metrics(
        OUT / "multimodal_085_joint_multitask_final_test_metrics.json", result
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
