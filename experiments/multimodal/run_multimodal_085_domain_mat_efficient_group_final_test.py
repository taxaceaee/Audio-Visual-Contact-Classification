"""One-shot robot: locked 0.837 contact stack + domain-aug material ResNet fusion."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

import multimodal_085_protocol as proto
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_domain_mat_efficient_group_selection",
    )
)
V2 = Path(
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalDS(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224))
        return self.tf(im)


def make_model():
    m = resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(512, 3))
    return m.to(DEVICE)


@torch.no_grad()
def predict_windows(model, paths, bs=128):
    ld = DataLoader(EvalDS(paths), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    outs = []
    for x in ld:
        outs.append(torch.softmax(model(x.to(DEVICE, non_blocking=True)), 1).cpu().numpy())
    return np.concatenate(outs, axis=0)


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


def pool_files(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


def fuse_material(contact_pred, mat_p, rule, conf_th):
    out = contact_pred.copy()
    c = out > 0
    if rule == "none" or not c.any():
        return out
    mat = 1 + mat_p.argmax(1)
    conf = mat_p.max(1)
    if rule == "always":
        out[c] = mat[c]
    elif rule == "conf_gate":
        m = c & (conf >= conf_th)
        out[m] = mat[m]
    elif rule == "disagree_gate":
        m = c & (conf >= conf_th) & (mat != out)
        out[m] = mat[m]
    elif rule == "wood_only":
        wood = c & ((out == 2) | (out == 3))
        m = wood & (conf >= conf_th)
        out[m] = mat[m]
    return out


def main():
    lock = proto.load_selection_lock(OUT / "selection_lock.json")
    sc = lock["selected_candidate"]
    primary = lock["primary_contact"]
    secondary = lock["secondary_img"]
    rule = sc["rule"]
    conf_th = float(sc["conf_th"])

    # material model
    model = make_model()
    state = torch.load(OUT / "final_material_resnet18.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # fit contact detectors on full hand
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
    multi_h = np.hstack([hx[k] for k in IMG_MULTI])
    ha = {k: np.load(p).astype(np.float32) for k, p in AUDIO_H.items()}
    ha_s = {k: np.vstack([ha[k][hc == i].mean(0) for i in range(len(hu))]) for k in ha}
    hw = np.load(W2V_H).astype(np.float32)
    hw_s = np.vstack([hw[hc == i].mean(0) for i in range(len(hu))])

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
    Xw = np.vstack(
        [np.hstack([ha_s[v][wood], hw_s[wood], multi_h[wood]]) for v in AUDIO_H]
    )
    m_tt = fit_bin(Xw, np.concatenate([(hy[wood] == 2).astype(int)] * 3), 0.1)

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    emb_t = {n: np.load(p / "robot_test/X.npy").astype(np.float32) for n, p in IMG_MULTI.items()}
    tu, tc, _, ty = pool_files(tfr.audio_file, emb_t["clip"], yt)
    tx = {n: np.vstack([emb_t[n][tc == i].mean(0) for i in range(len(tu))]) for n in emb_t}
    multi_t = np.hstack([tx[k] for k in IMG_MULTI])
    ta = np.load(AUDIO_R).astype(np.float32)
    ta_s = np.vstack([ta[tc == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_R).astype(np.float32)
    tw_s = np.vstack([tw[tc == i].mean(0) for i in range(len(tu))])
    Xr = np.hstack([ta_s, tw_s, multi_t])

    bins = pos(m_bin, tx["clip"])
    tr = pos(m_tr, tx["clip"])
    ts = pos(m_ts, Xr)
    cs = pos(m_cs, Xr)
    tt = pos(m_tt, Xr)

    z = np.load(V2, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred_seg = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"])
    order = {k: i for i, k in enumerate(ids)}
    pred = np.array([pred_seg[order[str(k)]] for k in tu])

    # primary + secondary contact
    pred[(pred == 0) & (ts >= primary["ts_th"]) & (np.maximum(cs, bins) >= primary["cs_th"])] = 2
    if primary.get("do_tt", True):
        pred[
            (pred == 3)
            & (tt >= primary["tt_th"])
            & (ts >= primary["ts_th"] * 0.9)
        ] = 2
    pred[(pred == 0) & (tr >= secondary["th"]) & (bins >= secondary["cth"])] = 2

    # material predictions: window then pool
    img_paths = tfr.image_path.astype(str).to_numpy()
    p_win = predict_windows(model, img_paths)
    # pool to segments
    mat_seg = np.zeros((len(tu), 3), dtype=np.float64)
    for i in range(len(tu)):
        mask = tc == i
        mat_seg[i] = p_win[mask].mean(0)

    pred = fuse_material(pred, mat_seg, rule, conf_th)

    sk = sel.seg(tfr.audio_file).to_numpy()
    ou = {k: i for i, k in enumerate(tu)}
    pred_rows = np.array([pred[ou[k]] for k in sk])

    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "locked_candidate": sc,
        "primary_contact": primary,
        "secondary_img": secondary,
        **proto.metrics_bundle(yt, pred_rows),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "no_test_hp_tuning": True,
            "filename_class_features": False,
            "robot_uda": False,
            "domain_aug_material_ft": True,
            "contact_locked_from_best_strict": True,
        },
    }
    proto.write_final_metrics(
        OUT / "multimodal_085_domain_mat_efficient_final_test_metrics.json", result
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
