"""Hand-only: efficient domain-aug material ResNet18 FT + locked 0.837 contact stack.

Protocol: specimen_group OOF; never loads robot/test.
Primary contact is the locked best-strict stack (v2 + stress amb-lift + collapse-selected
secondary CLIP trunk). Material head is fine-tuned with camera_heavy augmentation on
contact windows only; fusion selected on hand worst-fold macro F1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedGroupKFold
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
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
BUNDLE = Path(
    "outputs/audio_feature_benchmarks/multimodal_085_material_logit_bias_group_selection/"
    "hand_oof_bundle.npz"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Locked primary+secondary from best strict 0.837
PRIMARY = dict(ts_th=0.5, cs_th=0.1, tt_th=0.45, do_tt=True)
SECONDARY = dict(th=0.625, cth=0.75)


class MaterialDS(Dataset):
    def __init__(self, paths, y_mat, train: bool):
        self.paths = list(paths)
        self.y = np.asarray(y_mat, dtype=np.int64)
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.7, 1.4)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.5, 0.5, 0.4, 0.1)], p=0.9
                    ),
                    transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.5))], p=0.5),
                    transforms.RandomGrayscale(p=0.15),
                    transforms.RandomPerspective(distortion_scale=0.25, p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
                ]
            )
        else:
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
        return self.tf(im), int(self.y[i])


def make_model():
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(m.fc.in_features, 3))
    return m.to(DEVICE)


@torch.no_grad()
def predict_windows(model, paths, bs=128):
    ds = MaterialDS(paths, np.zeros(len(paths)), train=False)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    outs = []
    for x, _ in ld:
        outs.append(torch.softmax(model(x.to(DEVICE, non_blocking=True)), 1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_material(paths, y_mat, epochs=7, lr=1e-4, trunk_w=2.5, seed=0):
    torch.manual_seed(seed)
    model = make_model()
    counts = np.bincount(y_mat, minlength=3).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1.0)
    w[1] *= trunk_w  # trunk among material {leaf,trunk,twig}
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    crit = nn.CrossEntropyLoss(weight=w)
    ld = DataLoader(
        MaterialDS(paths, y_mat, train=True),
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=len(paths) > 128,
    )
    for ep in range(epochs):
        model.train()
        tot = 0.0
        nstep = 0
        for x, t in ld:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x.to(DEVICE, non_blocking=True)), t.to(DEVICE, non_blocking=True))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nstep += 1
        print(f"  ep {ep+1}/{epochs} loss={tot/max(nstep,1):.4f}", flush=True)
    return model


def pool_seg_from_windows(p_win, codes, n_seg):
    """Mean-pool window material proba to segments (vectorized)."""
    out = np.zeros((n_seg, p_win.shape[1]), dtype=np.float64)
    counts = np.bincount(codes, minlength=n_seg).astype(np.float64)
    for c in range(p_win.shape[1]):
        out[:, c] = np.bincount(codes, weights=p_win[:, c], minlength=n_seg)
    out /= np.maximum(counts[:, None], 1.0)
    return out


def apply_contact(pred_v2, oof_ts, oof_cs, oof_bin, oof_tt, oof_tr):
    out = pred_v2.copy()
    out[
        (out == 0)
        & (oof_ts >= PRIMARY["ts_th"])
        & (np.maximum(oof_cs, oof_bin) >= PRIMARY["cs_th"])
    ] = 2
    if PRIMARY["do_tt"]:
        out[
            (out == 3)
            & (oof_tt >= PRIMARY["tt_th"])
            & (oof_ts >= PRIMARY["ts_th"] * 0.9)
        ] = 2
    out[(out == 0) & (oof_tr >= SECONDARY["th"]) & (oof_bin >= SECONDARY["cth"])] = 2
    return out


def fuse_material(contact_pred, mat_p, rule, conf_th):
    out = contact_pred.copy()
    c = out > 0
    if not c.any():
        return out
    mat = 1 + mat_p.argmax(1)
    conf = mat_p.max(1)
    if rule == "always":
        out[c] = mat[c]
    elif rule == "conf_gate":
        m = c & (conf >= conf_th)
        out[m] = mat[m]
    elif rule == "disagree_gate":
        # only override when material disagrees with current and conf high
        m = c & (conf >= conf_th) & (mat != out)
        out[m] = mat[m]
    elif rule == "wood_only":
        # only reclass among wood (trunk/twig) predictions
        wood = c & ((out == 2) | (out == 3))
        m = wood & (conf >= conf_th)
        # map material only if predicted wood class in mat
        mat_wood = mat.copy()
        mat_wood[mat == 1] = out[mat == 1]  # keep leaf as was if mat says leaf unless conf very high
        out[m] = mat[m]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y = fr.y.to_numpy(np.int64)
    paths = fr.image_path.astype(str).to_numpy()
    files = fr.audio_file.astype(str)
    sk = sel.seg(files).to_numpy()
    sg = sel.spec(files).to_numpy()
    useg, codes = np.unique(sk, return_inverse=True)
    n = len(useg)
    sy = np.array([y[codes == i][0] for i in range(n)], np.int64)
    ss = np.array([sg[codes == i][0] for i in range(n)])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(n), sy, ss)
    )

    z = np.load(BUNDLE)
    pred_v2 = z["pred_v2"]
    oof_ts, oof_cs, oof_bin, oof_tt, oof_tr = (
        z["oof_ts"],
        z["oof_cs"],
        z["oof_bin"],
        z["oof_tt"],
        z["oof_tr_img"],
    )
    assert len(pred_v2) == n
    contact_pred = apply_contact(pred_v2, oof_ts, oof_cs, oof_bin, oof_tt, oof_tr)
    print("locked contact hand F1", proto.fast_macro_f1(sy, contact_pred), flush=True)

    # OOF material proba
    oof_mat = np.zeros((n, 3), dtype=np.float64)
    contact_w = np.where(y > 0)[0]
    print(f"contact windows {len(contact_w)} / {len(y)}", flush=True)

    for fi, (tr, va) in enumerate(folds):
        print(f"Fold {fi+1}/5", flush=True)
        tr_set = set(tr.tolist())
        tr_idx = [i for i in contact_w if codes[i] in tr_set]
        ytr = y[tr_idx] - 1
        model = train_material(paths[tr_idx], ytr, epochs=6, trunk_w=2.5, seed=42 + fi)
        # predict ALL val windows once, then pool
        va_win = np.where(np.isin(codes, va))[0]
        p_win = predict_windows(model, paths[va_win])
        # map to segment indices
        va_codes = codes[va_win]
        # only fill va segments
        for j, si in enumerate(va):
            mask = va_codes == si
            if mask.any():
                oof_mat[si] = p_win[mask].mean(0)
            else:
                oof_mat[si] = 1.0 / 3.0
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    np.save(OUT / "hand_oof_material3.npy", oof_mat)
    mat_pred = 1 + oof_mat.argmax(1)
    print(
        "material OOF among true contact",
        proto.fast_macro_f1(sy[sy > 0] - 1, mat_pred[sy > 0] - 1)
        if (sy > 0).any()
        else None,
        flush=True,
    )

    # fusion grid
    rows = []
    for rule in ("always", "conf_gate", "disagree_gate", "wood_only", "none"):
        confs = [0.0] if rule in ("always", "none") else list(np.linspace(0.35, 0.9, 12))
        for conf_th in confs:
            fold_mins = []
            oof_pred = np.zeros(n, np.int64)
            for _, va in folds:
                if rule == "none":
                    pred = contact_pred
                else:
                    pred = fuse_material(contact_pred, oof_mat, rule, conf_th)
                fold_mins.append(proto.fast_macro_f1(sy[va], pred[va]))
                oof_pred[va] = pred[va]
            clean = proto.fast_macro_f1(sy, oof_pred)
            if clean < 0.90:
                continue
            false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
            if false_tr > 25:
                continue
            amb = float(np.sum((sy == 0) & (oof_pred == 0)) / max((sy == 0).sum(), 1))
            if amb < 0.97:
                continue
            trunk_rec = float(
                np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1)
            )
            twig_rec = float(
                np.sum((sy == 3) & (oof_pred == 3)) / max((sy == 3).sum(), 1)
            )
            leaf_rec = float(
                np.sum((sy == 1) & (oof_pred == 1)) / max((sy == 1).sum(), 1)
            )
            worst = float(np.min(fold_mins))
            mean_f = float(np.mean(fold_mins))
            bal = min(trunk_rec, twig_rec, leaf_rec)
            # prefer diversity vs pure contact if material helps
            disagree = float(np.mean(oof_pred != contact_pred))
            comp = (
                0.4 * worst
                + 0.2 * mean_f
                + 0.15 * clean
                + 0.15 * bal
                + 0.1 * trunk_rec
                - 0.01 * false_tr
            )
            rows.append(
                dict(
                    rule=rule,
                    conf_th=float(conf_th),
                    worst_fold=worst,
                    mean_fold=mean_f,
                    clean_macro_f1=clean,
                    trunk_recall=trunk_rec,
                    twig_recall=twig_rec,
                    leaf_recall=leaf_rec,
                    bal_recall=bal,
                    false_trunk=false_tr,
                    ambient_recall=amb,
                    disagree_vs_contact=disagree,
                    composite=comp,
                )
            )

    board = pd.DataFrame(rows).sort_values(
        ["composite", "worst_fold", "bal_recall", "trunk_recall"], ascending=False
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_domain_mat_efficient_leaderboard.csv", index=False)
    selected = board.iloc[0].to_dict()
    print(board.head(15).to_string(index=False), flush=True)
    print("SELECTED", selected, flush=True)

    # final model on all contact
    y_all = y[contact_w] - 1
    final = train_material(paths[contact_w], y_all, epochs=8, trunk_w=2.5, seed=123)
    torch.save(final.state_dict(), OUT / "final_material_resnet18.pt")

    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_domain_mat_efficient_contact_locked_group_OOF",
            "primary_contact": PRIMARY,
            "secondary_img": SECONDARY,
            "selected_candidate": selected,
            "model": "ResNet18 camera_heavy material-only 3-class trunk_w=2.5",
            "base_contact_hand_macro_f1": proto.fast_macro_f1(sy, contact_pred),
            "base_robot_artifact": (
                "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
                "segment_rule_stack_v2_base_segment_outputs.npz"
            ),
        },
    )
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
