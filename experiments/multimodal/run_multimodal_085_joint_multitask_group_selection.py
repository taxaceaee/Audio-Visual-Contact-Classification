"""Hand-only joint multi-task multimodal net with modality dropout + multi-view stress.

Paper-pure path (catalog E1/E3/E4):
- Inputs: multi-view audio (clean/robot_mix/bandlimit) + wav2vec2 + multi-bb image emb
- Shared MLP + heads: 4-class, binary contact, 3-class material
- Train with modality dropout, class-balanced CE, trunk-upweighted material
- Select by worst-fold × worst-view macro F1; optionally fuse with locked v2 contact
- Never loads robot/test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import multimodal_085_protocol as proto
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_joint_multitask_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
}
AUDIO = {
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
W2V = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pool(x, codes, n):
    return np.vstack([x[codes == i].mean(0) for i in range(n)]).astype(np.float32)


class JointMultiTask(nn.Module):
    def __init__(self, d_audio, d_w2v, d_img, hidden=512, drop=0.35):
        super().__init__()
        self.d_audio = d_audio
        self.d_w2v = d_w2v
        self.d_img = d_img
        d_in = d_audio + d_w2v + d_img
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        h2 = hidden // 2
        self.head4 = nn.Linear(h2, 4)
        self.head_bin = nn.Linear(h2, 2)
        self.head_mat = nn.Linear(h2, 3)

    def forward(self, x, mod_drop_p=0.0):
        # x: [B, d_audio+d_w2v+d_img]
        if self.training and mod_drop_p > 0:
            B = x.size(0)
            # drop audio block, w2v block, or image block independently
            a = x[:, : self.d_audio]
            w = x[:, self.d_audio : self.d_audio + self.d_w2v]
            im = x[:, self.d_audio + self.d_w2v :]
            if torch.rand(1).item() < mod_drop_p:
                which = torch.randint(0, 3, (1,)).item()
                if which == 0:
                    a = torch.zeros_like(a)
                elif which == 1:
                    w = torch.zeros_like(w)
                else:
                    im = torch.zeros_like(im)
            x = torch.cat([a, w, im], dim=1)
        h = self.trunk(x)
        return self.head4(h), self.head_bin(h), self.head_mat(h)


def train_one(
    Xtr,
    ytr,
    Xva_views,
    yva,
    d_audio,
    d_w2v,
    d_img,
    epochs=40,
    lr=1e-3,
    mod_drop=0.25,
    trunk_w=2.5,
    seed=0,
):
    torch.manual_seed(seed)
    model = JointMultiTask(d_audio, d_w2v, d_img).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    # class weights
    counts = np.bincount(ytr, minlength=4).astype(np.float64)
    w4 = counts.sum() / np.maximum(counts, 1.0)
    w4[2] *= trunk_w  # extra trunk
    w4 = torch.tensor(w4 / w4.mean(), dtype=torch.float32, device=DEVICE)
    ybin = (ytr > 0).astype(np.int64)
    bcounts = np.bincount(ybin, minlength=2).astype(np.float64)
    wb = torch.tensor(bcounts.sum() / np.maximum(bcounts, 1.0), dtype=torch.float32, device=DEVICE)
    wb = wb / wb.mean()
    contact = ytr > 0
    ymat = ytr[contact] - 1
    mcounts = np.bincount(ymat, minlength=3).astype(np.float64)
    wm = mcounts.sum() / np.maximum(mcounts, 1.0)
    wm[1] *= trunk_w  # trunk among material
    wm = torch.tensor(wm / wm.mean(), dtype=torch.float32, device=DEVICE)

    ce4 = nn.CrossEntropyLoss(weight=w4)
    ceb = nn.CrossEntropyLoss(weight=wb)
    cem = nn.CrossEntropyLoss(weight=wm)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    ybt = torch.tensor(ybin, dtype=torch.long, device=DEVICE)

    best_state = None
    best_score = -1.0
    n = len(ytr)
    bs = min(256, max(32, n // 4))

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            x = Xt[idx]
            # feature noise as domain randomization
            if torch.rand(1).item() < 0.5:
                x = x + 0.05 * torch.randn_like(x)
            logits4, logitsb, logitsm = model(x, mod_drop_p=mod_drop)
            loss = ce4(logits4, yt[idx]) + 0.5 * ceb(logitsb, ybt[idx])
            # material head only on contact samples in batch
            yb = yt[idx]
            cmask = yb > 0
            if cmask.any():
                loss = loss + 0.75 * cem(logitsm[cmask], yb[cmask] - 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # val: worst over views
        model.eval()
        scores = []
        with torch.no_grad():
            for Xv in Xva_views:
                xv = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
                logits4, _, _ = model(xv, mod_drop_p=0.0)
                pred = logits4.argmax(1).cpu().numpy()
                scores.append(proto.fast_macro_f1(yva, pred))
        worst = float(min(scores))
        if worst > best_score:
            best_score = worst
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_score


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    xv = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits4, logitsb, logitsm = model(xv, mod_drop_p=0.0)
    p4 = torch.softmax(logits4, 1).cpu().numpy()
    pb = torch.softmax(logitsb, 1).cpu().numpy()[:, 1]
    pm = torch.softmax(logitsm, 1).cpu().numpy()
    return p4, pb, pm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y = fr.y.to_numpy(np.int64)
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

    img = {
        k: pool(np.load(p / "hand_train_full/X.npy").astype(np.float32), codes, n)
        for k, p in IMG.items()
    }
    multi = np.hstack([img[k] for k in ("clip", "eff", "dino", "convnext")])
    aud = {k: pool(np.load(p).astype(np.float32), codes, n) for k, p in AUDIO.items()}
    w2v = pool(np.load(W2V).astype(np.float32), codes, n)

    d_audio = aud["clean"].shape[1]
    d_w2v = w2v.shape[1]
    d_img = multi.shape[1]
    print(f"dims audio={d_audio} w2v={d_w2v} img={d_img} n={n}", flush=True)

    # scale per block on full hand (fit inside folds too)
    oof_p4 = np.zeros((n, 4), np.float64)
    oof_pb = np.zeros(n, np.float64)
    oof_pm = np.zeros((n, 3), np.float64)
    fold_scores = []

    for fold_id, (tr, va) in enumerate(folds):
        print(f"Fold {fold_id+1}/5 tr={len(tr)} va={len(va)}", flush=True)
        # fit scalers on train clean
        sc_a = StandardScaler().fit(aud["clean"][tr])
        sc_w = StandardScaler().fit(w2v[tr])
        sc_i = StandardScaler().fit(multi[tr])

        def pack(view, idx):
            return np.hstack(
                [sc_a.transform(aud[view][idx]), sc_w.transform(w2v[idx]), sc_i.transform(multi[idx])]
            ).astype(np.float32)

        # multi-view bag training: stack views
        Xtr = np.vstack([pack(v, tr) for v in ("clean", "robot_mix", "bandlimit")])
        ytr = np.concatenate([sy[tr]] * 3)
        Xva_views = [pack(v, va) for v in ("clean", "robot_mix", "bandlimit")]

        model, worst = train_one(
            Xtr,
            ytr,
            Xva_views,
            sy[va],
            d_audio,
            d_w2v,
            d_img,
            epochs=35,
            lr=1e-3,
            mod_drop=0.3,
            trunk_w=2.5,
            seed=42 + fold_id,
        )
        p4, pb, pm = predict_proba(model, pack("clean", va))
        # average clean + robot_mix for robustness
        p4b, pbb, pmb = predict_proba(model, pack("robot_mix", va))
        oof_p4[va] = 0.5 * (p4 + p4b)
        oof_pb[va] = 0.5 * (pb + pbb)
        oof_pm[va] = 0.5 * (pm + pmb)
        pred = oof_p4[va].argmax(1)
        f1 = proto.fast_macro_f1(sy[va], pred)
        fold_scores.append({"fold": fold_id, "worst_train_select": worst, "oof_clean_macro": f1})
        print(f"  oof clean macro {f1:.4f} select_worst {worst:.4f}", flush=True)

    joint_pred = oof_p4.argmax(1)
    joint_f1 = proto.fast_macro_f1(sy, joint_pred)
    print("joint OOF macro", joint_f1, flush=True)

    # v2 base for fusion
    oof_meta = np.load(STACK / "hand_oof_meta.npy")
    oof_hier = np.load(STACK / "hand_oof_hier.npy")
    pred_v2 = np.where(oof_hier.argmax(1) == 2, oof_hier.argmax(1), oof_meta.argmax(1))
    print("v2 OOF macro", proto.fast_macro_f1(sy, pred_v2), flush=True)

    # Fusion rules: keep v2 ambient/contact structure; material from joint when contact
    # Also soft blend and gated rules selected on hand worst-fold
    rows = []
    for rule in ("joint_only", "v2_contact_joint_mat", "soft_blend", "hier_trunk_joint_else", "max_trunk"):
        for wj in ([0.0] if rule != "soft_blend" else np.linspace(0.1, 0.9, 9)):
            fold_mins = []
            oof_pred = np.zeros(n, np.int64)
            for _, va in folds:
                scores = []
                # score clean joint already in oof; for soft_blend use clean oof
                if rule == "joint_only":
                    pred = joint_pred.copy()
                elif rule == "v2_contact_joint_mat":
                    pred = pred_v2.copy()
                    c = pred > 0
                    mat = 1 + oof_pm.argmax(1)
                    # if joint says ambient for a contact pred, keep v2 mat; else joint mat
                    pred[c] = mat[c]
                elif rule == "soft_blend":
                    p = proto.normalize((1 - wj) * oof_meta + wj * oof_p4)
                    pred = p.argmax(1)
                elif rule == "hier_trunk_joint_else":
                    pred = joint_pred.copy()
                    pred[pred_v2 == 2] = 2
                elif rule == "max_trunk":
                    pred = joint_pred.copy()
                    # if either says trunk and joint contact high, trunk
                    m = ((joint_pred == 2) | (pred_v2 == 2)) & (oof_pb >= 0.4)
                    pred[m] = 2
                else:
                    pred = joint_pred
                scores.append(proto.fast_macro_f1(sy[va], pred[va]))
                fold_mins.append(min(scores))
                oof_pred[va] = pred[va]
            clean = proto.fast_macro_f1(sy, oof_pred)
            worst = float(np.min(fold_mins))
            trunk_rec = float(np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1))
            false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
            amb = float(np.sum((sy == 0) & (oof_pred == 0)) / max((sy == 0).sum(), 1))
            if amb < 0.95 or clean < 0.88:
                continue
            comp = 0.5 * worst + 0.25 * clean + 0.2 * trunk_rec - 0.01 * false_tr
            rows.append(
                dict(
                    rule=rule,
                    wj=float(wj),
                    worst_fold=worst,
                    clean_macro_f1=clean,
                    trunk_recall=trunk_rec,
                    false_trunk=false_tr,
                    ambient_recall=amb,
                    composite=comp,
                )
            )

    board = pd.DataFrame(rows).sort_values(
        ["composite", "worst_fold", "trunk_recall"], ascending=False
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_joint_multitask_leaderboard.csv", index=False)
    selected = board.iloc[0].to_dict() if len(board) else {
        "rule": "joint_only",
        "wj": 0.0,
        "worst_fold": joint_f1,
        "clean_macro_f1": joint_f1,
    }
    print(board.head(15).to_string(index=False) if len(board) else "empty board", flush=True)

    # train final model on full hand multi-view for export
    sc_a = StandardScaler().fit(aud["clean"])
    sc_w = StandardScaler().fit(w2v)
    sc_i = StandardScaler().fit(multi)

    def pack_full(view):
        return np.hstack(
            [sc_a.transform(aud[view]), sc_w.transform(w2v), sc_i.transform(multi)]
        ).astype(np.float32)

    Xtr = np.vstack([pack_full(v) for v in ("clean", "robot_mix", "bandlimit")])
    ytr = np.concatenate([sy] * 3)
    # use clean as va proxy for early stop score
    model, _ = train_one(
        Xtr,
        ytr,
        [pack_full("clean"), pack_full("robot_mix"), pack_full("bandlimit")],
        sy,
        d_audio,
        d_w2v,
        d_img,
        epochs=40,
        lr=1e-3,
        mod_drop=0.3,
        trunk_w=2.5,
        seed=123,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_audio_mean": sc_a.mean_,
            "scaler_audio_scale": sc_a.scale_,
            "scaler_w2v_mean": sc_w.mean_,
            "scaler_w2v_scale": sc_w.scale_,
            "scaler_img_mean": sc_i.mean_,
            "scaler_img_scale": sc_i.scale_,
            "d_audio": d_audio,
            "d_w2v": d_w2v,
            "d_img": d_img,
        },
        OUT / "final_joint_multitask.pt",
    )
    np.savez_compressed(
        OUT / "hand_oof_joint.npz",
        sy=sy,
        oof_p4=oof_p4,
        oof_pb=oof_pb,
        oof_pm=oof_pm,
        pred_v2=pred_v2,
        joint_pred=joint_pred,
    )

    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_joint_multitask_moddrop_stress_group_OOF",
            "selected_candidate": selected,
            "model": "JointMultiTask hidden=512 mod_drop=0.3 trunk_w=2.5 multi-view bag",
            "fold_scores": fold_scores,
            "joint_oof_macro_f1": joint_f1,
            "v2_oof_macro_f1": proto.fast_macro_f1(sy, pred_v2),
            "base_robot_artifact": (
                "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
                "segment_rule_stack_v2_base_segment_outputs.npz"
            ),
        },
    )
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
