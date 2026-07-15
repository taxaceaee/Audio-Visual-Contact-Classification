"""Hier v2: multi-backbone material + trunk confidence rescue (hand only)."""
from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_hier_v2_group_selection",
    )
)
BACKBONES = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "cnv": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
    "r18": Path("outputs/image_deep_features/resnet18_224"),
}


def seg(s):
    return s.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)


def spec(s):
    return seg(s).str.replace(r"_segment_.*$", "", regex=True)


def fast_macro(y, p):
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set("total240")
    fr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y = fr.y.to_numpy(np.int64)
    files = fr.audio_file.astype(str)
    sk = seg(files).to_numpy()
    sg = spec(files).to_numpy()
    useg, codes = np.unique(sk, return_inverse=True)
    sy = np.array([y[codes == i][0] for i in range(len(useg))])
    ss = np.array([sg[codes == i][0] for i in range(len(useg))])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(sy)), sy, ss
        )
    )

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
    assignment = np.full(len(y), -1, np.int64)
    for k, (_, va) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(y)), y, sg
        )
    ):
        assignment[va] = k
    src = broad.load_train_sources(Path("outputs"), y, assignment, 42)
    audio = lift.postprocess(
        af,
        lift.normalize(0.95 * anchor + 0.05 * src["report_gate_onehot"]),
        "segment_lift",
    )
    pc, lg = avr.avr_inputs(audio)
    seg_pc = np.array([pc[codes == i].mean() for i in range(len(useg))])
    sa_mat = np.vstack(
        [
            suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0]
            for i in range(len(useg))
        ]
    )

    xs = {}
    for name, path in BACKBONES.items():
        x = np.load(path / "hand_train_full/X.npy").astype(np.float32)
        assert len(x) == len(y), name
        xs[name] = np.vstack([x[codes == i].mean(0) for i in range(len(useg))])
    xs["cat_ce"] = np.hstack([xs["clip"], xs["eff"]])
    xs["cat_all"] = np.hstack([xs["clip"], xs["eff"], xs["dino"], xs["cnv"], xs["r18"]])

    def oof_mat(x):
        o = np.zeros((len(sy), 3), np.float32)
        for tr, va in folds:
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.03,
                    max_iter=1500,
                    class_weight="balanced",
                    multi_class="multinomial",
                    random_state=42,
                ),
            )
            ci = tr[sy[tr] > 0]
            m.fit(x[ci], sy[ci] - 1)
            o[va] = m.predict_proba(x[va])
        return o

    def oof_bin(x):
        o = np.zeros(len(sy), np.float32)
        for tr, va in folds:
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.03, max_iter=1500, class_weight="balanced", random_state=42
                ),
            )
            m.fit(x[tr], (sy[tr] > 0).astype(int))
            o[va] = m.predict_proba(x[va])[:, 1]
        return o

    def oof_trunk(x):
        o = np.zeros(len(sy), np.float32)
        for tr, va in folds:
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1, max_iter=1200, class_weight="balanced", random_state=42
                ),
            )
            m.fit(x[tr], (sy[tr] == 2).astype(int))
            o[va] = m.predict_proba(x[va])[:, 1]
        return o

    print("OOF multi-backbone...", flush=True)
    mats = {k: oof_mat(v) for k, v in xs.items() if k in ("clip", "eff", "dino", "cnv", "cat_ce", "cat_all")}
    bins = {k: oof_bin(xs[k]) for k in ("clip", "cat_ce", "cat_all")}
    trunks = {k: oof_trunk(xs[k]) for k in ("clip", "cat_all")}
    # material ensemble
    mats["ens"] = suite.normalize(
        np.mean([mats[k] for k in ("clip", "eff", "dino", "cnv")], axis=0)
    )

    np.savez_compressed(
        OUT / "hand_oof.npz",
        sy=sy,
        ss=ss,
        seg_pc=seg_pc,
        sa_mat=sa_mat,
        **{f"mat_{k}": v for k, v in mats.items()},
        **{f"bin_{k}": v for k, v in bins.items()},
        **{f"trunk_{k}": v for k, v in trunks.items()},
    )

    ths = np.linspace(0.40, 0.62, 12)
    rows = []
    for csrc, alpha in [
        ("audio", 1.0),
        ("clip", 0.5),
        ("clip", 0.7),
        ("cat_ce", 0.5),
        ("cat_all", 0.5),
    ]:
        if csrc == "audio":
            contact = seg_pc
        else:
            contact = alpha * seg_pc + (1 - alpha) * bins[csrc]
        for msrc in ("cat_ce", "cat_all", "ens", "clip", "dino"):
            for wl, wt, ww in product([0.3, 0.55, 0.7], repeat=3):
                w = np.array([wl, wt, ww], dtype=np.float64)
                mat = suite.normalize(sa_mat * (1 - w) + mats[msrc] * w)
                for tb, tsrc in ((0.0, "clip"), (0.45, "clip"), (0.45, "cat_all")):
                    mat_b = mat
                    if tb:
                        logits = np.log(np.clip(mat, 1e-8, 1))
                        logits[:, 1] += tb * trunks[tsrc]
                        mat_b = suite.normalize(np.exp(logits))
                    mat_pred = mat_b.argmax(1) + 1
                    trunk_score = trunks[tsrc]
                    for th in ths:
                        base_pred = np.where(contact >= th, mat_pred, 0)
                        for rescue_th in (0.0, 0.60, 0.72):
                            pred2 = base_pred.copy()
                            if rescue_th > 0:
                                rmask = (pred2 == 0) & (trunk_score >= rescue_th)
                                pred2[rmask] = 2
                                rmask2 = (pred2 == 3) & (trunk_score >= rescue_th + 0.05)
                                pred2[rmask2] = 2
                            f1 = fast_macro(sy, pred2)
                            rows.append(
                                {
                                    "contact_source": csrc,
                                    "contact_alpha_audio": float(alpha),
                                    "material_source": msrc,
                                    "leaf_w": float(wl),
                                    "trunk_w": float(wt),
                                    "twig_w": float(ww),
                                    "trunk_bias": float(tb),
                                    "trunk_source": tsrc,
                                    "contact_threshold": float(th),
                                    "trunk_rescue_th": float(rescue_th),
                                    "macro_f1_4class": float(f1),
                                    "binary_macro_f1": float(
                                        f1_score(
                                            sy > 0,
                                            pred2 > 0,
                                            average="macro",
                                            zero_division=0,
                                        )
                                    ),
                                    "trunk_recall": float(
                                        np.sum((sy == 2) & (pred2 == 2))
                                        / max(np.sum(sy == 2), 1)
                                    ),
                                }
                            )
    board = (
        pd.DataFrame(rows)
        .sort_values(
            ["macro_f1_4class", "trunk_recall", "binary_macro_f1"], ascending=False
        )
        .reset_index(drop=True)
    )
    board.to_csv(OUT / "hand_hier_v2_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    lock = {
        "protocol": "segment_hier_v2_multibb_trunk_rescue_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "audio_source": "locked group-aware OOF audio",
        "image_source": "CLIP+Eff+DINO+ConvNeXt+R18 segment OOF",
        "test_loaded": False,
        "selected_candidate": best,
        "n_candidates": int(len(board)),
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
