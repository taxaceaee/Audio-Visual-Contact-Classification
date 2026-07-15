"""Hand-only: primary stress amb-lift + secondary image trunk lift on residual ambient.

Selects img_th / img_cs under dual-collapse worst-fold on hand (specimen groups).
Never loads robot/test.
"""
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
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_img_secondary_lift_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
IMG_MULTI = {
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


def pool(x, codes, n):
    return np.vstack([x[codes == i].mean(0) for i in range(n)]).astype(np.float32)


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
    OUT.mkdir(parents=True, exist_ok=True)
    # reuse detectors from material bias bundle if present
    bundle = Path(
        "outputs/audio_feature_benchmarks/multimodal_085_material_logit_bias_group_selection/"
        "hand_oof_bundle.npz"
    )
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

    if bundle.exists():
        z = np.load(bundle)
        pred_v2 = z["pred_v2"]
        oof_ts, oof_cs, oof_bin, oof_tt, oof_tr = (
            z["oof_ts"],
            z["oof_cs"],
            z["oof_bin"],
            z["oof_tt"],
            z["oof_tr_img"],
        )
        assert len(pred_v2) == n
        print("reused detector bundle", flush=True)
    else:
        oof_meta = np.load(STACK / "hand_oof_meta.npy")
        oof_hier = np.load(STACK / "hand_oof_hier.npy")
        pred_v2 = np.where(oof_hier.argmax(1) == 2, oof_hier.argmax(1), oof_meta.argmax(1))
        img = {
            k: pool(np.load(p / "hand_train_full/X.npy").astype(np.float32), codes, n)
            for k, p in IMG_MULTI.items()
        }
        multi = np.hstack([img[k] for k in IMG_MULTI])
        aud = {k: pool(np.load(p).astype(np.float32), codes, n) for k, p in AUDIO.items()}
        w2v = pool(np.load(W2V).astype(np.float32), codes, n)
        sx = img["clip"]
        oof_bin = np.zeros(n)
        oof_ts = np.zeros(n)
        oof_cs = np.zeros(n)
        oof_tt = np.zeros(n)
        oof_tr = np.zeros(n)
        for fold_id, (tr, va) in enumerate(folds):
            oof_bin[va] = pos(fit_bin(sx[tr], (sy[tr] > 0).astype(int), 0.03), sx[va])
            oof_tr[va] = pos(fit_bin(sx[tr], (sy[tr] == 2).astype(int), 0.1), sx[va])
            Xst = np.vstack(
                [
                    np.hstack([aud["robot_mix"][tr], w2v[tr], multi[tr]]),
                    np.hstack([aud["bandlimit"][tr], w2v[tr], multi[tr]]),
                ]
            )
            m_ts = fit_bin(Xst, np.concatenate([(sy[tr] == 2).astype(int)] * 2), 0.05)
            m_cs = fit_bin(Xst, np.concatenate([(sy[tr] > 0).astype(int)] * 2), 0.05)
            Xva = np.hstack([aud["clean"][va], w2v[va], multi[va]])
            Xvm = np.hstack([aud["robot_mix"][va], w2v[va], multi[va]])
            oof_ts[va] = 0.5 * (pos(m_ts, Xva) + pos(m_ts, Xvm))
            oof_cs[va] = 0.5 * (pos(m_cs, Xva) + pos(m_cs, Xvm))
            wood = tr[(sy[tr] == 2) | (sy[tr] == 3)]
            if len(wood) > 8:
                Xw = np.vstack(
                    [np.hstack([aud[v][wood], w2v[wood], multi[wood]]) for v in AUDIO]
                )
                yw = np.concatenate([(sy[wood] == 2).astype(int)] * 3)
                oof_tt[va] = pos(fit_bin(Xw, yw, 0.1), Xva)
            print("fold", fold_id + 1, flush=True)

    def apply(pred, ts_th, cs_th, tt_th, do_tt, img_th, img_cs, do_img):
        out = pred.copy()
        out[(out == 0) & (oof_ts >= ts_th) & (np.maximum(oof_cs, oof_bin) >= cs_th)] = 2
        if do_tt:
            out[(out == 3) & (oof_tt >= tt_th) & (oof_ts >= ts_th * 0.9)] = 2
        if do_img:
            out[(out == 0) & (oof_tr >= img_th) & (oof_bin >= img_cs)] = 2
        return out

    primary = [
        dict(ts_th=0.5, cs_th=0.1, tt_th=0.45, do_tt=True),
        dict(ts_th=0.4, cs_th=0.55, tt_th=0.5, do_tt=False),
        dict(ts_th=0.5, cs_th=0.1, tt_th=0.45, do_tt=False),
    ]
    rows = []
    for prim in primary:
        for do_img in (True, False):
            img_grid = (
                [(th, cs) for th in np.linspace(0.45, 0.9, 10) for cs in np.linspace(0.35, 0.85, 11)]
                if do_img
                else [(0.0, 0.0)]
            )
            for img_th, img_cs in img_grid:
                fold_mins = []
                oof_pred = np.zeros(n, np.int64)
                for _, va in folds:
                    scores = []
                    for view in ("clean", "collapse"):
                        base = pred_v2.copy()
                        if view == "collapse":
                            weak = (base > 0) & (oof_cs < 0.5) & (oof_bin < 0.55)
                            base[weak] = 0
                        pred = apply(
                            base,
                            prim["ts_th"],
                            prim["cs_th"],
                            prim["tt_th"],
                            prim["do_tt"],
                            img_th,
                            img_cs,
                            do_img,
                        )
                        scores.append(proto.fast_macro_f1(sy[va], pred[va]))
                    fold_mins.append(min(scores))
                    oof_pred[va] = apply(
                        pred_v2,
                        prim["ts_th"],
                        prim["cs_th"],
                        prim["tt_th"],
                        prim["do_tt"],
                        img_th,
                        img_cs,
                        do_img,
                    )[va]
                clean = proto.fast_macro_f1(sy, oof_pred)
                if clean < 0.93:
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
                # count how often secondary fires on clean hand
                base_p = apply(
                    pred_v2,
                    prim["ts_th"],
                    prim["cs_th"],
                    prim["tt_th"],
                    prim["do_tt"],
                    0,
                    0,
                    False,
                )
                n_secondary = int(np.sum((base_p == 0) & (oof_pred == 2)))
                worst = float(np.min(fold_mins))
                # Prefer worse-view lift + trunk; bonus for secondary that fires under collapse
                # (collapse fold scores already in worst)
                comp = (
                    0.45 * worst
                    + 0.25 * clean
                    + 0.2 * trunk_rec
                    + 0.05 * twig_rec
                    + 0.02 * min(n_secondary, 20) / 20
                    - 0.015 * false_tr
                )
                rows.append(
                    dict(
                        **prim,
                        do_img=do_img,
                        img_th=float(img_th),
                        img_cs=float(img_cs),
                        worst_fold_min=worst,
                        clean_macro_f1=clean,
                        trunk_recall=trunk_rec,
                        twig_recall=twig_rec,
                        ambient_recall=amb,
                        false_trunk=false_tr,
                        n_secondary_flips=n_secondary,
                        composite=comp,
                    )
                )

    board = pd.DataFrame(rows).sort_values(
        ["composite", "trunk_recall", "worst_fold_min"], ascending=False
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_img_secondary_lift_leaderboard.csv", index=False)
    selected = board.iloc[0].to_dict()
    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_img_secondary_lift_hand_group_OOF",
            "base_robot_artifact": (
                "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
                "segment_rule_stack_v2_base_segment_outputs.npz"
            ),
            "selected_candidate": selected,
            "selection_recipe": "primary stress amb-lift + optional secondary CLIP trunk head under dual-collapse worst-fold",
            "n_candidates": int(len(board)),
        },
    )
    print(board.head(15).to_string(index=False))
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
