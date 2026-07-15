"""Hand-only selection of post-hoc rescue rules on top of v2-like OOF.

Rules apply after hier_trunk_meta_else. Never loads robot/test.
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
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_085_exact_v2_rescue_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
IMG_CLIP = Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k")
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
    audio_base.configure_feature_set("total240")
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

    oof_meta = np.load(STACK / "hand_oof_meta.npy")
    oof_hier = np.load(STACK / "hand_oof_hier.npy")
    pred_v2 = np.where(oof_hier.argmax(1) == 2, oof_hier.argmax(1), oof_meta.argmax(1))
    print("base v2-like", proto.fast_macro_f1(sy, pred_v2))

    img = {
        k: pool(np.load(p / "hand_train_full/X.npy").astype(np.float32), codes, n)
        for k, p in IMG_MULTI.items()
    }
    multi = np.hstack([img[k] for k in ("clip", "eff", "dino", "convnext")])
    aud = {k: pool(np.load(p).astype(np.float32), codes, n) for k, p in AUDIO.items()}
    w2v = pool(np.load(W2V).astype(np.float32), codes, n)
    sx_c = img["clip"]

    oof_bin = np.zeros(n)
    oof_ts = np.zeros(n)
    oof_cs = np.zeros(n)
    oof_tt = np.zeros(n)
    oof_tr_img = np.zeros(n)

    print("OOF detectors...", flush=True)
    for fold_id, (tr, va) in enumerate(folds):
        oof_bin[va] = pos(fit_bin(sx_c[tr], (sy[tr] > 0).astype(int), 0.03), sx_c[va])
        oof_tr_img[va] = pos(fit_bin(sx_c[tr], (sy[tr] == 2).astype(int), 0.1), sx_c[va])
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
                [
                    np.hstack([aud[v][wood], w2v[wood], multi[wood]])
                    for v in ("clean", "robot_mix", "bandlimit")
                ]
            )
            yw = np.concatenate([(sy[wood] == 2).astype(int)] * 3)
            m_tt = fit_bin(Xw, yw, 0.1)
            oof_tt[va] = pos(m_tt, Xva)
        print(f" fold {fold_id+1}/5", flush=True)

    def rescue(pred, ts_th, cs_th, tt_th, do_amb_lift, do_twig_to_trunk, tscore_mode):
        out = pred.copy()
        if tscore_mode == "stress":
            tscore = oof_ts
        elif tscore_mode == "max":
            tscore = np.maximum(oof_ts, oof_tr_img)
        else:
            tscore = oof_tr_img
        if do_amb_lift:
            m = (out == 0) & (tscore >= ts_th) & (np.maximum(oof_cs, oof_bin) >= cs_th)
            out[m] = 2
        if do_twig_to_trunk:
            m = (out == 3) & (oof_tt >= tt_th) & (tscore >= ts_th * 0.9)
            out[m] = 2
        return out

    rows = []
    for tscore_mode in ("stress", "max", "img"):
        for ts_th in np.linspace(0.35, 0.85, 11):
            for cs_th in np.linspace(0.15, 0.6, 10):
                for tt_th in np.linspace(0.5, 0.85, 8):
                    for do_amb in (True, False):
                        for do_tt in (True, False):
                            if not do_amb and not do_tt:
                                continue
                            fold_mins = []
                            oof_pred = np.zeros(n, np.int64)
                            for _, va in folds:
                                scores = []
                                for view in ("clean", "collapse"):
                                    if view == "clean":
                                        base = pred_v2
                                    else:
                                        # weaken: force some contact preds to ambient then rescue
                                        base = pred_v2.copy()
                                        weak = (base > 0) & (oof_cs < 0.45) & (oof_bin < 0.5)
                                        base = base.copy()
                                        base[weak] = 0
                                    pred = rescue(
                                        base, ts_th, cs_th, tt_th, do_amb, do_tt, tscore_mode
                                    )
                                    scores.append(proto.fast_macro_f1(sy[va], pred[va]))
                                fold_mins.append(min(scores))
                                oof_pred[va] = rescue(
                                    pred_v2, ts_th, cs_th, tt_th, do_amb, do_tt, tscore_mode
                                )[va]
                            clean = proto.fast_macro_f1(sy, oof_pred)
                            false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
                            trunk_rec = float(
                                np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1)
                            )
                            twig_rec = float(
                                np.sum((sy == 3) & (oof_pred == 3)) / max((sy == 3).sum(), 1)
                            )
                            t2a = int(np.sum((sy == 2) & (oof_pred == 0)))
                            score = float(np.mean(fold_mins))
                            # require maintain near-v2 clean
                            if clean < 0.93 or false_tr > 30:
                                continue
                            composite = (
                                0.45 * score
                                + 0.25 * clean
                                + 0.2 * trunk_rec
                                + 0.1 * twig_rec
                                - 0.01 * false_tr
                            )
                            rows.append(
                                dict(
                                    tscore_mode=tscore_mode,
                                    ts_th=float(ts_th),
                                    cs_th=float(cs_th),
                                    tt_th=float(tt_th),
                                    do_amb_lift=do_amb,
                                    do_twig_to_trunk=do_tt,
                                    score_mean_min=score,
                                    clean_macro_f1=clean,
                                    trunk_recall=trunk_rec,
                                    twig_recall=twig_rec,
                                    false_trunk=false_tr,
                                    trunk_to_ambient=t2a,
                                    composite=composite,
                                )
                            )

    board = pd.DataFrame(rows).sort_values(
        ["composite", "trunk_recall", "clean_macro_f1"], ascending=False
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_exact_v2_rescue_leaderboard.csv", index=False)
    selected = board.iloc[0].to_dict()
    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_exact_v2_posthoc_rescue_group_OOF",
            "base_robot_artifact": "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/segment_rule_stack_v2_base_segment_outputs.npz",
            "selected_candidate": selected,
            "base_hand_macro_f1": proto.fast_macro_f1(sy, pred_v2),
        },
    )
    np.savez_compressed(
        OUT / "hand_oof_detectors.npz",
        sy=sy,
        pred_v2=pred_v2,
        oof_ts=oof_ts,
        oof_cs=oof_cs,
        oof_tt=oof_tt,
        oof_bin=oof_bin,
        oof_tr_img=oof_tr_img,
    )
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
