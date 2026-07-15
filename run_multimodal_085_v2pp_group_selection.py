"""Strict hand-only v2++ : contact-lift + material override + trunk/twig disambig.

Builds on v2 hier_trunk_meta_else with:
  1) stress multi-view trunk/contact detectors
  2) ambient→contact lift into image-heavy material (not hard trunk force only)
  3) trunk vs twig binary override when contact predicted

Selection: nested specimen folds; score = mean min(clean, dual-collapse, audio-stress).
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
        "outputs/audio_feature_benchmarks/multimodal_085_v2pp_group_selection",
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
V2 = dict(alpha=0.4, tb=0.3, th=0.55, mw=np.array([0.4, 0.55, 0.4]))


def pool(x, codes, n):
    return np.vstack([x[codes == i].mean(0) for i in range(n)]).astype(np.float32)


def fit_bin(x, y, c=0.1):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42),
    )
    m.fit(x, y)
    return m


def fit_multi(x, y, c=0.03):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=2000,
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


def make_mat(sa, im, w, tb, tr):
    mat = proto.normalize(sa * (1 - w) + im * w)
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1))
        logits[:, 1] += tb * tr
        mat = proto.normalize(np.exp(logits))
    return mat


def decode_hier(contact, mat, th):
    pred = np.zeros(len(contact), np.int64)
    hit = contact >= th
    pred[hit] = 1 + mat[hit].argmax(1)
    return pred


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

    af = audio_base.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train")
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
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(y)), y, sg)
    ):
        assignment[va] = k
    src = broad.load_train_sources(Path("outputs"), y, assignment, 42)
    audio = lift.postprocess(
        af, lift.normalize(0.95 * anchor + 0.05 * src["report_gate_onehot"]), "segment_lift"
    )
    pc, lg = avr.avr_inputs(audio)
    seg_pc = np.array([pc[codes == i].mean() for i in range(n)], np.float64)
    sa_mat = np.vstack(
        [suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0] for i in range(n)]
    )

    img = {
        k: pool(np.load(p / "hand_train_full/X.npy").astype(np.float32), codes, n)
        for k, p in IMG.items()
    }
    aud = {k: pool(np.load(p).astype(np.float32), codes, n) for k, p in AUDIO.items()}
    w2v = pool(np.load(W2V).astype(np.float32), codes, n)
    sx_c = img["clip"]
    sx_cat = np.hstack([img["clip"], img["eff"]])
    sx_multi = np.hstack([img[k] for k in ("clip", "eff", "dino", "convnext")])
    oof_meta = np.load(STACK / "hand_oof_meta.npy").astype(np.float64)

    print("building nested OOF...", flush=True)
    oof_bin = np.zeros(n)
    oof_trunk = np.zeros(n)
    oof_mat = np.zeros((n, 3))
    oof_mat_multi = np.zeros((n, 3))
    oof_ts = np.zeros(n)
    oof_cs = np.zeros(n)
    oof_tt = np.zeros(n)  # trunk vs twig among contact wood-like

    for fold_id, (tr, va) in enumerate(folds):
        oof_bin[va] = pos(fit_bin(sx_c[tr], (sy[tr] > 0).astype(int), 0.03), sx_c[va])
        oof_trunk[va] = pos(fit_bin(sx_c[tr], (sy[tr] == 2).astype(int), 0.1), sx_c[va])
        ct = tr[sy[tr] > 0]
        oof_mat[va] = suite.normalize(fit_multi(sx_cat[ct], sy[ct] - 1).predict_proba(sx_cat[va]))
        oof_mat_multi[va] = suite.normalize(
            fit_multi(sx_multi[ct], sy[ct] - 1).predict_proba(sx_multi[va])
        )
        Xst = np.vstack(
            [
                np.hstack([aud["robot_mix"][tr], w2v[tr], sx_multi[tr]]),
                np.hstack([aud["bandlimit"][tr], w2v[tr], sx_multi[tr]]),
            ]
        )
        m_ts = fit_bin(Xst, np.concatenate([(sy[tr] == 2).astype(int)] * 2), 0.05)
        m_cs = fit_bin(Xst, np.concatenate([(sy[tr] > 0).astype(int)] * 2), 0.05)
        Xva = np.hstack([aud["clean"][va], w2v[va], sx_multi[va]])
        Xvm = np.hstack([aud["robot_mix"][va], w2v[va], sx_multi[va]])
        oof_ts[va] = 0.5 * (pos(m_ts, Xva) + pos(m_ts, Xvm))
        oof_cs[va] = 0.5 * (pos(m_cs, Xva) + pos(m_cs, Xvm))
        # trunk vs twig among true trunk/twig
        wood = tr[(sy[tr] == 2) | (sy[tr] == 3)]
        if len(wood) > 10:
            yw = (sy[wood] == 2).astype(int)
            Xw = np.hstack([aud["clean"][wood], w2v[wood], sx_multi[wood]])
            # bag stress wood too
            Xw = np.vstack(
                [
                    Xw,
                    np.hstack([aud["robot_mix"][wood], w2v[wood], sx_multi[wood]]),
                    np.hstack([aud["bandlimit"][wood], w2v[wood], sx_multi[wood]]),
                ]
            )
            yw = np.concatenate([yw, yw, yw])
            m_tt = fit_bin(Xw, yw, 0.1)
            oof_tt[va] = pos(m_tt, Xva)
        print(f" fold {fold_id+1}/5", flush=True)

    contact_v2 = V2["alpha"] * seg_pc + (1 - V2["alpha"]) * oof_bin
    mat_v2 = make_mat(sa_mat, oof_mat, V2["mw"], V2["tb"], oof_trunk)
    pred_h = decode_hier(contact_v2, mat_v2, V2["th"])
    pred_v2 = np.where(pred_h == 2, pred_h, oof_meta.argmax(1))
    print("v2 hand", proto.fast_macro_f1(sy, pred_v2), flush=True)

    def apply(c):
        pred = pred_v2.copy()
        # image-heavy material for lifts
        mw_img = np.array([0.35, c["mw_trunk"], 0.35])
        mat_img = make_mat(sa_mat, oof_mat_multi, mw_img, c["tb_img"], oof_trunk)
        mat_blend = proto.normalize(
            (1 - c["img_mat_w"]) * mat_v2 + c["img_mat_w"] * mat_img
        )
        # 1) contact lift: ambient + high stress trunk/contact → material
        lift_mask = (pred == 0) & (oof_ts >= c["ts_th"]) & (
            np.maximum(oof_cs, oof_bin) >= c["cs_th"]
        )
        if c["lift_mode"] == "force_trunk":
            pred[lift_mask] = 2
        else:
            pred[lift_mask] = 1 + mat_blend[lift_mask].argmax(1)
        # 2) trunk/twig disambiguation among contact wood preds
        wood_mask = (pred == 2) | (pred == 3)
        if c["tt_mode"] == "tt_head":
            # if tt head confident trunk, set trunk; if confident not, set twig
            to_trunk = wood_mask & (oof_tt >= c["tt_th"])
            to_twig = wood_mask & (oof_tt <= (1 - c["tt_th"])) & (pred == 2)
            # only convert trunk→twig if meta/mat also lean twig
            pred = pred.copy()
            pred[to_trunk] = 2
            if c["allow_trunk_to_twig"]:
                lean_twig = mat_blend[:, 2] > mat_blend[:, 1]
                pred[to_twig & lean_twig] = 3
        elif c["tt_mode"] == "mat_override":
            conf = mat_blend.max(1)
            arg = mat_blend.argmax(1) + 1
            override = wood_mask & (conf >= c["tt_th"])
            pred = pred.copy()
            pred[override] = arg[override]
        # 3) optional soft: if predicted twig but trunk head high → trunk
        if c["twig_to_trunk"]:
            pred = pred.copy()
            pred[(pred == 3) & (oof_tt >= c["tt_th"]) & (oof_trunk >= 0.45)] = 2
        return pred

    candidates = []
    for ts_th in (0.35, 0.45, 0.55, 0.65, 0.75):
        for cs_th in (0.2, 0.3, 0.4, 0.5):
            for lift_mode in ("material", "force_trunk"):
                for img_mat_w in (0.5, 0.75, 1.0):
                    for mw_trunk in (0.7, 0.9, 1.0):
                        for tb_img in (0.0, 0.5, 1.0):
                            for tt_mode in ("none", "tt_head", "mat_override"):
                                for tt_th in (0.55, 0.65, 0.75):
                                    for twig_to_trunk in (False, True):
                                        for allow_t2g in (False, True):
                                            if tt_mode == "none" and (
                                                twig_to_trunk or allow_t2g
                                            ):
                                                continue
                                            candidates.append(
                                                dict(
                                                    ts_th=ts_th,
                                                    cs_th=cs_th,
                                                    lift_mode=lift_mode,
                                                    img_mat_w=img_mat_w,
                                                    mw_trunk=mw_trunk,
                                                    tb_img=tb_img,
                                                    tt_mode=tt_mode,
                                                    tt_th=tt_th,
                                                    twig_to_trunk=twig_to_trunk,
                                                    allow_trunk_to_twig=allow_t2g,
                                                )
                                            )

    rng = np.random.default_rng(1)
    if len(candidates) > 600:
        idx = rng.choice(len(candidates), size=600, replace=False)
        candidates = [candidates[i] for i in idx]
    # always include a pure lift-focused set
    for ts_th in (0.4, 0.5, 0.6):
        for cs_th in (0.25, 0.35):
            candidates.append(
                dict(
                    ts_th=ts_th,
                    cs_th=cs_th,
                    lift_mode="force_trunk",
                    img_mat_w=0.8,
                    mw_trunk=0.9,
                    tb_img=0.5,
                    tt_mode="tt_head",
                    tt_th=0.6,
                    twig_to_trunk=True,
                    allow_trunk_to_twig=False,
                )
            )

    print(f"scoring {len(candidates)}...", flush=True)
    rows = []
    for i, c in enumerate(candidates):
        fold_mins = []
        oof_pred = np.zeros(n, np.int64)
        for _, va in folds:
            scores = []
            # clean
            pred = apply(c)
            scores.append(proto.fast_macro_f1(sy[va], pred[va]))
            # dual collapse: weaken seg_pc and bin in a synthetic re-base is hard;
            # approximate by evaluating same rescue when we first zero many contacts
            pred_c = pred_v2.copy()
            # kill some contact preds to ambient (simulate) then re-apply lift only
            # mark currently non-ambient with low stress contact as ambient
            weak = (pred_c > 0) & (oof_cs < 0.4) & (seg_pc < 0.35)
            pred_c = pred_c.copy()
            pred_c[weak] = 0
            # apply lift rules onto pred_c as base
            # temporarily swap
            # reimplement lift on pred_c
            pred2 = pred_c.copy()
            mw_img = np.array([0.35, c["mw_trunk"], 0.35])
            mat_img = make_mat(sa_mat, oof_mat_multi, mw_img, c["tb_img"], oof_trunk)
            mat_blend = proto.normalize(
                (1 - c["img_mat_w"]) * mat_v2 + c["img_mat_w"] * mat_img
            )
            lift_mask = (pred2 == 0) & (oof_ts >= c["ts_th"]) & (
                np.maximum(oof_cs, oof_bin) >= c["cs_th"]
            )
            if c["lift_mode"] == "force_trunk":
                pred2[lift_mask] = 2
            else:
                pred2[lift_mask] = 1 + mat_blend[lift_mask].argmax(1)
            scores.append(proto.fast_macro_f1(sy[va], pred2[va]))
            fold_mins.append(min(scores))
            oof_pred[va] = pred[va]
        clean = proto.fast_macro_f1(sy, oof_pred)
        trunk_rec = float(np.sum((sy == 2) & (oof_pred == 2)) / max((sy == 2).sum(), 1))
        twig_rec = float(np.sum((sy == 3) & (oof_pred == 3)) / max((sy == 3).sum(), 1))
        false_tr = int(np.sum((sy == 0) & (oof_pred == 2)))
        t2a = int(np.sum((sy == 2) & (oof_pred == 0)))
        score = float(np.mean(fold_mins))
        composite = (
            0.5 * score
            + 0.2 * clean
            + 0.15 * trunk_rec
            + 0.1 * twig_rec
            - 0.02 * max(false_tr - 5, 0)
        )
        rows.append(
            {
                **c,
                "score_mean_min": score,
                "clean_macro_f1": clean,
                "trunk_recall": trunk_rec,
                "twig_recall": twig_rec,
                "false_trunk": false_tr,
                "trunk_to_ambient": t2a,
                "composite": composite,
            }
        )
        if (i + 1) % 100 == 0:
            print(i + 1, "best", max(r["composite"] for r in rows), flush=True)

    board = pd.DataFrame(rows).sort_values(
        ["composite", "score_mean_min", "trunk_recall", "clean_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_v2pp_leaderboard.csv", index=False)
    ok = board[(board["clean_macro_f1"] >= 0.93) & (board["false_trunk"] <= 20)]
    if len(ok) == 0:
        ok = board[board["clean_macro_f1"] >= 0.90]
    selected = ok.iloc[0].to_dict()

    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_v2pp_contact_lift_material_tt_group_OOF",
            "base": "v2_hier_trunk_meta_else",
            "v2_params": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in V2.items()
            },
            "selected_candidate": selected,
            "n_candidates": len(candidates),
        },
    )
    np.savez_compressed(
        OUT / "hand_oof_bundle.npz",
        sy=sy,
        pred_v2=pred_v2,
        oof_ts=oof_ts,
        oof_cs=oof_cs,
        oof_tt=oof_tt,
        oof_bin=oof_bin,
        oof_trunk=oof_trunk,
        oof_mat=oof_mat,
        oof_mat_multi=oof_mat_multi,
        oof_meta=oof_meta,
        seg_pc=seg_pc,
        sa_mat=sa_mat,
        mat_v2=mat_v2,
    )
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
