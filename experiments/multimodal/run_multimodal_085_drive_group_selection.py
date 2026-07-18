"""Strict hand-only selection for multimodal Macro F1 > 0.85 drive.

Builds multi-view multi-backbone OOF at segment level, then selects contact/material
fusion under clean + multi-view audio stress + dual contact-collapse surrogates.
Writes selection_lock.json with test_loaded=false. Never loads robot/test.
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
        "outputs/audio_feature_benchmarks/multimodal_085_drive_group_selection",
    )
)
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
}
AUDIO_VIEWS = {
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
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
META_FEAT = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_features.npy")
META_Y = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_y.npy")


def pool(x, codes, n):
    return np.vstack([x[codes == i].mean(0) for i in range(n)]).astype(np.float32)


def fit_lr(x, y, c=0.1, multi=True, balanced=True):
    cw = "balanced" if balanced else None
    if multi and len(np.unique(y)) > 2:
        clf = LogisticRegression(
            C=c,
            max_iter=2500,
            class_weight=cw,
            multi_class="multinomial",
            random_state=42,
        )
    else:
        clf = LogisticRegression(C=c, max_iter=2500, class_weight=cw, random_state=42)
    m = make_pipeline(StandardScaler(), clf)
    m.fit(x, y)
    return m


def proba4(model, x) -> np.ndarray:
    p = model.predict_proba(x)
    out = np.zeros((len(x), 4), np.float64)
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, j]
    return proto.normalize(out)


def proba1(model, x) -> np.ndarray:
    p = model.predict_proba(x)
    # positive class column
    if p.shape[1] == 1:
        return p[:, 0]
    # find class 1
    classes = list(model.classes_)
    if 1 in classes:
        return p[:, classes.index(1)]
    return p[:, -1]


def decode_hier(contact, mat3, th):
    pred = np.zeros(len(contact), np.int64)
    hit = contact >= th
    pred[hit] = 1 + mat3[hit].argmax(1)
    return pred


def make_mat(sa_mat, imat, w, tb, trunk_s):
    mat = proto.normalize(sa_mat * (1.0 - w) + imat * w)
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1.0))
        logits[:, 1] += tb * trunk_s
        mat = proto.normalize(np.exp(logits))
    return mat


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
    nseg = len(useg)
    sy = np.array([y[codes == i][0] for i in range(nseg)], np.int64)
    ss = np.array([sg[codes == i][0] for i in range(nseg)])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(nseg), sy, ss
        )
    )

    # locked audio OOF (window) → segment
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
    seg_pc = np.array([pc[codes == i].mean() for i in range(nseg)], np.float64)
    sa_mat = np.vstack(
        [
            suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0]
            for i in range(nseg)
        ]
    )
    sa4 = np.vstack(
        [suite.normalize(audio[codes == i].mean(0, keepdims=True))[0] for i in range(nseg)]
    )

    # embeddings
    img_seg = {
        n: pool(np.load(p / "hand_train_full/X.npy").astype(np.float32), codes, nseg)
        for n, p in IMG.items()
    }
    audio_seg = {v: pool(np.load(p).astype(np.float32), codes, nseg) for v, p in AUDIO_VIEWS.items()}
    w2v_seg = pool(np.load(W2V).astype(np.float32), codes, nseg)
    sx_c = img_seg["clip"]
    sx_cat = np.hstack([img_seg["clip"], img_seg["eff"]])
    sx_multi = np.hstack([img_seg[k] for k in ("clip", "eff", "dino", "convnext")])
    # multi-view audio concat for joint
    sx_audio_joint = {
        v: np.hstack([audio_seg[v], w2v_seg]) for v in audio_seg
    }

    print("building nested OOF heads...", flush=True)
    oof = {
        "clip4": np.zeros((nseg, 4), np.float64),
        "w2v4": np.zeros((nseg, 4), np.float64),
        "joint4": {v: np.zeros((nseg, 4), np.float64) for v in audio_seg},
        "bin_clip": np.zeros(nseg, np.float64),
        "bin_multi": np.zeros(nseg, np.float64),
        "trunk_clip": np.zeros(nseg, np.float64),
        "trunk_joint": {v: np.zeros(nseg, np.float64) for v in audio_seg},
        "mat_clip_eff": np.zeros((nseg, 3), np.float64),
        "mat_multi": np.zeros((nseg, 3), np.float64),
        "audio_view4": {v: np.zeros((nseg, 4), np.float64) for v in audio_seg},
    }

    for fold_id, (tr, va) in enumerate(folds):
        m_clip4 = fit_lr(sx_c[tr], sy[tr], 0.03)
        m_w2v = fit_lr(w2v_seg[tr], sy[tr], 0.1)
        m_bin = fit_lr(sx_c[tr], (sy[tr] > 0).astype(int), 0.03, multi=False)
        m_bin_m = fit_lr(sx_multi[tr], (sy[tr] > 0).astype(int), 0.03, multi=False)
        m_trunk = fit_lr(sx_c[tr], (sy[tr] == 2).astype(int), 0.1, multi=False)
        ct = tr[sy[tr] > 0]
        m_mat = fit_lr(sx_cat[ct], sy[ct] - 1, 0.03)
        m_mat_m = fit_lr(sx_multi[ct], sy[ct] - 1, 0.03)

        oof["clip4"][va] = proba4(m_clip4, sx_c[va])
        oof["w2v4"][va] = proba4(m_w2v, w2v_seg[va])
        oof["bin_clip"][va] = proba1(m_bin, sx_c[va])
        oof["bin_multi"][va] = proba1(m_bin_m, sx_multi[va])
        oof["trunk_clip"][va] = proba1(m_trunk, sx_c[va])
        oof["mat_clip_eff"][va] = suite.normalize(m_mat.predict_proba(sx_cat[va]))
        oof["mat_multi"][va] = suite.normalize(m_mat_m.predict_proba(sx_multi[va]))

        for v in audio_seg:
            Xtr = np.hstack([sx_audio_joint[v][tr], sx_multi[tr]])
            Xva = np.hstack([sx_audio_joint[v][va], sx_multi[va]])
            # multi-view bag train: stack all audio views for this fold train
            Xbag = np.vstack(
                [
                    np.hstack([sx_audio_joint[vv][tr], sx_multi[tr]])
                    for vv in audio_seg
                ]
            )
            ybag = np.concatenate([sy[tr] for _ in audio_seg])
            m_j = fit_lr(Xbag, ybag, 0.05)
            oof["joint4"][v][va] = proba4(m_j, Xva)
            m_tj = fit_lr(Xbag, (ybag == 2).astype(int), 0.1, multi=False)
            oof["trunk_joint"][v][va] = proba1(m_tj, Xva)
            m_a = fit_lr(audio_seg[v][tr], sy[tr], 0.1)
            oof["audio_view4"][v][va] = proba4(m_a, audio_seg[v][va])
        print(f" fold {fold_id+1}/5", flush=True)

    # meta from prior nested features (hand OOF)
    meta = fit_lr(np.load(META_FEAT), np.load(META_Y), 0.1)
    # rebuild meta proba at segment using clip material + audio like v2
    # use stored stack OOF if same length
    if (STACK / "hand_oof_meta.npy").exists():
        oof_meta = np.load(STACK / "hand_oof_meta.npy").astype(np.float64)
        oof_hier_base = np.load(STACK / "hand_oof_hier.npy").astype(np.float64)
        assert len(oof_meta) == nseg
    else:
        oof_meta = oof["clip4"].copy()
        oof_hier_base = oof["clip4"].copy()

    # persist OOF
    np.savez_compressed(
        OUT / "hand_oof_bundle.npz",
        sy=sy,
        ss=ss.astype(str),
        useg=useg.astype(str),
        seg_pc=seg_pc,
        sa_mat=sa_mat,
        sa4=sa4,
        clip4=oof["clip4"],
        w2v4=oof["w2v4"],
        bin_clip=oof["bin_clip"],
        bin_multi=oof["bin_multi"],
        trunk_clip=oof["trunk_clip"],
        mat_clip_eff=oof["mat_clip_eff"],
        mat_multi=oof["mat_multi"],
        joint_clean=oof["joint4"]["clean"],
        joint_mix=oof["joint4"]["robot_mix"],
        joint_band=oof["joint4"]["bandlimit"],
        trunk_joint_clean=oof["trunk_joint"]["clean"],
        meta=oof_meta,
        hier_base=oof_hier_base,
    )

    # candidate generators
    candidates = []

    # 1) image-heavy hier + meta rules
    for alpha in (0.05, 0.15, 0.25, 0.35, 0.45):
        for th in (0.35, 0.45, 0.55, 0.65):
            for tb in (0.0, 0.4, 0.8, 1.2):
                for mw_t in (0.55, 0.75, 0.95):
                    for bin_src in ("clip", "multi", "max", "mean"):
                        for mat_src in ("clip_eff", "multi", "blend"):
                            for rule in (
                                "hier_trunk_meta_else",
                                "hier_only",
                                "max_contact_then_mat",
                                "joint_if_conf",
                            ):
                                candidates.append(
                                    dict(
                                        family="hier",
                                        alpha=alpha,
                                        th=th,
                                        tb=tb,
                                        mw=np.array([0.45, mw_t, 0.45]),
                                        bin_src=bin_src,
                                        mat_src=mat_src,
                                        rule=rule,
                                        jw=0.0,
                                        trunk_force=-1.0,
                                    )
                                )

    # 2) soft blends with joint
    for jw in (0.2, 0.4, 0.6, 0.8, 1.0):
        for rule in ("soft_v2_joint", "joint_only", "hier_trunk_joint_else"):
            candidates.append(
                dict(
                    family="joint",
                    alpha=0.25,
                    th=0.55,
                    tb=0.3,
                    mw=np.array([0.4, 0.7, 0.4]),
                    bin_src="max",
                    mat_src="blend",
                    rule=rule,
                    jw=jw,
                    trunk_force=-1.0,
                )
            )

    # 3) trunk force on ambient when multi-signal
    for tf in (0.55, 0.65, 0.75, 0.85):
        candidates.append(
            dict(
                family="trunk_force",
                alpha=0.2,
                th=0.5,
                tb=0.6,
                mw=np.array([0.4, 0.85, 0.4]),
                bin_src="max",
                mat_src="multi",
                rule="hier_trunk_meta_else",
                jw=0.35,
                trunk_force=tf,
            )
        )

    def bin_contact(bin_src, fac_audio=1.0, fac_img=1.0):
        a = seg_pc * fac_audio
        if bin_src == "clip":
            b = oof["bin_clip"] * fac_img
        elif bin_src == "multi":
            b = oof["bin_multi"] * fac_img
        elif bin_src == "max":
            b = np.maximum(oof["bin_clip"], oof["bin_multi"]) * fac_img
        else:
            b = 0.5 * (oof["bin_clip"] + oof["bin_multi"]) * fac_img
        return a, b

    def mat_of(mat_src, mw, tb, trunk_s, sa=None):
        sa = sa_mat if sa is None else sa
        if mat_src == "clip_eff":
            im = oof["mat_clip_eff"]
        elif mat_src == "multi":
            im = oof["mat_multi"]
        else:
            im = proto.normalize(0.5 * oof["mat_clip_eff"] + 0.5 * oof["mat_multi"])
        return make_mat(sa, im, mw, tb, trunk_s)

    def predict_cand(c, view="clean", collapse=None):
        # collapse: (fac_audio, fac_img) or None
        fac_a, fac_i = (1.0, 1.0) if collapse is None else collapse
        a, b = bin_contact(c["bin_src"], fac_a, fac_i)
        contact = c["alpha"] * a + (1 - c["alpha"]) * b
        trunk_s = oof["trunk_clip"]
        if view != "clean":
            trunk_s = np.maximum(trunk_s, oof["trunk_joint"][view])
        mat = mat_of(c["mat_src"], c["mw"], c["tb"], trunk_s)
        pred_h = decode_hier(contact, mat, c["th"])
        pj = oof["joint4"][view]
        meta_arg = oof_meta.argmax(1)
        rule = c["rule"]
        if rule == "hier_only":
            pred = pred_h.copy()
        elif rule == "hier_trunk_meta_else":
            pred = np.where(pred_h == 2, pred_h, meta_arg)
        elif rule == "max_contact_then_mat":
            # contact from max sources; material argmax
            contact2 = np.maximum(a, b)
            pred = np.where(contact2 >= c["th"], 1 + mat.argmax(1), 0)
        elif rule == "joint_if_conf":
            conf_j = pj.max(1)
            conf_m = oof_meta.max(1)
            pred = np.where(conf_j >= conf_m, pj.argmax(1), meta_arg)
            pred = np.where(pred_h == 2, 2, pred)
        elif rule == "soft_v2_joint":
            # v2 soft = hier trunk else meta
            p_v2 = oof_meta.copy()
            mask = pred_h == 2
            # one-hot-ish from hier
            p_h = np.zeros_like(p_v2)
            p_h[np.arange(len(p_h)), pred_h] = 1.0
            p_v2[mask] = p_h[mask]
            p = proto.normalize((1 - c["jw"]) * p_v2 + c["jw"] * pj)
            pred = p.argmax(1)
        elif rule == "joint_only":
            pred = pj.argmax(1)
        elif rule == "hier_trunk_joint_else":
            pred = np.where(pred_h == 2, 2, pj.argmax(1))
        else:
            raise ValueError(rule)
        tf = c["trunk_force"]
        if tf is not None and tf > 0:
            # multi-signal trunk force on ambient predictions
            tscore = np.maximum(oof["trunk_clip"], oof["trunk_joint"][view])
            bscore = np.maximum(oof["bin_clip"], oof["bin_multi"])
            force = (pred == 0) & (tscore >= tf) & (bscore >= 0.35)
            pred = pred.copy()
            pred[force] = 2
        return pred

    # score candidates (subsample if too many)
    rng = np.random.default_rng(0)
    if len(candidates) > 800:
        # keep all joint/trunk_force + sample hier
        keep = [c for c in candidates if c["family"] != "hier"]
        hier = [c for c in candidates if c["family"] == "hier"]
        pick = rng.choice(len(hier), size=500, replace=False)
        candidates = keep + [hier[i] for i in pick]

    print(f"scoring {len(candidates)} candidates...", flush=True)
    rows = []
    views = ("clean", "robot_mix", "bandlimit")
    collapses = [None, (0.2, 0.35), (0.1, 0.2), (0.05, 0.15)]

    best = None
    best_score = -1.0

    for i, c in enumerate(candidates):
        fold_mins = []
        for _, va in folds:
            scores = []
            for view in views:
                pred = predict_cand(c, view=view, collapse=None)
                scores.append(proto.fast_macro_f1(sy[va], pred[va]))
            for col in collapses[1:]:
                pred = predict_cand(c, view="clean", collapse=col)
                scores.append(proto.fast_macro_f1(sy[va], pred[va]))
            fold_mins.append(min(scores))
        pred_clean = predict_cand(c, "clean", None)
        # dual hard trunk recall
        pred_dual = predict_cand(c, "clean", (0.1, 0.2))
        score = float(np.mean(fold_mins))
        worst = float(np.min(fold_mins))
        clean_f1 = proto.fast_macro_f1(sy, pred_clean)
        dual_f1 = proto.fast_macro_f1(sy, pred_dual)
        trunk_clean = float(np.sum((sy == 2) & (pred_clean == 2)) / max(np.sum(sy == 2), 1))
        trunk_dual = float(np.sum((sy == 2) & (pred_dual == 2)) / max(np.sum(sy == 2), 1))
        t2a_dual = int(np.sum((sy == 2) & (pred_dual == 0)))
        # composite: prioritize worst-view/surrogate while keeping clean high
        composite = 0.55 * score + 0.25 * dual_f1 + 0.10 * trunk_dual + 0.10 * clean_f1
        if clean_f1 < 0.88:
            composite -= 0.15
        row = {
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()},
            "score_mean_min": score,
            "score_worst_min": worst,
            "clean_macro_f1": clean_f1,
            "dual_macro_f1": dual_f1,
            "trunk_recall_clean": trunk_clean,
            "trunk_recall_dual": trunk_dual,
            "trunk_to_ambient_dual": t2a_dual,
            "composite": composite,
        }
        rows.append(row)
        if composite > best_score:
            best_score = composite
            best = row
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(candidates)} best_composite={best_score:.4f}", flush=True)

    board = pd.DataFrame(rows).sort_values(
        ["composite", "score_mean_min", "trunk_recall_dual", "clean_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_085_drive_leaderboard.csv", index=False)

    # also pick max dual trunk with clean>=0.90
    viable = board[board["clean_macro_f1"] >= 0.90]
    if len(viable) == 0:
        viable = board
    selected = viable.iloc[0].to_dict()
    # prefer best composite among clean>=0.92 if exists
    strong = board[board["clean_macro_f1"] >= 0.92]
    if len(strong):
        selected = strong.iloc[0].to_dict()

    # v2 baseline on hand for reference
    pred_v2 = np.where(oof_hier_base.argmax(1) == 2, oof_hier_base.argmax(1), oof_meta.argmax(1))
    v2_clean = proto.fast_macro_f1(sy, pred_v2)

    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_drive_multiview_joint_hier_group_OOF",
            "selected_candidate": selected,
            "v2_hand_clean_macro_f1": v2_clean,
            "n_candidates_scored": len(candidates),
            "selection_objective": "composite of min(clean,audio-stress,dual-collapse) folds + dual trunk",
            "feature_sources": {
                "audio": list(AUDIO_VIEWS),
                "image": list(IMG),
                "wav2vec2": True,
                "locked_audio_lift": True,
            },
        },
    )
    # save selected clean predictions
    # rebuild cand dict
    sc = selected
    cand = dict(
        family=sc["family"],
        alpha=float(sc["alpha"]),
        th=float(sc["th"]),
        tb=float(sc["tb"]),
        mw=np.array(sc["mw"], dtype=np.float64),
        bin_src=sc["bin_src"],
        mat_src=sc["mat_src"],
        rule=sc["rule"],
        jw=float(sc["jw"]),
        trunk_force=float(sc["trunk_force"]),
    )
    pred_sel = predict_cand(cand, "clean", None)
    np.save(OUT / "hand_selected_pred.npy", pred_sel)
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(12).to_string(index=False))
    print("selected clean", selected["clean_macro_f1"], "dual trunk", selected["trunk_recall_dual"])


if __name__ == "__main__":
    main()
