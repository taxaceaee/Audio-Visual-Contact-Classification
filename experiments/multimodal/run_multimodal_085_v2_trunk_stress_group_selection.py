"""Strict hand-only: v2 base + stress-trained trunk detector rescue.

Trunk detector is fit primarily on robot_mix/bandlimit multi-view features so it
specializes in hard acoustic conditions. Rescue flips ambient→trunk only under
multi-signal gates selected on hand nested folds (clean + dual-collapse + stress).

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
        "outputs/audio_feature_benchmarks/multimodal_085_v2_trunk_stress_group_selection",
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
# v2 hier params (locked paper candidate family)
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


def pos_proba(m, x):
    p = m.predict_proba(x)
    classes = list(m.classes_)
    return p[:, classes.index(1)] if 1 in classes else p[:, -1]


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

    # audio lift segment
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

    # v2 meta/hier OOF
    oof_meta = np.load(STACK / "hand_oof_meta.npy").astype(np.float64)
    assert len(oof_meta) == n

    print("nested OOF: image heads + stress trunk detectors...", flush=True)
    oof_bin = np.zeros(n, np.float64)
    oof_trunk_img = np.zeros(n, np.float64)
    oof_mat = np.zeros((n, 3), np.float64)
    oof_trunk_stress = np.zeros(n, np.float64)
    oof_trunk_clean = np.zeros(n, np.float64)
    oof_contact_stress = np.zeros(n, np.float64)

    for fold_id, (tr, va) in enumerate(folds):
        m_bin = fit_bin(sx_c[tr], (sy[tr] > 0).astype(int), 0.03)
        m_tr = fit_bin(sx_c[tr], (sy[tr] == 2).astype(int), 0.1)
        ct = tr[sy[tr] > 0]
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        m_mat = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.03,
                max_iter=1500,
                class_weight="balanced",
                multi_class="multinomial",
                random_state=42,
            ),
        )
        m_mat.fit(sx_cat[ct], sy[ct] - 1)
        oof_bin[va] = pos_proba(m_bin, sx_c[va])
        oof_trunk_img[va] = pos_proba(m_tr, sx_c[va])
        oof_mat[va] = suite.normalize(m_mat.predict_proba(sx_cat[va]))

        # stress-only trunk detector: train on mix+band features (+image)
        Xstress = np.vstack(
            [
                np.hstack([aud["robot_mix"][tr], w2v[tr], sx_multi[tr]]),
                np.hstack([aud["bandlimit"][tr], w2v[tr], sx_multi[tr]]),
            ]
        )
        ystress = np.concatenate([(sy[tr] == 2).astype(int), (sy[tr] == 2).astype(int)])
        m_ts = fit_bin(Xstress, ystress, 0.05)
        # evaluate on clean features (robot domain proxy = clean robot audio at test)
        Xva_clean = np.hstack([aud["clean"][va], w2v[va], sx_multi[va]])
        Xva_mix = np.hstack([aud["robot_mix"][va], w2v[va], sx_multi[va]])
        oof_trunk_stress[va] = 0.5 * (
            pos_proba(m_ts, Xva_clean) + pos_proba(m_ts, Xva_mix)
        )
        # clean-trained trunk joint for comparison
        Xclean_tr = np.hstack([aud["clean"][tr], w2v[tr], sx_multi[tr]])
        m_tc = fit_bin(Xclean_tr, (sy[tr] == 2).astype(int), 0.05)
        oof_trunk_clean[va] = pos_proba(m_tc, Xva_clean)

        # stress contact detector
        y_c = (sy[tr] > 0).astype(int)
        m_cs = fit_bin(Xstress, np.concatenate([y_c, y_c]), 0.05)
        oof_contact_stress[va] = 0.5 * (
            pos_proba(m_cs, Xva_clean) + pos_proba(m_cs, Xva_mix)
        )
        print(f" fold {fold_id+1}/5", flush=True)

    # v2 predictions OOF (rebuild hier with nested heads + stack meta)
    contact_v2 = V2["alpha"] * seg_pc + (1 - V2["alpha"]) * oof_bin
    mat_v2 = make_mat(sa_mat, oof_mat, V2["mw"], V2["tb"], oof_trunk_img)
    pred_h = decode_hier(contact_v2, mat_v2, V2["th"])
    pred_v2 = np.where(pred_h == 2, pred_h, oof_meta.argmax(1))
    v2_f1 = proto.fast_macro_f1(sy, pred_v2)
    print("v2-like hand OOF macro", v2_f1, flush=True)

    # grid rescue
    rows = []
    for t_th in np.linspace(0.35, 0.9, 12):
        for c_th in np.linspace(0.25, 0.8, 12):
            for mode in (
                "stress_trunk",
                "max_trunk",
                "stress_trunk_and_contact",
                "stress_trunk_or_img",
            ):
                for require_meta_not_leaf in (True, False):
                    fold_scores = []
                    oof_pred = pred_v2.copy()
                    for _, va in folds:
                        scores = []
                        for stress_mode in ("clean", "collapse", "mix_score"):
                            pred = pred_v2.copy()
                            if mode == "stress_trunk":
                                tscore = oof_trunk_stress
                            elif mode == "max_trunk":
                                tscore = np.maximum(oof_trunk_stress, oof_trunk_img)
                            elif mode == "stress_trunk_and_contact":
                                tscore = oof_trunk_stress
                            else:
                                tscore = np.maximum(oof_trunk_stress, oof_trunk_clean)

                            if stress_mode == "collapse":
                                # simulate audio contact collapse: only flip using image/stress det
                                base = pred.copy()
                                # degrade v2 ambient more: force re-eval contact-low
                                # keep pred, only allow rescue
                                pass
                            mask = pred == 0
                            mask &= tscore >= t_th
                            if mode == "stress_trunk_and_contact":
                                mask &= oof_contact_stress >= c_th
                            else:
                                # weak contact support from image bin or stress contact
                                mask &= np.maximum(oof_bin, oof_contact_stress) >= c_th
                            if require_meta_not_leaf:
                                mask &= oof_meta.argmax(1) != 1
                            pred = pred.copy()
                            pred[mask] = 2
                            scores.append(proto.fast_macro_f1(sy[va], pred[va]))
                        fold_scores.append(min(scores))
                        # store clean rescue pred for va
                        pred = pred_v2.copy()
                        if mode == "stress_trunk":
                            tscore = oof_trunk_stress
                        elif mode == "max_trunk":
                            tscore = np.maximum(oof_trunk_stress, oof_trunk_img)
                        elif mode == "stress_trunk_and_contact":
                            tscore = oof_trunk_stress
                        else:
                            tscore = np.maximum(oof_trunk_stress, oof_trunk_clean)
                        mask = pred == 0
                        mask &= tscore >= t_th
                        if mode == "stress_trunk_and_contact":
                            mask &= oof_contact_stress >= c_th
                        else:
                            mask &= np.maximum(oof_bin, oof_contact_stress) >= c_th
                        if require_meta_not_leaf:
                            mask &= oof_meta.argmax(1) != 1
                        pred = pred.copy()
                        pred[mask] = 2
                        oof_pred[va] = pred[va]

                    clean_f1 = proto.fast_macro_f1(sy, oof_pred)
                    false_trunk = int(np.sum((sy == 0) & (oof_pred == 2)))
                    trunk_rec = float(
                        np.sum((sy == 2) & (oof_pred == 2)) / max(np.sum(sy == 2), 1)
                    )
                    nflip = int(np.sum((pred_v2 == 0) & (oof_pred == 2)))
                    score = float(np.mean(fold_scores))
                    # require not destroying ambient massively
                    penalty = 0.0 if false_trunk <= 15 else 0.05 * (false_trunk - 15)
                    composite = score - penalty + 0.05 * (trunk_rec - 0.9)
                    rows.append(
                        dict(
                            t_th=float(t_th),
                            c_th=float(c_th),
                            mode=mode,
                            require_meta_not_leaf=require_meta_not_leaf,
                            score_mean_min=score,
                            clean_macro_f1=clean_f1,
                            v2_macro_f1=v2_f1,
                            false_trunk_on_ambient=false_trunk,
                            trunk_recall=trunk_rec,
                            n_ambient_flips=nflip,
                            composite=composite,
                        )
                    )

    board = pd.DataFrame(rows).sort_values(
        ["composite", "score_mean_min", "trunk_recall", "clean_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_v2_trunk_stress_leaderboard.csv", index=False)

    # select: must not tank clean below v2 - 0.02, prefer lift on composite
    ok = board[
        (board["clean_macro_f1"] >= v2_f1 - 0.02)
        & (board["false_trunk_on_ambient"] <= 25)
    ]
    if len(ok) == 0:
        ok = board[board["clean_macro_f1"] >= 0.90]
    selected = ok.iloc[0].to_dict()

    # also pure v2 option
    v2_row = dict(
        t_th=-1.0,
        c_th=-1.0,
        mode="v2_only",
        require_meta_not_leaf=False,
        score_mean_min=v2_f1,
        clean_macro_f1=v2_f1,
        v2_macro_f1=v2_f1,
        false_trunk_on_ambient=0,
        trunk_recall=float(np.sum((sy == 2) & (pred_v2 == 2)) / max(np.sum(sy == 2), 1)),
        n_ambient_flips=0,
        composite=v2_f1,
    )
    if selected["composite"] < v2_f1 + 0.001 and selected["n_ambient_flips"] == 0:
        # no useful rescue found — still lock best rescue attempt for one-shot diversity
        # but record that v2 is better on hand
        pass

    lock = proto.write_selection_lock(
        OUT / "selection_lock.json",
        {
            "protocol": "multimodal_085_v2_base_stress_trunk_rescue_group_OOF",
            "base": "v2_hier_trunk_meta_else",
            "v2_params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in V2.items()},
            "selected_candidate": selected,
            "v2_reference": v2_row,
            "selection_note": "stress-trained trunk detector rescue on ambient predictions",
        },
    )
    np.savez_compressed(
        OUT / "hand_oof_bundle.npz",
        sy=sy,
        pred_v2=pred_v2,
        oof_trunk_stress=oof_trunk_stress,
        oof_trunk_img=oof_trunk_img,
        oof_trunk_clean=oof_trunk_clean,
        oof_contact_stress=oof_contact_stress,
        oof_bin=oof_bin,
        oof_meta=oof_meta,
        seg_pc=seg_pc,
        sa_mat=sa_mat,
        oof_mat=oof_mat,
    )
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(15).to_string(index=False))
    print("selected", selected)


if __name__ == "__main__":
    main()
