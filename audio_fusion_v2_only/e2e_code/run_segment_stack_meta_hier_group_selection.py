"""Stack meta-weighted + hierarchical contact-rescue OOF (hand only).

Two base segment models produce OOF 4-class probs; a multinomial meta-learner
is selected with specimen-group StratifiedGroupKFold. Never loads robot/test.
"""
from __future__ import annotations

import json
import os
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
        "outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection",
    )
)
IMG_CLIP = Path(
    "outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"
)
IMG_EFF = Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k")


def seg(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)


def spec(s: pd.Series) -> pd.Series:
    return seg(s).str.replace(r"_segment_.*$", "", regex=True)


def feat_meta(a, i, p):
    return np.hstack(
        [
            np.log(np.clip(a, 1e-6, 1)),
            np.log(np.clip(i, 1e-6, 1)),
            a,
            i,
            p[:, None],
            a.max(1, keepdims=True),
            i.max(1, keepdims=True),
        ]
    ).astype(np.float32)


def main() -> None:
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

    # Locked audio OOF
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
    sa4 = np.vstack(
        [
            suite.normalize(audio[codes == i].mean(0, keepdims=True))[0]
            for i in range(len(useg))
        ]
    )

    x_clip = np.load(IMG_CLIP / "hand_train_full/X.npy").astype(np.float32)
    x_eff = np.load(IMG_EFF / "hand_train_full/X.npy").astype(np.float32)
    sx_c = np.vstack([x_clip[codes == i].mean(0) for i in range(len(useg))])
    sx_e = np.vstack([x_eff[codes == i].mean(0) for i in range(len(useg))])
    sx_cat = np.hstack([sx_c, sx_e])

    # Nested OOF base predictions for stacking
    oof_meta = np.zeros((len(sy), 4), np.float32)
    oof_hier = np.zeros((len(sy), 4), np.float32)
    oof_blend = np.zeros((len(sy), 4), np.float32)

    # Fixed base HPs from prior hand-selected winners (not from robot test):
    # meta-weighted: C=0.1, trunk_w=1.2, twig_w=1.2
    # hier: alpha=0.5 clip contact, cat material, weights, th=0.56, trunk_bias=0.45
    META_C, META_TW, META_GW = 0.1, 1.2, 1.2
    HIER = dict(alpha=0.5, leaf_w=0.4, trunk_w=0.55, twig_w=0.4, th=0.56, tb=0.45)

    print("building nested OOF base models...", flush=True)
    for fold_id, (tr, va) in enumerate(folds):
        # --- image heads on train folds only ---
        def fit_4(x, ytr_idx):
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
            m.fit(x[ytr_idx], sy[ytr_idx])
            return m

        def fit_mat(x, ytr_idx):
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
            ci = ytr_idx[sy[ytr_idx] > 0]
            m.fit(x[ci], sy[ci] - 1)
            return m

        def fit_bin(x, ytr_idx):
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.03, max_iter=1500, class_weight="balanced", random_state=42
                ),
            )
            m.fit(x[ytr_idx], (sy[ytr_idx] > 0).astype(int))
            return m

        def fit_trunk(x, ytr_idx):
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1, max_iter=1200, class_weight="balanced", random_state=42
                ),
            )
            m.fit(x[ytr_idx], (sy[ytr_idx] == 2).astype(int))
            return m

        # inner OOF for meta features on train portion (to train meta head without leak)
        inner_folds = list(
            StratifiedGroupKFold(5, shuffle=True, random_state=fold_id + 7).split(
                np.arange(len(tr)), sy[tr], ss[tr]
            )
        )
        sio = np.zeros((len(tr), 3), np.float32)
        for itr, iva in inner_folds:
            tr_i = tr[itr]
            va_i = tr[iva]
            m = fit_4(sx_c, tr_i)
            sio[iva] = suite.normalize(m.predict_proba(sx_c[va_i])[:, 1:])
        z_tr = feat_meta(sa_mat[tr], sio, seg_pc[tr])
        # meta head OOF on train via same inner folds
        meta_oof_tr = np.zeros((len(tr), 4), np.float32)
        for itr, iva in inner_folds:
            cw = {0: 1.0, 1: 1.0, 2: META_TW, 3: META_GW}
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=META_C,
                    max_iter=4000,
                    class_weight=cw,
                    multi_class="multinomial",
                    random_state=42,
                ),
            )
            m.fit(z_tr[itr], sy[tr[itr]])
            meta_oof_tr[iva] = suite.normalize(m.predict_proba(z_tr[iva]))

        # apply meta to val: fit image on full tr, fit meta on z_tr
        m_img4 = fit_4(sx_c, tr)
        img_va = suite.normalize(m_img4.predict_proba(sx_c[va])[:, 1:])
        z_va = feat_meta(sa_mat[va], img_va, seg_pc[va])
        cw = {0: 1.0, 1: 1.0, 2: META_TW, 3: META_GW}
        meta = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=META_C,
                max_iter=4000,
                class_weight=cw,
                multi_class="multinomial",
                random_state=42,
            ),
        )
        meta.fit(z_tr, sy[tr])
        oof_meta[va] = suite.normalize(meta.predict_proba(z_va))

        # --- hierarchical base ---
        m_bin = fit_bin(sx_c, tr)
        m_mat_cat = fit_mat(sx_cat, tr)
        m_tr = fit_trunk(sx_c, tr)
        contact = HIER["alpha"] * seg_pc[va] + (1 - HIER["alpha"]) * m_bin.predict_proba(
            sx_c[va]
        )[:, 1]
        w = np.array([HIER["leaf_w"], HIER["trunk_w"], HIER["twig_w"]])
        imat = m_mat_cat.predict_proba(sx_cat[va])
        mat = suite.normalize(sa_mat[va] * (1 - w) + imat * w)
        tb = HIER["tb"]
        if tb:
            logits = np.log(np.clip(mat, 1e-8, 1))
            logits[:, 1] += tb * m_tr.predict_proba(sx_c[va])[:, 1]
            mat = suite.normalize(np.exp(logits))
        th = HIER["th"]
        p_h = np.zeros((len(va), 4), np.float64)
        for i in range(len(va)):
            if contact[i] >= th:
                cmass = max(float(contact[i]), 0.55)
                p_h[i, 0] = 1 - cmass
                p_h[i, 1:] = cmass * mat[i]
            else:
                p_h[i, 0] = 1.0
        oof_hier[va] = suite.normalize(p_h)

        # simple log-blend of meta & hier as third base
        for bw in (0.4, 0.5, 0.6):
            pass
        oof_blend[va] = suite.normalize(0.45 * oof_meta[va] + 0.55 * oof_hier[va])
        print(f"fold {fold_id+1}/5 done", flush=True)

    # Stack features
    def stack_x(pm, ph, pb, a4, pc_, trunk_proxy):
        return np.hstack(
            [
                np.log(np.clip(pm, 1e-6, 1)),
                np.log(np.clip(ph, 1e-6, 1)),
                pm,
                ph,
                pb,
                a4,
                pc_[:, None],
                (pm.argmax(1) == ph.argmax(1)).astype(np.float32)[:, None],
                pm.max(1, keepdims=True),
                ph.max(1, keepdims=True),
                trunk_proxy[:, None],
            ]
        ).astype(np.float32)

    trunk_proxy = oof_hier[:, 2]  # soft trunk mass from hier
    Z = stack_x(oof_meta, oof_hier, oof_blend, sa4, seg_pc, trunk_proxy)
    np.save(OUT / "hand_stack_oof_features.npy", Z)
    np.save(OUT / "hand_stack_oof_y.npy", sy)
    np.save(OUT / "hand_oof_meta.npy", oof_meta)
    np.save(OUT / "hand_oof_hier.npy", oof_hier)

    rows = []
    for C in (0.03, 0.1, 0.3, 1.0):
        for tw in (1.0, 1.5, 2.0, 2.5):
            for gw in (1.0, 1.2, 1.5, 2.0):
                cw = {0: 1.0, 1: 1.0, 2: tw, 3: gw}
                fs = []
                for tr, va in folds:
                    m = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=C,
                            max_iter=4000,
                            class_weight=cw,
                            multi_class="multinomial",
                            random_state=42,
                        ),
                    )
                    m.fit(Z[tr], sy[tr])
                    fs.append(
                        f1_score(
                            sy[va], m.predict(Z[va]), average="macro", zero_division=0
                        )
                    )
                rows.append(
                    {
                        "C": C,
                        "trunk_class_weight": tw,
                        "twig_class_weight": gw,
                        "mean_cv_macro_f1": float(np.mean(fs)),
                        "worst_cv_macro_f1": float(np.min(fs)),
                    }
                )
    # also pure blend weights without stack LR
    for bw in np.linspace(0, 1, 21):
        pred = suite.normalize((1 - bw) * oof_meta + bw * oof_hier).argmax(1)
        rows.append(
            {
                "C": -1,
                "trunk_class_weight": -1,
                "twig_class_weight": -1,
                "blend_hier_weight": float(bw),
                "mean_cv_macro_f1": float(
                    f1_score(sy, pred, average="macro", zero_division=0)
                ),
                "worst_cv_macro_f1": float(
                    f1_score(sy, pred, average="macro", zero_division=0)
                ),
            }
        )

    # Prefer worst-fold (domain-robust) then mean — hand→robot shift is severe.
    board = (
        pd.DataFrame(rows)
        .sort_values(["worst_cv_macro_f1", "mean_cv_macro_f1"], ascending=False)
        .reset_index(drop=True)
    )
    board.to_csv(OUT / "hand_stack_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    # base model F1s
    base_scores = {
        "meta_oof_macro_f1": float(
            f1_score(sy, oof_meta.argmax(1), average="macro", zero_division=0)
        ),
        "hier_oof_macro_f1": float(
            f1_score(sy, oof_hier.argmax(1), average="macro", zero_division=0)
        ),
        "blend_oof_macro_f1": float(
            f1_score(sy, oof_blend.argmax(1), average="macro", zero_division=0)
        ),
    }
    lock = {
        "protocol": "segment_stack_meta_weighted_plus_hier_contact_rescue_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "base_models": {
            "meta_weighted": {
                "C": META_C,
                "trunk_class_weight": META_TW,
                "twig_class_weight": META_GW,
            },
            "hier_contact_rescue": HIER,
        },
        "base_oof_scores": base_scores,
        "test_loaded": False,
        "selected_candidate": best,
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
