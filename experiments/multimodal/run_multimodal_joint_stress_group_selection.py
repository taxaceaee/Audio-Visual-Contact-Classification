"""Phase 2/3: multi-view joint audio+image model (hand only).

Train segment-level multinomial heads on multi-view audio features
(clean / robot_mix / bandlimit) concatenated with frozen image embeddings.
Select fusion with v2 meta/hier soft stack by worst-view macro F1.

Never loads robot/test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/multimodal_joint_stress_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
AUDIO_CLEAN = Path(
    "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"
)
AUDIO_VIEWS = {
    "clean": AUDIO_CLEAN,
    "robot_mix": Path(
        "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/robot_mix/X.npy"
    ),
    "bandlimit": Path(
        "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/bandlimit/X.npy"
    ),
}
IMG = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k/hand_train_full/X.npy"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m/hand_train_full/X.npy"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k/hand_train_full/X.npy"),
}
W2V = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")
HIGHSR = {
    "clean": Path(
        "outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/features/hand_train_full/clean/X.npy"
    ),
    "robot_mix": Path(
        "outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/features/hand_train_full/robot_mix/X.npy"
    ),
    "bandlimit": Path(
        "outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/features/hand_train_full/bandlimit/X.npy"
    ),
}


def fast_macro(y, p):
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def pool(x, codes, n):
    return np.vstack([x[codes == i].mean(0) for i in range(n)]).astype(np.float32)


def v2_soft(pm, ph):
    """Soft version of hier_trunk_meta_else."""
    out = pm.copy()
    mask = ph.argmax(1) == 2
    out[mask] = ph[mask]
    return suite.normalize(out)


def fit_logreg(x, y, c=0.1, trunk_w=1.5):
    cw = {0: 1.0, 1: 1.1, 2: trunk_w, 3: 1.2}
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=3000,
            class_weight=cw,
            multi_class="multinomial",
            random_state=42,
        ),
    )
    m.fit(x, y)
    return m


def fit_hgb(x, y, seed=0):
    # balanced via sample weights
    counts = np.bincount(y, minlength=4).astype(np.float64)
    w_c = counts.sum() / np.maximum(counts, 1.0)
    w_c = w_c / w_c.mean()
    w_c[2] *= 1.4
    sw = w_c[y]
    m = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=1.0,
        random_state=42 + seed,
    )
    m.fit(x, y, sample_weight=sw)
    return m


def build_feat(audio_seg, img_parts, w2v_seg, highsr_seg=None, use_highsr=False):
    parts = [audio_seg]
    for name in sorted(img_parts):
        parts.append(img_parts[name])
    parts.append(w2v_seg)
    if use_highsr and highsr_seg is not None:
        # highsr is high-dim; use first 256 PCA-less projection via fixed random
        # better: just L2-normalize and take mean pools already; subsample dims
        rng = np.random.default_rng(0)
        idx = rng.choice(highsr_seg.shape[1], size=min(256, highsr_seg.shape[1]), replace=False)
        parts.append(highsr_seg[:, np.sort(idx)])
    return np.hstack(parts).astype(np.float32)


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
    nseg = len(useg)
    sy = np.array([y[codes == i][0] for i in range(nseg)], np.int64)
    ss = np.array([sg[codes == i][0] for i in range(nseg)])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(nseg), sy, ss)
    )

    # stack OOF for v2 soft
    pm = np.load(STACK / "hand_oof_meta.npy").astype(np.float64)
    ph = np.load(STACK / "hand_oof_hier.npy").astype(np.float64)
    assert len(pm) == nseg
    p_v2 = v2_soft(pm, ph)
    pred_v2 = np.where(ph.argmax(1) == 2, ph.argmax(1), pm.argmax(1))

    img_win = {n: np.load(p).astype(np.float32) for n, p in IMG.items()}
    img_seg = {n: pool(x, codes, nseg) for n, x in img_win.items()}
    w2v_seg = pool(np.load(W2V).astype(np.float32), codes, nseg)
    audio_seg = {}
    highsr_seg = {}
    for view, path in AUDIO_VIEWS.items():
        audio_seg[view] = pool(np.load(path).astype(np.float32), codes, nseg)
    for view, path in HIGHSR.items():
        highsr_seg[view] = pool(np.load(path).astype(np.float32), codes, nseg)

    feature_recipes = {
        "a240_clip_eff": lambda av: build_feat(
            audio_seg[av], {"clip": img_seg["clip"], "eff": img_seg["eff"]}, w2v_seg, None, False
        ),
        "a240_multibb_w2v": lambda av: build_feat(audio_seg[av], img_seg, w2v_seg, None, False),
        "a240_hsr_clip": lambda av: build_feat(
            audio_seg[av],
            {"clip": img_seg["clip"]},
            w2v_seg,
            highsr_seg[av],
            True,
        ),
    }

    modelspecs = [
        ("logreg_C0.03_tw1.5", lambda x, y, s: fit_logreg(x, y, 0.03, 1.5)),
        ("logreg_C0.1_tw2.0", lambda x, y, s: fit_logreg(x, y, 0.1, 2.0)),
        ("logreg_C0.3_tw1.8", lambda x, y, s: fit_logreg(x, y, 0.3, 1.8)),
        ("hgb", lambda x, y, s: fit_hgb(x, y, s)),
    ]

    rows = []
    # store best joint oof for later fusion grid
    best_joint = None
    best_joint_score = -1.0
    best_joint_meta = None

    for feat_name, feat_fn in feature_recipes.items():
        # precompute view features
        Xv = {v: feat_fn(v) for v in ("clean", "robot_mix", "bandlimit")}
        for mname, mfit in modelspecs:
            oof = {v: np.zeros((nseg, 4), np.float64) for v in Xv}
            for fold_id, (tr, va) in enumerate(folds):
                # multi-view bagging on train
                Xtr = np.vstack([Xv[v][tr] for v in Xv])
                ytr = np.concatenate([sy[tr] for _ in Xv])
                model = mfit(Xtr, ytr, fold_id)
                for v in Xv:
                    proba = model.predict_proba(Xv[v][va])
                    # align columns to 0..3
                    full = np.zeros((len(va), 4), np.float64)
                    classes = list(model.classes_)
                    for j, c in enumerate(classes):
                        full[:, int(c)] = proba[:, j]
                    oof[v][va] = suite.normalize(full)
            # view scores
            view_f1 = {v: fast_macro(sy, oof[v].argmax(1)) for v in oof}
            fold_mins = []
            for _, va in folds:
                vs = [fast_macro(sy[va], oof[v][va].argmax(1)) for v in oof]
                fold_mins.append(min(vs))
            score = float(np.mean(fold_mins))
            worst = float(np.min(fold_mins))
            row = {
                "feature": feat_name,
                "model": mname,
                "score_mean_min_view": score,
                "score_worst_min_view": worst,
                "clean_macro_f1": view_f1["clean"],
                "robot_mix_macro_f1": view_f1["robot_mix"],
                "bandlimit_macro_f1": view_f1["bandlimit"],
                "trunk_recall_clean": float(
                    np.sum((sy == 2) & (oof["clean"].argmax(1) == 2)) / max(np.sum(sy == 2), 1)
                ),
            }
            rows.append(row)
            print(
                f"{feat_name:18s} {mname:18s} score={score:.4f} worst={worst:.4f} "
                f"clean={view_f1['clean']:.4f} mix={view_f1['robot_mix']:.4f} "
                f"band={view_f1['bandlimit']:.4f}",
                flush=True,
            )
            if score > best_joint_score:
                best_joint_score = score
                best_joint = {v: oof[v].copy() for v in oof}
                best_joint_meta = row

    board_j = pd.DataFrame(rows).sort_values(
        ["score_mean_min_view", "score_worst_min_view", "clean_macro_f1"], ascending=False
    ).reset_index(drop=True)
    board_j.to_csv(OUT / "hand_joint_stress_leaderboard.csv", index=False)
    assert best_joint is not None

    # save joint OOF
    for v, p in best_joint.items():
        np.save(OUT / f"hand_oof_joint_{v}.npy", p)
    np.save(OUT / "hand_oof_y.npy", sy)
    np.save(OUT / "hand_specimen.npy", ss.astype(str))

    # fusion with v2 soft: select weight + optional trunk force rule
    fusion_rows = []
    pj = best_joint["clean"]
    # also evaluate fusion under stress by replacing joint view
    for wj in np.linspace(0, 1, 21):
        for mode in ("soft", "hier_trunk_joint_else", "joint_trunk_v2_else", "max_conf"):
            fold_mins = []
            oof_pred = np.zeros(nseg, np.int64)
            for fold_id, (tr, va) in enumerate(folds):
                view_scores = []
                preds_by_view = {}
                for v in ("clean", "robot_mix", "bandlimit"):
                    p_j = best_joint[v]
                    if mode == "soft":
                        p = suite.normalize((1 - wj) * p_v2 + wj * p_j)
                        pred = p.argmax(1)
                    elif mode == "hier_trunk_joint_else":
                        pred_h = ph.argmax(1)
                        pred = np.where(pred_h == 2, 2, p_j.argmax(1))
                    elif mode == "joint_trunk_v2_else":
                        pred_j = p_j.argmax(1)
                        pred_v = pred_v2
                        pred = pred_v.copy()
                        pred[pred_j == 2] = 2
                    elif mode == "max_conf":
                        conf_v = p_v2.max(1)
                        conf_j = p_j.max(1)
                        pred = np.where(conf_j >= conf_v, p_j.argmax(1), pred_v2)
                    else:
                        raise ValueError(mode)
                    preds_by_view[v] = pred
                    view_scores.append(fast_macro(sy[va], pred[va]))
                fold_mins.append(min(view_scores))
                oof_pred[va] = preds_by_view["clean"][va]
            disagree = float(np.mean(oof_pred != pred_v2))
            fusion_rows.append(
                {
                    "mode": mode,
                    "joint_weight": float(wj),
                    "score_mean_min_view": float(np.mean(fold_mins)),
                    "score_worst_min_view": float(np.min(fold_mins)),
                    "clean_macro_f1": fast_macro(sy, oof_pred),
                    "trunk_recall": float(
                        np.sum((sy == 2) & (oof_pred == 2)) / max(np.sum(sy == 2), 1)
                    ),
                    "disagreement_vs_v2": disagree,
                    "joint_base": best_joint_meta,
                }
            )
            # soft modes depend on wj; rule modes don't — break after first wj for pure rules
            if mode != "soft":
                break

    # fix: above breaks wrong — recompute pure rules once
    fusion_rows = [r for r in fusion_rows if r["mode"] == "soft"]
    for mode in ("hier_trunk_joint_else", "joint_trunk_v2_else", "max_conf", "v2_only", "joint_only"):
        fold_mins = []
        oof_pred = np.zeros(nseg, np.int64)
        for fold_id, (tr, va) in enumerate(folds):
            view_scores = []
            for v in ("clean", "robot_mix", "bandlimit"):
                p_j = best_joint[v]
                if mode == "hier_trunk_joint_else":
                    pred = np.where(ph.argmax(1) == 2, 2, p_j.argmax(1))
                elif mode == "joint_trunk_v2_else":
                    pred = pred_v2.copy()
                    pred[p_j.argmax(1) == 2] = 2
                elif mode == "max_conf":
                    pred = np.where(p_j.max(1) >= p_v2.max(1), p_j.argmax(1), pred_v2)
                elif mode == "v2_only":
                    pred = pred_v2
                elif mode == "joint_only":
                    pred = p_j.argmax(1)
                view_scores.append(fast_macro(sy[va], pred[va]))
            fold_mins.append(min(view_scores))
            # clean pred store
            p_j = best_joint["clean"]
            if mode == "hier_trunk_joint_else":
                oof_pred[va] = np.where(ph[va].argmax(1) == 2, 2, p_j[va].argmax(1))
            elif mode == "joint_trunk_v2_else":
                tmp = pred_v2[va].copy()
                tmp[p_j[va].argmax(1) == 2] = 2
                oof_pred[va] = tmp
            elif mode == "max_conf":
                oof_pred[va] = np.where(
                    p_j[va].max(1) >= p_v2[va].max(1), p_j[va].argmax(1), pred_v2[va]
                )
            elif mode == "v2_only":
                oof_pred[va] = pred_v2[va]
            else:
                oof_pred[va] = p_j[va].argmax(1)
        fusion_rows.append(
            {
                "mode": mode,
                "joint_weight": -1.0,
                "score_mean_min_view": float(np.mean(fold_mins)),
                "score_worst_min_view": float(np.min(fold_mins)),
                "clean_macro_f1": fast_macro(sy, oof_pred),
                "trunk_recall": float(
                    np.sum((sy == 2) & (oof_pred == 2)) / max(np.sum(sy == 2), 1)
                ),
                "disagreement_vs_v2": float(np.mean(oof_pred != pred_v2)),
                "joint_base": best_joint_meta,
            }
        )

    # recompute soft rows properly with fold-min
    soft_rows = []
    for wj in np.linspace(0, 1, 21):
        fold_mins = []
        oof_pred = np.zeros(nseg, np.int64)
        for _, va in folds:
            vs = []
            for v in ("clean", "robot_mix", "bandlimit"):
                p = suite.normalize((1 - wj) * p_v2 + wj * best_joint[v])
                pred = p.argmax(1)
                vs.append(fast_macro(sy[va], pred[va]))
            fold_mins.append(min(vs))
            oof_pred[va] = suite.normalize((1 - wj) * p_v2[va] + wj * best_joint["clean"][va]).argmax(1)
        soft_rows.append(
            {
                "mode": "soft",
                "joint_weight": float(wj),
                "score_mean_min_view": float(np.mean(fold_mins)),
                "score_worst_min_view": float(np.min(fold_mins)),
                "clean_macro_f1": fast_macro(sy, oof_pred),
                "trunk_recall": float(
                    np.sum((sy == 2) & (oof_pred == 2)) / max(np.sum(sy == 2), 1)
                ),
                "disagreement_vs_v2": float(np.mean(oof_pred != pred_v2)),
                "joint_base": best_joint_meta,
            }
        )

    board_f = pd.DataFrame(soft_rows + [r for r in fusion_rows if r["mode"] != "soft"]).sort_values(
        ["score_mean_min_view", "score_worst_min_view", "trunk_recall"], ascending=False
    ).reset_index(drop=True)
    board_f.to_csv(OUT / "hand_fusion_leaderboard.csv", index=False)
    best = board_f.iloc[0].to_dict()
    # strip nested non-json-friendly if needed
    if isinstance(best.get("joint_base"), dict):
        pass

    # v2 baseline stress score for gate
    v2_fold_mins = []
    for _, va in folds:
        # v2 has no multi-view OOF; stress proxy via temperature on proba
        vs = [fast_macro(sy[va], pred_v2[va])]
        for seed, gamma in ((1, 0.75), (2, 0.6)):
            rng = np.random.default_rng(seed)
            logp = gamma * np.log(np.clip(p_v2[va], 1e-8, 1))
            logp = logp + 0.12 * rng.standard_normal(logp.shape)
            pred_s = suite.normalize(np.exp(logp)).argmax(1)
            vs.append(fast_macro(sy[va], pred_s))
        v2_fold_mins.append(min(vs))
    v2_gate = {
        "score_mean_min_proxy": float(np.mean(v2_fold_mins)),
        "score_worst_min_proxy": float(np.min(v2_fold_mins)),
        "clean_macro_f1": fast_macro(sy, pred_v2),
    }

    lock = {
        "protocol": "multimodal_joint_multiview_stress_fusion_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "test_loaded": False,
        "selected_joint_model": best_joint_meta,
        "selected_fusion": {
            k: best[k]
            for k in best
            if k != "joint_base"
        },
        "selected_joint_feature": best_joint_meta["feature"],
        "selected_joint_model_name": best_joint_meta["model"],
        "v2_baseline_hand_gate": v2_gate,
        "beats_v2_hand_gate": bool(
            best["score_mean_min_view"] >= v2_gate["score_mean_min_proxy"] - 1e-9
        ),
        "invariants": {
            "robot_not_used": True,
            "filename_class_features": False,
            "multiview_audio_train": True,
            "image_frozen_embeddings": True,
        },
    }
    # attach joint base cleanly
    lock["selected_fusion"]["joint_base"] = best_joint_meta
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print("\nJOINT TOP:")
    print(board_j.head(10).to_string(index=False))
    print("\nFUSION TOP:")
    print(board_f.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
