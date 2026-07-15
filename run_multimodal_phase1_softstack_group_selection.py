"""Phase 0+1 hand-only multimodal selection (no robot/test). Fast path.

Reuses locked stack OOF for meta/hier; builds single-level group OOF for:
  - multi-backbone material (CLIP/Eff/DINO/ConvNeXt)
  - wav2vec2 4-class head
  - joint MLP on rich probability features
Then selects soft-blend / conf-gate / rule variants by nested-style
worst-fold + stress-proxy score. Diversity gate vs v2 rule baseline.

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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        "outputs/audio_feature_benchmarks/multimodal_phase1_softstack_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")
IMG_ROOTS = {
    "clip": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "eff": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
    "dino": Path("outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m"),
    "convnext": Path("outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k"),
}
W2V = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")
DISAGREE_MIN = float(os.environ.get("PHASE1_DISAGREE_MIN", "0.05"))
HIER = dict(alpha=0.4, leaf_w=0.4, trunk_w=0.55, twig_w=0.4, th=0.55, tb=0.3)
META = dict(C=0.1, trunk_class_weight=1.2, twig_class_weight=1.2)


def fast_macro(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def pool_emb(x: np.ndarray, codes: np.ndarray, nseg: int) -> np.ndarray:
    return np.vstack([x[codes == i].mean(0) for i in range(nseg)]).astype(np.float32)


def stress_proxy_proba(p: np.ndarray, seed: int, gamma: float = 0.75, noise: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logp = gamma * np.log(np.clip(p, 1e-8, 1.0))
    logp = logp + noise * rng.standard_normal(logp.shape)
    return suite.normalize(np.exp(logp))


def apply_log_bias(p: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return suite.normalize(np.exp(np.log(np.clip(p, 1e-8, 1.0)) + bias[None, :]))


def soft_blend(pm, ph, pj, wm, wh, wj):
    wsum = max(wm + wh + wj, 1e-8)
    return suite.normalize((wm * pm + wh * ph + wj * pj) / wsum)


def conf_gate_blend(pm, ph, pj, beta: float):
    w = np.exp(beta * np.stack([pm.max(1), ph.max(1), pj.max(1)], axis=1))
    w = w / w.sum(1, keepdims=True)
    return suite.normalize(w[:, 0:1] * pm + w[:, 1:2] * ph + w[:, 2:3] * pj)


def search_log_bias(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    grid = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    best_b = np.zeros(4, np.float64)
    best = -1.0
    for bl in grid:
        for bt in grid:
            for bg in grid:
                b = np.array([0.0, bl, bt, bg], np.float64)
                s = fast_macro(y, apply_log_bias(p, b).argmax(1))
                if s > best:
                    best, best_b = s, b
    return best_b


def fit_lr4(x, y, c=0.03):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
        ),
    )
    m.fit(x, y)
    return m


def fit_lr_mat(x, y3, c=0.03):
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
        ),
    )
    m.fit(x, y3)
    return m


def fit_mlp(x, y, hidden=(128,), alpha=2e-3, seed=0):
    rng = np.random.default_rng(42 + seed)
    counts = np.bincount(y, minlength=4).astype(np.float64)
    target = int(max(counts.max(), 1))
    targets = {0: target, 1: target, 2: int(target * 1.35), 3: target}
    idxs = []
    for c in range(4):
        ci = np.where(y == c)[0]
        if len(ci) == 0:
            continue
        need = targets[c]
        take = ci[rng.choice(len(ci), size=need, replace=len(ci) < need)]
        idxs.append(take)
    idxs = np.concatenate(idxs)
    rng.shuffle(idxs)
    m = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=350,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=42 + seed,
        ),
    )
    m.fit(x[idxs], y[idxs])
    return m


def joint_features(sa4, sa_mat, seg_pc, p_clip4, p_mats, p_w2v, p_meta, p_hier):
    parts = [
        np.log(np.clip(sa4, 1e-6, 1)),
        sa4,
        sa_mat,
        seg_pc[:, None],
        np.log(np.clip(p_clip4, 1e-6, 1)),
        p_clip4,
        np.log(np.clip(p_w2v, 1e-6, 1)),
        p_w2v,
        np.log(np.clip(p_meta, 1e-6, 1)),
        p_meta,
        np.log(np.clip(p_hier, 1e-6, 1)),
        p_hier,
    ]
    for name in sorted(p_mats):
        pm = p_mats[name]
        parts.append(np.log(np.clip(pm, 1e-6, 1)))
        parts.append(pm)
    ent = lambda p: -(p * np.log(np.clip(p, 1e-8, 1))).sum(1, keepdims=True)
    parts += [ent(sa4), ent(p_clip4), ent(p_meta)]
    parts.append((sa4.argmax(1) == p_clip4.argmax(1)).astype(np.float32)[:, None])
    parts.append((p_meta.argmax(1) == p_hier.argmax(1)).astype(np.float32)[:, None])
    return np.hstack(parts).astype(np.float32)


def decode_family(pm, ph, pj, mode: str, params: dict) -> np.ndarray:
    if mode == "v2_rule":
        pred_h = ph.argmax(1)
        return np.where(pred_h == 2, pred_h, pm.argmax(1))
    if mode == "soft_fixed":
        p = soft_blend(pm, ph, pj, params["wm"], params["wh"], params["wj"])
        if "bias" in params:
            p = apply_log_bias(p, params["bias"])
        return p.argmax(1)
    if mode == "conf_gate":
        p = conf_gate_blend(pm, ph, pj, params["beta"])
        if "bias" in params:
            p = apply_log_bias(p, params["bias"])
        return p.argmax(1)
    if mode == "joint_only":
        p = pj if "bias" not in params else apply_log_bias(pj, params["bias"])
        return p.argmax(1)
    if mode == "hier_trunk_joint_else":
        pred_h = ph.argmax(1)
        pred_j = (apply_log_bias(pj, params["bias"]).argmax(1) if "bias" in params else pj.argmax(1))
        return np.where(pred_h == 2, pred_h, pred_j)
    if mode == "hier_trunk_soft_else":
        p = soft_blend(pm, ph, pj, params.get("wm", 0.4), params.get("wh", 0.2), params.get("wj", 0.4))
        if "bias" in params:
            p = apply_log_bias(p, params["bias"])
        pred = p.argmax(1)
        pred_h = ph.argmax(1)
        return np.where(pred_h == 2, pred_h, pred)
    if mode == "max_conf3":
        conf = np.stack([pm.max(1), ph.max(1), pj.max(1)], axis=1)
        arg = conf.argmax(1)
        pred = pm.argmax(1)
        pred[arg == 1] = ph.argmax(1)[arg == 1]
        pred[arg == 2] = pj.argmax(1)[arg == 2]
        return pred
    raise ValueError(mode)


def blend_proba(pm, ph, pj, mode, params):
    if mode == "soft_fixed":
        p = soft_blend(pm, ph, pj, params["wm"], params["wh"], params["wj"])
    elif mode == "conf_gate":
        p = conf_gate_blend(pm, ph, pj, params["beta"])
    elif mode == "joint_only":
        p = pj.copy()
    elif mode == "hier_trunk_soft_else":
        p = soft_blend(pm, ph, pj, params.get("wm", 0.4), params.get("wh", 0.2), params.get("wj", 0.4))
    else:
        p = soft_blend(pm, ph, pj, 0.4, 0.3, 0.3)
    if "bias" in params:
        p = apply_log_bias(p, params["bias"])
    return p


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
    sk = sel.seg(files).to_numpy()
    sg = sel.spec(files).to_numpy()
    useg, codes = np.unique(sk, return_inverse=True)
    nseg = len(useg)
    sy = np.array([y[codes == i][0] for i in range(nseg)], dtype=np.int64)
    ss = np.array([sg[codes == i][0] for i in range(nseg)])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(nseg), sy, ss)
    )

    # Reuse stack meta/hier OOF (aligned by construction)
    oof_meta = np.load(STACK / "hand_oof_meta.npy").astype(np.float32)
    oof_hier = np.load(STACK / "hand_oof_hier.npy").astype(np.float32)
    sy_chk = np.load(STACK / "hand_stack_oof_y.npy")
    assert np.array_equal(sy, sy_chk), "segment label alignment failed"
    assert len(oof_meta) == nseg

    # Audio segment aggregates (locked OOF)
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
    seg_pc = np.array([pc[codes == i].mean() for i in range(nseg)], dtype=np.float64)
    sa_mat = np.vstack(
        [suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0] for i in range(nseg)]
    )
    sa4 = np.vstack([suite.normalize(audio[codes == i].mean(0, keepdims=True))[0] for i in range(nseg)])

    emb = {
        name: pool_emb(np.load(root / "hand_train_full/X.npy").astype(np.float32), codes, nseg)
        for name, root in IMG_ROOTS.items()
    }
    x_w2v = pool_emb(np.load(W2V).astype(np.float32), codes, nseg)
    sx_c = emb["clip"]
    sx_multi = np.hstack([emb[k] for k in ("clip", "eff", "dino", "convnext")])

    print("building single-level OOF: clip4 / multi-mat / w2v / joint...", flush=True)
    oof_clip4 = np.zeros((nseg, 4), np.float32)
    oof_w2v = np.zeros((nseg, 4), np.float32)
    oof_mats = {k: np.zeros((nseg, 3), np.float32) for k in list(emb) + ["multi"]}
    oof_joint = np.zeros((nseg, 4), np.float32)

    # First pass: base heads OOF
    for fold_id, (tr, va) in enumerate(folds):
        m4 = fit_lr4(sx_c[tr], sy[tr], 0.03)
        mw = fit_lr4(x_w2v[tr], sy[tr], 0.1)
        oof_clip4[va] = suite.normalize(m4.predict_proba(sx_c[va])).astype(np.float32)
        oof_w2v[va] = suite.normalize(mw.predict_proba(x_w2v[va])).astype(np.float32)
        ct = tr[sy[tr] > 0]
        if len(ct):
            for name, xe in emb.items():
                mh = fit_lr_mat(xe[ct], sy[ct] - 1, 0.03)
                oof_mats[name][va] = suite.normalize(mh.predict_proba(xe[va])).astype(np.float32)
            mm = fit_lr_mat(sx_multi[ct], sy[ct] - 1, 0.03)
            oof_mats["multi"][va] = suite.normalize(mm.predict_proba(sx_multi[va])).astype(
                np.float32
            )
        print(f"  base fold {fold_id+1}/5", flush=True)

    # Second pass: joint MLP OOF using OOF features (no leakage — features are OOF)
    # For train fold features we need features only from other folds: use current OOF
    # which for train rows are OOF w.r.t. their own fold (standard stacking).
    Z = joint_features(sa4, sa_mat, seg_pc, oof_clip4, oof_mats, oof_w2v, oof_meta, oof_hier)
    for fold_id, (tr, va) in enumerate(folds):
        mlp = fit_mlp(Z[tr], sy[tr], hidden=(160,), alpha=2e-3, seed=fold_id)
        oof_joint[va] = suite.normalize(mlp.predict_proba(Z[va])).astype(np.float32)
        print(f"  joint fold {fold_id+1}/5", flush=True)

    np.save(OUT / "hand_oof_meta.npy", oof_meta)
    np.save(OUT / "hand_oof_hier.npy", oof_hier)
    np.save(OUT / "hand_oof_joint.npy", oof_joint)
    np.save(OUT / "hand_oof_w2v.npy", oof_w2v)
    np.save(OUT / "hand_oof_clip4.npy", oof_clip4)
    np.save(OUT / "hand_oof_y.npy", sy)
    np.save(OUT / "hand_oof_joint_features.npy", Z)
    np.save(OUT / "hand_segment_ids.npy", useg.astype(str))
    np.save(OUT / "hand_specimen.npy", ss.astype(str))

    v2_pred = decode_family(oof_meta, oof_hier, oof_joint, "v2_rule", {})
    v2_fold = [fast_macro(sy[va], v2_pred[va]) for _, va in folds]
    v2_stats = {
        "mean_fold_macro_f1": float(np.mean(v2_fold)),
        "worst_fold_macro_f1": float(np.min(v2_fold)),
        "global_macro_f1": fast_macro(sy, v2_pred),
    }

    candidates = []
    for wm, wh, wj in [
        (0.55, 0.45, 0.0),
        (0.5, 0.3, 0.2),
        (0.4, 0.3, 0.3),
        (0.35, 0.25, 0.4),
        (0.3, 0.2, 0.5),
        (0.25, 0.15, 0.6),
        (0.2, 0.2, 0.6),
        (0.33, 0.33, 0.34),
        (0.15, 0.15, 0.7),
        (0.1, 0.1, 0.8),
    ]:
        candidates.append(
            {"mode": "soft_fixed", "params": {"wm": wm, "wh": wh, "wj": wj}, "name": f"soft_{wm}_{wh}_{wj}"}
        )
    for beta in (1.0, 2.0, 3.5, 5.0, 7.0):
        candidates.append({"mode": "conf_gate", "params": {"beta": beta}, "name": f"conf_b{beta}"})
    candidates += [
        {"mode": "joint_only", "params": {}, "name": "joint_only"},
        {"mode": "hier_trunk_joint_else", "params": {}, "name": "hier_trunk_joint_else"},
        {
            "mode": "hier_trunk_soft_else",
            "params": {"wm": 0.35, "wh": 0.2, "wj": 0.45},
            "name": "hier_trunk_soft_else",
        },
        {"mode": "max_conf3", "params": {}, "name": "max_conf3"},
        {"mode": "v2_rule", "params": {}, "name": "v2_rule_ref"},
    ]

    rows = []
    for cand in candidates:
        for use_bias in (False, True):
            fold_clean, fold_stress = [], []
            oof_pred = np.zeros(nseg, np.int64)
            for fold_id, (tr, va) in enumerate(folds):
                params = dict(cand["params"])
                if use_bias:
                    p_tr = blend_proba(oof_meta[tr], oof_hier[tr], oof_joint[tr], cand["mode"], params)
                    params["bias"] = search_log_bias(p_tr, sy[tr])
                pred_va = decode_family(
                    oof_meta[va], oof_hier[va], oof_joint[va], cand["mode"], params
                )
                oof_pred[va] = pred_va
                fold_clean.append(fast_macro(sy[va], pred_va))
                pred_s = decode_family(
                    stress_proxy_proba(oof_meta[va], 1000 + fold_id),
                    stress_proxy_proba(oof_hier[va], 2000 + fold_id),
                    stress_proxy_proba(oof_joint[va], 3000 + fold_id),
                    cand["mode"],
                    params,
                )
                fold_stress.append(fast_macro(sy[va], pred_s))
            fold_min = [min(c, s) for c, s in zip(fold_clean, fold_stress)]
            disagree = float(np.mean(oof_pred != v2_pred))
            rows.append(
                {
                    "name": cand["name"],
                    "mode": cand["mode"],
                    "use_bias": use_bias,
                    "params": json.dumps(cand["params"]),
                    "score_mean_min_clean_stress": float(np.mean(fold_min)),
                    "score_worst_min_clean_stress": float(np.min(fold_min)),
                    "mean_clean_macro_f1": float(np.mean(fold_clean)),
                    "worst_clean_macro_f1": float(np.min(fold_clean)),
                    "mean_stress_macro_f1": float(np.mean(fold_stress)),
                    "worst_stress_macro_f1": float(np.min(fold_stress)),
                    "global_macro_f1": fast_macro(sy, oof_pred),
                    "trunk_recall": float(
                        np.sum((sy == 2) & (oof_pred == 2)) / max(np.sum(sy == 2), 1)
                    ),
                    "disagreement_vs_v2": disagree,
                    "passes_diversity": bool(
                        disagree >= DISAGREE_MIN or cand["mode"] == "v2_rule"
                    ),
                }
            )

    board = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "score_mean_min_clean_stress",
                "score_worst_min_clean_stress",
                "worst_clean_macro_f1",
                "trunk_recall",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    board.to_csv(OUT / "hand_phase1_leaderboard.csv", index=False)

    diverse = board[(board["passes_diversity"]) & (board["mode"] != "v2_rule")]
    if len(diverse) == 0:
        selected_row = board[board["mode"] != "v2_rule"].iloc[0]
        diversity_note = "no candidate met disagreement floor; took best non-v2"
    else:
        selected_row = diverse.iloc[0]
        diversity_note = "ok"
    selected = selected_row.to_dict()

    lock = {
        "protocol": "multimodal_phase1_softstack_nested_group_OOF_fast",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "test_loaded": False,
        "hier_locked": HIER,
        "meta_locked": META,
        "stack_oof_source": str(STACK),
        "v2_baseline_hand": v2_stats,
        "diversity_min": DISAGREE_MIN,
        "diversity_note": diversity_note,
        "selected_candidate": selected,
        "image_backbones": list(IMG_ROOTS.keys()),
        "joint_uses_wav2vec2": True,
        "stress_proxy": "temperature_gamma=0.75 + gaussian_log_noise=0.15 on OOF proba",
        "invariants": {
            "robot_not_used": True,
            "filename_class_features": False,
            "meta_hier_from_prior_group_oof": True,
            "joint_stacked_on_oof_features": True,
        },
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))

    # store selected OOF preds
    oof_sel = np.zeros(nseg, np.int64)
    sel_mode = selected["mode"]
    sel_params = json.loads(selected["params"]) if isinstance(selected["params"], str) else selected["params"]
    for fold_id, (tr, va) in enumerate(folds):
        params = dict(sel_params)
        if selected["use_bias"] in (True, "True", "true", 1):
            p_tr = blend_proba(oof_meta[tr], oof_hier[tr], oof_joint[tr], sel_mode, params)
            params["bias"] = search_log_bias(p_tr, sy[tr])
        oof_sel[va] = decode_family(oof_meta[va], oof_hier[va], oof_joint[va], sel_mode, params)
    np.save(OUT / "hand_oof_selected_pred.npy", oof_sel)

    print(json.dumps(lock, indent=2, default=float))
    print("\nTOP 15:")
    print(board.head(15).to_string(index=False))
    print("\nv2 baseline:", v2_stats)


if __name__ == "__main__":
    main()
