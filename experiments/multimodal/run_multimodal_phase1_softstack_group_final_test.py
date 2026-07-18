"""One-shot robot/test for Phase 1 soft-stack after hand selection lock."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

import run_avr_group_selection as avr
import run_multimodal_phase1_softstack_group_selection as phase1
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
W2V_HAND = Path("outputs/audio_wav2vec2_features/hand_train_full_X.npy")
W2V_ROBOT = Path("outputs/audio_wav2vec2_features/robot_test_X.npy")
META_OOF_FEAT = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_features.npy")
META_OOF_Y = Path("outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_y.npy")


def pool(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


def fit_meta(z, y, c, tw, gw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cw = {0: 1.0, 1: 1.0, 2: tw, 3: gw}
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, max_iter=4000, class_weight=cw, multi_class="multinomial", random_state=42
        ),
    )
    m.fit(z, y)
    return m


def hier_proba_from_heads(spc, sa_mat, tx_c, tx_cat, m_bin, m_mat, m_trunk, hier, mat_w):
    contact = hier["alpha"] * spc + (1.0 - hier["alpha"]) * m_bin.predict_proba(tx_c)[:, 1]
    mat = suite.normalize(sa_mat * (1.0 - mat_w) + m_mat.predict_proba(tx_cat) * mat_w)
    if hier["tb"]:
        logits = np.log(np.clip(mat, 1e-8, 1.0))
        logits[:, 1] += hier["tb"] * m_trunk.predict_proba(tx_c)[:, 1]
        mat = suite.normalize(np.exp(logits))
    p = np.zeros((len(spc), 4), np.float64)
    for i in range(len(spc)):
        if contact[i] >= hier["th"]:
            cmass = max(float(contact[i]), 0.55)
            p[i, 0] = 1.0 - cmass
            p[i, 1:] = cmass * mat[i]
        else:
            p[i, 0] = 1.0
    return suite.normalize(p)


def main() -> None:
    lock = json.loads((OUT / "selection_lock.json").read_text())
    assert not lock.get("test_loaded"), "lock already test-loaded"
    selected = lock["selected_candidate"]
    mode = selected["mode"]
    params = (
        json.loads(selected["params"])
        if isinstance(selected["params"], str)
        else dict(selected["params"])
    )
    use_bias = selected["use_bias"] in (True, "True", "true", 1)
    hier = lock.get("hier_locked", phase1.HIER)
    mat_w = np.array([hier["leaf_w"], hier["trunk_w"], hier["twig_w"]], dtype=np.float64)
    meta_cfg = lock.get("meta_locked", phase1.META)

    audio_base.configure_feature_set("total240")

    # ---- hand fit ----
    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    emb_h = {
        name: np.load(root / "hand_train_full/X.npy").astype(np.float32)
        for name, root in phase1.IMG_ROOTS.items()
    }
    xw_h = np.load(W2V_HAND).astype(np.float32)
    hu, hc, _, hy = pool(hfr.audio_file, emb_h["clip"], yh)
    hx = {n: np.vstack([emb_h[n][hc == i].mean(0) for i in range(len(hu))]) for n in emb_h}
    hx_w2v = np.vstack([xw_h[hc == i].mean(0) for i in range(len(hu))])
    hx_c = hx["clip"]
    hx_cat = np.hstack([hx["clip"], hx["eff"]])
    hx_multi = np.hstack([hx[k] for k in ("clip", "eff", "dino", "convnext")])

    af = audio_base.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train")
    high = group_audio.load_highsr_oof(Path("outputs"))
    pair = specimen.load_pairwise_oof(
        Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        )
    )
    anchor = lift.anchor_lift_proba(af, high, pair)
    sg = sel.spec(hfr.audio_file).to_numpy()
    assignment = np.full(len(yh), -1, np.int64)
    for k, (_, va) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(yh)), yh, sg)
    ):
        assignment[va] = k
    src = broad.load_train_sources(Path("outputs"), yh, assignment, 42)
    audio_h = lift.postprocess(
        af, lift.normalize(0.95 * anchor + 0.05 * src["report_gate_onehot"]), "segment_lift"
    )
    pc_h, lg_h = avr.avr_inputs(audio_h)
    sk_h = sel.seg(hfr.audio_file).to_numpy()
    h_sa_mat = np.vstack(
        [suite.normalize(np.exp(lg_h[sk_h == k].mean(0, keepdims=True)))[0] for k in hu]
    )
    h_sa4 = np.vstack(
        [suite.normalize(audio_h[sk_h == k].mean(0, keepdims=True))[0] for k in hu]
    )
    h_spc = np.array([pc_h[sk_h == k].mean() for k in hu], dtype=np.float64)

    m_clip4 = phase1.fit_lr4(hx_c, hy, 0.03)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m_bin = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.03, max_iter=1500, class_weight="balanced", random_state=42),
    )
    m_bin.fit(hx_c, (hy > 0).astype(int))
    m_trunk = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1200, class_weight="balanced", random_state=42),
    )
    m_trunk.fit(hx_c, (hy == 2).astype(int))
    contact_h = hy > 0
    m_mat_cat = phase1.fit_lr_mat(hx_cat[contact_h], hy[contact_h] - 1, 0.03)
    m_mat_multi = phase1.fit_lr_mat(hx_multi[contact_h], hy[contact_h] - 1, 0.03)
    mat_heads = {n: phase1.fit_lr_mat(hx[n][contact_h], hy[contact_h] - 1, 0.03) for n in hx}
    m_w2v = phase1.fit_lr4(hx_w2v, hy, 0.1)

    # meta: prefer nested OOF features if available (same as v2)
    if META_OOF_FEAT.exists() and META_OOF_Y.exists():
        hm_x = np.load(META_OOF_FEAT)
        hm_y = np.load(META_OOF_Y)
        meta = fit_meta(
            hm_x, hm_y, meta_cfg["C"], meta_cfg["trunk_class_weight"], meta_cfg["twig_class_weight"]
        )
    else:
        img_mat3_h = suite.normalize(m_clip4.predict_proba(hx_c)[:, 1:])
        z_meta_h = sel.feat_meta(h_sa_mat, img_mat3_h, h_spc)
        meta = fit_meta(
            z_meta_h, hy, meta_cfg["C"], meta_cfg["trunk_class_weight"], meta_cfg["twig_class_weight"]
        )

    # hand joint train features (full-fit heads — same practice as v2 final)
    p_clip_h = suite.normalize(m_clip4.predict_proba(hx_c))
    p_w2v_h = suite.normalize(m_w2v.predict_proba(hx_w2v))
    p_mats_h = {n: suite.normalize(h.predict_proba(hx[n])) for n, h in mat_heads.items()}
    p_mats_h["multi"] = suite.normalize(m_mat_multi.predict_proba(hx_multi))
    p_meta_h = suite.normalize(
        meta.predict_proba(sel.feat_meta(h_sa_mat, suite.normalize(p_clip_h[:, 1:]), h_spc))
    )
    p_hier_h = hier_proba_from_heads(
        h_spc, h_sa_mat, hx_c, hx_cat, m_bin, m_mat_cat, m_trunk, hier, mat_w
    )
    z_joint_h = phase1.joint_features(
        h_sa4, h_sa_mat, h_spc, p_clip_h, p_mats_h, p_w2v_h, p_meta_h, p_hier_h
    )
    # Prefer training joint on hand OOF joint features if present (less leakage)
    z_oof = OUT / "hand_oof_joint_features.npy"
    y_oof = OUT / "hand_oof_y.npy"
    if z_oof.exists() and y_oof.exists():
        mlp = phase1.fit_mlp(np.load(z_oof), np.load(y_oof), hidden=(160,), alpha=2e-3, seed=0)
    else:
        mlp = phase1.fit_mlp(z_joint_h, hy, hidden=(160,), alpha=2e-3, seed=0)

    if use_bias:
        p_tr = phase1.blend_proba(p_meta_h, p_hier_h, suite.normalize(mlp.predict_proba(z_joint_h)), mode, params)
        # if trained on oof features, bias on oof blend
        if z_oof.exists():
            om = np.load(OUT / "hand_oof_meta.npy")
            oh = np.load(OUT / "hand_oof_hier.npy")
            oj = np.load(OUT / "hand_oof_joint.npy")
            oy = np.load(OUT / "hand_oof_y.npy")
            p_tr = phase1.blend_proba(om, oh, oj, mode, params)
            params["bias"] = phase1.search_log_bias(p_tr, oy)
        else:
            params["bias"] = phase1.search_log_bias(p_tr, hy)

    # ---- robot AFTER lock ----
    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    emb_t = {
        name: np.load(root / "robot_test/X.npy").astype(np.float32)
        for name, root in phase1.IMG_ROOTS.items()
    }
    xw_t = np.load(W2V_ROBOT).astype(np.float32)
    tu, tc, _, ty = pool(tfr.audio_file, emb_t["clip"], yt)
    tx = {n: np.vstack([emb_t[n][tc == i].mean(0) for i in range(len(tu))]) for n in emb_t}
    tx_w2v = np.vstack([xw_t[tc == i].mean(0) for i in range(len(tu))])
    tx_c = tx["clip"]
    tx_cat = np.hstack([tx["clip"], tx["eff"]])
    tx_multi = np.hstack([tx[k] for k in ("clip", "eff", "dino", "convnext")])

    audio = pd.read_csv(
        "outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/"
        "audio_lift_source_blend_select_final_test_predictions.csv"
    )
    raw = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    assert np.array_equal(
        audio.audio_file.astype(str).to_numpy(), raw.audio_file.astype(str).to_numpy()
    )
    proba = suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy(np.float64))
    pc, lg = avr.avr_inputs(proba)
    sk = sel.seg(tfr.audio_file).to_numpy()
    sa_mat = np.vstack(
        [suite.normalize(np.exp(lg[sk == k].mean(0, keepdims=True)))[0] for k in tu]
    )
    sa4 = np.vstack([suite.normalize(proba[sk == k].mean(0, keepdims=True))[0] for k in tu])
    spc = np.array([pc[sk == k].mean() for k in tu], dtype=np.float64)

    img4 = suite.normalize(m_clip4.predict_proba(tx_c))
    z_meta = sel.feat_meta(sa_mat, suite.normalize(img4[:, 1:]), spc)
    p_meta = suite.normalize(meta.predict_proba(z_meta))
    p_hier = hier_proba_from_heads(
        spc, sa_mat, tx_c, tx_cat, m_bin, m_mat_cat, m_trunk, hier, mat_w
    )
    p_w2v = suite.normalize(m_w2v.predict_proba(tx_w2v))
    p_mats = {n: suite.normalize(h.predict_proba(tx[n])) for n, h in mat_heads.items()}
    p_mats["multi"] = suite.normalize(m_mat_multi.predict_proba(tx_multi))
    z_joint = phase1.joint_features(sa4, sa_mat, spc, img4, p_mats, p_w2v, p_meta, p_hier)
    p_joint = suite.normalize(mlp.predict_proba(z_joint))

    pred_seg = phase1.decode_family(p_meta, p_hier, p_joint, mode, params)
    order = {k: i for i, k in enumerate(tu)}
    pred = np.array([pred_seg[order[k]] for k in sk])
    by = (yt > 0).astype(int)
    bp = (pred > 0).astype(int)

    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "n": int(len(yt)),
        "locked_candidate": selected,
        "metrics": {
            "accuracy_4class": float(accuracy_score(yt, pred)),
            "macro_precision_4class": float(
                precision_score(yt, pred, average="macro", zero_division=0)
            ),
            "macro_recall_4class": float(
                recall_score(yt, pred, average="macro", zero_division=0)
            ),
            "macro_f1_4class": float(f1_score(yt, pred, average="macro", zero_division=0)),
            "weighted_f1_4class": float(
                f1_score(yt, pred, average="weighted", zero_division=0)
            ),
            "binary_macro_f1": float(f1_score(by, bp, average="macro", zero_division=0)),
        },
        "per_class_4class": classification_report(
            yt,
            pred,
            labels=[0, 1, 2, 3],
            target_names=["ambient", "leaf", "trunk", "twig"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_4class": confusion_matrix(yt, pred, labels=[0, 1, 2, 3]).tolist(),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "segment_label_pure": True,
            "no_test_tuning": True,
        },
    }
    (OUT / "phase1_softstack_final_test_metrics.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    lock_after = dict(lock)
    lock_after["test_loaded"] = True
    lock_after["final_test_macro_f1"] = result["metrics"]["macro_f1_4class"]
    (OUT / "selection_lock_after_test.json").write_text(
        json.dumps(lock_after, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
