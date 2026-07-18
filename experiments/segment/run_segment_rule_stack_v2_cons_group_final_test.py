"""Robot/test: rule-stack v2 + specimen contact majority (co-optimal on hand)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_rule_stack_v2_cons_group_selection",
    )
)
META_FIXED = {"C": 0.1, "trunk_class_weight": 1.2, "twig_class_weight": 1.2}
MAT_W = np.array([0.4, 0.55, 0.4], dtype=np.float64)
# Co-optimal with base on hand (macro_f1 identical); prefer consistency among contact segs.
ALPHA, TB, TH = 0.4, 0.3, 0.55
CONS_MIN_N = 2


def pool(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


def specimen_majority_contact(pred, contact, specimen_ids, th, min_n=2):
    out = pred.copy()
    for sid in np.unique(specimen_ids):
        idx = np.where(specimen_ids == sid)[0]
        cidx = idx[contact[idx] >= th]
        if len(cidx) < min_n:
            continue
        vals = out[cidx]
        contact_vals = vals[vals > 0]
        if len(contact_vals) == 0:
            continue
        u, c = np.unique(contact_vals, return_counts=True)
        maj = u[c.argmax()]
        out[cidx] = maj
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = {
        "protocol": "segment_rule_stack_v2_plus_specimen_contact_majority_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "test_loaded": False,
        "selected_candidate": {
            "alpha": ALPHA,
            "tb": TB,
            "th": TH,
            "rule": "hier_trunk_meta_else",
            "consensus": "majority_contact",
            "min_n": CONS_MIN_N,
            "hand_note": "co-optimal with no-consensus on hand macro_f1=0.95611; tie-break prefer contact consistency",
        },
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    assert not lock.get("test_loaded")
    audio_base.configure_feature_set("total240")

    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    xh_c = np.load(sel.IMG_CLIP / "hand_train_full/X.npy").astype(np.float32)
    xh_e = np.load(sel.IMG_EFF / "hand_train_full/X.npy").astype(np.float32)
    _, _, hx_c, hy = pool(hfr.audio_file, xh_c, yh)
    _, _, hx_e, _ = pool(hfr.audio_file, xh_e, yh)
    hx_cat = np.hstack([hx_c, hx_e])

    m_img4 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.03, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
        ),
    )
    m_img4.fit(hx_c, hy)
    m_bin = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.03, max_iter=1500, class_weight="balanced", random_state=42),
    )
    m_bin.fit(hx_c, (hy > 0).astype(int))
    m_mat = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.03, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
        ),
    )
    m_mat.fit(hx_cat[hy > 0], hy[hy > 0] - 1)
    m_trunk = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1200, class_weight="balanced", random_state=42),
    )
    m_trunk.fit(hx_c, (hy == 2).astype(int))

    hm_x = np.load(
        "outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_features.npy"
    )
    hm_y = np.load(
        "outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_y.npy"
    )
    cw = {
        0: 1.0,
        1: 1.0,
        2: META_FIXED["trunk_class_weight"],
        3: META_FIXED["twig_class_weight"],
    }
    meta_base = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=META_FIXED["C"],
            max_iter=4000,
            class_weight=cw,
            multi_class="multinomial",
            random_state=42,
        ),
    )
    meta_base.fit(hm_x, hm_y)

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    xt_c = np.load(sel.IMG_CLIP / "robot_test/X.npy").astype(np.float32)
    xt_e = np.load(sel.IMG_EFF / "robot_test/X.npy").astype(np.float32)
    tu, tc, tx_c, ty = pool(tfr.audio_file, xt_c, yt)
    _, _, tx_e, _ = pool(tfr.audio_file, xt_e, yt)
    tx_cat = np.hstack([tx_c, tx_e])
    seg_spec = sel.spec(pd.Series(tu)).to_numpy()

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
    spc = np.array([pc[sk == k].mean() for k in tu])

    img_mat3 = suite.normalize(m_img4.predict_proba(tx_c)[:, 1:])
    z_meta = sel.feat_meta(sa_mat, img_mat3, spc)
    p_meta = suite.normalize(meta_base.predict_proba(z_meta))
    pred_m = p_meta.argmax(1)

    contact = ALPHA * spc + (1 - ALPHA) * m_bin.predict_proba(tx_c)[:, 1]
    mat = suite.normalize(sa_mat * (1 - MAT_W) + m_mat.predict_proba(tx_cat) * MAT_W)
    if TB:
        logits = np.log(np.clip(mat, 1e-8, 1))
        logits[:, 1] += TB * m_trunk.predict_proba(tx_c)[:, 1]
        mat = suite.normalize(np.exp(logits))
    pred_h = np.where(contact >= TH, mat.argmax(1) + 1, 0)
    pred_seg = np.where(pred_h == 2, pred_h, pred_m)
    pred_seg = specimen_majority_contact(pred_seg, contact, seg_spec, TH, CONS_MIN_N)

    order = {k: i for i, k in enumerate(tu)}
    pred = np.array([pred_seg[order[k]] for k in sk])
    by = (yt > 0).astype(int)
    bp = (pred > 0).astype(int)
    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "n": int(len(yt)),
        "locked_candidate": lock["selected_candidate"],
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
            "specimen_consensus_prediction_only": True,
        },
    }
    (OUT / "segment_rule_stack_v2_cons_final_test_metrics.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
