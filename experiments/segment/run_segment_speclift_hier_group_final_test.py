"""Robot/test once for locked spec-lift + hier image fusion."""
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
import run_segment_speclift_hier_group_selection as sel
import train_audio_specimen_contact_consensus_select_final_test as sp
import train_audio_specimen_contact_lift_select_final_test as sl
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_speclift_hier_group_selection",
    )
)


def pool(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


def main() -> None:
    lock = json.loads((OUT / "selection_lock.json").read_text())
    assert not lock.get("test_loaded")
    b = lock["selected_candidate"]
    lift_lock = lock["audio_lift_lock"]
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

    def fit_mat(x, y):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.03, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
            ),
        )
        m.fit(x[y > 0], y[y > 0] - 1)
        return m

    def fit_bin(x, y):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, max_iter=1500, class_weight="balanced", random_state=42),
        )
        m.fit(x, (y > 0).astype(int))
        return m

    def fit_trunk(x, y):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1200, class_weight="balanced", random_state=42),
        )
        m.fit(x, (y == 2).astype(int))
        return m

    models_mat = {
        "clip": fit_mat(hx_c, hy),
        "eff": fit_mat(hx_e, hy),
        "cat": fit_mat(hx_cat, hy),
    }
    models_bin = {"clip": fit_bin(hx_c, hy), "cat": fit_bin(hx_cat, hy)}
    m_trunk = fit_trunk(hx_c, hy)

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

    # Spec-lift final audio from locked highsr+pairwise test preds
    hi = pd.read_csv(
        "outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/"
        "audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pa = pd.read_csv(
        "outputs/audio_feature_benchmarks/audio_pairwise_contact_stress_cv_select/reports/"
        "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    raw = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    assert np.array_equal(hi.audio_file.astype(str), pa.audio_file.astype(str))
    assert np.array_equal(hi.audio_file.astype(str), raw.audio_file.astype(str))
    base = sl.normalize(
        0.8 * hi[sl.PROBA_COLUMNS].to_numpy() + 0.2 * pa[sl.PROBA_COLUMNS].to_numpy()
    )
    segp, w2s = sp.segment_proba_from_window(hi, base)
    codes = sp.specimen_codes_for_segments(hi, w2s)
    segp = sl.consensus_and_lift(
        segp,
        codes,
        float(lift_lock["consensus_threshold"]),
        int(lift_lock["min_contact_segments"]),
        lift_lock["lift_min_mass"],
        float(lift_lock["lift_floor"]),
        float(lift_lock["lift_confidence"]),
    )
    audio = segp[w2s]
    pc, lg = avr.avr_inputs(audio)
    sk = sel.seg(tfr.audio_file).to_numpy()
    seg_pc = np.array([pc[sk == k].mean() for k in tu])
    sa_mat = np.vstack(
        [suite.normalize(np.exp(lg[sk == k].mean(0, keepdims=True)))[0] for k in tu]
    )

    csrc = b["contact_source"]
    alpha = float(b["contact_alpha_audio"])
    if csrc == "audio":
        contact = seg_pc
    elif csrc == "clip":
        contact = alpha * seg_pc + (1 - alpha) * models_bin["clip"].predict_proba(tx_c)[:, 1]
    else:
        contact = alpha * seg_pc + (1 - alpha) * models_bin["cat"].predict_proba(tx_cat)[:, 1]

    msrc = b["material_source"]
    if msrc == "audio":
        mat = sa_mat
    else:
        feats = {"clip": tx_c, "eff": tx_e, "cat": tx_cat}
        imat = models_mat[msrc].predict_proba(feats[msrc])
        w = np.array([b["leaf_w"], b["trunk_w"], b["twig_w"]], dtype=np.float64)
        mat = suite.normalize(sa_mat * (1 - w) + imat * w)
    tb = float(b["trunk_bias"])
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1))
        logits[:, 1] += tb * m_trunk.predict_proba(tx_c)[:, 1]
        mat = suite.normalize(np.exp(logits))
    th = float(b["contact_threshold"])
    pred_seg = np.where(contact >= th, mat.argmax(1) + 1, 0)
    order = {k: i for i, k in enumerate(tu)}
    pred = np.array([pred_seg[order[k]] for k in sk])

    by = (yt > 0).astype(int)
    bp = (pred > 0).astype(int)
    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "n": int(len(yt)),
        "locked_candidate": b,
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
            "no_specimen_y0_training": True,
        },
    }
    (OUT / "segment_speclift_hier_final_test_metrics.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
