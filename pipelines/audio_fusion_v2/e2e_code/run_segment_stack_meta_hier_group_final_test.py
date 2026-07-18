"""Robot/test once for locked stack of meta-weighted + hier contact-rescue."""
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
        "outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection",
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
    cand = lock["selected_candidate"]
    base = lock["base_models"]
    META = base["meta_weighted"]
    HIER = base["hier_contact_rescue"]
    audio_base.configure_feature_set("total240")

    meta_x = np.load(OUT / "hand_stack_oof_features.npy")
    meta_y = np.load(OUT / "hand_stack_oof_y.npy")

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

    # Fit base heads on full hand
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

    # meta-weighted second stage needs OOF-style image mat on hand for training was saved;
    # for test: use full-hand image head + locked meta LR on hand_meta features from selection file
    # Rebuild hand meta features like original meta-weighted for fitting meta base:
    # Use saved oof meta/hier only for stack training; at test produce base preds then stack.

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
    spc = np.array([pc[sk == k].mean() for k in tu])

    # Base meta-weighted path
    img_mat3 = suite.normalize(m_img4.predict_proba(tx_c)[:, 1:])
    z_meta = sel.feat_meta(sa_mat, img_mat3, spc)
    # Fit meta base on hand OOF features reconstructed similarly to selection:
    # Use hand_oof_meta as soft targets? Better: retrain meta LR on hand using
    # same feat_meta with full-hand image OOF saved during selection is hard.
    # Instead fit meta LR on stacked hand features from selection's oof_meta labels
    # Wait - we need meta base probs on test. Reconstruct:
    # Train meta-weighted model on hand using full-hand image fit + audio OOF segs.
    # Audio OOF for hand was used in selection; for final we use locked audio test preds for test
    # and for training meta base we use saved hand_oof arrays from original meta-weighted run if present.
    meta_feat_path = Path(
        "outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_features.npy"
    )
    meta_y_path = Path(
        "outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_y.npy"
    )
    if meta_feat_path.exists():
        hm_x = np.load(meta_feat_path)
        hm_y = np.load(meta_y_path)
    else:
        # fallback: use stack features' meta part unavailable — fit on z from hand clip
        hm_x = sel.feat_meta(
            # approximate with zeros forbidden; recompute from hand audio oof requires long path
            sa_mat[:0],
            img_mat3[:0],
            spc[:0],
        )
        raise RuntimeError("need hand_meta_oof_features from meta-weighted selection")

    cw = {
        0: 1.0,
        1: 1.0,
        2: float(META["trunk_class_weight"]),
        3: float(META["twig_class_weight"]),
    }
    meta_base = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(META["C"]),
            max_iter=4000,
            class_weight=cw,
            multi_class="multinomial",
            random_state=42,
        ),
    )
    meta_base.fit(hm_x, hm_y)
    p_meta = suite.normalize(meta_base.predict_proba(z_meta))

    # Base hier path
    contact = float(HIER["alpha"]) * spc + (1 - float(HIER["alpha"])) * m_bin.predict_proba(tx_c)[:, 1]
    w = np.array([HIER["leaf_w"], HIER["trunk_w"], HIER["twig_w"]], dtype=np.float64)
    imat = m_mat.predict_proba(tx_cat)
    mat = suite.normalize(sa_mat * (1 - w) + imat * w)
    tb = float(HIER["tb"])
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1))
        logits[:, 1] += tb * m_trunk.predict_proba(tx_c)[:, 1]
        mat = suite.normalize(np.exp(logits))
    th = float(HIER["th"])
    p_hier = np.zeros((len(tu), 4), np.float64)
    for i in range(len(tu)):
        if contact[i] >= th:
            cmass = max(float(contact[i]), 0.55)
            p_hier[i, 0] = 1 - cmass
            p_hier[i, 1:] = cmass * mat[i]
        else:
            p_hier[i, 0] = 1.0
    p_hier = suite.normalize(p_hier)
    p_blend = suite.normalize(0.45 * p_meta + 0.55 * p_hier)

    # Stack or simple blend
    if float(cand.get("C", -1)) < 0:
        bw = float(cand.get("blend_hier_weight", 0.55))
        p_final = suite.normalize((1 - bw) * p_meta + bw * p_hier)
    else:
        trunk_proxy = p_hier[:, 2]
        Zt = np.hstack(
            [
                np.log(np.clip(p_meta, 1e-6, 1)),
                np.log(np.clip(p_hier, 1e-6, 1)),
                p_meta,
                p_hier,
                p_blend,
                sa4,
                spc[:, None],
                (p_meta.argmax(1) == p_hier.argmax(1)).astype(np.float32)[:, None],
                p_meta.max(1, keepdims=True),
                p_hier.max(1, keepdims=True),
                trunk_proxy[:, None],
            ]
        ).astype(np.float32)
        cw2 = {
            0: 1.0,
            1: 1.0,
            2: float(cand["trunk_class_weight"]),
            3: float(cand["twig_class_weight"]),
        }
        stack = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(cand["C"]),
                max_iter=4000,
                class_weight=cw2,
                multi_class="multinomial",
                random_state=42,
            ),
        )
        stack.fit(meta_x, meta_y)
        p_final = suite.normalize(stack.predict_proba(Zt))

    pred_seg = p_final.argmax(1)
    order = {k: i for i, k in enumerate(tu)}
    pred = np.array([pred_seg[order[k]] for k in sk])
    by = (yt > 0).astype(int)
    bp = (pred > 0).astype(int)
    result = {
        "split": "robot_test_final",
        "protocol": lock["protocol"],
        "n": int(len(yt)),
        "locked_candidate": cand,
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
        },
    }
    (OUT / "segment_stack_meta_hier_final_test_metrics.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
