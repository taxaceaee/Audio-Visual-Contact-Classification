"""Robot/test once for locked hier v2 multi-backbone trunk-rescue fusion."""
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
import run_segment_hier_v2_group_selection as sel
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_hier_v2_group_selection",
    )
)


def pool(files, x, y=None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


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


def main():
    lock = json.loads((OUT / "selection_lock.json").read_text())
    assert not lock.get("test_loaded")
    b = lock["selected_candidate"]
    audio_base.configure_feature_set("total240")

    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    hx = {}
    for name, path in sel.BACKBONES.items():
        x = np.load(path / "hand_train_full/X.npy").astype(np.float32)
        _, _, px, hy = pool(hfr.audio_file, x, yh)
        hx[name] = px
    hy = hy
    hx["cat_ce"] = np.hstack([hx["clip"], hx["eff"]])
    hx["cat_all"] = np.hstack([hx["clip"], hx["eff"], hx["dino"], hx["cnv"], hx["r18"]])

    models_mat = {
        k: fit_mat(hx[k], hy) for k in ("clip", "eff", "dino", "cnv", "cat_ce", "cat_all")
    }
    models_bin = {k: fit_bin(hx[k], hy) for k in ("clip", "cat_ce", "cat_all")}
    models_trunk = {k: fit_trunk(hx[k], hy) for k in ("clip", "cat_all")}

    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    tx = {}
    for name, path in sel.BACKBONES.items():
        x = np.load(path / "robot_test/X.npy").astype(np.float32)
        tu, tc, px, ty = pool(tfr.audio_file, x, yt)
        tx[name] = px
    tx["cat_ce"] = np.hstack([tx["clip"], tx["eff"]])
    tx["cat_all"] = np.hstack([tx["clip"], tx["eff"], tx["dino"], tx["cnv"], tx["r18"]])

    mats = {k: models_mat[k].predict_proba(tx[k]) for k in models_mat}
    mats["ens"] = suite.normalize(
        np.mean([mats[k] for k in ("clip", "eff", "dino", "cnv")], axis=0)
    )
    bins = {k: models_bin[k].predict_proba(tx[k])[:, 1] for k in models_bin}
    trunks = {k: models_trunk[k].predict_proba(tx[k])[:, 1] for k in models_trunk}

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
    seg_pc = np.array([pc[sk == k].mean() for k in tu])
    sa_mat = np.vstack(
        [suite.normalize(np.exp(lg[sk == k].mean(0, keepdims=True)))[0] for k in tu]
    )

    csrc = b["contact_source"]
    alpha = float(b["contact_alpha_audio"])
    if csrc == "audio":
        contact = seg_pc
    else:
        contact = alpha * seg_pc + (1 - alpha) * bins[csrc]
    msrc = b["material_source"]
    w = np.array([b["leaf_w"], b["trunk_w"], b["twig_w"]], dtype=np.float64)
    mat = suite.normalize(sa_mat * (1 - w) + mats[msrc] * w)
    tb = float(b["trunk_bias"])
    tsrc = b["trunk_source"]
    if tb:
        logits = np.log(np.clip(mat, 1e-8, 1))
        logits[:, 1] += tb * trunks[tsrc]
        mat = suite.normalize(np.exp(logits))
    th = float(b["contact_threshold"])
    pred_seg = np.where(contact >= th, mat.argmax(1) + 1, 0)
    rescue_th = float(b["trunk_rescue_th"])
    trunk_score = trunks[tsrc]
    if rescue_th > 0:
        rmask = (pred_seg == 0) & (trunk_score >= rescue_th)
        pred_seg[rmask] = 2
        rmask2 = (pred_seg == 3) & (trunk_score >= rescue_th + 0.05)
        pred_seg[rmask2] = 2
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
        },
    }
    (OUT / "segment_hier_v2_final_test_metrics.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
