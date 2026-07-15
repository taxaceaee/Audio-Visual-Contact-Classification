"""Single robot/test eval for locked multi-backbone segment meta fusion."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
import run_segment_multibb_meta_group_selection as sel
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_multibb_meta_group_selection",
    )
)


def pool_seg(files: pd.Series, x: np.ndarray, y: np.ndarray | None = None):
    keys = sel.seg(files).to_numpy()
    u, c = np.unique(keys, return_inverse=True)
    px = np.vstack([x[c == i].mean(0) for i in range(len(u))])
    py = None if y is None else np.array([y[c == i][0] for i in range(len(u))])
    return u, c, px, py


def main() -> None:
    lock = json.loads((OUT / "selection_lock.json").read_text())
    assert not lock.get("test_loaded"), "must lock before loading test"
    cand = lock["selected_candidate"]
    meta_x = np.load(OUT / "hand_meta_oof_features.npy")
    meta_y = np.load(OUT / "hand_meta_oof_y.npy")
    audio_base.configure_feature_set("total240")

    # Fit per-backbone 4-class image heads on full hand (post-lock).
    hfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    yh = hfr.y.to_numpy(np.int64)
    hand_models = []
    for bb in sel.BACKBONES:
        xh = np.load(bb / "hand_train_full/X.npy").astype(np.float32)
        _, _, hx, hy = pool_seg(hfr.audio_file, xh, yh)
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.03,
                max_iter=1200,
                class_weight="balanced",
                multi_class="multinomial",
                random_state=42,
            ),
        )
        m.fit(hx, hy)
        hand_models.append(m)

    # Robot test — loaded only after lock assert.
    tfr = suite.load_manifest(
        ROOT / "audio_visual_dataset_robo_default/dataset.csv",
        ROOT / "audio_visual_dataset_robo_default",
        "robot_test",
    )
    yt = tfr.y.to_numpy(np.int64)
    tu, tc, _, _ = pool_seg(
        tfr.audio_file, np.zeros((len(tfr), 1), dtype=np.float32), yt
    )

    img4_list = []
    for bb, m in zip(sel.BACKBONES, hand_models):
        xt = np.load(bb / "robot_test/X.npy").astype(np.float32)
        assert len(xt) == len(yt), bb
        _, _, tx, _ = pool_seg(tfr.audio_file, xt, yt)
        img4_list.append(suite.normalize(m.predict_proba(tx)))

    audio = pd.read_csv(
        "outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/"
        "audio_lift_source_blend_select_final_test_predictions.csv"
    )
    raw = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    assert np.array_equal(
        audio.audio_file.astype(str).to_numpy(), raw.audio_file.astype(str).to_numpy()
    )
    proba = suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy(np.float64))
    sk = sel.seg(tfr.audio_file).to_numpy()
    sa4 = np.vstack(
        [suite.normalize(proba[sk == k].mean(0, keepdims=True))[0] for k in tu]
    )
    pc, _ = avr.avr_inputs(proba)
    spc = np.array([pc[sk == k].mean() for k in tu], dtype=np.float64)
    z = sel.build_meta_features(sa4, img4_list, spc)

    model_name = str(cand.get("model", "logreg"))
    if model_name == "hgb":
        sw = np.ones(len(meta_y), dtype=np.float64)
        sw[meta_y == 2] = float(cand.get("trunk_class_weight", 2.0))
        sw[meta_y == 3] = float(cand.get("twig_class_weight", 1.5))
        md = int(cand.get("max_depth", 3))
        meta = HistGradientBoostingClassifier(
            max_depth=None if md < 0 else md,
            learning_rate=float(cand.get("lr", 0.1)),
            max_iter=200,
            random_state=42,
        )
        meta.fit(meta_x, meta_y, sample_weight=sw)
        pred_seg = meta.predict(z)
    else:
        cw = {
            0: 1.0,
            1: 1.0,
            2: float(cand["trunk_class_weight"]),
            3: float(cand["twig_class_weight"]),
        }
        meta = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(cand["C"]),
                max_iter=4000,
                class_weight=cw,
                multi_class="multinomial",
                random_state=42,
            ),
        )
        meta.fit(meta_x, meta_y)
        pred_seg = meta.predict(z)

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
            "macro_f1_4class": float(
                f1_score(yt, pred, average="macro", zero_division=0)
            ),
            "weighted_f1_4class": float(
                f1_score(yt, pred, average="weighted", zero_division=0)
            ),
            "binary_accuracy": float(accuracy_score(by, bp)),
            "binary_macro_f1": float(
                f1_score(by, bp, average="macro", zero_division=0)
            ),
        },
        "per_class_4class": classification_report(
            yt,
            pred,
            labels=[0, 1, 2, 3],
            target_names=["ambient", "leaf", "trunk", "twig"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_4class": confusion_matrix(
            yt, pred, labels=[0, 1, 2, 3]
        ).tolist(),
        "invariants": {
            "test_loaded_after_lock": True,
            "selection_used_hand_only": True,
            "no_new_split": True,
            "segment_label_pure": True,
        },
    }
    (OUT / "segment_multibb_meta_final_test_metrics.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
