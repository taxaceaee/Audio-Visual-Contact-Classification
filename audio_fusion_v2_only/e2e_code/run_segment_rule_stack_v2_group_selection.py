"""Hand-only, group-OOF selection for the v2 segment rule stack.

This is the selection half of ``run_segment_rule_stack_v2_group_final_test``.
It deliberately never loads robot/test data.  The companion final-test runner
must only be invoked after this script has written ``selection_lock.json``.
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
import run_segment_meta_weighted_group_selection as meta_selection
import run_segment_stack_meta_hier_group_selection as stack
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection",
    )
)
META_OUT = Path("outputs/audio_feature_meta_weighted_group_selection")
MAT_W = np.array([0.4, 0.55, 0.4], dtype=np.float64)
META_FIXED = {"C": 0.1, "trunk_class_weight": 1.2, "twig_class_weight": 1.2}


def _pool(files: pd.Series, x: np.ndarray, y: np.ndarray):
    keys = stack.seg(files).to_numpy()
    unique, codes = np.unique(keys, return_inverse=True)
    pooled = np.vstack([x[codes == i].mean(0) for i in range(len(unique))])
    labels = np.array([y[codes == i][0] for i in range(len(unique))])
    return unique, codes, pooled, labels


def _normalize(p: np.ndarray) -> np.ndarray:
    return suite.normalize(np.asarray(p, dtype=np.float64))


def _fit_meta(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=META_FIXED["C"],
            max_iter=4000,
            class_weight={
                0: 1.0,
                1: 1.0,
                2: META_FIXED["trunk_class_weight"],
                3: META_FIXED["twig_class_weight"],
            },
            multi_class="multinomial",
            random_state=42,
        ),
    ).fit(x, y)


def _fast_macro(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # This writes the two hand-only meta OOF arrays consumed by the v2 runner.
    # Its own protocol has no robot/test load.
    meta_selection.main()

    audio_base.configure_feature_set("total240")
    frame = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y_rows = frame.y.to_numpy(np.int64)
    segments, codes, clip, y = _pool(
        frame.audio_file,
        np.load(stack.IMG_CLIP / "hand_train_full/X.npy").astype(np.float32),
        y_rows,
    )
    _, _, eff, _ = _pool(
        frame.audio_file,
        np.load(stack.IMG_EFF / "hand_train_full/X.npy").astype(np.float32),
        y_rows,
    )
    groups = stack.spec(pd.Series(segments)).to_numpy()
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(y)), y, groups
        )
    )

    # Rebuild the locked hand OOF audio view used by the base stack.
    audio_frame = audio_base.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train"
    )
    high = group_audio.load_highsr_oof(Path("outputs"))
    pair = specimen.load_pairwise_oof(
        Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        )
    )
    anchor = lift.anchor_lift_proba(audio_frame, high, pair)
    row_groups = stack.spec(frame.audio_file).to_numpy()
    assignment = np.full(len(y_rows), -1, np.int64)
    for fold_id, (_, valid) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(y_rows)), y_rows, row_groups
        )
    ):
        assignment[valid] = fold_id
    sources = broad.load_train_sources(Path("outputs"), y_rows, assignment, 42)
    audio = lift.postprocess(
        audio_frame,
        lift.normalize(0.95 * anchor + 0.05 * sources["report_gate_onehot"]),
        "segment_lift",
    )
    row_contact, row_logits = avr.avr_inputs(audio)
    segment_contact = np.array(
        [row_contact[codes == i].mean() for i in range(len(segments))]
    )
    audio_material = np.vstack(
        [
            _normalize(np.exp(row_logits[codes == i].mean(0, keepdims=True)))[0]
            for i in range(len(segments))
        ]
    )

    meta_x = np.load(META_OUT / "hand_meta_oof_features.npy")
    meta_y = np.load(META_OUT / "hand_meta_oof_y.npy")
    if not np.array_equal(meta_y, y):
        raise AssertionError("hand meta OOF labels do not align with v2 segments")

    meta_oof = np.zeros((len(y), 4), dtype=np.float64)
    binary_oof = np.zeros(len(y), dtype=np.float64)
    material_oof = np.zeros((len(y), 3), dtype=np.float64)
    trunk_oof = np.zeros(len(y), dtype=np.float64)
    clip_eff = np.hstack([clip, eff])
    for fold_id, (train, valid) in enumerate(folds, 1):
        meta_oof[valid] = _normalize(
            _fit_meta(meta_x[train], y[train]).predict_proba(meta_x[valid])
        )
        binary = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, max_iter=1500, class_weight="balanced", random_state=42),
        ).fit(clip[train], (y[train] > 0).astype(int))
        material = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.03,
                max_iter=1500,
                class_weight="balanced",
                multi_class="multinomial",
                random_state=42,
            ),
        ).fit(clip_eff[train][y[train] > 0], y[train][y[train] > 0] - 1)
        trunk = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1200, class_weight="balanced", random_state=42),
        ).fit(clip[train], (y[train] == 2).astype(int))
        binary_oof[valid] = binary.predict_proba(clip[valid])[:, 1]
        material_oof[valid] = material.predict_proba(clip_eff[valid])
        trunk_oof[valid] = trunk.predict_proba(clip[valid])[:, 1]
        print(f"v2 OOF fold {fold_id}/5 done", flush=True)

    rows = []
    # The grid and tie-break reproduce the pre-registered v2 hand selection.
    for alpha in (0.3, 0.4, 0.5):
        contact = alpha * segment_contact + (1.0 - alpha) * binary_oof
        for trunk_bias in (0.3, 0.45, 0.6):
            material = _normalize(audio_material * (1.0 - MAT_W) + material_oof * MAT_W)
            logits = np.log(np.clip(material, 1e-8, 1.0))
            logits[:, 1] += trunk_bias * trunk_oof
            material = _normalize(np.exp(logits))
            for threshold in np.round(np.arange(0.50, 0.651, 0.01), 2):
                pred_hier = np.where(contact >= threshold, material.argmax(1) + 1, 0)
                for rule in ("hier_trunk_meta_else", "hier_only"):
                    pred = (
                        np.where(pred_hier == 2, pred_hier, meta_oof.argmax(1))
                        if rule == "hier_trunk_meta_else"
                        else pred_hier
                    )
                    rows.append(
                        {
                            "alpha": float(alpha),
                            "tb": float(trunk_bias),
                            "th": float(threshold),
                            "macro_f1": _fast_macro(y, pred),
                            "trunk_recall": float(np.mean(pred[y == 2] == 2)),
                            "binary": float(
                                f1_score(y > 0, pred > 0, average="macro", zero_division=0)
                            ),
                            "rule": rule,
                        }
                    )
    board = pd.DataFrame(rows).sort_values(
        ["macro_f1", "trunk_recall", "binary", "rule", "alpha", "tb", "th"],
        ascending=[False, False, False, True, True, True, True],
    ).reset_index(drop=True)
    board.to_csv(OUT / "hand_rule_v2_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    lock = {
        "protocol": "segment_rule_stack_v2_retuned_hier_plus_meta_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "test_loaded": False,
        "selected_candidate": best,
        "global_best_hand": best,
        "base_meta_from": "meta_weighted lock C=0.1 tw=1.2 gw=1.2",
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
