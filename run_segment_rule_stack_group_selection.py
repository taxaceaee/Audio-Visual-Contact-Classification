"""Rule-based stack of meta-weighted OOF + hier OOF (hand only).

Selects a small menu of deterministic combination rules on specimen-group OOF.
Never loads robot/test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_rule_stack_group_selection",
    )
)
STACK = Path("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection")


def fast_macro(y, p):
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def apply_rule(pm, ph, rule: str, bw: float = 0.5):
    pred_m = pm.argmax(1)
    pred_h = ph.argmax(1)
    out = pred_h.copy()
    if rule == "hier_only":
        out = pred_h
    elif rule == "meta_only":
        out = pred_m
    elif rule == "blend":
        out = suite_norm_argmax(pm, ph, bw)
    elif rule == "hier_trunk_meta_else":
        out = np.where(pred_h == 2, pred_h, pred_m)
    elif rule == "hier_trunk_meta_twig_else_hier":
        out = pred_h.copy()
        out[pred_h == 2] = 2
        out[pred_m == 3] = 3
        # if conflict trunk/twig prefer hier trunk
        out[(pred_h == 2) & (pred_m == 3)] = 2
    elif rule == "max_conf":
        conf_m = pm.max(1)
        conf_h = ph.max(1)
        out = np.where(conf_h >= conf_m, pred_h, pred_m)
    elif rule == "hier_unless_meta_twig_high":
        out = pred_h.copy()
        mask = (pm[:, 3] > 0.45) & (pm[:, 3] > ph[:, 3] + 0.05)
        out[mask] = 3
    elif rule == "hier_trunk_boost_meta_twig":
        out = pred_h.copy()
        out[pred_h == 2] = 2
        mask = (pred_m == 3) & (pred_h != 2)
        out[mask] = 3
    else:
        raise ValueError(rule)
    return out


def suite_norm_argmax(pm, ph, bw):
    p = (1 - bw) * pm + bw * ph
    p = p / p.sum(1, keepdims=True)
    return p.argmax(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pm = np.load(STACK / "hand_oof_meta.npy")
    ph = np.load(STACK / "hand_oof_hier.npy")
    sy = np.load(STACK / "hand_stack_oof_y.npy")
    assert len(pm) == len(sy) == len(ph)

    rules = [
        "hier_only",
        "meta_only",
        "blend",
        "hier_trunk_meta_else",
        "hier_trunk_meta_twig_else_hier",
        "max_conf",
        "hier_unless_meta_twig_high",
        "hier_trunk_boost_meta_twig",
    ]
    rows = []
    for rule in rules:
        if rule == "blend":
            for bw in np.linspace(0, 1, 21):
                pred = apply_rule(pm, ph, rule, bw)
                rows.append(
                    {
                        "rule": rule,
                        "blend_hier_weight": float(bw),
                        "macro_f1_4class": fast_macro(sy, pred),
                        "binary_macro_f1": float(
                            f1_score(sy > 0, pred > 0, average="macro", zero_division=0)
                        ),
                        "trunk_recall": float(
                            np.sum((sy == 2) & (pred == 2)) / max(np.sum(sy == 2), 1)
                        ),
                        "twig_recall": float(
                            np.sum((sy == 3) & (pred == 3)) / max(np.sum(sy == 3), 1)
                        ),
                    }
                )
        else:
            pred = apply_rule(pm, ph, rule)
            rows.append(
                {
                    "rule": rule,
                    "blend_hier_weight": -1.0,
                    "macro_f1_4class": fast_macro(sy, pred),
                    "binary_macro_f1": float(
                        f1_score(sy > 0, pred > 0, average="macro", zero_division=0)
                    ),
                    "trunk_recall": float(
                        np.sum((sy == 2) & (pred == 2)) / max(np.sum(sy == 2), 1)
                    ),
                    "twig_recall": float(
                        np.sum((sy == 3) & (pred == 3)) / max(np.sum(sy == 3), 1)
                    ),
                }
            )
    board = (
        pd.DataFrame(rows)
        .sort_values(
            ["macro_f1_4class", "trunk_recall", "twig_recall"], ascending=False
        )
        .reset_index(drop=True)
    )
    board.to_csv(OUT / "hand_rule_stack_leaderboard.csv", index=False)
    best = board.iloc[0].to_dict()
    # freeze base model HPs from prior hand locks
    base = json.loads((STACK / "selection_lock.json").read_text())["base_models"]
    lock = {
        "protocol": "segment_rule_stack_meta_hier_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "base_models": base,
        "test_loaded": False,
        "selected_candidate": best,
        "base_oof_from": str(STACK),
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print(board.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
