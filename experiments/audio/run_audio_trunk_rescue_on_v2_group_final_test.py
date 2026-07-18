"""Replace only the v2 trunk rescue head with locked raw high-SR audio.

All decoder parameters are selected on persisted hand specimen-group OOF
arrays. The v2 robot/test segment probabilities are loaded only after the
selection lock is written.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


OUT = Path("outputs/audio_feature_benchmarks/audio_trunk_rescue_on_v2_robust_group_selection")
V2 = Path("outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection")
BASE = Path("outputs/audio_feature_benchmarks/segment_hier_contact_rescue_group_selection")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
RAW = Path("outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/oof_proba/highsr_hgb_default__all_aug")
RAW_TEST = Path("outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports")


def norm(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-9, None)
    return x / x.sum(axis=1, keepdims=True)


def segment_codes(files):
    key = pd.Series(files).astype(str).map(lambda s: re.sub(r"_window_\d+.*$", "", Path(s).stem)).to_numpy()
    return np.unique(key, return_inverse=True)


def segment_raw(path, files, mode):
    p = np.load(path)
    _, code = segment_codes(files)
    if mode == "mean":
        return norm(np.vstack([p[code == i].mean(0) for i in range(int(code.max()) + 1)]))
    if mode == "logmean":
        return norm(np.vstack([np.exp(np.log(np.clip(p[code == i], 1e-9, 1)).mean(0)) for i in range(int(code.max()) + 1)]))
    raise ValueError(mode)


def decode(contact, material, trunk, meta, alpha, tb, th, rule):
    c = alpha * contact[:, 0] + (1.0 - alpha) * contact[:, 1]
    mat = norm(material[:, 0, :] * 0.6 + material[:, 1, :] * 0.4)
    logits = np.log(np.clip(mat, 1e-9, 1.0))
    logits[:, 1] += tb * trunk
    mat = norm(np.exp(logits))
    hier = np.where(c >= th, mat.argmax(1) + 1, 0)
    if rule == "hier_trunk_meta_else":
        return np.where(hier == 2, 2, meta.argmax(1)).astype(np.int64)
    if rule == "hier_only":
        return hier.astype(np.int64)
    raise ValueError(rule)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q = np.load(BASE / "hand_oof_arrays.npz")
    y = q["sy"].astype(np.int64)
    contact = np.stack([q["seg_pc"], q["img_contact_clip"]], axis=1)
    material = np.stack([q["seg_audio_mat"], q["img_mat_cat"]], axis=1)
    meta = np.load("outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection/hand_oof_meta.npy")
    hand = pd.read_csv(ROOT / "audio_visual_dataset_default/dataset.csv")
    hand_seg, _ = segment_codes(hand.audio_file)
    if len(hand_seg) != len(y):
        raise AssertionError("hand segment order mismatch")
    modes = ("mean", "logmean")
    sources = ("clean", "robot_mix", "bandlimit")
    view_probs = {
        mode: {source: segment_raw(RAW / f"{source}_oof_proba.npy", hand.audio_file, mode) for source in sources}
        for mode in modes
    }
    rows = []
    for mode in modes:
        for alpha in (0.25, 0.4, 0.5, 0.6, 0.75):
            for tb in (0.0, 0.2, 0.3, 0.45, 0.6):
                for th in (0.45, 0.50, 0.55, 0.60, 0.65):
                    for rule in ("hier_trunk_meta_else", "hier_only"):
                        scores = []
                        for source in sources:
                            p = view_probs[mode][source]
                            cc = np.stack([p[:, 1:].sum(axis=1), q["img_contact_clip"]], axis=1)
                            mm = np.stack([p[:, 1:], q["img_mat_cat"]], axis=1)
                            pred = decode(cc, mm, p[:, 2], meta, alpha, tb, th, rule)
                            scores.append(float(f1_score(y, pred, average="macro", zero_division=0)))
                        rows.append({"mode": mode, "alpha": alpha, "trunk_bias": tb, "threshold": th, "rule": rule, "mean_view_macro_f1": float(np.mean(scores)), "worst_view_macro_f1": float(np.min(scores)), "clean_macro_f1": scores[0], "robot_mix_macro_f1": scores[1], "bandlimit_macro_f1": scores[2]})
    board = pd.DataFrame(rows).sort_values(["worst_view_macro_f1", "mean_view_macro_f1"], ascending=False).reset_index(drop=True)
    selected = board.iloc[0].to_dict()
    lock = {"protocol": "v2_rule_stack_audio_highsr_trunk_rescue_worst_stress_group_OOF", "selection_data": "hand/default only", "group_column": "specimen_group", "test_loaded": False, "selected_candidate": selected, "final_test_view": "robot_mix"}
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    board.to_csv(OUT / "hand_audio_trunk_rescue_leaderboard.csv", index=False)
    print("Selection lock written before robot/test", json.dumps(lock, indent=2, default=float), flush=True)

    test = pd.read_csv(ROOT / "audio_visual_dataset_robo_default/dataset.csv")
    tbase = np.load(V2 / "segment_rule_stack_v2_base_segment_outputs.npz", allow_pickle=True)
    ids = tbase["segment_ids"].astype(str)
    row_ids = tbase["row_segment_ids"].astype(str)
    p_h = norm(tbase["p_h"])
    contact_t = 1.0 - p_h[:, 0]
    mat_t = norm(p_h[:, 1:] / np.clip(contact_t[:, None], 1e-9, None))
    p_meta_t = norm(tbase["p_meta"])
    _, code = segment_codes(test.audio_file)
    v = np.load(RAW_TEST / "audio_highsr_temporal_tta_select_robot_mix_test_proba.npy")
    if selected["mode"] == "mean":
        pt = norm(np.vstack([v[code == i].mean(0) for i in range(len(ids))]))
    else:
        pt = norm(np.vstack([np.exp(np.log(np.clip(v[code == i], 1e-9, 1)).mean(0)) for i in range(len(ids))]))
    audio_contact_t = pt[:, 1:].sum(axis=1)
    c = float(selected["alpha"]) * audio_contact_t + (1.0 - float(selected["alpha"])) * contact_t
    mat = norm(pt[:, 1:] * 0.6 + mat_t * 0.4)
    logits = np.log(np.clip(mat, 1e-9, 1.0)); logits[:, 1] += float(selected["trunk_bias"]) * pt[:, 2]; mat = norm(np.exp(logits))
    hier = np.where(c >= float(selected["threshold"]), mat.argmax(1) + 1, 0)
    pred_seg = np.where(hier == 2, 2, p_meta_t.argmax(1)) if selected["rule"] == "hier_trunk_meta_else" else hier
    order = {k: i for i, k in enumerate(ids)}
    pred = np.asarray([pred_seg[order[k]] for k in row_ids], dtype=np.int64)
    yt = test.category.map({"ambient": 0, "leaf": 1, "trunk": 2, "twig": 3}).to_numpy(np.int64)
    result = {"split": "robot_test_final", "protocol": lock["protocol"], "selected_candidate": selected, "metrics": {"macro_f1_4class": float(f1_score(yt, pred, average="macro", zero_division=0)), "binary_macro_f1": float(f1_score(yt > 0, pred > 0, average="macro", zero_division=0))}, "confusion_matrix_4class": confusion_matrix(yt, pred, labels=[0, 1, 2, 3]).tolist(), "per_class_4class": classification_report(yt, pred, labels=[0, 1, 2, 3], target_names=["ambient", "leaf", "trunk", "twig"], output_dict=True, zero_division=0), "invariants": {"test_loaded_after_lock": True, "selection_used_hand_only": True, "segment_label_pure": True}}
    (OUT / "audio_trunk_rescue_on_v2_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
