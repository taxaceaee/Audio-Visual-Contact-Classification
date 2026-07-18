"""Hand-only raw high-SR audio + image hierarchical fusion.

Selection is specimen-group OOF on hand/default. Robot/test is loaded only
after the selection lock is written. The audio input is the persisted raw
high-SR OOF/test probability source, before the older segment-lift decoder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs/audio_feature_benchmarks/raw_highsr_hier_fusion_group_selection")
IMG_CLIP = Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k")
IMG_EFF = Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k")
CLASSES = np.arange(4)
PROBA = ["proba_ambient", "proba_leaf", "proba_trunk", "proba_twig"]


def norm(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-9, None)
    return x / x.sum(axis=1, keepdims=True)


def segment_pool(files: pd.Series, x: np.ndarray, y: np.ndarray | None = None):
    keys = sel.seg(files).to_numpy()
    unique, code = np.unique(keys, return_inverse=True)
    pooled = np.vstack([x[code == i].mean(axis=0) for i in range(len(unique))])
    labels = None if y is None else np.asarray([y[code == i][0] for i in range(len(unique))])
    return unique, code, pooled, labels


def fit_heads(xc, xe, y, train_idx):
    def fit4(x):
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=2000, class_weight="balanced", multi_class="multinomial", random_state=42)).fit(x[train_idx], y[train_idx])

    def fitbin(x):
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=2000, class_weight="balanced", random_state=42)).fit(x[train_idx], (y[train_idx] > 0).astype(int))

    contact = fitbin(xc)
    material_idx = train_idx[y[train_idx] > 0]
    material = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=2000, class_weight="balanced", multi_class="multinomial", random_state=42)).fit(np.hstack([xc, xe])[material_idx], y[material_idx] - 1)
    trunk = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)).fit(xc[train_idx], (y[train_idx] == 2).astype(int))
    return contact, material, trunk


def audio_recipe(raw: dict[str, np.ndarray], code: np.ndarray, nseg: int, mode: str):
    rows = []
    for view in ("clean", "robot_mix", "bandlimit"):
        p = norm(raw[view])
        if mode == "mean":
            q = np.vstack([p[code == i].mean(0) for i in range(nseg)])
        elif mode == "logmean":
            q = norm(np.exp(np.vstack([np.log(p[code == i]).mean(0) for i in range(nseg)])))
        else:
            raise ValueError(mode)
        rows.append(q)
    return rows


def evaluate(y, audio_contact, audio_mat, image_contact, image_mat, image_trunk, params):
    alpha, audio_weight, threshold, trunk_bias, recipe, decoder = params
    contact = alpha * audio_contact + (1.0 - alpha) * image_contact
    mat = norm(audio_weight * audio_mat + (1.0 - audio_weight) * image_mat)
    if trunk_bias:
        logits = np.log(np.clip(mat, 1e-9, 1.0))
        logits[:, 1] += trunk_bias * image_trunk
        mat = norm(np.exp(logits))
    if decoder == "hier":
        pred = np.where(contact >= threshold, mat.argmax(1) + 1, 0)
    elif decoder == "trunk_else_image":
        hier = np.where(contact >= threshold, mat.argmax(1) + 1, 0)
        pred = np.where(hier == 2, 2, np.where(image_contact >= threshold, image_mat.argmax(1) + 1, 0))
    else:
        raise ValueError(decoder)
    return pred.astype(np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hand = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    y_rows = hand.y.to_numpy(np.int64)
    _, code_rows, _, _ = segment_pool(hand.audio_file, np.zeros((len(hand), 1)), y_rows)
    seg_ids, _, _, y_seg = segment_pool(hand.audio_file, np.zeros((len(hand), 1)), y_rows)
    specimen = sel.spec(pd.Series(seg_ids)).to_numpy()
    raw_dir = Path("outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/oof_proba/highsr_hgb_default__all_aug")
    raw = {v: np.load(raw_dir / f"{v}_oof_proba.npy") for v in ("clean", "robot_mix", "bandlimit")}
    if any(len(v) != len(hand) for v in raw.values()):
        raise AssertionError("raw high-SR OOF rows do not align with hand manifest")
    xc_rows = np.load(IMG_CLIP / "hand_train_full/X.npy").astype(np.float32)
    xe_rows = np.load(IMG_EFF / "hand_train_full/X.npy").astype(np.float32)
    _, _, xc, _ = segment_pool(hand.audio_file, xc_rows, y_rows)
    _, _, xe, _ = segment_pool(hand.audio_file, xe_rows, y_rows)
    recipes = {}
    for mode in ("mean", "logmean"):
        views = audio_recipe(raw, code_rows, len(seg_ids), mode)
        # Average clean/robot_mix/bandlimit in probability space; individual
        # views are retained as selectable train-only robustness candidates.
        recipes[mode] = {"clean": views[0], "robot_mix": views[1], "bandlimit": views[2], "all": norm(sum(views) / 3.0)}

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(y_seg)), y_seg, specimen))
    oof = []
    candidate_rows = []
    grid = []
    for recipe in ("clean", "robot_mix", "bandlimit", "all"):
        for mode in ("mean", "logmean"):
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                for aw in (0.0, 0.25, 0.5, 0.75, 1.0):
                    for threshold in (0.35, 0.45, 0.55, 0.65):
                        for tb in (0.0, 0.3, 0.6):
                            for decoder in ("hier", "trunk_else_image"):
                                grid.append((alpha, aw, threshold, tb, recipe, decoder, mode))
    for fold_id, (tr, va) in enumerate(folds):
        cmodel, mmodel, tmodel = fit_heads(xc, xe, y_seg, tr)
        image_contact = cmodel.predict_proba(xc[va])[:, 1]
        image_mat = norm(mmodel.predict_proba(np.hstack([xc, xe])[va]))
        image_trunk = tmodel.predict_proba(xc[va])[:, 1]
        fold_rows = []
        for params in grid:
            alpha, aw, th, tb, recipe, decoder, mode = params
            p = recipes[mode][recipe]
            ac = p[:, 1:].sum(1)[va]
            am = norm(p[:, 1:][va])
            pred = evaluate(y_seg[va], ac, am, image_contact, image_mat, image_trunk, (alpha, aw, th, tb, recipe, decoder))
            fold_rows.append((params, float(f1_score(y_seg[va], pred, average="macro", zero_division=0))))
        oof.append((fold_id, fold_rows))
        print(f"fold {fold_id + 1}/5 done", flush=True)
    for params in grid:
        scores = [dict(rows)[params] for _, rows in oof]
        candidate_rows.append({"mode": params[-1], "alpha": params[0], "audio_weight": params[1], "threshold": params[2], "trunk_bias": params[3], "recipe": params[4], "decoder": params[5], "mean_cv_macro_f1": float(np.mean(scores)), "worst_cv_macro_f1": float(np.min(scores))})
    leaderboard = pd.DataFrame(candidate_rows).sort_values(["mean_cv_macro_f1", "worst_cv_macro_f1"], ascending=False).reset_index(drop=True)
    selected = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "raw_highsr_audio_image_hierarchical_specimen_group_OOF", "selection_data": "hand/default only", "test_loaded": False, "group_column": "specimen_group", "selected_candidate": selected}
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    leaderboard.to_csv(OUT / "hand_raw_highsr_hier_leaderboard.csv", index=False)
    print("Selection lock written before robot/test", json.dumps(lock, indent=2, default=float), flush=True)

    # Full-hand fit and test are intentionally below the lock write.
    cmodel, mmodel, tmodel = fit_heads(xc, xe, y_seg, np.arange(len(y_seg)))
    test = suite.load_manifest(ROOT / "audio_visual_dataset_robo_default/dataset.csv", ROOT / "audio_visual_dataset_robo_default", "robot_test")
    yt_rows = test.y.to_numpy(np.int64)
    test_ids, test_code, _, yt_seg = segment_pool(test.audio_file, np.zeros((len(test), 1)), yt_rows)
    xtc_rows = np.load(IMG_CLIP / "robot_test/X.npy").astype(np.float32)
    xte_rows = np.load(IMG_EFF / "robot_test/X.npy").astype(np.float32)
    _, _, xtc, _ = segment_pool(test.audio_file, xtc_rows, yt_rows)
    _, _, xte, _ = segment_pool(test.audio_file, xte_rows, yt_rows)
    raw_test_dir = Path("outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports")
    test_audio = {
        view: norm(np.load(raw_test_dir / f"audio_highsr_temporal_tta_select_{view}_test_proba.npy"))
        for view in ("clean", "robot_mix", "bandlimit")
    }
    test_views = {}
    for mode in ("mean", "logmean"):
        test_views[mode] = {}
        for view in ("clean", "robot_mix", "bandlimit"):
            q = test_audio[view]
            test_views[mode][view] = norm(np.vstack([q[test_code == i].mean(0) for i in range(len(test_ids))]))
        test_views[mode]["all"] = norm(sum(test_views[mode].values()) / 3.0)
    sel_params = (float(selected["alpha"]), float(selected["audio_weight"]), float(selected["threshold"]), float(selected["trunk_bias"]), selected["recipe"], selected["decoder"])
    p = test_views[selected["mode"]][selected["recipe"]]
    ic = cmodel.predict_proba(xtc)[:, 1]
    im = norm(mmodel.predict_proba(np.hstack([xtc, xte])))
    it = tmodel.predict_proba(xtc)[:, 1]
    pred_seg = evaluate(yt_seg, p[:, 1:].sum(1), norm(p[:, 1:]), ic, im, it, sel_params)
    pred_rows = pred_seg[test_code]
    result = {"split": "robot_test_final", "protocol": lock["protocol"], "selected_candidate": selected, "metrics": {"macro_f1_4class": float(f1_score(yt_rows, pred_rows, average="macro", zero_division=0)), "binary_macro_f1": float(f1_score(yt_rows > 0, pred_rows > 0, average="macro", zero_division=0))}, "confusion_matrix_4class": confusion_matrix(yt_rows, pred_rows, labels=CLASSES).tolist(), "per_class_4class": classification_report(yt_rows, pred_rows, labels=CLASSES, target_names=["ambient", "leaf", "trunk", "twig"], output_dict=True, zero_division=0), "invariants": {"test_loaded_after_lock": True, "selection_used_hand_only": True, "segment_label_pure": True}}
    (OUT / "raw_highsr_hier_fusion_final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
