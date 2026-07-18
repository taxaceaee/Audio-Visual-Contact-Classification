from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_stress_robust_group_selection as image_base
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_val_select_final_test as audio_base


OUT = Path("outputs")
ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
VIEWS = ["raw", "segment_lift"]


def main() -> None:
    report = OUT / "audio_feature_benchmarks/multimodal_exact_audio_lift_group_selection"
    report.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set("total240")
    train_df = audio_base.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train")
    y = train_df.y.to_numpy(dtype=np.int64)
    frame = suite.load_manifest(ROOT / "audio_visual_dataset_default/dataset.csv", ROOT / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = suite.specimen_group(frame.audio_file)
    if not np.array_equal(y, frame.y.to_numpy(dtype=np.int64)) or not np.array_equal(train_df.audio_file.astype(str).to_numpy(), frame.audio_file.astype(str).to_numpy()):
        raise AssertionError("manifest order mismatch")
    groups = frame.specimen_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))
    fold_summary = [{"fold": i, "train_rows": len(tr), "val_rows": len(va), "train_groups": int(pd.Series(groups[tr]).nunique()), "val_groups": int(pd.Series(groups[va]).nunique()), "overlap": len(set(groups[tr]) & set(groups[va]))} for i, (tr, va) in enumerate(folds)]
    if any(x["overlap"] for x in fold_summary):
        raise AssertionError("group overlap")
    (report / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")

    # Exact locked audio-only anchor used by audio_lift_source_blend.
    highsr = group_audio.load_highsr_oof(OUT)
    pair = specimen.load_pairwise_oof(OUT / "audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy")
    anchor = lift.anchor_lift_proba(train_df, highsr, pair)
    broad_sources = broad.load_train_sources(OUT, y, np.array([next(i for i, (_, va) in enumerate(folds) if j in va) for j in range(len(y))]), 42)
    audio_candidates = {"anchor_only": anchor}
    for name, source in broad_sources.items():
        for w in [0.05, 0.1, 0.2, 0.33, 0.5]:
            blended = lift.normalize((1.0 - w) * anchor + w * source)
            audio_candidates[f"anchor+{name}@{w:.2f}"] = lift.postprocess(train_df, blended, "segment_lift")
    audio_candidates["anchor_raw"] = anchor

    image_features = {
        "resnet18": OUT / "image_deep_features/resnet18_224",
        "efficientnet_b3": OUT / "image_timm_features/efficientnet_b3.ra2_in1k",
        "clip_vit_base": OUT / "image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k",
    }
    image_oof = {}
    for name, root in image_features.items():
        X = np.load(root / "hand_train_full/X.npy")
        yc = np.load(root / "hand_train_full/y.npy").astype(np.int64)
        if not np.array_equal(y, yc):
            raise AssertionError(f"image alignment {name}")
        p = np.zeros((len(y), 4), dtype=np.float64)
        for tr, va in folds:
            model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X[tr], y[tr])
            p[va] = image_base.image_proba(model, X[va])
        image_oof[name] = p

    rows = []
    for audio_name, pa in audio_candidates.items():
        for image_name, pi in image_oof.items():
            for iw in np.linspace(0.0, 1.0, 21):
                for kind in ["linear", "log"]:
                    if kind == "linear":
                        p = suite.normalize((1.0-iw) * pa + iw * pi)
                    else:
                        p = suite.normalize(np.exp((1.0-iw) * np.log(suite.normalize(pa)) + iw * np.log(suite.normalize(pi))))
                    fs = [f1_score(y[va], p[va].argmax(1), average="macro", zero_division=0) for _, va in folds]
                    rows.append({"audio_candidate": audio_name, "image_backbone": image_name, "image_weight": float(iw), "kind": kind, "mean_fold_macro_f1": float(np.mean(fs)), "worst_fold_macro_f1": float(np.min(fs)), "std_fold_macro_f1": float(np.std(fs))})
    leaderboard = pd.DataFrame(rows).sort_values(["worst_fold_macro_f1", "mean_fold_macro_f1", "std_fold_macro_f1"], ascending=[False, False, True]).reset_index(drop=True)
    leaderboard.to_csv(report / "hand_exact_audio_lift_multimodal_leaderboard.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "multimodal_exact_audio_lift_specimen_group_hand_only", "group_column": "specimen_group", "n_folds": 5, "test_loaded": False, "candidate_count": len(rows), "selected_candidate": best}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
