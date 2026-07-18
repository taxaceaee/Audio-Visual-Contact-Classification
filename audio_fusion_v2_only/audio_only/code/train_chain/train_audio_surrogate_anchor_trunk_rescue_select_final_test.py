from __future__ import annotations

"""Surrogate-anchor trunk-rescue selection (audio-only, non-leaky).

Problem this attacks: the trunk->ambient failure mode of the locked 0.7027
anchor only exists under robot domain shift and never appears on hand/default
holdouts, so no direct hand-only validation can certify a rescue threshold.

Surrogate idea:
1. Train a cheap 4-class PROXY model on the domain-holdout TRAIN rows using
   clean + stress views.
2. Predict the holdout VAL rows under heavy robot-like views
   (`robot_heavy`, `robot_mix`, `bandlimit`). Under this forced shift the
   proxy exhibits trunk->ambient/twig collapse similar to the robot test.
3. Select the trunk-rescue rule (detector, rescue-from, threshold) that best
   repairs the proxy's shifted predictions, measured by WORST-view macro-F1.
4. Guard: the same rule applied to the REAL anchor's holdout-val predictions
   (clean view) must not reduce macro-F1 or leaf-F1 beyond small margins.
5. Lock the rule BEFORE loading robot/test, refit the detector on full train,
   and apply the rule to the real locked anchor pipeline on robot/test.

Selection uses only hand/default data. No image features. No class words
parsed from filenames for prediction.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score

import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_domain_holdout_select_final_test as domain
import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_specimen_contact_consensus_select_final_test as spec
import train_audio_trunk_rescue_domain_robust_select_final_test as rescue
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
TRUNK_ID = 2

SURROGATE_VIEWS = ("robot_heavy", "robot_mix", "bandlimit")
RESCUE_THRESHOLDS = (0.45, 0.55, 0.65, 0.75, 0.85)
RESCUE_TARGETS: dict[str, tuple[int, ...]] = {
    "ambient": (0,),
    "twig": (3,),
    "ambient_twig": (0, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Surrogate-anchor trunk-rescue selection. A proxy 4-class model is "
            "shifted with heavy robot-like views to reproduce trunk->ambient "
            "errors on a hand/default holdout; the rescue rule that repairs the "
            "proxy (and does not hurt the real anchor on clean holdout) is "
            "locked, then applied once to robot/test."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_surrogate_anchor_trunk_rescue_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--leaf-val-frac", type=float, default=0.28)
    parser.add_argument("--letwig-val-frac", type=float, default=0.18)
    parser.add_argument("--anchor-guard-macro-drop", type=float, default=0.002)
    parser.add_argument("--anchor-guard-leaf-drop", type=float, default=0.01)
    parser.add_argument(
        "--clean-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument(
        "--heavy-feature-dir",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_trunk_rescue_domain_robust_select/heavy_features"
        ),
    )
    parser.add_argument(
        "--pairwise-oof-cache",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def make_proxy_model(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        class_weight={0: 0.6, 1: 1.1, 2: 1.8, 3: 1.3},
        random_state=random_state,
    )


def macro_and_leaf(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    macro = float(f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0))
    contact = float(f1_score(y_true, pred, labels=CONTACT_LABELS, average="macro", zero_division=0))
    per_class = f1_score(y_true, pred, labels=LABELS, average=None, zero_division=0)
    return macro, contact, float(per_class[1])


def confusion_text(y_true: np.ndarray, pred: np.ndarray) -> str:
    matrix = confusion_matrix(y_true, pred, labels=LABELS)
    lines = []
    for row_id, row in zip(LABELS, matrix):
        cells = ", ".join(f"{base.ID2LABEL[int(col_id)]} {int(count)}" for col_id, count in zip(LABELS, row) if count)
        lines.append(f"    {base.ID2LABEL[int(row_id)]:>7} -> {cells}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve())
    print("RUN_DIR   =", run_dir.resolve())
    print("Protocol  = surrogate-anchor trunk-rescue selection; robot/test after lock")

    # ---- hand/default train only from here ----
    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)
    train_df["prefix_alpha"] = train_df["audio_file"].map(domain.prefix_alpha)

    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    y = clean_feat["y"]

    X_by_view: dict[str, np.ndarray] = {"clean": clean_feat["X"]}
    for view in ("robot_mix", "bandlimit"):
        payload, _ = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.stress_feature_dir,
            force_rebuild=False,
        )
        X_by_view[view] = payload["X"]
    for view in rescue.HEAVY_VIEWS:
        payload = rescue.build_or_load_heavy_cache(
            train_df,
            view=view,
            heavy_feature_dir=args.heavy_feature_dir,
            force_rebuild=False,
        )
        X_by_view[view] = payload["X"]

    # Locked anchor baseline on train OOF probabilities.
    split_path = (
        args.output
        / "audio_feature_benchmarks"
        / "audio_tta_grid_hgb_select"
        / "splits"
        / "hand_train_full_hgb_tta_grid_folds.csv"
    )
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)
    train_sources = broad.load_train_sources(args.output, y, fold_assignment, args.random_state)
    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = spec.load_pairwise_oof(args.pairwise_oof_cache)
    anchor_proba = rescue.locked_anchor_pipeline(
        train_df, highsr_oof, pairwise_oof, train_sources[rescue.ANCHOR_SOURCE_NAME]
    )
    anchor_pred = anchor_proba.argmax(axis=1).astype(np.int64)

    # Domain-holdout split.
    train_idx, val_idx, split_info = domain.make_domain_holdout_split(
        train_df,
        args.random_state,
        args.leaf_val_frac,
        args.letwig_val_frac,
    )
    split_df = train_df.copy()
    split_df["domain_split"] = "train"
    split_df.loc[val_idx, "domain_split"] = "val"
    split_df.to_csv(split_dir / "hand_train_domain_holdout_split.csv", index=False)

    y_val = y[val_idx]
    val_frame = train_df.iloc[val_idx].reset_index(drop=True)
    anchor_pred_val = anchor_pred[val_idx]
    anchor_macro, anchor_contact, anchor_leaf = macro_and_leaf(y_val, anchor_pred_val)
    print(f"\nReal anchor on holdout-val (clean): macro={anchor_macro:.4f} leaf={anchor_leaf:.4f}")

    # ---- Surrogate proxy: 4-class model shifted by heavy views ----
    proxy_views = ("clean", "robot_mix", "bandlimit")
    X_proxy_train = np.vstack([X_by_view[view][train_idx] for view in proxy_views])
    y_proxy_train = np.concatenate([y[train_idx] for _ in proxy_views])
    proxy = make_proxy_model(args.random_state)
    start = time.perf_counter()
    proxy.fit(X_proxy_train, y_proxy_train)
    print(f"Proxy fit on holdout-train ({X_proxy_train.shape[0]} rows) in {time.perf_counter() - start:.1f}s")

    proxy_pred_by_view: dict[str, np.ndarray] = {}
    proxy_base_scores: dict[str, tuple[float, float, float]] = {}
    for view in SURROGATE_VIEWS:
        proba = cv.proba_aligned(proxy, X_by_view[view][val_idx])
        proba = rescue.segment_lift_postprocess(val_frame, rescue.normalize(proba))
        pred = proba.argmax(axis=1).astype(np.int64)
        proxy_pred_by_view[view] = pred
        proxy_base_scores[view] = macro_and_leaf(y_val, pred)
        trunk_mask = y_val == TRUNK_ID
        trunk_to_ambient = int(((pred == 0) & trunk_mask).sum())
        trunk_to_twig = int(((pred == 3) & trunk_mask).sum())
        print(
            f"Proxy under {view}: macro={proxy_base_scores[view][0]:.4f} "
            f"trunk->ambient={trunk_to_ambient} trunk->twig={trunk_to_twig} "
            f"(trunk support {int(trunk_mask.sum())})"
        )
        print(confusion_text(y_val, pred))

    proxy_worst_base = float(min(score[0] for score in proxy_base_scores.values()))

    # ---- Detectors and rescue-rule grid ----
    factories = rescue.make_detector_factories(args.random_state)
    detector_specs = rescue.make_detector_specs()
    leaderboard_rows = [
        {
            "candidate": "anchor_baseline",
            "detector": "none",
            "rescue_from": "none",
            "threshold": np.nan,
            "eligible": True,
            "surrogate_worst_macro_after": proxy_worst_base,
            "surrogate_mean_macro_after": float(np.mean([score[0] for score in proxy_base_scores.values()])),
            "surrogate_worst_improvement": 0.0,
            "anchor_macro_after": anchor_macro,
            "anchor_leaf_after": anchor_leaf,
        }
    ]

    for detector in detector_specs.values():
        start = time.perf_counter()
        model = rescue.fit_detector(detector, factories, X_by_view, y, train_idx)
        fit_time = time.perf_counter() - start
        seg_trunk_by_view: dict[str, np.ndarray] = {}
        for view in SURROGATE_VIEWS + ("clean",):
            trunk_window = rescue.trunk_probability(model, X_by_view[view][val_idx])
            seg_trunk_by_view[view] = rescue.segment_mean_trunk(val_frame, trunk_window)
        print(f"\nDetector {detector.name} views={detector.train_views} fit={fit_time:.1f}s")
        for target_name, target_ids in RESCUE_TARGETS.items():
            for threshold in RESCUE_THRESHOLDS:
                macro_after_by_view = {}
                for view in SURROGATE_VIEWS:
                    fixed = rescue.apply_trunk_rescue(
                        proxy_pred_by_view[view], seg_trunk_by_view[view], target_ids, threshold
                    )
                    macro_after_by_view[view], _, _ = macro_and_leaf(y_val, fixed)
                worst_after = float(min(macro_after_by_view.values()))
                mean_after = float(np.mean(list(macro_after_by_view.values())))
                worst_improvement = float(
                    min(macro_after_by_view[view] - proxy_base_scores[view][0] for view in SURROGATE_VIEWS)
                )

                anchor_fixed = rescue.apply_trunk_rescue(
                    anchor_pred_val, seg_trunk_by_view["clean"], target_ids, threshold
                )
                anchor_macro_after, _, anchor_leaf_after = macro_and_leaf(y_val, anchor_fixed)

                eligible = (
                    worst_improvement > 0.0
                    and anchor_macro_after >= anchor_macro - args.anchor_guard_macro_drop
                    and anchor_leaf_after >= anchor_leaf - args.anchor_guard_leaf_drop
                )
                leaderboard_rows.append(
                    {
                        "candidate": f"{detector.name}__{target_name}__t{threshold:.2f}",
                        "detector": detector.name,
                        "rescue_from": target_name,
                        "threshold": float(threshold),
                        "eligible": bool(eligible),
                        "detector_fit_time_sec": fit_time,
                        "surrogate_worst_macro_after": worst_after,
                        "surrogate_mean_macro_after": mean_after,
                        "surrogate_worst_improvement": worst_improvement,
                        "anchor_macro_after": anchor_macro_after,
                        "anchor_leaf_after": anchor_leaf_after,
                        **{f"surrogate_{view}_macro_after": macro_after_by_view[view] for view in SURROGATE_VIEWS},
                    }
                )
                print(
                    f"  {target_name:>12} t={threshold:.2f}: surrogate_worst_after={worst_after:.4f} "
                    f"worst_impr={worst_improvement:+.4f} anchor_after={anchor_macro_after:.4f} "
                    f"anchor_leaf={anchor_leaf_after:.4f} eligible={eligible}"
                )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["eligible", "surrogate_worst_macro_after", "surrogate_mean_macro_after", "anchor_macro_after"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_surrogate_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    eligible_board = leaderboard[leaderboard["eligible"] & (leaderboard["candidate"] != "anchor_baseline")]
    selected = eligible_board.iloc[0].to_dict() if len(eligible_board) else leaderboard_rows[0]

    method_card = {
        "protocol": "audio_only_surrogate_anchor_trunk_rescue_no_test_until_lock",
        "anchor_recipe": {
            "source_name": rescue.ANCHOR_SOURCE_NAME,
            "source_weight": rescue.ANCHOR_SOURCE_WEIGHT,
            "blend_mode": rescue.ANCHOR_BLEND_MODE,
        },
        "surrogate_views": list(SURROGATE_VIEWS),
        "proxy_model": "direct HGB total240, trained on holdout-train clean+robot_mix+bandlimit",
        "selection_rule": (
            "maximize worst-view macro-F1 of rescued surrogate proxy; require positive "
            "worst-view improvement AND real-anchor clean-holdout macro/leaf guards"
        ),
        "anchor_guard_macro_drop": float(args.anchor_guard_macro_drop),
        "anchor_guard_leaf_drop": float(args.anchor_guard_leaf_drop),
        "allowed_selection_data": "hand/default labels, locked OOF probabilities, hand/default domain holdout",
        "forbidden_selection_data": "robot/test labels or probabilities, image/multimodal features, filename class words",
        "known_risk": "surrogate/anchor mismatch: rules that fix the proxy may not transfer to the real anchor",
        "candidate_count": int(len(leaderboard)),
        "split_info": split_info,
        "proxy_base_scores": {view: proxy_base_scores[view][0] for view in SURROGATE_VIEWS},
        "real_anchor_holdout_macro": anchor_macro,
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)

    selection_summary = {
        "selected_without_test": selected,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "method_card": str(method_card_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(
        json.dumps(
            {
                key: selected.get(key)
                for key in [
                    "candidate",
                    "detector",
                    "rescue_from",
                    "threshold",
                    "surrogate_worst_macro_after",
                    "surrogate_worst_improvement",
                    "anchor_macro_after",
                ]
            },
            indent=2,
            default=float,
        ),
        flush=True,
    )

    # ---- robot/test only from here ----
    test_frame, final_sources = broad.load_final_sources(broad.final_prediction_paths(args.output))
    final_root = args.output / "audio_feature_benchmarks"
    highsr_frame = pd.read_csv(
        final_root / "audio_highsr_temporal_tta_select" / "reports" / "audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pair_frame = pd.read_csv(
        final_root / "audio_pairwise_contact_stress_cv_select" / "reports" / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    proba_columns = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
    highsr_final = rescue.normalize(highsr_frame[proba_columns].to_numpy(dtype=np.float64))
    pair_final = rescue.normalize(pair_frame[proba_columns].to_numpy(dtype=np.float64))
    if not np.array_equal(highsr_frame["audio_file"].astype(str).to_numpy(), test_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("High-SR final frame is not aligned with source frame")

    final_baseline_proba = rescue.locked_anchor_pipeline(
        test_frame, highsr_final, pair_final, final_sources[rescue.ANCHOR_SOURCE_NAME]
    )
    final_baseline_pred = final_baseline_proba.argmax(axis=1).astype(np.int64)

    selected_candidate = str(selected["candidate"])
    if selected_candidate == "anchor_baseline":
        final_pred = final_baseline_pred
        final_seg_trunk = np.zeros(len(final_baseline_pred), dtype=np.float64)
        final_detector_artifact = None
    else:
        test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
        test_df = base.load_manifest(test_csv, "robot_test")
        if not np.array_equal(test_df["audio_file"].astype(str).to_numpy(), test_frame["audio_file"].astype(str).to_numpy()):
            raise AssertionError("Robot/test manifest is not aligned with final source frame")
        test_feat, _ = base.build_or_load_feature_cache(
            test_df,
            "robot_test",
            feature_dir=args.clean_feature_cache_dir,
            force_rebuild=False,
        )
        detector = detector_specs[str(selected["detector"])]
        full_idx = np.arange(len(y))
        start = time.perf_counter()
        final_detector_artifact = rescue.fit_detector(detector, factories, X_by_view, y, full_idx)
        print(f"Refit final detector {detector.name} on full train in {time.perf_counter() - start:.1f}s")
        trunk_window_test = rescue.trunk_probability(final_detector_artifact, test_feat["X"])
        final_seg_trunk = rescue.segment_mean_trunk(test_frame, trunk_window_test)
        target_ids = RESCUE_TARGETS[str(selected["rescue_from"])]
        final_pred = rescue.apply_trunk_rescue(
            final_baseline_pred, final_seg_trunk, target_ids, float(selected["threshold"])
        )

    y_test = test_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_surrogate_anchor_trunk_rescue",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    baseline_macro = float(f1_score(y_test, final_baseline_pred, labels=LABELS, average="macro", zero_division=0))
    final_row.update(
        {
            "selected_by": "hand_default_surrogate_anchor_worst_view",
            "selected_candidate": selected_candidate,
            "selected_detector": selected.get("detector"),
            "selected_rescue_from": selected.get("rescue_from"),
            "selected_threshold": selected.get("threshold"),
            "selected_surrogate_worst_macro_after": selected.get("surrogate_worst_macro_after"),
            "selected_surrogate_worst_improvement": selected.get("surrogate_worst_improvement"),
            "baseline_test_macro_f1": baseline_macro,
            "n_rescued": int((final_pred != final_baseline_pred).sum()),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_frame[[column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_frame]].copy()
    prediction_frame["baseline_pred_y"] = final_baseline_pred.astype(int)
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    prediction_frame["segment_trunk_proba"] = final_seg_trunk
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_baseline_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(y_test, final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": method_card["protocol"],
            "method_card": method_card,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "final_detector_artifact": final_detector_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": method_card["protocol"],
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "method_card": method_card,
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "method_card": str(method_card_path.resolve()),
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen surrogate-anchor rescue selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "baseline_test_macro_f1",
                "selected_candidate",
                "n_rescued",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
