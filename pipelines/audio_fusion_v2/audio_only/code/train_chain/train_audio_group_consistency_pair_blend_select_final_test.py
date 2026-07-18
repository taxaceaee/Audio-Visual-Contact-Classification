from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_highsr_temporal_tta_select_final_test as highsr
import train_audio_pairwise_contact_stress_cv_select_final_test as pairwise
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
HIGHSR_WEIGHT_CANDIDATES = [0.65, 0.70, 0.75, 0.80]
GROUP_RULES = ["mean_proba", "sum_log_proba"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only segment-consistency blend. Rebuilds pairwise contact OOF, "
            "blends it with locked high-SR OOF, selects group decoding by train OOF, "
            "then loads robot/test after the selection lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-slug", default="audio_group_consistency_pair_blend_select")
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
    parser.add_argument("--force-rebuild-pairwise-oof", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-12, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray = LABELS) -> float:
    scores = []
    for label in labels:
        true_mask = y_true == label
        pred_mask = pred == label
        tp = float(np.sum(true_mask & pred_mask))
        fp = float(np.sum(~true_mask & pred_mask))
        fn = float(np.sum(true_mask & ~pred_mask))
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {base.ID2LABEL[int(label)]: int(np.sum(y == label)) for label in LABELS}


def group_decode(frame: pd.DataFrame, proba: np.ndarray, rule: str) -> np.ndarray:
    group_codes, _ = pd.factorize(frame["group_key"].astype(str), sort=False)
    n_groups = int(group_codes.max()) + 1
    values = np.log(np.clip(proba, 1e-12, 1.0)) if rule == "sum_log_proba" else proba
    sums = np.vstack(
        [np.bincount(group_codes, weights=values[:, class_id], minlength=n_groups) for class_id in LABELS]
    ).T
    if rule == "mean_proba":
        counts = np.bincount(group_codes, minlength=n_groups).astype(np.float64)
        sums = sums / counts[:, None]
    elif rule != "sum_log_proba":
        raise KeyError(f"Unknown group rule: {rule}")
    group_pred = np.argmax(sums, axis=1).astype(np.int64)
    return group_pred[group_codes]


def build_splits_from_assignment(fold_assignment: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(fold_assignment))
    splits = []
    for fold_id in sorted(set(fold_assignment.tolist())):
        val_idx = indices[fold_assignment == fold_id]
        train_idx = indices[fold_assignment != fold_id]
        splits.append((train_idx, val_idx))
    return splits


def load_highsr_oof(output: Path) -> np.ndarray:
    run = output / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    lock = json.loads((run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    candidate = str(lock["selected_candidate"])
    weights = lock["selected_tta_weights"]
    oof_by_view = {
        view: np.load(run / "oof_proba" / candidate / f"{view}_oof_proba.npy")
        for view in highsr.STRESS_VIEWS
    }
    return normalize(highsr.weighted_proba(oof_by_view, weights))


def load_or_rebuild_pairwise_oof(
    args: argparse.Namespace,
    run_dir: Path,
    train_df: pd.DataFrame,
    y: np.ndarray,
    fold_assignment: np.ndarray,
) -> np.ndarray:
    cache_path = run_dir / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force_rebuild_pairwise_oof:
        print(f"Loaded cached pairwise OOF: {cache_path}")
        return normalize(np.load(cache_path))

    payloads = {}
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    payloads["clean"] = clean_feat
    for view in pairwise.STRESS_VIEWS:
        if view == "clean":
            continue
        payload, _ = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.stress_feature_dir,
            force_rebuild=False,
        )
        payloads[view] = payload
    for view in pairwise.STRESS_VIEWS:
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")
    X_by_view = {view: payloads[view]["X"] for view in pairwise.STRESS_VIEWS}

    pair_lock = json.loads(
        (
            args.output
            / "audio_feature_benchmarks"
            / "audio_pairwise_contact_stress_cv_select"
            / "reports"
            / "audio_pairwise_contact_stress_cv_select_selected_without_test.json"
        ).read_text(encoding="utf-8")
    )
    selected_candidate = str(pair_lock["selected_candidate"])
    candidates = pairwise.make_candidates(args.random_state)
    spec = candidates[selected_candidate]

    oof = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    splits = build_splits_from_assignment(fold_assignment)
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = pairwise.fit_pairwise_candidate(spec, X_by_view, y, train_idx)
        oof[val_idx] = pairwise.predict_pairwise_artifact(artifact, X_by_view["clean"][val_idx])
        print(
            f"  rebuilt pairwise fold {fold_id}: val={len(val_idx)} time={time.perf_counter() - start:.2f}s",
            flush=True,
        )
    oof = normalize(oof)
    np.save(cache_path, oof)
    return oof


def evaluate_candidates(
    frame: pd.DataFrame,
    y: np.ndarray,
    highsr_oof: np.ndarray,
    pairwise_oof: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for highsr_weight in HIGHSR_WEIGHT_CANDIDATES:
        pairwise_weight = 1.0 - highsr_weight
        blended = normalize(highsr_weight * highsr_oof + pairwise_weight * pairwise_oof)
        for rule in GROUP_RULES:
            pred = group_decode(frame, blended, rule)
            macro = fast_macro_f1(y, pred, LABELS)
            contact = fast_macro_f1(y, pred, CONTACT_LABELS)
            binary = fast_macro_f1((y > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
            rows.append(
                {
                    "recipe_kind": "audio_segment_consistency_pair_blend",
                    "highsr_weight": float(highsr_weight),
                    "pairwise_weight": float(pairwise_weight),
                    "group_rule": rule,
                    "macro_f1": macro,
                    "contact_macro_f1": contact,
                    "binary_macro_f1": binary,
                    "selection_score": float(0.65 * macro + 0.25 * contact + 0.10 * binary),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)


def load_final_frame_and_proba(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve())
    print("RUN_DIR   =", run_dir.resolve())
    print("Protocol  = audio-only OOF-selected segment consistency; robot/test after lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    y = clean_feat["y"]
    pair_split_path = (
        args.output
        / "audio_feature_benchmarks"
        / "audio_pairwise_contact_stress_cv_select"
        / "splits"
        / "hand_train_full_pairwise_contact_stress_cv_folds.csv"
    )
    fold_assignment = pd.read_csv(pair_split_path)["cv_fold"].to_numpy(dtype=np.int64)

    highsr_oof = load_highsr_oof(args.output)
    pairwise_oof = load_or_rebuild_pairwise_oof(args, run_dir, train_df, y, fold_assignment)
    leaderboard = evaluate_candidates(train_df, y, highsr_oof, pairwise_oof)

    method_card = {
        "protocol": "audio_only_segment_consistency_pair_blend_no_test_until_lock",
        "allowed_selection_data": "hand/default train labels, train group_key, locked high-SR OOF, rebuilt pairwise OOF",
        "forbidden_selection_data": "robot/test labels or robot/test predictions before selection lock; image/multimodal features",
        "candidate_count": int(len(leaderboard)),
        "highsr_weight_candidates": HIGHSR_WEIGHT_CANDIDATES,
        "group_rules": GROUP_RULES,
        "label_counts": label_counts(y),
        "train_groups": int(train_df["group_key"].nunique()),
        "pairwise_split_path": str(pair_split_path.resolve()),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_group_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    selection_summary = {
        "selection_rule": "best train-only OOF segment-consistency blend of high-SR and pairwise audio sources",
        "selected_without_test": selected,
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    final_root = args.output / "audio_feature_benchmarks"
    highsr_frame, highsr_final = load_final_frame_and_proba(
        final_root
        / "audio_highsr_temporal_tta_select"
        / "reports"
        / "audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pair_frame, pair_final = load_final_frame_and_proba(
        final_root
        / "audio_pairwise_contact_stress_cv_select"
        / "reports"
        / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    for column in ["audio_file", "y"]:
        if not np.array_equal(highsr_frame[column].astype(str).to_numpy(), pair_frame[column].astype(str).to_numpy()):
            raise AssertionError(f"Final source frames are not aligned on {column}")

    final_blend = normalize(float(selected["highsr_weight"]) * highsr_final + float(selected["pairwise_weight"]) * pair_final)
    final_pred = group_decode(highsr_frame, final_blend, str(selected["group_rule"]))
    y_test = highsr_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_group_consistency_highsr_pairwise_blend",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_segment_consistency",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_highsr_weight": selected["highsr_weight"],
            "selected_pairwise_weight": selected["pairwise_weight"],
            "selected_group_rule": selected["group_rule"],
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = highsr_frame[
        [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in highsr_frame]
    ].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_blend[:, class_id]
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

    print("\nFinal robot/test result after frozen audio segment-consistency blend:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_highsr_weight",
                "selected_pairwise_weight",
                "selected_group_rule",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Method card:", method_card_path.resolve())
    print("Leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
