from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_highsr_contact_source_consensus_select_final_test as hsrc
import train_audio_lift_source_blend_select_final_test as lift_blend
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]

BLEND_WEIGHTS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.33, 0.50]
TRUNK_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
TRUNK_MARGINS = [-0.10, 0.00, 0.10, 0.20, 0.30, 0.40]
MAX_ANCHOR_LEAF = [0.20, 0.35, 0.50, 0.65]
ANCHOR_ALLOWED = ["ambient_twig", "not_leaf", "all_nontrunk"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only anchor + high-SR contact-source guarded correction. "
            "The correction/blend rule is selected using hand/default OOF only; "
            "robot/test is loaded after selected_without_test.json is written."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_anchor_hsrc_guard_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--pairwise-oof-cache",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        ),
    )
    parser.add_argument(
        "--highsr-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/features"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray = LABELS) -> float:
    return specimen.fast_macro_f1(y_true, pred, labels)


def binary_macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    return fast_macro_f1(
        (y_true > 0).astype(np.int64),
        (pred > 0).astype(np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )


def score_prediction(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = fast_macro_f1(y, pred, LABELS)
    contact = fast_macro_f1(y, pred, CONTACT_LABELS)
    binary = binary_macro_f1(y, pred)
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "binary_macro_f1": binary,
        "selection_score": float(0.60 * macro + 0.30 * contact + 0.10 * binary),
    }


def load_current_anchor_oof(
    output: Path,
    train_df: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
    pairwise_oof_cache: Path,
) -> np.ndarray:
    split_path = output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select" / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)
    train_sources = broad.load_train_sources(output, y, fold_assignment, random_state)
    highsr_oof = group_impl.load_highsr_oof(output)
    pairwise_oof = specimen.load_pairwise_oof(pairwise_oof_cache)
    anchor = lift_blend.anchor_lift_proba(train_df, highsr_oof, pairwise_oof)

    lock_path = (
        output
        / "audio_feature_benchmarks"
        / "audio_lift_source_blend_select"
        / "reports"
        / "audio_lift_source_blend_select_selected_without_test.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))["selected_without_test"]
    source_name = str(lock["source_name"])
    if source_name != "anchor_only":
        anchor = normalize((1.0 - float(lock["source_weight"])) * anchor + float(lock["source_weight"]) * train_sources[source_name])
    return lift_blend.postprocess(train_df, anchor, str(lock["blend_mode"]))


def selected_hsrc_oof(
    output: Path,
    feature_dir: Path,
    pairwise_oof_cache: Path,
    train_df: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
) -> np.ndarray:
    split_path = (
        output
        / "audio_feature_benchmarks"
        / "audio_highsr_temporal_tta_select"
        / "splits"
        / "hand_train_full_highsr_temporal_folds.csv"
    )
    split_df = pd.read_csv(split_path)
    if not np.array_equal(split_df["audio_file"].astype(str).to_numpy(), train_df["audio_file"].astype(str).to_numpy()):
        raise AssertionError("High-SR split manifest is not aligned with hand/default train manifest")
    fold_assignment = split_df["cv_fold"].to_numpy(dtype=np.int64)

    lock_path = (
        output
        / "audio_feature_benchmarks"
        / "audio_highsr_contact_source_consensus_select"
        / "reports"
        / "audio_highsr_contact_source_consensus_select_selected_without_test.json"
    )
    selected = json.loads(lock_path.read_text(encoding="utf-8"))["selected_without_test"]
    contact_source = str(selected["contact_source"])
    contact_oof = None
    if contact_source != "none":
        X_by_view, y_features = hsrc.load_highsr_features(feature_dir, "hand_train_full")
        if not np.array_equal(y, y_features):
            raise AssertionError("High-SR feature labels are not aligned with hand/default train labels")
        contact_specs = hsrc.contact_candidates(random_state)
        if contact_source not in contact_specs:
            raise KeyError(f"Unknown locked high-SR contact source: {contact_source}")
        contact_oof_dict, _ = hsrc.build_contact_oof(
            {contact_source: contact_specs[contact_source]},
            X_by_view,
            y,
            fold_assignment,
        )
        contact_oof = contact_oof_dict[contact_source]

    highsr_oof = group_impl.load_highsr_oof(output)
    pairwise_oof = specimen.load_pairwise_oof(pairwise_oof_cache)
    base_window = normalize(float(selected["highsr_weight"]) * highsr_oof + float(selected["pairwise_weight"]) * pairwise_oof)
    segment, window_to_segment = hsrc.build_segment_proba_with_contact_source(
        train_df,
        base_window,
        contact_oof,
        float(selected["contact_source_weight"]),
        str(selected["contact_segment_rule"]),
    )
    specimen_codes = specimen.specimen_codes_for_segments(train_df, window_to_segment)
    segment = specimen.apply_specimen_contact_consensus(
        segment,
        specimen_codes,
        threshold=float(selected["contact_threshold"]),
        alpha=float(selected["consensus_alpha"]),
        min_contact_segments=int(selected["min_contact_segments"]),
        agg_rule=str(selected["agg_rule"]),
    )
    return normalize(segment[window_to_segment])


def allowed_anchor_mask(anchor_pred: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ambient_twig":
        return np.isin(anchor_pred, [0, 3])
    if mode == "not_leaf":
        return anchor_pred != 1
    if mode == "all_nontrunk":
        return anchor_pred != 2
    raise KeyError(f"Unknown anchor allowed mode: {mode}")


def apply_candidate(anchor: np.ndarray, hsrc_proba: np.ndarray, candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    anchor = normalize(anchor)
    hsrc_proba = normalize(hsrc_proba)
    kind = str(candidate["candidate_kind"])
    if kind == "anchor_only":
        proba = anchor
        return proba, proba.argmax(axis=1).astype(np.int64)
    if kind == "convex_blend":
        weight = float(candidate["hsrc_weight"])
        proba = normalize((1.0 - weight) * anchor + weight * hsrc_proba)
        return proba, proba.argmax(axis=1).astype(np.int64)
    if kind == "guarded_trunk_rescue":
        pred = anchor.argmax(axis=1).astype(np.int64)
        proba = anchor.copy()
        trunk_margin = hsrc_proba[:, 2] - np.maximum.reduce([hsrc_proba[:, 0], hsrc_proba[:, 1], hsrc_proba[:, 3]])
        mask = (
            (hsrc_proba[:, 2] >= float(candidate["hsrc_trunk_threshold"]))
            & (trunk_margin >= float(candidate["hsrc_trunk_margin"]))
            & (anchor[:, 1] <= float(candidate["max_anchor_leaf_proba"]))
            & allowed_anchor_mask(pred, str(candidate["anchor_allowed"]))
        )
        proba[mask] = np.asarray([1e-12, 1e-12, 1.0, 1e-12], dtype=np.float64)
        return normalize(proba), normalize(proba).argmax(axis=1).astype(np.int64)
    raise KeyError(f"Unknown candidate kind: {kind}")


def candidate_grid() -> list[dict]:
    rows = [
        {
            "candidate_kind": "anchor_only",
            "hsrc_weight": 0.0,
            "hsrc_trunk_threshold": None,
            "hsrc_trunk_margin": None,
            "max_anchor_leaf_proba": None,
            "anchor_allowed": "none",
        }
    ]
    for weight in BLEND_WEIGHTS:
        rows.append(
            {
                "candidate_kind": "convex_blend",
                "hsrc_weight": float(weight),
                "hsrc_trunk_threshold": None,
                "hsrc_trunk_margin": None,
                "max_anchor_leaf_proba": None,
                "anchor_allowed": "all",
            }
        )
    for threshold in TRUNK_THRESHOLDS:
        for margin in TRUNK_MARGINS:
            for max_leaf in MAX_ANCHOR_LEAF:
                for allowed in ANCHOR_ALLOWED:
                    rows.append(
                        {
                            "candidate_kind": "guarded_trunk_rescue",
                            "hsrc_weight": 1.0,
                            "hsrc_trunk_threshold": float(threshold),
                            "hsrc_trunk_margin": float(margin),
                            "max_anchor_leaf_proba": float(max_leaf),
                            "anchor_allowed": allowed,
                        }
                    )
    return rows


def evaluate_candidates(y: np.ndarray, anchor: np.ndarray, hsrc_proba: np.ndarray) -> pd.DataFrame:
    rows = []
    anchor_pred = anchor.argmax(axis=1).astype(np.int64)
    for spec in candidate_grid():
        _, pred = apply_candidate(anchor, hsrc_proba, spec)
        scores = score_prediction(y, pred)
        changed = int(np.sum(pred != anchor_pred))
        rows.append(
            {
                "recipe_kind": "audio_anchor_hsrc_guard",
                **spec,
                **scores,
                "changed_from_anchor": changed,
                "changed_fraction": float(changed / len(pred)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "binary_macro_f1", "changed_fraction"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def load_prediction_proba(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))


def main() -> None:
    args = parse_args()
    base.configure_feature_set("total240")
    root_path = base.resolve_root(args.root)
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve())
    print("RUN_DIR   =", run_dir.resolve())
    print("Protocol  = audio-only guarded correction; robot/test after lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    y = train_df["y"].to_numpy(dtype=np.int64)
    anchor_oof = load_current_anchor_oof(args.output, train_df, y, args.random_state, args.pairwise_oof_cache)
    hsrc_oof = selected_hsrc_oof(args.output, args.highsr_feature_dir, args.pairwise_oof_cache, train_df, y, args.random_state)

    leaderboard = evaluate_candidates(y, anchor_oof, hsrc_oof)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()

    method_card = {
        "protocol": "audio_only_anchor_hsrc_guard_no_test_until_lock",
        "allowed_selection_data": "hand/default labels, locked OOF anchor probabilities, locked high-SR contact-source OOF probabilities",
        "forbidden_selection_data": "robot/test labels, robot/test predictions before selection lock, image/multimodal features, parsing class/contact words from filenames",
        "anchor_checkpoint": "checkpoints/audio_only_paper_safe_current_0702672_20260706",
        "extra_source": "audio_highsr_contact_source_consensus_select selected by its own hand-OOF lock",
        "candidate_count": int(len(leaderboard)),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact score over anchor plus guarded high-SR contact-source candidates",
        "selected_without_test": selected,
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    final_root = args.output / "audio_feature_benchmarks"
    anchor_frame, anchor_final = load_prediction_proba(
        final_root
        / "audio_lift_source_blend_select"
        / "reports"
        / "audio_lift_source_blend_select_final_test_predictions.csv"
    )
    hsrc_frame, hsrc_final = load_prediction_proba(
        final_root
        / "audio_highsr_contact_source_consensus_select"
        / "reports"
        / "audio_highsr_contact_source_consensus_select_final_test_predictions.csv"
    )
    if not np.array_equal(anchor_frame["audio_file"].astype(str).to_numpy(), hsrc_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("Final anchor and high-SR contact-source predictions are not aligned")

    final_proba, final_pred = apply_candidate(anchor_final, hsrc_final, selected)
    y_test = anchor_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_anchor_hsrc_guard",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_anchor_hsrc_guard",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_candidate_kind": selected["candidate_kind"],
            "selected_hsrc_weight": selected["hsrc_weight"],
            "selected_hsrc_trunk_threshold": selected["hsrc_trunk_threshold"],
            "selected_hsrc_trunk_margin": selected["hsrc_trunk_margin"],
            "selected_max_anchor_leaf_proba": selected["max_anchor_leaf_proba"],
            "selected_anchor_allowed": selected["anchor_allowed"],
            "selected_oof_changed_fraction": selected["changed_fraction"],
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = anchor_frame[
        [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in anchor_frame]
    ].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
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

    print("\nFinal robot/test result after frozen guarded correction:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_candidate_kind",
                "selected_anchor_allowed",
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
