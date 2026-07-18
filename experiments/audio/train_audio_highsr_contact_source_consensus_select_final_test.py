from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")
HIGHSR_WEIGHT_CANDIDATES = [0.75, 0.80, 0.85]
CONTACT_SOURCE_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]
SEGMENT_CONTACT_RULES = ["mean_contact_dist", "sum_log_contact_dist"]


@dataclass(frozen=True)
class ContactCandidate:
    name: str
    train_views: tuple[str, ...]
    factory: Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only high-SR contact-subclass source. Trains a contact-only "
            "classical classifier on cached high-SR audio features, selects blend "
            "with train OOF only, then evaluates robot/test once."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_highsr_contact_source_consensus_select")
    parser.add_argument(
        "--highsr-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/features"),
    )
    parser.add_argument(
        "--pairwise-oof-cache",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    return specimen.normalize(proba)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray = LABELS) -> float:
    return specimen.fast_macro_f1(y_true, pred, labels)


def contact_candidates(random_state: int) -> dict[str, ContactCandidate]:
    return {
        "highsr_contact_extra_trees_all_aug": ContactCandidate(
            name="highsr_contact_extra_trees_all_aug",
            train_views=("clean", "robot_mix", "bandlimit"),
            factory=lambda: ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "highsr_contact_extra_trees_clean": ContactCandidate(
            name="highsr_contact_extra_trees_clean",
            train_views=("clean",),
            factory=lambda: ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "highsr_contact_random_forest_all_aug": ContactCandidate(
            name="highsr_contact_random_forest_all_aug",
            train_views=("clean", "robot_mix", "bandlimit"),
            factory=lambda: RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "highsr_contact_logistic_clean": ContactCandidate(
            name="highsr_contact_logistic_clean",
            train_views=("clean",),
            factory=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.7,
                            class_weight="balanced",
                            max_iter=2500,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    }


def proba_aligned_contact(model: object, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    output = np.zeros((len(X), 3), dtype=np.float64)
    for col, class_id in enumerate(classes):
        if class_id in CONTACT_LABELS:
            output[:, int(class_id) - 1] = raw[:, col]
    return normalize(output)


def load_highsr_features(feature_dir: Path, split_name: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_view = {}
    y_ref = None
    for view in STRESS_VIEWS:
        X = np.load(feature_dir / split_name / view / "X.npy", mmap_mode="r")
        y = np.load(feature_dir / split_name / view / "y.npy")
        by_view[view] = X
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise AssertionError(f"High-SR label mismatch for {split_name}/{view}")
    return by_view, np.asarray(y_ref, dtype=np.int64)


def make_training_matrix(
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    train_views: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    contact_idx = train_idx[y[train_idx] > 0]
    X_train = np.vstack([np.asarray(X_by_view[view][contact_idx]) for view in train_views])
    y_train = np.concatenate([y[contact_idx] for _ in train_views])
    return X_train, y_train


def build_contact_oof(
    specs: dict[str, ContactCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    outputs = {}
    rows = []
    indices = np.arange(len(y))
    for spec in specs.values():
        print(f"\nHigh-SR contact OOF candidate: {spec.name}", flush=True)
        oof = np.zeros((len(y), 3), dtype=np.float64)
        total_time = 0.0
        for fold_id in sorted(set(fold_assignment.tolist())):
            val_idx = indices[fold_assignment == fold_id]
            train_idx = indices[fold_assignment != fold_id]
            X_train, y_train = make_training_matrix(X_by_view, y, train_idx, spec.train_views)
            start = time.perf_counter()
            model = spec.factory()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            oof[val_idx] = proba_aligned_contact(model, np.asarray(X_by_view["clean"][val_idx]))
            contact_val = val_idx[y[val_idx] > 0]
            pred = CONTACT_LABELS[oof[contact_val].argmax(axis=1)]
            macro = fast_macro_f1(y[contact_val], pred, CONTACT_LABELS)
            print(f"  fold {fold_id}: contact_macro={macro:.4f} time={elapsed:.2f}s", flush=True)
        outputs[spec.name] = normalize(oof)
        rows.append({"candidate": spec.name, "train_views": json.dumps(list(spec.train_views)), "train_time_sec": total_time})
    return outputs, rows


def segment_contact_dist_from_window(
    frame: pd.DataFrame,
    contact_window_proba: np.ndarray,
    rule: str,
) -> np.ndarray:
    group_codes, _ = pd.factorize(frame["group_key"].astype(str), sort=False)
    n_groups = int(group_codes.max()) + 1
    if rule == "mean_contact_dist":
        sums = np.vstack(
            [
                np.bincount(group_codes, weights=contact_window_proba[:, class_idx], minlength=n_groups)
                for class_idx in range(3)
            ]
        ).T
        counts = np.bincount(group_codes, minlength=n_groups).astype(np.float64)
        return normalize(sums / counts[:, None])
    if rule == "sum_log_contact_dist":
        logs = np.log(np.clip(contact_window_proba, 1e-12, 1.0))
        sums = np.vstack(
            [np.bincount(group_codes, weights=logs[:, class_idx], minlength=n_groups) for class_idx in range(3)]
        ).T
        exp_scores = np.exp(sums - sums.max(axis=1, keepdims=True))
        return normalize(exp_scores)
    raise KeyError(f"Unknown contact segment rule: {rule}")


def build_segment_proba_with_contact_source(
    frame: pd.DataFrame,
    base_window_proba: np.ndarray,
    contact_window_proba: np.ndarray | None,
    contact_source_weight: float,
    contact_segment_rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_segment, window_to_segment = specimen.segment_proba_from_window(frame, base_window_proba)
    if contact_window_proba is None or contact_source_weight <= 0.0:
        return base_segment, window_to_segment
    base_contact_dist = normalize(base_segment[:, 1:4])
    source_contact_dist = segment_contact_dist_from_window(frame, contact_window_proba, contact_segment_rule)
    contact_mass = base_segment[:, 1:4].sum(axis=1)
    blended_contact = normalize(
        (1.0 - contact_source_weight) * base_contact_dist
        + contact_source_weight * source_contact_dist
    )
    segment_proba = np.zeros_like(base_segment)
    segment_proba[:, 0] = base_segment[:, 0]
    segment_proba[:, 1:4] = contact_mass[:, None] * blended_contact
    return normalize(segment_proba), window_to_segment


def evaluate_candidates(
    frame: pd.DataFrame,
    y: np.ndarray,
    highsr_oof: np.ndarray,
    pairwise_oof: np.ndarray,
    contact_oof: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for highsr_weight in HIGHSR_WEIGHT_CANDIDATES:
        pairwise_weight = 1.0 - highsr_weight
        base_window = normalize(highsr_weight * highsr_oof + pairwise_weight * pairwise_oof)
        contact_source_items: list[tuple[str, np.ndarray | None]] = [("none", None)]
        contact_source_items.extend(sorted(contact_oof.items()))
        for contact_name, contact_proba in contact_source_items:
            for source_weight in CONTACT_SOURCE_WEIGHTS:
                if contact_name == "none" and source_weight > 0.0:
                    continue
                if contact_name != "none" and source_weight <= 0.0:
                    continue
                for contact_rule in SEGMENT_CONTACT_RULES:
                    segment_proba, window_to_segment = build_segment_proba_with_contact_source(
                        frame,
                        base_window,
                        contact_proba,
                        source_weight,
                        contact_rule,
                    )
                    specimen_codes = specimen.specimen_codes_for_segments(frame, window_to_segment)
                    segment_proba = specimen.apply_specimen_contact_consensus(
                        segment_proba,
                        specimen_codes,
                        threshold=0.45,
                        alpha=1.0,
                        min_contact_segments=1,
                        agg_rule="mean_contact_dist",
                    )
                    pred = segment_proba[window_to_segment].argmax(axis=1).astype(np.int64)
                    macro = fast_macro_f1(y, pred, LABELS)
                    contact = fast_macro_f1(y, pred, CONTACT_LABELS)
                    binary = fast_macro_f1((y > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
                    rows.append(
                        {
                            "recipe_kind": "audio_highsr_contact_source_consensus",
                            "highsr_weight": float(highsr_weight),
                            "pairwise_weight": float(pairwise_weight),
                            "contact_source": contact_name,
                            "contact_source_weight": float(source_weight),
                            "contact_segment_rule": contact_rule,
                            "specimen_rule": "contact_subclass_consensus",
                            "contact_threshold": 0.45,
                            "consensus_alpha": 1.0,
                            "min_contact_segments": 1,
                            "agg_rule": "mean_contact_dist",
                            "macro_f1": macro,
                            "contact_macro_f1": contact,
                            "binary_macro_f1": binary,
                            "selection_score": float(0.60 * macro + 0.30 * contact + 0.10 * binary),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)


def fit_final_contact_model(
    spec: ContactCandidate,
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
) -> object:
    train_idx = np.arange(len(y))
    X_train, y_train = make_training_matrix(X_by_view, y, train_idx, spec.train_views)
    model = spec.factory()
    model.fit(X_train, y_train)
    return model


def load_final_frame_and_proba(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
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
    print("Protocol  = audio-only high-SR contact source; robot/test after lock")

    split_path = (
        args.output
        / "audio_feature_benchmarks"
        / "audio_highsr_temporal_tta_select"
        / "splits"
        / "hand_train_full_highsr_temporal_folds.csv"
    )
    train_df = pd.read_csv(split_path)
    y = train_df["y"].to_numpy(dtype=np.int64)
    fold_assignment = train_df["cv_fold"].to_numpy(dtype=np.int64)
    X_by_view, y_features = load_highsr_features(args.highsr_feature_dir, "hand_train_full")
    if not np.array_equal(y, y_features):
        raise AssertionError("High-SR feature cache labels do not align with split manifest")

    contact_specs = contact_candidates(args.random_state)
    contact_oof, contact_rows = build_contact_oof(contact_specs, X_by_view, y, fold_assignment)
    pd.DataFrame(contact_rows).to_csv(report_dir / f"{args.run_slug}_contact_oof_sources.csv", index=False)

    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = specimen.load_pairwise_oof(args.pairwise_oof_cache)
    leaderboard = evaluate_candidates(train_df, y, highsr_oof, pairwise_oof, contact_oof)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()

    method_card = {
        "protocol": "audio_only_highsr_contact_source_consensus_no_test_until_lock",
        "allowed_selection_data": "hand/default high-SR audio features, hand labels, grouped OOF probabilities",
        "forbidden_selection_data": "robot/test labels, robot/test features before selection lock, image/multimodal features",
        "checkpoint_before_this_run": "checkpoints/audio_specimen_contact_consensus_0690682",
        "candidate_count": int(len(leaderboard)),
        "contact_sources": list(contact_specs),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact score over high-SR contact-source blend grid",
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
    if not np.array_equal(highsr_frame["audio_file"].astype(str).to_numpy(), pair_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("Final source frames are not aligned")
    X_test_by_view, y_test_features = load_highsr_features(args.highsr_feature_dir, "robot_test")
    y_test = highsr_frame["y"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, y_test_features):
        raise AssertionError("Robot/test high-SR labels do not align with final predictions")

    contact_source_name = str(selected["contact_source"])
    final_contact_proba = None
    final_artifact = None
    if contact_source_name != "none":
        start = time.perf_counter()
        selected_contact_spec = contact_specs[contact_source_name]
        final_artifact = fit_final_contact_model(selected_contact_spec, X_by_view, y)
        final_contact_proba = proba_aligned_contact(final_artifact, np.asarray(X_test_by_view["clean"]))
        print(f"Final contact source fit {contact_source_name}: {time.perf_counter() - start:.2f}s", flush=True)

    final_blend = normalize(float(selected["highsr_weight"]) * highsr_final + float(selected["pairwise_weight"]) * pair_final)
    final_segment, final_window_to_segment = build_segment_proba_with_contact_source(
        highsr_frame,
        final_blend,
        final_contact_proba,
        float(selected["contact_source_weight"]),
        str(selected["contact_segment_rule"]),
    )
    final_specimen_codes = specimen.specimen_codes_for_segments(highsr_frame, final_window_to_segment)
    final_segment = specimen.apply_specimen_contact_consensus(
        final_segment,
        final_specimen_codes,
        threshold=float(selected["contact_threshold"]),
        alpha=float(selected["consensus_alpha"]),
        min_contact_segments=int(selected["min_contact_segments"]),
        agg_rule=str(selected["agg_rule"]),
    )
    final_window_proba = final_segment[final_window_to_segment]
    final_pred = final_window_proba.argmax(axis=1).astype(np.int64)
    final_row = base.make_report_row(
        model_name="audio_highsr_contact_source_consensus",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_highsr_contact_source_consensus",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_highsr_weight": selected["highsr_weight"],
            "selected_pairwise_weight": selected["pairwise_weight"],
            "selected_contact_source": selected["contact_source"],
            "selected_contact_source_weight": selected["contact_source_weight"],
            "selected_contact_segment_rule": selected["contact_segment_rule"],
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
        prediction_frame[f"proba_{class_name}"] = final_window_proba[:, class_id]
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
            "final_contact_artifact": final_artifact,
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
    print("\nFinal robot/test result after frozen high-SR contact-source consensus:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_contact_source",
                "selected_contact_source_weight",
                "selected_contact_segment_rule",
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
