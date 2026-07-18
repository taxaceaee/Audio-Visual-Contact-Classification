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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_specimen_contact_lift_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as spec
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
HIGHSR_WEIGHT_CANDIDATES = [0.75, 0.80, 0.85]
TRUNK_SOURCE_WEIGHTS = [0.0, 0.15, 0.30, 0.45, 0.60]
TRUNK_SEGMENT_RULES = ["mean", "max"]
TRUNK_FLOORS = [0.0, 0.45, 0.55, 0.65]
TRUNK_THRESHOLDS = [0.45, 0.55, 0.65]


@dataclass(frozen=True)
class TrunkCandidate:
    name: str
    factory: Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only trunk binary source. Trains trunk-vs-other-contact models "
            "on cached total240 audio features, selects blend by train OOF, then "
            "evaluates robot/test once."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_trunk_binary_source_lift_select")
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--fold-path",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/"
            "splits/hand_train_full_highsr_temporal_folds.csv"
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
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    return spec.normalize(proba)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray = LABELS) -> float:
    return spec.fast_macro_f1(y_true, pred, labels)


def trunk_candidates(random_state: int) -> dict[str, TrunkCandidate]:
    return {
        "trunk_lr_clean_total240": TrunkCandidate(
            name="trunk_lr_clean_total240",
            factory=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.7,
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "trunk_extra_trees_clean_total240": TrunkCandidate(
            name="trunk_extra_trees_clean_total240",
            factory=lambda: ExtraTreesClassifier(
                n_estimators=800,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "trunk_hgb_clean_total240": TrunkCandidate(
            name="trunk_hgb_clean_total240",
            factory=lambda: HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.08,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    }


def positive_proba(model: object, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if 1 not in classes:
        return np.zeros(len(X), dtype=np.float64)
    return raw[:, int(np.where(classes == 1)[0][0])].astype(np.float64)


def load_total240_features(feature_cache_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(feature_cache_dir / split_name / "X.npy", mmap_mode="r")
    y = np.load(feature_cache_dir / split_name / "y.npy")
    return X, np.asarray(y, dtype=np.int64)


def build_trunk_oof(
    candidates: dict[str, TrunkCandidate],
    X: np.ndarray,
    y: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    indices = np.arange(len(y))
    outputs = {}
    rows = []
    for candidate in candidates.values():
        print(f"\nTrunk binary OOF candidate: {candidate.name}", flush=True)
        oof = np.zeros(len(y), dtype=np.float64)
        total_time = 0.0
        for fold_id in sorted(set(fold_assignment.tolist())):
            train_idx = indices[fold_assignment != fold_id]
            val_idx = indices[fold_assignment == fold_id]
            contact_train_idx = train_idx[y[train_idx] > 0]
            y_train = (y[contact_train_idx] == 2).astype(np.int64)
            start = time.perf_counter()
            model = candidate.factory()
            model.fit(np.asarray(X[contact_train_idx]), y_train)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            oof[val_idx] = positive_proba(model, np.asarray(X[val_idx]))
            contact_val_idx = val_idx[y[val_idx] > 0]
            pred = np.where(oof[contact_val_idx] >= 0.5, 2, 1)
            binary_macro = fast_macro_f1((y[contact_val_idx] == 2).astype(np.int64), (pred == 2).astype(np.int64), np.asarray([0, 1]))
            print(f"  fold {fold_id}: trunk_binary_macro={binary_macro:.4f} time={elapsed:.2f}s", flush=True)
        outputs[candidate.name] = np.clip(oof, 1e-6, 1.0 - 1e-6)
        rows.append({"candidate": candidate.name, "train_time_sec": total_time})
    return outputs, rows


def segment_trunk_score(
    frame: pd.DataFrame,
    trunk_window_proba: np.ndarray,
    rule: str,
) -> np.ndarray:
    group_codes, _ = pd.factorize(frame["group_key"].astype(str), sort=False)
    n_groups = int(group_codes.max()) + 1
    if rule == "mean":
        sums = np.bincount(group_codes, weights=trunk_window_proba, minlength=n_groups)
        counts = np.bincount(group_codes, minlength=n_groups).astype(np.float64)
        return np.clip(sums / counts, 1e-6, 1.0 - 1e-6)
    if rule == "max":
        out = np.full(n_groups, 1e-6, dtype=np.float64)
        for idx, value in zip(group_codes, trunk_window_proba):
            if value > out[idx]:
                out[idx] = value
        return np.clip(out, 1e-6, 1.0 - 1e-6)
    raise KeyError(f"Unknown trunk segment rule: {rule}")


def apply_trunk_source(
    frame: pd.DataFrame,
    segment_proba: np.ndarray,
    trunk_window_proba: np.ndarray | None,
    source_weight: float,
    segment_rule: str,
    trunk_threshold: float,
    trunk_floor: float,
) -> np.ndarray:
    if trunk_window_proba is None or source_weight <= 0.0:
        return normalize(segment_proba)
    output = normalize(segment_proba.copy())
    contact_mass = output[:, 1:4].sum(axis=1)
    contact_dist = normalize(output[:, 1:4])
    trunk_score = segment_trunk_score(frame, trunk_window_proba, segment_rule)

    leaf_twig = normalize(np.column_stack([contact_dist[:, 0], contact_dist[:, 2]]))
    source_dist = np.zeros_like(contact_dist)
    source_dist[:, 1] = trunk_score
    source_dist[:, 0] = (1.0 - trunk_score) * leaf_twig[:, 0]
    source_dist[:, 2] = (1.0 - trunk_score) * leaf_twig[:, 1]
    blended = normalize((1.0 - source_weight) * contact_dist + source_weight * source_dist)
    if trunk_floor > 0.0:
        mask = trunk_score >= trunk_threshold
        if np.any(mask):
            target = np.maximum(blended[mask, 1], trunk_floor)
            target = np.clip(target, 1e-6, 0.98)
            rest = 1.0 - target
            leaf_twig_blend = normalize(np.column_stack([blended[mask, 0], blended[mask, 2]]))
            blended[mask, 1] = target
            blended[mask, 0] = rest * leaf_twig_blend[:, 0]
            blended[mask, 2] = rest * leaf_twig_blend[:, 1]
    output[:, 1:4] = contact_mass[:, None] * normalize(blended)
    output[:, 0] = 1.0 - contact_mass
    return normalize(output)


def base_segment_lift(
    frame: pd.DataFrame,
    highsr_proba: np.ndarray,
    pairwise_proba: np.ndarray,
    highsr_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    pairwise_weight = 1.0 - highsr_weight
    window = normalize(highsr_weight * highsr_proba + pairwise_weight * pairwise_proba)
    segment, window_to_segment = spec.segment_proba_from_window(frame, window)
    specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
    segment = lift.consensus_and_lift(
        segment,
        specimen_codes,
        consensus_threshold=0.45,
        min_contact_segments=1,
        lift_min_mass=0.35,
        lift_floor=0.58,
        lift_confidence=0.45,
    )
    return segment, window_to_segment


def evaluate_candidates(
    frame: pd.DataFrame,
    y: np.ndarray,
    highsr_oof: np.ndarray,
    pairwise_oof: np.ndarray,
    trunk_oof: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    base_cache = {}
    for highsr_weight in HIGHSR_WEIGHT_CANDIDATES:
        base_name = f"h{int(round(highsr_weight * 100)):02d}_p{int(round((1-highsr_weight) * 100)):02d}"
        if base_name not in base_cache:
            base_cache[base_name] = base_segment_lift(frame, highsr_oof, pairwise_oof, highsr_weight)
        segment, window_to_segment = base_cache[base_name]
        source_items: list[tuple[str, np.ndarray | None]] = [("none", None)]
        source_items.extend(sorted(trunk_oof.items()))
        for source_name, trunk_proba in source_items:
            for source_weight in TRUNK_SOURCE_WEIGHTS:
                if source_name == "none" and source_weight > 0:
                    continue
                if source_name != "none" and source_weight <= 0:
                    continue
                for segment_rule in TRUNK_SEGMENT_RULES:
                    for trunk_floor in TRUNK_FLOORS:
                        if trunk_floor <= 0.0:
                            threshold_values = [0.5]
                        else:
                            threshold_values = TRUNK_THRESHOLDS
                        for trunk_threshold in threshold_values:
                            adjusted = apply_trunk_source(
                                frame,
                                segment,
                                trunk_proba,
                                source_weight,
                                segment_rule,
                                trunk_threshold,
                                trunk_floor,
                            )
                            pred = adjusted[window_to_segment].argmax(axis=1).astype(np.int64)
                            macro = fast_macro_f1(y, pred, LABELS)
                            contact = fast_macro_f1(y, pred, CONTACT_LABELS)
                            binary = fast_macro_f1((y > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
                            rows.append(
                                {
                                    "recipe_kind": "audio_trunk_binary_source_lift",
                                    "base_name": base_name,
                                    "highsr_weight": float(highsr_weight),
                                    "pairwise_weight": float(1.0 - highsr_weight),
                                    "trunk_source": source_name,
                                    "trunk_source_weight": float(source_weight),
                                    "trunk_segment_rule": segment_rule,
                                    "trunk_threshold": float(trunk_threshold),
                                    "trunk_floor": float(trunk_floor),
                                    "macro_f1": macro,
                                    "contact_macro_f1": contact,
                                    "binary_macro_f1": binary,
                                    "selection_score": float(0.60 * macro + 0.30 * contact + 0.10 * binary),
                                }
                            )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "binary_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)


def fit_final_trunk_model(candidate: TrunkCandidate, X: np.ndarray, y: np.ndarray) -> object:
    contact_idx = np.where(y > 0)[0]
    model = candidate.factory()
    model.fit(np.asarray(X[contact_idx]), (y[contact_idx] == 2).astype(np.int64))
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
    print("Protocol  = audio-only trunk binary source; robot/test after lock")

    split_df = pd.read_csv(args.fold_path)
    y = split_df["y"].to_numpy(dtype=np.int64)
    fold_assignment = split_df["cv_fold"].to_numpy(dtype=np.int64)
    X_train, y_features = load_total240_features(args.feature_cache_dir, "hand_train_full")
    if not np.array_equal(y, y_features):
        raise AssertionError("Feature labels do not align with split manifest")
    trunk_specs = trunk_candidates(args.random_state)
    trunk_oof, trunk_rows = build_trunk_oof(trunk_specs, X_train, y, fold_assignment)
    pd.DataFrame(trunk_rows).to_csv(report_dir / f"{args.run_slug}_trunk_oof_sources.csv", index=False)

    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = spec.load_pairwise_oof(args.pairwise_oof_cache)
    leaderboard = evaluate_candidates(split_df, y, highsr_oof, pairwise_oof, trunk_oof)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()

    method_card = {
        "protocol": "audio_only_trunk_binary_source_lift_no_test_until_lock",
        "allowed_selection_data": "hand/default total240 audio features, hand labels, train OOF probabilities",
        "forbidden_selection_data": "robot/test labels, robot/test predictions before lock, image/multimodal features",
        "checkpoint_before_this_run": "checkpoints/audio_specimen_contact_lift_0693839",
        "candidate_count": int(len(leaderboard)),
        "trunk_sources": list(trunk_specs),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "highest train-only OOF macro/contact score over trunk-source lift grid",
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
    X_test, y_test_features = load_total240_features(args.feature_cache_dir, "robot_test")
    y_test = highsr_frame["y"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, y_test_features):
        raise AssertionError("Robot/test feature labels do not align")

    trunk_source_name = str(selected["trunk_source"])
    trunk_final_proba = None
    final_artifact = None
    if trunk_source_name != "none":
        start = time.perf_counter()
        final_artifact = fit_final_trunk_model(trunk_specs[trunk_source_name], X_train, y)
        trunk_final_proba = positive_proba(final_artifact, np.asarray(X_test))
        print(f"Final trunk source fit {trunk_source_name}: {time.perf_counter() - start:.2f}s", flush=True)

    segment, window_to_segment = base_segment_lift(
        highsr_frame,
        highsr_final,
        pair_final,
        float(selected["highsr_weight"]),
    )
    segment = apply_trunk_source(
        highsr_frame,
        segment,
        trunk_final_proba,
        float(selected["trunk_source_weight"]),
        str(selected["trunk_segment_rule"]),
        float(selected["trunk_threshold"]),
        float(selected["trunk_floor"]),
    )
    final_window_proba = segment[window_to_segment]
    final_pred = final_window_proba.argmax(axis=1).astype(np.int64)
    final_row = base.make_report_row(
        model_name="audio_trunk_binary_source_lift",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "train_only_oof_audio_trunk_binary_source_lift",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_highsr_weight": selected["highsr_weight"],
            "selected_pairwise_weight": selected["pairwise_weight"],
            "selected_trunk_source": selected["trunk_source"],
            "selected_trunk_source_weight": selected["trunk_source_weight"],
            "selected_trunk_segment_rule": selected["trunk_segment_rule"],
            "selected_trunk_threshold": selected["trunk_threshold"],
            "selected_trunk_floor": selected["trunk_floor"],
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
            "final_trunk_artifact": final_artifact,
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
    print("\nFinal robot/test result after frozen trunk-source lift:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_trunk_source",
                "selected_trunk_source_weight",
                "selected_trunk_segment_rule",
                "selected_trunk_floor",
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
