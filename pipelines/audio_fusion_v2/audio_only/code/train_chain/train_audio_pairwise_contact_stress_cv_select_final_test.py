from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lightgbm import LGBMClassifier

import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PAIR_LABELS = ((1, 2), (1, 3), (2, 3))
STRESS_VIEWS = stress.STRESS_VIEWS


@dataclass(frozen=True)
class PairwiseCandidate:
    name: str
    train_views: tuple[str, ...]
    binary_factory: Callable[[], object]
    pair_factory: Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only hierarchical pairwise contact model. Uses only hand/default "
            "clean plus train-only stress views for selection; robot/test after lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
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
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def make_training_matrix(
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    train_views: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([X_by_view[view][train_idx] for view in train_views]),
        np.concatenate([y[train_idx] for _ in train_views]),
    )


def proba_positive(model: object, X: np.ndarray, positive_label: int = 1) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if positive_label not in classes:
        return np.zeros(len(X), dtype=np.float64)
    col = int(np.where(classes == positive_label)[0][0])
    return raw[:, col].astype(np.float64)


def fit_pairwise_candidate(
    spec: PairwiseCandidate,
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
) -> dict:
    X_train, y_train = make_training_matrix(X_by_view, y, train_idx, spec.train_views)

    binary = spec.binary_factory()
    binary.fit(X_train, (y_train > 0).astype(np.int64))

    pair_models = {}
    for a, b in PAIR_LABELS:
        mask = (y_train == a) | (y_train == b)
        pair_y = (y_train[mask] == a).astype(np.int64)
        model = spec.pair_factory()
        model.fit(X_train[mask], pair_y)
        pair_models[(a, b)] = model
    return {"binary": binary, "pairs": pair_models}


def predict_pairwise_artifact(artifact: dict, X: np.ndarray) -> np.ndarray:
    contact_prob = np.clip(proba_positive(artifact["binary"], X, positive_label=1), 1e-12, 1.0)
    contact_scores = np.zeros((len(X), 3), dtype=np.float64)

    for a, b in PAIR_LABELS:
        p_a = np.clip(proba_positive(artifact["pairs"][(a, b)], X, positive_label=1), 1e-6, 1.0 - 1e-6)
        ai = int(a - 1)
        bi = int(b - 1)
        contact_scores[:, ai] += p_a
        contact_scores[:, bi] += 1.0 - p_a

    contact_scores = np.clip(contact_scores, 1e-12, None)
    contact_class_proba = contact_scores / contact_scores.sum(axis=1, keepdims=True)
    output = np.zeros((len(X), 4), dtype=np.float64)
    output[:, 0] = 1.0 - contact_prob
    output[:, 1:4] = contact_prob[:, None] * contact_class_proba
    return output / output.sum(axis=1, keepdims=True)


def class_f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    tp = np.diag(cm).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    true_count = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return cv.predict_with_bias(proba, bias)


def view_scores(y_true: np.ndarray, proba: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    pred = predict_with_bias(proba, bias)
    f1 = class_f1_scores(y_true, pred)
    macro = float(np.mean(f1))
    contact = float(np.mean(f1[1:4]))
    min_contact = float(np.min(f1[1:4]))
    balanced = float(0.35 * macro + 0.35 * contact + 0.30 * min_contact)
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "min_contact_f1": min_contact,
        "balanced_contact_score": balanced,
        "ambient_f1": float(f1[0]),
        "leaf_f1": float(f1[1]),
        "trunk_f1": float(f1[2]),
        "twig_f1": float(f1[3]),
    }


def multiview_scores(proba_by_view: dict[str, np.ndarray], y_true: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    per_view = {view: view_scores(y_true, proba, bias) for view, proba in proba_by_view.items()}
    output = {}
    for metric in ["macro_f1", "contact_macro_f1", "min_contact_f1", "balanced_contact_score", "leaf_f1", "trunk_f1", "twig_f1"]:
        values = [per_view[view][metric] for view in STRESS_VIEWS]
        output[f"worst_{metric}"] = float(min(values))
        output[f"mean_{metric}"] = float(np.mean(values))
        for view in STRESS_VIEWS:
            output[f"{view}_{metric}"] = per_view[view][metric]
    return output


def tune_bias(proba_by_view: dict[str, np.ndarray], y_true: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    grid = np.asarray([-1.2, -0.6, 0.0, 0.6, 1.2], dtype=np.float64)
    best_bias = np.zeros(4, dtype=np.float64)
    best_scores = multiview_scores(proba_by_view, y_true, best_bias)
    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        scores = multiview_scores(proba_by_view, y_true, bias)
        if (
            scores["worst_balanced_contact_score"] > best_scores["worst_balanced_contact_score"]
            or (
                scores["worst_balanced_contact_score"] == best_scores["worst_balanced_contact_score"]
                and scores["mean_balanced_contact_score"] > best_scores["mean_balanced_contact_score"]
            )
        ):
            best_bias = bias
            best_scores = scores
    for ambient_bias in np.asarray([-0.4, 0.0, 0.4], dtype=np.float64):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        scores = multiview_scores(proba_by_view, y_true, bias)
        if (
            scores["worst_balanced_contact_score"] > best_scores["worst_balanced_contact_score"]
            or (
                scores["worst_balanced_contact_score"] == best_scores["worst_balanced_contact_score"]
                and scores["mean_balanced_contact_score"] > best_scores["mean_balanced_contact_score"]
            )
        ):
            best_bias = bias
            best_scores = scores
    return best_bias, best_scores


def make_candidates(random_state: int) -> dict[str, PairwiseCandidate]:
    def binary_hgb() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=random_state,
        )

    def binary_extra() -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def pair_hgb() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=420,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=18,
            l2_regularization=0.10,
            class_weight="balanced",
            random_state=random_state,
        )

    def pair_lgbm() -> LGBMClassifier:
        return LGBMClassifier(
            objective="binary",
            n_estimators=560,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=18,
            subsample=0.86,
            colsample_bytree=0.82,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def pair_svm() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=5.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    specs = [
        PairwiseCandidate("pairwise_extra_hgb_all_aug", STRESS_VIEWS, binary_extra, pair_hgb),
        PairwiseCandidate("pairwise_hgb_hgb_all_aug", STRESS_VIEWS, binary_hgb, pair_hgb),
        PairwiseCandidate("pairwise_hgb_lgbm_all_aug", STRESS_VIEWS, binary_hgb, pair_lgbm),
        PairwiseCandidate("pairwise_hgb_svm_all_aug", STRESS_VIEWS, binary_hgb, pair_svm),
    ]
    return {spec.name: spec for spec in specs}


def evaluate_candidate(
    spec: PairwiseCandidate,
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, list[dict]]:
    print(f"\nPairwise candidate: {spec.name}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = fit_pairwise_candidate(spec, X_by_view, y, train_idx)
        train_time = time.perf_counter() - start
        view_parts = []
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = predict_pairwise_artifact(artifact, X_by_view[view][val_idx])
            scores = view_scores(y[val_idx], oof_by_view[view][val_idx], np.zeros(4, dtype=np.float64))
            view_parts.append(
                f"{view}=M{scores['macro_f1']:.4f}/C{scores['contact_macro_f1']:.4f}/min{scores['min_contact_f1']:.4f}"
            )
        fold_rows.append(
            {
                "candidate": spec.name,
                "fold": fold_id,
                "train_time_sec": train_time,
                "val_samples": int(len(val_idx)),
            }
        )
        print(f"  fold {fold_id}: " + " | ".join(view_parts) + f" time={train_time:.2f}s", flush=True)

    zero = np.zeros(4, dtype=np.float64)
    zero_scores = multiview_scores(oof_by_view, y, zero)
    tuned_bias, tuned_scores = tune_bias(oof_by_view, y)
    candidates = [("no_bias", zero, zero_scores), ("tuned_balanced_contact_score", tuned_bias, tuned_scores)]
    variant_name, bias, scores = max(
        candidates,
        key=lambda item: (item[2]["worst_balanced_contact_score"], item[2]["mean_balanced_contact_score"]),
    )
    clean_pred = predict_with_bias(oof_by_view["clean"], bias)
    row = base.make_report_row(
        model_name=f"{spec.name}__{variant_name}",
        split_name="pairwise_contact_stress_oof_cv",
        y_true=y,
        y_pred=clean_pred,
        train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
        predict_time_sec=0.0,
    )
    row.update(
        {
            "candidate": spec.name,
            "bias_variant": variant_name,
            "class_bias_json": json.dumps(bias.tolist()),
            **scores,
            "selection_worst_balanced_score": scores["worst_balanced_contact_score"],
            "selection_mean_balanced_score": scores["mean_balanced_contact_score"],
        }
    )
    print(
        f"  selected variant={variant_name} worst balanced={scores['worst_balanced_contact_score']:.4f} "
        f"worst min={scores['worst_min_contact_f1']:.4f} bias={np.round(bias, 3).tolist()}",
        flush=True,
    )
    return row, fold_rows


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {base.ID2LABEL[int(label)]: int((y == label).sum()) for label in LABELS}


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_pairwise_contact_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH               =", root_path.resolve())
    print("RUN_DIR                 =", run_dir.resolve())
    print("CLEAN_FEATURE_CACHE_DIR =", args.clean_feature_cache_dir.resolve())
    print("STRESS_FEATURE_DIR      =", args.stress_feature_dir.resolve())
    print("Selection               = train-only pairwise contact stress-CV")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    clean_feat, clean_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    payloads = {"clean": clean_feat}
    timing = {"clean": clean_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, view_timing = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.stress_feature_dir,
            force_rebuild=False,
        )
        payloads[view] = payload
        timing[view] = view_timing

    X_by_view = {view: payloads[view]["X"] for view in STRESS_VIEWS}
    y = clean_feat["y"]
    for view in STRESS_VIEWS:
        if not np.array_equal(y, payloads[view]["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")

    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    split_path = split_dir / "hand_train_full_pairwise_contact_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only pairwise contact stress-CV; robot/test after lock",
        "n_train_samples": int(len(train_df)),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": int(args.n_folds),
        "stress_views": list(STRESS_VIEWS),
        "label_counts": label_counts(y),
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    candidates = make_candidates(args.random_state)
    leaderboard_rows = []
    fold_rows = []
    for spec in candidates.values():
        row, rows = evaluate_candidate(spec, X_by_view, y, splits)
        leaderboard_rows.append(row)
        fold_rows.extend(rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [
            "selection_worst_balanced_score",
            "selection_mean_balanced_score",
            "worst_macro_f1",
            "worst_contact_macro_f1",
            "worst_min_contact_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_pairwise_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_spec = candidates[str(selected["candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_candidate": selected["candidate"],
        "selected_bias_variant": selected["bias_variant"],
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
    test_df = base.load_manifest(test_csv, "robot_test")
    test_feat, test_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )

    full_idx = np.arange(len(y))
    start = time.perf_counter()
    final_artifact = fit_pairwise_candidate(selected_spec, X_by_view, y, full_idx)
    final_fit_time = time.perf_counter() - start
    final_proba = predict_pairwise_artifact(final_artifact, test_feat["X"])
    final_pred = predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=test_feat["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_pairwise_contact_stress_cv",
            "selected_worst_balanced_score": selected["selection_worst_balanced_score"],
            "selected_worst_macro_f1": selected["worst_macro_f1"],
            "selected_worst_contact_macro_f1": selected["worst_contact_macro_f1"],
            "selected_worst_min_contact_f1": selected["worst_min_contact_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_feat["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_pairwise_contact_stress_cv_select_no_test_until_final",
            "feature_set": base.FEATURE_SET,
            "feature_spec": base.FEATURE_SPEC,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_pairwise_contact_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "clean_feature_cache_dir": str(args.clean_feature_cache_dir.resolve()),
        "stress_feature_dir": str(args.stress_feature_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "fold_report": str(fold_report_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen pairwise selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_worst_balanced_score",
                "selected_worst_min_contact_f1",
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
