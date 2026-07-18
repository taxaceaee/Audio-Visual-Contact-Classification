from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_tta_grid_hgb_select_final_test as grid_hgb
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    feature_set: str
    source_run: Path
    clean_feature_cache_dir: Path
    train_stress_feature_dir: Path
    test_stress_feature_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only multi-feature HGB TTA ensemble selection. It combines saved "
            "train-only OOF proba from feature-set grid runs, locks the ensemble "
            "before robot/test is loaded, then evaluates once on robot/test."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-per-base", type=int, default=4)
    parser.add_argument("--max-pair-members", type=int, default=18)
    parser.add_argument("--max-trio-members", type=int, default=10)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=sorted(base.FEATURE_SPECS),
        default=["total240", "total120", "mfcc40"],
        help="Feature-set grid runs to ensemble. Each source run must already exist.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def grid_run_slug(feature_set: str) -> str:
    return "audio_tta_grid_hgb_select" if feature_set == "total240" else f"audio_{feature_set}_tta_grid_hgb_select"


def source_spec_for(feature_set: str, output: Path) -> SourceSpec:
    source_run = output / "audio_feature_benchmarks" / grid_run_slug(feature_set)
    if feature_set == "total240":
        return SourceSpec(
            source_id="total240",
            feature_set=feature_set,
            source_run=source_run,
            clean_feature_cache_dir=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
            train_stress_feature_dir=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
            test_stress_feature_dir=Path("outputs/audio_feature_benchmarks/audio_tta_contact_stress_cv_select/test_tta_features"),
        )
    return SourceSpec(
        source_id=feature_set,
        feature_set=feature_set,
        source_run=source_run,
        clean_feature_cache_dir=source_run / "features",
        train_stress_feature_dir=source_run / "train_stress_features",
        test_stress_feature_dir=source_run / "test_tta_features",
    )


def leaderboard_path(source: SourceSpec) -> Path:
    return source.source_run / "reports" / f"{source.source_run.name}_oof_tta_grid_leaderboard.csv"


def split_path(source: SourceSpec) -> Path:
    return source.source_run / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"


def load_oof_by_base(source: SourceSpec, base_candidate: str) -> dict[str, np.ndarray]:
    base_dir = source.source_run / "oof_proba" / base_candidate
    return {
        view: np.load(base_dir / f"{view}_oof_proba.npy")
        for view in STRESS_VIEWS
    }


def make_member_pool(source_frames: dict[str, pd.DataFrame], top_per_base: int) -> list[dict]:
    pool: list[dict] = []
    for source_id, frame in source_frames.items():
        for _, sub in frame.groupby("base_candidate", sort=False):
            ranked = sub.sort_values(
                ["selection_score", "tta_contact_macro_f1", "tta_macro_f1"],
                ascending=False,
            ).head(top_per_base)
            for _, row in ranked.iterrows():
                item = row.to_dict()
                item["source_id"] = source_id
                item["feature_set"] = str(item["feature_set"])
                item["member_id"] = f"{source_id}::{item['model']}"
                pool.append(item)
    return pool


def make_ensemble_specs(
    pool: list[dict],
    max_pair_members: int,
    max_trio_members: int,
) -> list[list[dict]]:
    specs: list[list[dict]] = [[{**member, "member_weight": 1.0}] for member in pool]

    top_pair_pool = sorted(pool, key=lambda item: item["selection_score"], reverse=True)[:max_pair_members]
    pair_weights = [(0.5, 0.5), (0.65, 0.35), (0.35, 0.65), (0.8, 0.2), (0.2, 0.8)]
    for left, right in itertools.combinations(top_pair_pool, 2):
        for left_weight, right_weight in pair_weights:
            specs.append(
                [
                    {**left, "member_weight": left_weight},
                    {**right, "member_weight": right_weight},
                ]
            )

    top_trio_pool = sorted(pool, key=lambda item: item["selection_score"], reverse=True)[:max_trio_members]
    for members in itertools.combinations(top_trio_pool, 3):
        specs.append([{**member, "member_weight": 1.0 / 3.0} for member in members])
        for heavy_index in range(3):
            weighted = []
            for index, member in enumerate(members):
                weighted.append({**member, "member_weight": 0.5 if index == heavy_index else 0.25})
            specs.append(weighted)

    return specs


def weighted_by_member(
    oof_by_member: dict[str, dict[str, np.ndarray]],
    member: dict,
) -> np.ndarray:
    return grid_hgb.weighted_by_recipe(
        oof_by_member[str(member["member_id"])],
        json.loads(str(member["tta_weights_json"])),
    )


def ensemble_probas(
    oof_by_member: dict[str, dict[str, np.ndarray]],
    members: list[dict],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    total = sum(float(member["member_weight"]) for member in members)
    tta_output = None
    view_output = {view: None for view in STRESS_VIEWS}
    for member in members:
        weight = float(member["member_weight"]) / total
        member_views = oof_by_member[str(member["member_id"])]
        member_tta = grid_hgb.weighted_by_recipe(member_views, json.loads(str(member["tta_weights_json"])))
        tta_output = weight * member_tta if tta_output is None else tta_output + weight * member_tta
        for view in STRESS_VIEWS:
            part = weight * member_views[view]
            view_output[view] = part if view_output[view] is None else view_output[view] + part
    if tta_output is None:
        raise RuntimeError("Empty ensemble spec")
    return tta_output, {view: np.asarray(proba) for view, proba in view_output.items()}


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for label in labels:
        true_mask = y_true == label
        pred_mask = pred == label
        tp = float(np.sum(true_mask & pred_mask))
        fp = float(np.sum(~true_mask & pred_mask))
        fn = float(np.sum(true_mask & ~pred_mask))
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else (2.0 * tp / denom))
    return float(np.mean(scores))


def fast_score_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = fast_macro_f1(y_true, pred, LABELS)
    contact = fast_macro_f1(y_true, pred, np.asarray([1, 2, 3], dtype=np.int64))
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def fast_score_proba(y_true: np.ndarray, proba: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    return fast_score_pred(y_true, grid_hgb.tta.predict_with_bias(proba, bias))


def fast_view_scores(
    y_true: np.ndarray,
    proba_by_view: dict[str, np.ndarray],
    bias: np.ndarray,
) -> dict[str, float]:
    output = {}
    for view, proba in proba_by_view.items():
        scores = fast_score_proba(y_true, proba, bias)
        for metric, value in scores.items():
            output[f"{view}_{metric}"] = value
    output["worst_single_view_hybrid"] = min(
        output[f"{view}_hybrid_macro_contact"] for view in STRESS_VIEWS
    )
    return output


def fast_fold_scores(
    y_true: np.ndarray,
    proba: np.ndarray,
    bias: np.ndarray,
    fold_assignment: np.ndarray,
) -> dict[str, float]:
    hybrids = []
    macros = []
    contacts = []
    for fold_id in sorted(set(fold_assignment.tolist())):
        mask = fold_assignment == fold_id
        scores = fast_score_proba(y_true[mask], proba[mask], bias)
        macros.append(scores["macro_f1"])
        contacts.append(scores["contact_macro_f1"])
        hybrids.append(scores["hybrid_macro_contact"])
    return {
        "fold_mean_macro_f1": float(np.mean(macros)),
        "fold_mean_contact_macro_f1": float(np.mean(contacts)),
        "fold_mean_hybrid_macro_contact": float(np.mean(hybrids)),
        "fold_std_hybrid_macro_contact": float(np.std(hybrids)),
        "worst_fold_hybrid_macro_contact": float(np.min(hybrids)),
    }


def fast_selection_score(row: dict[str, float]) -> float:
    stability_penalty = 0.5 * row["fold_std_hybrid_macro_contact"]
    return float(
        0.55 * row["hybrid_macro_contact"]
        + 0.25 * row["worst_single_view_hybrid"]
        + 0.20 * row["worst_fold_hybrid_macro_contact"]
        - stability_penalty
    )


def fast_tune_bias_for_selection(
    y: np.ndarray,
    proba_by_view: dict[str, np.ndarray],
    tta_proba: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    candidates = [
        np.asarray(values, dtype=np.float64)
        for values in [
            [0.0, 0.0, 0.0, 0.0],
            [-0.4, 0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.2],
            [0.0, 0.4, 0.0, 0.4],
            [0.2, 0.4, 0.0, 0.4],
            [-0.2, 0.4, 0.0, 0.4],
            [0.0, 0.0, -0.2, 0.0],
            [0.0, 0.0, -0.4, 0.0],
            [0.0, 0.2, -0.2, 0.2],
            [0.0, 0.4, -0.4, 0.4],
            [0.2, 0.8, 0.4, 0.8],
            [0.0, 0.8, 0.4, 0.8],
        ]
    ]
    best_bias = np.zeros(4, dtype=np.float64)
    best_metrics: dict[str, float] | None = None
    for bias in candidates:
        metrics = fast_score_proba(y, tta_proba, bias)
        metrics.update(fast_view_scores(y, proba_by_view, bias))
        metrics.update(fast_fold_scores(y, tta_proba, bias, fold_assignment))
        metrics["selection_score"] = fast_selection_score(metrics)
        if best_metrics is None or (
            metrics["selection_score"],
            metrics["worst_single_view_hybrid"],
            metrics["worst_fold_hybrid_macro_contact"],
            metrics["macro_f1"],
        ) > (
            best_metrics["selection_score"],
            best_metrics["worst_single_view_hybrid"],
            best_metrics["worst_fold_hybrid_macro_contact"],
            best_metrics["macro_f1"],
        ):
            best_bias = bias
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Bias grid produced no candidates")
    return best_bias, best_metrics


def score_ensemble(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    oof_by_member: dict[str, dict[str, np.ndarray]],
    members: list[dict],
) -> dict:
    tta_proba, view_proba = ensemble_probas(oof_by_member, members)
    bias, metrics = fast_tune_bias_for_selection(y, view_proba, tta_proba, fold_assignment)
    pred = grid_hgb.tta.predict_with_bias(tta_proba, bias)
    row = base.make_report_row(
        model_name="mf_ens__" + "__".join(str(member["member_id"]) for member in members),
        split_name="audio_multifeature_hgb_tta_ensemble_oof_cv",
        y_true=y,
        y_pred=pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    row.update(
        {
            "n_members": len(members),
            "members_json": json.dumps(
                [
                    {
                        "member_id": str(member["member_id"]),
                        "source_id": str(member["source_id"]),
                        "feature_set": str(member["feature_set"]),
                        "base_candidate": str(member["base_candidate"]),
                        "tta_weights_json": str(member["tta_weights_json"]),
                        "member_weight": float(member["member_weight"]),
                    }
                    for member in members
                ]
            ),
            "class_bias_json": json.dumps(bias.tolist()),
            "tta_macro_f1": metrics["macro_f1"],
            "tta_contact_macro_f1": metrics["contact_macro_f1"],
            "tta_hybrid_macro_contact": metrics["hybrid_macro_contact"],
            "selection_score": metrics["selection_score"],
            **{
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "macro_f1",
                    "contact_macro_f1",
                    "hybrid_macro_contact",
                    "selection_score",
                }
            },
        }
    )
    return row


def load_train_payloads(source: SourceSpec, train_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    base.configure_feature_set(source.feature_set)
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=source.clean_feature_cache_dir,
        force_rebuild=False,
    )
    payloads = {"clean": clean_feat}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, _ = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=source.train_stress_feature_dir,
            force_rebuild=False,
        )
        payloads[view] = payload
    return payloads


def load_test_payloads(source: SourceSpec, test_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    base.configure_feature_set(source.feature_set)
    clean_feat, _ = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=source.clean_feature_cache_dir,
        force_rebuild=False,
    )
    payloads = {"clean": clean_feat}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, _ = stress.build_or_load_stress_cache(
            test_df,
            view=view,
            stress_feature_dir=source.test_stress_feature_dir,
            force_rebuild=False,
        )
        payloads[view] = payload
    return payloads


def final_member_proba(
    final_proba_by_key: dict[tuple[str, str], dict[str, np.ndarray]],
    member: dict,
) -> np.ndarray:
    return grid_hgb.weighted_by_recipe(
        final_proba_by_key[(str(member["source_id"]), str(member["base_candidate"]))],
        json.loads(str(member["tta_weights_json"])),
    )


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state

    root_path = base.resolve_root(args.root)
    sources = [source_spec_for(feature_set, args.output) for feature_set in args.feature_sets]
    run_slug = "audio_multifeature_tta_grid_hgb_ensemble_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH  =", root_path.resolve())
    print("RUN_DIR    =", run_dir.resolve())
    print("FEATURES   =", ", ".join(source.feature_set for source in sources))
    print("Selection  = train-only multi-feature saved OOF HGB TTA ensemble")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")

    source_frames: dict[str, pd.DataFrame] = {}
    fold_assignment: np.ndarray | None = None
    source_summary = []
    for source in sources:
        lb_path = leaderboard_path(source)
        sp_path = split_path(source)
        if not lb_path.exists():
            raise FileNotFoundError(f"Missing source leaderboard for {source.source_id}: {lb_path}")
        if not sp_path.exists():
            raise FileNotFoundError(f"Missing source split file for {source.source_id}: {sp_path}")
        frame = pd.read_csv(lb_path)
        frame["source_id"] = source.source_id
        frame["feature_set"] = source.feature_set
        source_frames[source.source_id] = frame
        current_fold = pd.read_csv(sp_path)["cv_fold"].to_numpy(dtype=np.int64)
        if fold_assignment is None:
            fold_assignment = current_fold
        elif not np.array_equal(fold_assignment, current_fold):
            raise AssertionError(f"Fold assignment differs for {source.source_id}")
        source_summary.append(
            {
                "source_id": source.source_id,
                "feature_set": source.feature_set,
                "source_run": str(source.source_run.resolve()),
                "leaderboard_path": str(lb_path.resolve()),
            }
        )
    if fold_assignment is None:
        raise RuntimeError("No source fold assignment was loaded")

    reference_source = sources[0]
    base.configure_feature_set(reference_source.feature_set)
    reference_clean, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=reference_source.clean_feature_cache_dir,
        force_rebuild=False,
    )
    y = reference_clean["y"]

    member_pool = make_member_pool(source_frames, args.top_per_base)
    oof_by_member = {}
    for member in member_pool:
        source = next(item for item in sources if item.source_id == str(member["source_id"]))
        oof_by_member[str(member["member_id"])] = load_oof_by_base(source, str(member["base_candidate"]))

    ensemble_specs = make_ensemble_specs(member_pool, args.max_pair_members, args.max_trio_members)
    print(f"Scoring {len(ensemble_specs)} train-only ensemble specs...", flush=True)
    rows = []
    for index, members in enumerate(ensemble_specs, start=1):
        rows.append(score_ensemble(y, fold_assignment, oof_by_member, members))
        if index % 500 == 0:
            print(f"  scored {index}/{len(ensemble_specs)}", flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "tta_contact_macro_f1",
            "worst_single_view_hybrid",
            "worst_fold_hybrid_macro_contact",
            "tta_macro_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)
    ensemble_leaderboard_path = report_dir / f"{run_slug}_oof_ensemble_leaderboard.csv"
    leaderboard.to_csv(ensemble_leaderboard_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_members = json.loads(str(selected["members_json"]))
    selected_bias = np.asarray(json.loads(str(selected["class_bias_json"])), dtype=np.float64)
    selection_summary = {
        "selection_rule": "best train-only multi-feature HGB TTA ensemble score from saved OOF proba",
        "source_summary": source_summary,
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_members": selected_members,
        "selected_bias": selected_bias.tolist(),
        "ensemble_leaderboard_path": str(ensemble_leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
    test_df = base.load_manifest(test_csv, "robot_test")

    train_payloads_by_source = {}
    test_payloads_by_source = {}
    for source in sources:
        if source.source_id in {str(member["source_id"]) for member in selected_members}:
            train_payloads_by_source[source.source_id] = load_train_payloads(source, train_df)
            test_payloads_by_source[source.source_id] = load_test_payloads(source, test_df)

    hgb_specs = grid_hgb.make_hgb_specs()
    base_specs = cv.make_candidates(args.random_state)
    final_artifacts = {}
    final_proba_by_key = {}
    start = time.perf_counter()
    for member in selected_members:
        key = (str(member["source_id"]), str(member["base_candidate"]))
        if key in final_proba_by_key:
            continue
        source = next(item for item in sources if item.source_id == key[0])
        base.configure_feature_set(source.feature_set)
        y_source = train_payloads_by_source[source.source_id]["clean"]["y"]
        if not np.array_equal(y, y_source):
            raise AssertionError(f"Label mismatch for {source.source_id}")
        X_by_view = {
            view: train_payloads_by_source[source.source_id][view]["X"]
            for view in STRESS_VIEWS
        }
        X_test_by_view = {
            view: test_payloads_by_source[source.source_id][view]["X"]
            for view in STRESS_VIEWS
        }
        artifact = stress.fit_stress_candidate(
            hgb_specs[key[1]],
            base_specs,
            hgb_specs,
            X_by_view,
            y,
            np.arange(len(y)),
        )
        final_artifacts[f"{key[0]}::{key[1]}"] = artifact
        final_proba_by_key[key] = {
            view: stress.predict_stress_artifact(artifact, X_test_by_view[view])
            for view in STRESS_VIEWS
        }
    final_fit_time = time.perf_counter() - start

    total_weight = sum(float(member["member_weight"]) for member in selected_members)
    final_proba = None
    for member in selected_members:
        member_proba = final_member_proba(final_proba_by_key, member)
        weight = float(member["member_weight"]) / total_weight
        final_proba = weight * member_proba if final_proba is None else final_proba + weight * member_proba
    if final_proba is None:
        raise RuntimeError("No final proba was produced")
    final_pred = grid_hgb.tta.predict_with_bias(final_proba, selected_bias)

    test_y = test_payloads_by_source[selected_members[0]["source_id"]]["clean"]["y"]
    final_row = base.make_report_row(
        model_name=str(selected["model"]),
        split_name="robot_test_final",
        y_true=test_y,
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_multifeature_hgb_tta_ensemble_oof",
            "selected_score": selected["selection_score"],
            "selected_tta_macro_f1": selected["tta_macro_f1"],
            "selected_tta_contact_macro_f1": selected["tta_contact_macro_f1"],
            "selected_worst_single_view_hybrid": selected["worst_single_view_hybrid"],
            "selected_worst_fold_hybrid_macro_contact": selected["worst_fold_hybrid_macro_contact"],
            "selected_members_json": json.dumps(selected_members),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    audio_columns = [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_df]
    prediction_frame = test_df[audio_columns].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_y, final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_multifeature_hgb_tta_ensemble_select_no_test_until_lock",
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifacts": final_artifacts,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_multifeature_hgb_tta_ensemble_select_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "ensemble_leaderboard": str(ensemble_leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen multi-feature HGB TTA ensemble selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Ensemble leaderboard:", ensemble_leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
