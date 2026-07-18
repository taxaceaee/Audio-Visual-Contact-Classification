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

import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


@dataclass(frozen=True)
class EnsembleSpec:
    name: str
    members: tuple[str, ...]
    weights: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only OOF ensemble selection with balanced contact-class objective. "
            "Selection uses only hand/default clean plus train-only stress views; "
            "robot/test is loaded only after the selection lock."
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
    parser.add_argument("--force-rebuild-stress", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def make_base_candidates() -> dict[str, stress.StressCandidate]:
    specs = [
        stress.StressCandidate("ens_hgb_default__all_aug", "direct_hgb_default", "single", STRESS_VIEWS),
        stress.StressCandidate("ens_hgb_regularized__all_aug", "direct_hgb_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("ens_lgbm__all_aug", "direct_lightgbm_regularized", "single", STRESS_VIEWS),
        stress.StressCandidate("ens_hier_extra_hgb__all_aug", "hier_extra_hgb", "single", STRESS_VIEWS),
        stress.StressCandidate("ens_hier_hgb_lgbm__all_aug", "hier_hgb_lightgbm", "single", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return cv.predict_with_bias(proba, bias)


def view_scores(y_true: np.ndarray, proba: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    pred = predict_with_bias(proba, bias)
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), pred.astype(np.int64)), 1)
    tp = np.diag(cm).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    true_count = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision + recall
    class_f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    macro = float(np.mean(class_f1))
    contact = float(np.mean(class_f1[1:4]))
    min_contact = float(np.min(class_f1[1:4]))
    # Balanced objective is deliberately conservative: high only if all contact classes hold up.
    balanced = float(0.35 * macro + 0.35 * contact + 0.30 * min_contact)
    return {
        "macro_f1": float(macro),
        "contact_macro_f1": float(contact),
        "min_contact_f1": min_contact,
        "balanced_contact_score": balanced,
        "ambient_f1": float(class_f1[0]),
        "leaf_f1": float(class_f1[1]),
        "trunk_f1": float(class_f1[2]),
        "twig_f1": float(class_f1[3]),
    }


def multiview_scores(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
    bias: np.ndarray,
) -> dict[str, float]:
    per_view = {view: view_scores(y_true, proba, bias) for view, proba in proba_by_view.items()}
    output = {}
    metric_names = [
        "macro_f1",
        "contact_macro_f1",
        "min_contact_f1",
        "balanced_contact_score",
        "leaf_f1",
        "trunk_f1",
        "twig_f1",
    ]
    for metric in metric_names:
        values = [per_view[view][metric] for view in STRESS_VIEWS]
        output[f"worst_{metric}"] = float(min(values))
        output[f"mean_{metric}"] = float(np.mean(values))
        for view in STRESS_VIEWS:
            output[f"{view}_{metric}"] = per_view[view][metric]
    return output


def tune_bias(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
    objective: str,
) -> tuple[np.ndarray, dict[str, float]]:
    grid = np.asarray([-1.2, -0.6, 0.0, 0.6, 1.2], dtype=np.float64)
    objective_key = f"worst_{objective}"
    mean_key = f"mean_{objective}"
    best_bias = np.zeros(4, dtype=np.float64)
    best_scores = multiview_scores(proba_by_view, y_true, best_bias)
    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        scores = multiview_scores(proba_by_view, y_true, bias)
        if (
            scores[objective_key] > best_scores[objective_key]
            or (scores[objective_key] == best_scores[objective_key] and scores[mean_key] > best_scores[mean_key])
        ):
            best_bias = bias
            best_scores = scores
    for ambient_bias in np.asarray([-0.4, 0.0, 0.4], dtype=np.float64):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        scores = multiview_scores(proba_by_view, y_true, bias)
        if (
            scores[objective_key] > best_scores[objective_key]
            or (scores[objective_key] == best_scores[objective_key] and scores[mean_key] > best_scores[mean_key])
        ):
            best_bias = bias
            best_scores = scores
    return best_bias, best_scores


def bias_variants(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> list[tuple[str, np.ndarray, dict[str, float]]]:
    zero = np.zeros(4, dtype=np.float64)
    variants = [("no_bias", zero, multiview_scores(proba_by_view, y_true, zero))]
    bias, scores = tune_bias(proba_by_view, y_true, "balanced_contact_score")
    variants.append(("tuned_balanced_contact_score", bias, scores))
    return variants


def evaluate_base_candidate(
    spec: stress.StressCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    all_specs: dict[str, stress.StressCandidate],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    print(f"\nBase candidate: {spec.name} train_views={spec.train_views}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = stress.fit_stress_candidate(spec, base_specs, all_specs, X_by_view, y, train_idx)
        train_time = time.perf_counter() - start
        fold_score_parts = []
        for view in STRESS_VIEWS:
            oof_by_view[view][val_idx] = stress.predict_stress_artifact(artifact, X_by_view[view][val_idx])
            scores = view_scores(y[val_idx], oof_by_view[view][val_idx], np.zeros(4, dtype=np.float64))
            fold_score_parts.append(
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
        print(f"  fold {fold_id}: " + " | ".join(fold_score_parts) + f" time={train_time:.2f}s", flush=True)
    return oof_by_view, fold_rows


def weighted_oof(
    member_names: tuple[str, ...],
    weights: tuple[float, ...],
    oof_by_candidate: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    weight_arr = np.asarray(weights, dtype=np.float64)
    weight_arr = weight_arr / weight_arr.sum()
    return {
        view: np.average(
            np.stack([oof_by_candidate[member][view] for member in member_names]),
            axis=0,
            weights=weight_arr,
        )
        for view in STRESS_VIEWS
    }


def make_ensemble_specs(single_leaderboard: pd.DataFrame) -> dict[str, EnsembleSpec]:
    ranked = single_leaderboard.sort_values(
        ["best_worst_balanced_contact_score", "best_mean_balanced_contact_score"],
        ascending=False,
    )["candidate"].tolist()
    specs: dict[str, EnsembleSpec] = {}

    for name in ranked:
        specs[f"single__{name}"] = EnsembleSpec(f"single__{name}", (name,), (1.0,))

    top = ranked[:5]
    for i, left in enumerate(top):
        for right in top[i + 1 :]:
            for w in [0.35, 0.5, 0.65]:
                name = f"pair__{left}__{right}__w{int(w * 100):02d}"
                specs[name] = EnsembleSpec(name, (left, right), (w, 1.0 - w))

    for k in [2, 3, 4]:
        members = tuple(ranked[:k])
        if len(members) == k:
            specs[f"top{k}_uniform"] = EnsembleSpec(f"top{k}_uniform", members, tuple([1.0] * k))
    return specs


def evaluate_oof_recipe(
    recipe: EnsembleSpec,
    proba_by_view: dict[str, np.ndarray],
    y: np.ndarray,
) -> list[dict]:
    rows = []
    for variant_name, bias, scores in bias_variants(proba_by_view, y):
        clean_pred = predict_with_bias(proba_by_view["clean"], bias)
        row = base.make_report_row(
            model_name=f"{recipe.name}__{variant_name}",
            split_name="ensemble_balanced_stress_oof_cv",
            y_true=y,
            y_pred=clean_pred,
            train_time_sec=0.0,
            predict_time_sec=0.0,
        )
        row.update(
            {
                "recipe": recipe.name,
                "members_json": json.dumps(list(recipe.members)),
                "weights_json": json.dumps(list(recipe.weights)),
                "bias_variant": variant_name,
                "class_bias_json": json.dumps(bias.tolist()),
                **scores,
                "selection_worst_balanced_score": scores["worst_balanced_contact_score"],
                "selection_mean_balanced_score": scores["mean_balanced_contact_score"],
            }
        )
        rows.append(row)
    return rows


def fit_final_member(
    member_name: str,
    candidate_specs: dict[str, stress.StressCandidate],
    base_specs: dict[str, cv.CandidateSpec],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
) -> object:
    full_idx = np.arange(len(y))
    return stress.fit_stress_candidate(
        candidate_specs[member_name],
        base_specs,
        candidate_specs,
        X_by_view,
        y,
        full_idx,
    )


def predict_final_ensemble(final_artifacts: dict[str, object], recipe: EnsembleSpec, X: np.ndarray) -> np.ndarray:
    weights = np.asarray(recipe.weights, dtype=np.float64)
    weights = weights / weights.sum()
    probas = [stress.predict_stress_artifact(final_artifacts[member], X) for member in recipe.members]
    return np.average(np.stack(probas), axis=0, weights=weights)


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {base.ID2LABEL[int(label)]: int((y == label).sum()) for label in LABELS}


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_ensemble_balanced_stress_cv_select"
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
    print("Selection               = train-only OOF ensemble, worst balanced contact score")

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
            force_rebuild=args.force_rebuild_stress,
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
    split_path = split_dir / "hand_train_full_ensemble_balanced_stress_cv_folds.csv"
    train_df.assign(cv_fold=fold_assignment).to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only total240 ensemble balanced stress-CV; robot/test after lock",
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

    base_specs = cv.make_candidates(args.random_state)
    candidate_specs = make_base_candidates()
    oof_by_candidate = {}
    fold_rows = []
    single_rows = []
    for spec in candidate_specs.values():
        oof_by_view, rows = evaluate_base_candidate(spec, base_specs, candidate_specs, X_by_view, y, splits)
        oof_by_candidate[spec.name] = oof_by_view
        fold_rows.extend(rows)
        single_eval_rows = evaluate_oof_recipe(EnsembleSpec(spec.name, (spec.name,), (1.0,)), oof_by_view, y)
        best_single = max(
            single_eval_rows,
            key=lambda row: (row["selection_worst_balanced_score"], row["selection_mean_balanced_score"]),
        )
        single_rows.append(
            {
                "candidate": spec.name,
                "best_worst_balanced_contact_score": best_single["selection_worst_balanced_score"],
                "best_mean_balanced_contact_score": best_single["selection_mean_balanced_score"],
                "best_model": best_single["model"],
            }
        )

    single_leaderboard = pd.DataFrame(single_rows).sort_values(
        ["best_worst_balanced_contact_score", "best_mean_balanced_contact_score"],
        ascending=False,
    ).reset_index(drop=True)
    single_leaderboard_path = report_dir / f"{run_slug}_single_candidate_leaderboard.csv"
    single_leaderboard.to_csv(single_leaderboard_path, index=False)
    print("\nSingle-candidate balanced leaderboard:")
    print(single_leaderboard.head(8).to_string(index=False), flush=True)

    recipes = make_ensemble_specs(single_leaderboard)
    leaderboard_rows = []
    for recipe in recipes.values():
        proba_by_view = weighted_oof(recipe.members, recipe.weights, oof_by_candidate)
        rows = evaluate_oof_recipe(recipe, proba_by_view, y)
        leaderboard_rows.extend(rows)

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
    leaderboard_path = report_dir / f"{run_slug}_ensemble_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_recipe = recipes[str(selected["recipe"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_recipe": selected_recipe.name,
        "selected_members": list(selected_recipe.members),
        "selected_weights": list(selected_recipe.weights),
        "selected_bias_variant": selected["bias_variant"],
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "single_leaderboard_path": str(single_leaderboard_path.resolve()),
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

    start = time.perf_counter()
    final_artifacts = {
        member: fit_final_member(member, candidate_specs, base_specs, X_by_view, y)
        for member in selected_recipe.members
    }
    final_fit_time = time.perf_counter() - start
    final_proba = predict_final_ensemble(final_artifacts, selected_recipe, test_feat["X"])
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
            "selected_by": "hand_default_audio_only_ensemble_balanced_stress_cv",
            "selected_worst_balanced_score": selected["selection_worst_balanced_score"],
            "selected_worst_macro_f1": selected["worst_macro_f1"],
            "selected_worst_contact_macro_f1": selected["worst_contact_macro_f1"],
            "selected_worst_min_contact_f1": selected["worst_min_contact_f1"],
            "selected_members_json": json.dumps(list(selected_recipe.members)),
            "selected_weights_json": json.dumps(list(selected_recipe.weights)),
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
            "protocol": "audio_only_ensemble_balanced_stress_cv_select_no_test_until_final",
            "feature_set": base.FEATURE_SET,
            "feature_spec": base.FEATURE_SPEC,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifacts": final_artifacts,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "audio_only_ensemble_balanced_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "clean_feature_cache_dir": str(args.clean_feature_cache_dir.resolve()),
        "stress_feature_dir": str(args.stress_feature_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "ensemble_leaderboard": str(leaderboard_path.resolve()),
            "single_candidate_leaderboard": str(single_leaderboard_path.resolve()),
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

    print("\nFinal robot/test result after frozen ensemble selection:")
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
    print("Ensemble leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
