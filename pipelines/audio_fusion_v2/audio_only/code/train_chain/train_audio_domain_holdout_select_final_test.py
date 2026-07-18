from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")


@dataclass(frozen=True)
class DomainCandidate:
    name: str
    base_name: str
    train_views: tuple[str, ...] = ("clean",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only domain-holdout selection. Validation is built only from "
            "hand/default by holding out non-le trunk/twig sources plus leaf groups. "
            "Robot/test is loaded only after the selected model and bias are locked."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--leaf-val-frac", type=float, default=0.28)
    parser.add_argument("--letwig-val-frac", type=float, default=0.18)
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


def prefix_alpha(audio_file: str) -> str:
    stem = Path(audio_file).stem
    match = re.match(r"^([^0-9._]+)", stem)
    return match.group(1) if match else stem


def label_counts_from_y(y: np.ndarray) -> dict[str, int]:
    return {base.ID2LABEL[int(label)]: int((y == label).sum()) for label in LABELS}


def make_domain_holdout_split(
    train_df: pd.DataFrame,
    random_state: int,
    leaf_val_frac: float,
    letwig_val_frac: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(random_state)
    groups = train_df["specimen_group"].to_numpy()
    prefix = train_df["prefix_alpha"].to_numpy()
    y = train_df["y"].to_numpy()

    val_groups = set(train_df.loc[train_df["prefix_alpha"].isin(["trunk", "twig"]), "specimen_group"])

    leaf_groups = np.asarray(sorted(train_df.loc[train_df["prefix_alpha"].eq("leaf"), "specimen_group"].unique()))
    if not 0.05 <= leaf_val_frac <= 0.6:
        raise ValueError("--leaf-val-frac must be between 0.05 and 0.6")
    n_leaf_val = max(1, int(round(len(leaf_groups) * leaf_val_frac)))
    val_leaf_groups = set(rng.choice(leaf_groups, size=n_leaf_val, replace=False).tolist())
    val_groups.update(val_leaf_groups)

    letwig_groups = np.asarray(sorted(train_df.loc[train_df["prefix_alpha"].eq("letwig"), "specimen_group"].unique()))
    if not 0.0 <= letwig_val_frac <= 0.5:
        raise ValueError("--letwig-val-frac must be between 0.0 and 0.5")
    n_letwig_val = max(1, int(round(len(letwig_groups) * letwig_val_frac))) if letwig_val_frac > 0 else 0
    val_letwig_groups = (
        set(rng.choice(letwig_groups, size=n_letwig_val, replace=False).tolist())
        if n_letwig_val
        else set()
    )
    val_groups.update(val_letwig_groups)

    val_mask = np.asarray([group in val_groups for group in groups], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    train_groups = set(groups[train_idx].tolist())
    overlap = train_groups.intersection(val_groups)
    if overlap:
        raise AssertionError(f"Domain holdout group leakage: {sorted(overlap)[:5]}")
    if len(set(y[val_idx].tolist())) != len(LABELS):
        raise AssertionError("Domain holdout validation is missing at least one class")
    if len(set(y[train_idx].tolist())) != len(LABELS):
        raise AssertionError("Domain holdout training is missing at least one class")

    split_info = {
        "rule": "hold out all specimen groups with prefix_alpha in ['trunk','twig'] plus random leaf specimen groups",
        "leaf_val_frac": float(leaf_val_frac),
        "letwig_val_frac": float(letwig_val_frac),
        "n_leaf_groups_total": int(len(leaf_groups)),
        "n_leaf_groups_val": int(len(val_leaf_groups)),
        "n_letwig_groups_total": int(len(letwig_groups)),
        "n_letwig_groups_val": int(len(val_letwig_groups)),
        "n_val_groups": int(len(val_groups)),
        "n_train_groups": int(len(train_groups)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_label_counts": label_counts_from_y(y[train_idx]),
        "val_label_counts": label_counts_from_y(y[val_idx]),
        "val_prefix_counts": train_df.iloc[val_idx]["prefix_alpha"].value_counts().to_dict(),
        "train_prefix_counts": train_df.iloc[train_idx]["prefix_alpha"].value_counts().to_dict(),
    }
    return train_idx, val_idx, split_info


def make_candidates() -> dict[str, DomainCandidate]:
    specs = [
        DomainCandidate("domain_hgb_default__clean", "direct_hgb_default", ("clean",)),
        DomainCandidate("domain_hgb_default__robot_aug", "direct_hgb_default", ("clean", "robot_mix")),
        DomainCandidate("domain_hgb_default__all_aug", "direct_hgb_default", STRESS_VIEWS),
        DomainCandidate("domain_hgb_regularized__clean", "direct_hgb_regularized", ("clean",)),
        DomainCandidate("domain_hgb_regularized__all_aug", "direct_hgb_regularized", STRESS_VIEWS),
        DomainCandidate("domain_lgbm__clean", "direct_lightgbm_regularized", ("clean",)),
        DomainCandidate("domain_lgbm__all_aug", "direct_lightgbm_regularized", STRESS_VIEWS),
        DomainCandidate("domain_hier_extra_hgb__clean", "hier_extra_hgb", ("clean",)),
        DomainCandidate("domain_hier_extra_hgb__all_aug", "hier_extra_hgb", STRESS_VIEWS),
        DomainCandidate("domain_hier_hgb_lgbm__all_aug", "hier_hgb_lightgbm", STRESS_VIEWS),
    ]
    return {spec.name: spec for spec in specs}


def score_view(y_true: np.ndarray, proba: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    pred = cv.predict_with_bias(proba, bias)
    macro = f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)
    contact = f1_score(y_true, pred, labels=CONTACT_LABELS, average="macro", zero_division=0)
    return {
        "macro_f1": float(macro),
        "contact_macro_f1": float(contact),
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def score_multiview(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
    bias: np.ndarray,
) -> dict[str, float]:
    per_view = {view: score_view(y_true, proba, bias) for view, proba in proba_by_view.items()}
    output = {}
    for metric in ["macro_f1", "contact_macro_f1", "hybrid_macro_contact"]:
        values = [per_view[view][metric] for view in proba_by_view]
        output[f"worst_{metric}"] = float(min(values))
        output[f"mean_{metric}"] = float(np.mean(values))
        for view in proba_by_view:
            output[f"{view}_{metric}"] = per_view[view][metric]
    return output


def tune_bias(
    proba_by_view: dict[str, np.ndarray],
    y_true: np.ndarray,
    objective: str,
) -> tuple[np.ndarray, dict[str, float]]:
    grid = np.asarray([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2], dtype=np.float64)
    objective_key = f"worst_{objective}"
    mean_key = f"mean_{objective}"
    best_bias = np.zeros(4, dtype=np.float64)
    best_scores = score_multiview(proba_by_view, y_true, best_bias)
    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        scores = score_multiview(proba_by_view, y_true, bias)
        if (
            scores[objective_key] > best_scores[objective_key]
            or (scores[objective_key] == best_scores[objective_key] and scores[mean_key] > best_scores[mean_key])
        ):
            best_bias = bias
            best_scores = scores
    for ambient_bias in np.linspace(-0.6, 0.6, 7):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        scores = score_multiview(proba_by_view, y_true, bias)
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
    variants = [("no_bias", zero, score_multiview(proba_by_view, y_true, zero))]
    for objective in ["macro_f1", "contact_macro_f1", "hybrid_macro_contact"]:
        bias, scores = tune_bias(proba_by_view, y_true, objective)
        variants.append((f"tuned_{objective}", bias, scores))
    return variants


def fit_candidate(
    spec: DomainCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
) -> object:
    stress_spec = stress.StressCandidate(
        name=spec.name,
        base_name=spec.base_name,
        kind="single",
        train_views=spec.train_views,
    )
    stress_specs = {spec.name: stress_spec}
    return stress.fit_stress_candidate(stress_spec, base_specs, stress_specs, X_by_view, y, train_idx)


def evaluate_candidate(
    spec: DomainCandidate,
    base_specs: dict[str, cv.CandidateSpec],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    print(f"\nDomain candidate: {spec.name} train_views={spec.train_views}", flush=True)
    start = time.perf_counter()
    artifact = fit_candidate(spec, base_specs, X_by_view, y, train_idx)
    train_time = time.perf_counter() - start
    proba_by_view = {}
    view_scores = {}
    for view in STRESS_VIEWS:
        proba_by_view[view] = stress.predict_stress_artifact(artifact, X_by_view[view][val_idx])
        no_bias = score_view(y[val_idx], proba_by_view[view], np.zeros(4, dtype=np.float64))
        view_scores[f"{view}_macro_f1"] = no_bias["macro_f1"]
        view_scores[f"{view}_contact_macro_f1"] = no_bias["contact_macro_f1"]
    print(
        "  no-bias val: "
        + " | ".join(f"{view}=M{view_scores[f'{view}_macro_f1']:.4f}/C{view_scores[f'{view}_contact_macro_f1']:.4f}" for view in STRESS_VIEWS)
        + f" time={train_time:.2f}s",
        flush=True,
    )

    fold_row = {
        "candidate": spec.name,
        "train_time_sec": train_time,
        "val_samples": int(len(val_idx)),
        **view_scores,
    }
    rows = []
    for variant_name, bias, scores in bias_variants(proba_by_view, y[val_idx]):
        pred = cv.predict_with_bias(proba_by_view["clean"], bias)
        row = base.make_report_row(
            model_name=f"{spec.name}__{variant_name}",
            split_name="domain_holdout_val",
            y_true=y[val_idx],
            y_pred=pred,
            train_time_sec=train_time,
            predict_time_sec=0.0,
        )
        row.update(
            {
                "candidate": spec.name,
                "base_name": spec.base_name,
                "bias_variant": variant_name,
                "train_views_json": json.dumps(list(spec.train_views)),
                "class_bias_json": json.dumps(bias.tolist()),
                **scores,
                "selection_worst_hybrid_score": scores["worst_hybrid_macro_contact"],
                "selection_mean_hybrid_score": scores["mean_hybrid_macro_contact"],
            }
        )
        rows.append(row)
        print(
            f"  {variant_name}: worst hybrid={scores['worst_hybrid_macro_contact']:.4f} "
            f"worst macro={scores['worst_macro_f1']:.4f} "
            f"worst contact={scores['worst_contact_macro_f1']:.4f} "
            f"bias={np.round(bias, 3).tolist()}",
            flush=True,
        )
    return rows, [fold_row]


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = "audio_domain_holdout_select"
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
    print("Selection               = train-only hand/domain holdout, then final robot/test")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)
    train_df["prefix_alpha"] = train_df["audio_file"].map(prefix_alpha)

    clean_feat, clean_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    payloads = {"clean": clean_feat}
    timing = {"clean": clean_timing}
    for view in ("robot_mix", "bandlimit"):
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

    train_idx, val_idx, split_info = make_domain_holdout_split(
        train_df,
        args.random_state,
        args.leaf_val_frac,
        args.letwig_val_frac,
    )
    split_df = train_df.copy()
    split_df["domain_split"] = "train"
    split_df.loc[val_idx, "domain_split"] = "val"
    split_path = split_dir / "hand_train_domain_holdout_split.csv"
    split_df.to_csv(split_path, index=False)

    split_summary = {
        "protocol": "audio-only hand/domain holdout selection; robot/test after lock",
        "split_info": split_info,
        "feature_timing": timing,
        "split_path": str(split_path.resolve()),
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)
    print("Domain split summary:")
    print(json.dumps(split_info, indent=2), flush=True)

    base_specs = cv.make_candidates(args.random_state)
    candidates = make_candidates()
    leaderboard_rows = []
    fold_rows = []
    for spec in candidates.values():
        rows, candidate_rows = evaluate_candidate(spec, base_specs, X_by_view, y, train_idx, val_idx)
        leaderboard_rows.extend(rows)
        fold_rows.extend(candidate_rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [
            "selection_worst_hybrid_score",
            "selection_mean_hybrid_score",
            "worst_macro_f1",
            "worst_contact_macro_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_domain_holdout_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_candidate_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_spec = candidates[str(selected["candidate"])]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected["model"],
        "selected_candidate": selected["candidate"],
        "selected_base_name": selected_spec.base_name,
        "selected_bias_variant": selected["bias_variant"],
        "selected_train_views": list(selected_spec.train_views),
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
    final_artifact = fit_candidate(selected_spec, base_specs, X_by_view, y, full_idx)
    final_fit_time = time.perf_counter() - start
    final_proba = stress.predict_stress_artifact(final_artifact, test_feat["X"])
    final_pred = cv.predict_with_bias(final_proba, selected_bias)
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
            "selected_by": "hand_default_audio_only_domain_holdout",
            "selected_worst_hybrid_score": selected["selection_worst_hybrid_score"],
            "selected_worst_macro_f1": selected["worst_macro_f1"],
            "selected_worst_contact_macro_f1": selected["worst_contact_macro_f1"],
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
            "protocol": "audio_only_domain_holdout_select_no_test_until_final",
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
        "protocol": "audio_only_domain_holdout_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "candidate_report": str(fold_report_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen domain-holdout audio selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_worst_hybrid_score",
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
