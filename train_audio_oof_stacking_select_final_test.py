from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_audio_multifeature_tta_grid_ensemble_select_final_test as mf_ens
import train_audio_tta_grid_hgb_select_final_test as grid_hgb
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


@dataclass(frozen=True)
class StackSpec:
    name: str
    top_k: int
    include_views: bool
    include_confidence: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only supervised OOF stacking. Base audio HGB/TTA probabilities are "
            "combined by a classical LogisticRegression meta-classifier selected only "
            "on hand/default OOF folds; robot/test is loaded after the lock is written."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-slug", default="audio_oof_stacking_select")
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_tta_grid_hgb_select"),
    )
    parser.add_argument(
        "--clean-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--train-stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument(
        "--test-stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_tta_contact_stress_cv_select/test_tta_features"),
    )
    parser.add_argument("--max-top-k", type=int, default=12)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def calibrate_proba(proba: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-12:
        return proba
    output = np.power(np.clip(proba, 1e-12, 1.0), gamma)
    return output / output.sum(axis=1, keepdims=True)


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(proba + bias.reshape(1, -1), axis=1).astype(np.int64)


def score_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = mf_ens.fast_macro_f1(y_true, pred, LABELS)
    contact = mf_ens.fast_macro_f1(y_true, pred, CONTACT_LABELS)
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def fold_scores(y_true: np.ndarray, pred: np.ndarray, fold_assignment: np.ndarray) -> dict[str, float]:
    macros = []
    contacts = []
    hybrids = []
    for fold_id in sorted(set(fold_assignment.tolist())):
        mask = fold_assignment == fold_id
        scores = score_pred(y_true[mask], pred[mask])
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


def selection_score(row: dict[str, float]) -> float:
    return float(
        0.60 * row["macro_f1"]
        + 0.20 * row["contact_macro_f1"]
        + 0.20 * row["worst_fold_hybrid_macro_contact"]
        - 0.5 * row["fold_std_hybrid_macro_contact"]
    )


def bias_candidates() -> list[np.ndarray]:
    return [
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
        ]
    ]


def tune_postprocess(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    proba: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float]]:
    best_bias = np.zeros(4, dtype=np.float64)
    best_gamma = 1.0
    best_row: dict[str, float] | None = None
    for gamma in [0.7, 0.85, 1.0, 1.2, 1.5, 2.0]:
        calibrated = calibrate_proba(proba, gamma)
        for bias in bias_candidates():
            pred = predict_with_bias(calibrated, bias)
            row = score_pred(y, pred)
            row.update(fold_scores(y, pred, fold_assignment))
            row["selection_score"] = selection_score(row)
            row["probability_gamma"] = float(gamma)
            row["class_bias_json"] = json.dumps(bias.tolist())
            if best_row is None or (
                row["selection_score"],
                row["macro_f1"],
                row["contact_macro_f1"],
                row["worst_fold_hybrid_macro_contact"],
            ) > (
                best_row["selection_score"],
                best_row["macro_f1"],
                best_row["contact_macro_f1"],
                best_row["worst_fold_hybrid_macro_contact"],
            ):
                best_bias = bias
                best_gamma = float(gamma)
                best_row = row
    if best_row is None:
        raise RuntimeError("Postprocess search produced no candidates")
    return best_bias, best_gamma, best_row


def load_oof_by_base(source_run: Path, base_candidate: str) -> dict[str, np.ndarray]:
    base_dir = source_run / "oof_proba" / base_candidate
    return {view: np.load(base_dir / f"{view}_oof_proba.npy") for view in STRESS_VIEWS}


def ranked_members(leaderboard: pd.DataFrame, max_top_k: int) -> list[dict]:
    rows = []
    seen = set()
    for _, row in leaderboard.sort_values("selection_score", ascending=False).iterrows():
        key = str(row["model"])
        if key in seen:
            continue
        seen.add(key)
        item = row.to_dict()
        item["member_id"] = key
        rows.append(item)
        if len(rows) >= max_top_k:
            break
    return rows


def add_probability_features(parts: list[np.ndarray], proba: np.ndarray, include_confidence: bool) -> None:
    parts.append(proba)
    parts.append(np.log(np.clip(proba, 1e-8, 1.0)))
    if include_confidence:
        contact = np.sum(proba[:, 1:], axis=1, keepdims=True)
        max_prob = np.max(proba, axis=1, keepdims=True)
        margin = np.sort(proba, axis=1)[:, -1:] - np.sort(proba, axis=1)[:, -2:-1]
        parts.extend([contact, max_prob, margin])


def stack_matrix(
    proba_by_base: dict[str, dict[str, np.ndarray]],
    members: list[dict],
    spec: StackSpec,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for member in members[: spec.top_k]:
        base_candidate = str(member["base_candidate"])
        weights = json.loads(str(member["tta_weights_json"]))
        tta_proba = grid_hgb.weighted_by_recipe(proba_by_base[base_candidate], weights)
        add_probability_features(parts, tta_proba, spec.include_confidence)
        if spec.include_views:
            for view in STRESS_VIEWS:
                add_probability_features(parts, proba_by_base[base_candidate][view], spec.include_confidence)
    return np.hstack(parts).astype(np.float64)


def stack_specs(max_top_k: int) -> list[StackSpec]:
    candidates = []
    for top_k in [4, 6, 8, 12]:
        if top_k > max_top_k:
            continue
        candidates.append(StackSpec(f"top{top_k}_tta", top_k, False, True))
        candidates.append(StackSpec(f"top{top_k}_tta_views", top_k, True, True))
    return candidates


def meta_candidates(random_state: int) -> list[dict]:
    candidates = []
    for class_weight in [None, "balanced"]:
        for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]:
            name = f"logreg_C{c_value:g}_cw{class_weight or 'none'}"
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=2000,
                    multi_class="auto",
                    random_state=random_state,
                    solver="lbfgs",
                ),
            )
            candidates.append(
                {
                    "name": name,
                    "model": model,
                    "C": c_value,
                    "class_weight": class_weight,
                    "params": {"family": "logreg", "C": c_value, "class_weight": class_weight},
                }
            )
    for max_leaf_nodes in [15, 31]:
        for l2_regularization in [0.0, 0.1, 1.0]:
            name = f"hgb_leaf{max_leaf_nodes}_l2{l2_regularization:g}"
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=max_leaf_nodes,
                l2_regularization=l2_regularization,
                random_state=random_state,
            )
            candidates.append(
                {
                    "name": name,
                    "model": model,
                    "C": "",
                    "class_weight": "none",
                    "params": {
                        "family": "hist_gradient_boosting",
                        "max_leaf_nodes": max_leaf_nodes,
                        "l2_regularization": l2_regularization,
                    },
                }
            )
    for min_samples_leaf in [1, 3, 8]:
        for max_features in ["sqrt", 0.5]:
            name = f"extratrees_leaf{min_samples_leaf}_mf{max_features}"
            model = ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
            candidates.append(
                {
                    "name": name,
                    "model": model,
                    "C": "",
                    "class_weight": "balanced",
                    "params": {
                        "family": "extra_trees",
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                    },
                }
            )
    return candidates


def cv_meta_proba(model, X: np.ndarray, y: np.ndarray, fold_assignment: np.ndarray) -> np.ndarray:
    oof = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    for fold_id in sorted(set(fold_assignment.tolist())):
        train_mask = fold_assignment != fold_id
        val_mask = fold_assignment == fold_id
        fold_model = clone(model)
        fold_model.fit(X[train_mask], y[train_mask])
        oof[val_mask] = fold_model.predict_proba(X[val_mask])
    return oof


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

    leaderboard_path = args.source_run / "reports" / f"{args.source_run.name}_oof_tta_grid_leaderboard.csv"
    split_path = args.source_run / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing leaderboard: {leaderboard_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing fold file: {split_path}")

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("SOURCE_RUN               =", args.source_run.resolve())
    print("Selection                = train-only OOF LogisticRegression stacking")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    y = clean_feat["y"]
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)

    leaderboard = pd.read_csv(leaderboard_path)
    members = ranked_members(leaderboard, args.max_top_k)
    base_candidates = sorted({str(member["base_candidate"]) for member in members})
    oof_by_base = {base_candidate: load_oof_by_base(args.source_run, base_candidate) for base_candidate in base_candidates}

    rows = []
    best_payload: dict | None = None
    for stack_spec in stack_specs(args.max_top_k):
        X_stack = stack_matrix(oof_by_base, members, stack_spec)
        for meta in meta_candidates(args.random_state):
            start = time.perf_counter()
            meta_oof = cv_meta_proba(meta["model"], X_stack, y, fold_assignment)
            cv_time = time.perf_counter() - start
            bias, gamma, metrics = tune_postprocess(y, fold_assignment, meta_oof)
            row = {
                "model": f"{stack_spec.name}__{meta['name']}",
                "stack_spec": stack_spec.name,
                "top_k": stack_spec.top_k,
                "include_views": stack_spec.include_views,
                "include_confidence": stack_spec.include_confidence,
                "meta_model": meta["name"],
                "meta_C": meta["C"],
                "meta_class_weight": meta["class_weight"] or "none",
                "meta_params_json": json.dumps(meta.get("params", {})),
                "n_stack_features": int(X_stack.shape[1]),
                "cv_time_sec": float(cv_time),
                **metrics,
            }
            rows.append(row)
            if best_payload is None or (
                row["selection_score"],
                row["macro_f1"],
                row["contact_macro_f1"],
                row["worst_fold_hybrid_macro_contact"],
            ) > (
                best_payload["row"]["selection_score"],
                best_payload["row"]["macro_f1"],
                best_payload["row"]["contact_macro_f1"],
                best_payload["row"]["worst_fold_hybrid_macro_contact"],
            ):
                best_payload = {
                    "row": row,
                    "stack_spec": stack_spec,
                    "meta": meta,
                    "bias": bias,
                    "gamma": gamma,
                    "X_stack": X_stack,
                }
        print(f"  checked stack={stack_spec.name} features={X_stack.shape[1]}", flush=True)

    if best_payload is None:
        raise RuntimeError("No stacking candidate was evaluated")

    stack_leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "worst_fold_hybrid_macro_contact"],
        ascending=False,
    ).reset_index(drop=True)
    stack_leaderboard_path = report_dir / f"{args.run_slug}_oof_stacking_leaderboard.csv"
    stack_leaderboard.to_csv(stack_leaderboard_path, index=False)

    selected_row = best_payload["row"]
    selected_members = members[: int(selected_row["top_k"])]
    selection_summary = {
        "selection_rule": "best train-only grouped OOF LogisticRegression stack plus train-only gamma/bias",
        "source_leaderboard": str(leaderboard_path.resolve()),
        "selected_without_test": selected_row,
        "selected_class_bias": best_payload["bias"].tolist(),
        "selected_probability_gamma": float(best_payload["gamma"]),
        "selected_members": [
            {
                "member_id": str(member["member_id"]),
                "base_candidate": str(member["base_candidate"]),
                "tta_weights_json": str(member["tta_weights_json"]),
            }
            for member in selected_members
        ],
        "leaderboard_path": str(stack_leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    train_payloads = {"clean": clean_feat}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, _ = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.train_stress_feature_dir,
            force_rebuild=False,
        )
        train_payloads[view] = payload
    X_by_view = {view: train_payloads[view]["X"] for view in STRESS_VIEWS}

    test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
    test_df = base.load_manifest(test_csv, "robot_test")
    test_clean, test_clean_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    test_payloads = {"clean": test_clean}
    test_timing = {"clean": test_clean_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, view_timing = stress.build_or_load_stress_cache(
            test_df,
            view=view,
            stress_feature_dir=args.test_stress_feature_dir,
            force_rebuild=False,
        )
        test_payloads[view] = payload
        test_timing[view] = view_timing
    X_test_by_view = {view: test_payloads[view]["X"] for view in STRESS_VIEWS}

    hgb_specs = grid_hgb.make_hgb_specs()
    base_specs = cv.make_candidates(args.random_state)
    final_artifacts = {}
    final_proba_by_base = {}
    start = time.perf_counter()
    for base_candidate in sorted({str(member["base_candidate"]) for member in selected_members}):
        artifact = stress.fit_stress_candidate(
            hgb_specs[base_candidate],
            base_specs,
            hgb_specs,
            X_by_view,
            y,
            np.arange(len(y)),
        )
        final_artifacts[base_candidate] = artifact
        final_proba_by_base[base_candidate] = {
            view: stress.predict_stress_artifact(artifact, X_test_by_view[view])
            for view in STRESS_VIEWS
        }
    final_base_time = time.perf_counter() - start

    final_stack_spec = best_payload["stack_spec"]
    X_train_stack = best_payload["X_stack"]
    X_test_stack = stack_matrix(final_proba_by_base, selected_members, final_stack_spec)
    final_meta = clone(best_payload["meta"]["model"])
    start = time.perf_counter()
    final_meta.fit(X_train_stack, y)
    final_meta_time = time.perf_counter() - start
    final_proba_raw = final_meta.predict_proba(X_test_stack)
    final_proba = calibrate_proba(final_proba_raw, float(best_payload["gamma"]))
    final_pred = predict_with_bias(final_proba, best_payload["bias"])

    final_row = base.make_report_row(
        model_name=f"{selected_row['model']}__oof_stack",
        split_name="robot_test_final",
        y_true=test_clean["y"],
        y_pred=final_pred,
        train_time_sec=final_base_time + final_meta_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_oof_stacking",
            "selected_score": selected_row["selection_score"],
            "selected_oof_macro_f1": selected_row["macro_f1"],
            "selected_oof_contact_macro_f1": selected_row["contact_macro_f1"],
            "selected_stack_spec": selected_row["stack_spec"],
            "selected_meta_model": selected_row["meta_model"],
            "selected_probability_gamma": best_payload["gamma"],
            "selected_class_bias_json": json.dumps(best_payload["bias"].tolist()),
            "selected_members_json": json.dumps(selection_summary["selected_members"]),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    audio_columns = [column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_df]
    prediction_frame = test_df[audio_columns].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_clean["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_oof_stacking_select_no_test_until_lock",
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifacts": final_artifacts,
            "meta_model": final_meta,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "audio_only_oof_stacking_select_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "leaderboard": str(stack_leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{args.run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen OOF stacking selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
                "selected_stack_spec",
                "selected_meta_model",
                "selected_probability_gamma",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Stacking leaderboard:", stack_leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
