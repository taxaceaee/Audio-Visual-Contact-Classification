from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import train_audio_multifeature_tta_grid_ensemble_select_final_test as mf_ens
import train_audio_tta_grid_ensemble_select_final_test as ens
import train_audio_tta_grid_hgb_select_final_test as grid_hgb
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
STRESS_VIEWS = stress.STRESS_VIEWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only gate on the selected HGB TTA ensemble. The gate thresholds "
            "are selected using train-only OOF predictions, then robot/test is "
            "loaded for final evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-slug", default="audio_tta_grid_hgb_ensemble_gate_select")
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_tta_grid_hgb_select"),
    )
    parser.add_argument(
        "--ensemble-run",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/audio_tta_grid_hgb_ensemble_select"),
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
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def selection_score(row: dict[str, float]) -> float:
    return float(
        0.55 * row["hybrid_macro_contact"]
        + 0.25 * row["worst_single_view_hybrid"]
        + 0.20 * row["worst_fold_hybrid_macro_contact"]
        - 0.5 * row["fold_std_hybrid_macro_contact"]
    )


def predict_with_gate(
    proba: np.ndarray,
    class_bias: np.ndarray,
    force_contact_threshold: float,
    force_ambient_threshold: float,
    contact_mode: str,
) -> np.ndarray:
    scores = proba + class_bias.reshape(1, -1)
    pred = np.argmax(scores, axis=1).astype(np.int64)
    contact_score = np.sum(proba[:, 1:], axis=1)
    if contact_mode == "biased":
        contact_pred = 1 + np.argmax(scores[:, 1:], axis=1)
    elif contact_mode == "raw":
        contact_pred = 1 + np.argmax(proba[:, 1:], axis=1)
    else:
        raise KeyError(f"Unknown contact_mode: {contact_mode}")

    if force_contact_threshold <= 1.0:
        mask = (pred == 0) & (contact_score >= force_contact_threshold)
        pred[mask] = contact_pred[mask]
    if force_ambient_threshold >= 0.0:
        mask = (pred != 0) & (contact_score <= force_ambient_threshold)
        pred[mask] = 0
    return pred


def calibrate_proba(proba: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-12:
        return proba
    output = np.power(np.clip(proba, 1e-12, 1.0), gamma)
    return output / output.sum(axis=1, keepdims=True)


def score_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = mf_ens.fast_macro_f1(y_true, pred, LABELS)
    contact = mf_ens.fast_macro_f1(y_true, pred, CONTACT_LABELS)
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "hybrid_macro_contact": float(0.5 * macro + 0.5 * contact),
    }


def score_gate(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_bias: np.ndarray,
    force_contact_threshold: float,
    force_ambient_threshold: float,
    contact_mode: str,
    probability_gamma: float,
) -> dict[str, float]:
    proba = calibrate_proba(proba, probability_gamma)
    pred = predict_with_gate(
        proba,
        class_bias,
        force_contact_threshold,
        force_ambient_threshold,
        contact_mode,
    )
    return score_pred(y_true, pred)


def fold_scores(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_bias: np.ndarray,
    force_contact_threshold: float,
    force_ambient_threshold: float,
    contact_mode: str,
    probability_gamma: float,
    fold_assignment: np.ndarray,
) -> dict[str, float]:
    macros = []
    contacts = []
    hybrids = []
    for fold_id in sorted(set(fold_assignment.tolist())):
        mask = fold_assignment == fold_id
        scores = score_gate(
            y_true[mask],
            proba[mask],
            class_bias,
            force_contact_threshold,
            force_ambient_threshold,
            contact_mode,
            probability_gamma,
        )
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


def evaluate_gate(
    y: np.ndarray,
    fold_assignment: np.ndarray,
    tta_proba: np.ndarray,
    view_proba: dict[str, np.ndarray],
    class_bias: np.ndarray,
    force_contact_threshold: float,
    force_ambient_threshold: float,
    contact_mode: str,
    probability_gamma: float,
) -> dict:
    row = score_gate(
        y,
        tta_proba,
        class_bias,
        force_contact_threshold,
        force_ambient_threshold,
        contact_mode,
        probability_gamma,
    )
    for view in STRESS_VIEWS:
        scores = score_gate(
            y,
            view_proba[view],
            class_bias,
            force_contact_threshold,
            force_ambient_threshold,
            contact_mode,
            probability_gamma,
        )
        for metric, value in scores.items():
            row[f"{view}_{metric}"] = value
    row["worst_single_view_hybrid"] = min(
        row[f"{view}_hybrid_macro_contact"] for view in STRESS_VIEWS
    )
    row.update(
        fold_scores(
            y,
            tta_proba,
            class_bias,
            force_contact_threshold,
            force_ambient_threshold,
            contact_mode,
            probability_gamma,
            fold_assignment,
        )
    )
    row["selection_score"] = selection_score(row)
    row["force_contact_threshold"] = force_contact_threshold
    row["force_ambient_threshold"] = force_ambient_threshold
    row["contact_mode"] = contact_mode
    row["probability_gamma"] = probability_gamma
    return row


def gate_grid() -> list[tuple[float, float, str, float]]:
    candidates = []
    gammas = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    for gamma in gammas:
        candidates.append((1.1, -0.1, "raw", gamma))
        candidates.append((1.1, -0.1, "biased", gamma))
    contact_thresholds = np.asarray([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ambient_thresholds = np.asarray([-0.1, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    for gamma in gammas:
        for mode in ["raw", "biased"]:
            for force_contact in contact_thresholds:
                for force_ambient in ambient_thresholds:
                    if force_ambient >= 0.0 and force_ambient >= force_contact:
                        continue
                    candidates.append((float(force_contact), float(force_ambient), mode, gamma))
    return candidates


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_slug = args.run_slug
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    ensemble_lock_path = args.ensemble_run / "reports" / f"{args.ensemble_run.name}_selected_without_test.json"
    split_path = args.source_run / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    if not ensemble_lock_path.exists():
        raise FileNotFoundError(f"Missing ensemble lock: {ensemble_lock_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing fold file: {split_path}")

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("SOURCE_RUN               =", args.source_run.resolve())
    print("ENSEMBLE_RUN             =", args.ensemble_run.resolve())
    print("Selection                = train-only OOF gate thresholds on selected HGB TTA ensemble")

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

    ensemble_lock = json.loads(ensemble_lock_path.read_text(encoding="utf-8"))
    selected_members = ensemble_lock["selected_members"]
    selected_bias = np.asarray(ensemble_lock["selected_bias"], dtype=np.float64)
    base_candidates = sorted({str(member["base_candidate"]) for member in selected_members})
    oof_by_base = {
        base_candidate: ens.load_oof_by_base(args.source_run, base_candidate)
        for base_candidate in base_candidates
    }
    tta_proba, view_proba = ens.ensemble_probas(oof_by_base, selected_members)

    rows = [
        evaluate_gate(
            y,
            fold_assignment,
            tta_proba,
            view_proba,
            selected_bias,
            force_contact,
            force_ambient,
            contact_mode,
            probability_gamma,
        )
        for force_contact, force_ambient, contact_mode, probability_gamma in gate_grid()
    ]
    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "macro_f1",
            "contact_macro_f1",
            "worst_single_view_hybrid",
            "worst_fold_hybrid_macro_contact",
        ],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_gate_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected_gate = leaderboard.iloc[0].to_dict()
    selection_summary = {
        "selection_rule": "best train-only contact/ambient gate on selected HGB TTA ensemble OOF proba",
        "base_ensemble_lock": str(ensemble_lock_path.resolve()),
        "selected_members": selected_members,
        "selected_base_bias": selected_bias.tolist(),
        "selected_without_test": selected_gate,
        "leaderboard_path": str(leaderboard_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
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
    for base_candidate in base_candidates:
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
    final_fit_time = time.perf_counter() - start

    total_weight = sum(float(member["member_weight"]) for member in selected_members)
    final_proba = None
    for member in selected_members:
        member_proba = ens.final_member_proba(final_proba_by_base, member)
        weight = float(member["member_weight"]) / total_weight
        final_proba = weight * member_proba if final_proba is None else final_proba + weight * member_proba
    if final_proba is None:
        raise RuntimeError("No final proba was produced")
    selected_gamma = float(selected_gate["probability_gamma"])
    calibrated_final_proba = calibrate_proba(final_proba, selected_gamma)
    final_pred = predict_with_gate(
        calibrated_final_proba,
        selected_bias,
        float(selected_gate["force_contact_threshold"]),
        float(selected_gate["force_ambient_threshold"]),
        str(selected_gate["contact_mode"]),
    )
    final_row = base.make_report_row(
        model_name=f"{ensemble_lock['selected_model']}__gate",
        split_name="robot_test_final",
        y_true=test_clean["y"],
        y_pred=final_pred,
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_audio_only_hgb_tta_ensemble_oof_gate",
            "selected_score": selected_gate["selection_score"],
            "selected_oof_macro_f1": selected_gate["macro_f1"],
            "selected_oof_contact_macro_f1": selected_gate["contact_macro_f1"],
            "selected_force_contact_threshold": selected_gate["force_contact_threshold"],
            "selected_force_ambient_threshold": selected_gate["force_ambient_threshold"],
            "selected_contact_mode": selected_gate["contact_mode"],
            "selected_probability_gamma": selected_gate["probability_gamma"],
            "selected_base_bias_json": json.dumps(selected_bias.tolist()),
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
        prediction_frame[f"calibrated_proba_{class_name}"] = calibrated_final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_clean["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "audio_only_hgb_tta_ensemble_gate_select_no_test_until_lock",
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifacts": final_artifacts,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "audio_only_hgb_tta_ensemble_gate_select_no_test_until_lock",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_feature_timing": test_timing,
        "artifacts": {
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen HGB TTA ensemble gate selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_score",
                "selected_force_contact_threshold",
                "selected_force_ambient_threshold",
                "selected_contact_mode",
                "selected_probability_gamma",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Gate leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
