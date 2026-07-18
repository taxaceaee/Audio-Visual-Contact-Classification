from __future__ import annotations

import argparse
import json
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

import train_audio_highsr_temporal_tta_select_final_test as highsr
import train_audio_locked_oof_blender_select_final_test as locked
import train_audio_report_grade_gate_select_final_test as report_gate
import train_cv_select_final_test as cv
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
PROBA_COLUMNS = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only broad OOF meta-stack. It builds meta features from locked "
            "audio-only OOF sources, selects the meta-model by grouped OOF only, "
            "writes a lock, then loads final robot/test source predictions."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-slug", default="audio_broad_oof_meta_select")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-12, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def one_hot(pred: np.ndarray) -> np.ndarray:
    output = np.zeros((len(pred), len(LABELS)), dtype=np.float64)
    output[np.arange(len(pred)), pred.astype(int)] = 1.0
    return output


def load_highsr_oof_from_run(output: Path, run_name: str) -> np.ndarray:
    run = output / "audio_feature_benchmarks" / run_name
    lock_path = run / "reports" / f"{run_name}_selected_without_test.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    candidate = str(lock["selected_candidate"])
    weights = lock["selected_tta_weights"]
    oof_by_view = {
        view: np.load(run / "oof_proba" / candidate / f"{view}_oof_proba.npy")
        for view in highsr.STRESS_VIEWS
    }
    return normalize(highsr.weighted_proba(oof_by_view, weights))


def load_train_sources(
    output: Path,
    y: np.ndarray,
    fold_assignment: np.ndarray,
    random_state: int,
) -> dict[str, np.ndarray]:
    sources = {
        "total240_grid": locked.load_grid_tta_oof(output, "audio_tta_grid_hgb_select"),
        "total240_ensemble": locked.load_total240_ensemble_oof(
            output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select",
            output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_ensemble_select",
        ),
        "total240_stack_lr": locked.load_total240_stack_oof(output, y, fold_assignment, random_state),
        "total240_tree_meta": locked.load_total240_stack_oof_from_run(
            output,
            "audio_oof_stacking_tree_meta_select",
            y,
            fold_assignment,
            random_state,
        ),
        "total120_grid": locked.load_grid_tta_oof(output, "audio_total120_tta_grid_hgb_select"),
        "mfcc40_grid": locked.load_grid_tta_oof(output, "audio_mfcc40_tta_grid_hgb_select"),
        "highsr_default": load_highsr_oof_from_run(output, "audio_highsr_temporal_tta_select"),
        "highsr_regularized": load_highsr_oof_from_run(output, "audio_highsr_temporal_hgb_regularized_select"),
        "highsr_extratrees": load_highsr_oof_from_run(output, "audio_highsr_temporal_extratrees_select"),
    }

    report_lock = json.loads(
        (
            output
            / "audio_feature_benchmarks"
            / "audio_report_grade_gate_select"
            / "reports"
            / "audio_report_grade_gate_select_selected_without_test.json"
        ).read_text(encoding="utf-8")
    )["selected_without_test"]
    report_pred, report_proba, _ = report_gate.apply_gate_recipe(
        {
            "total240_ensemble": sources["total240_ensemble"],
            "total240_stack_lr": sources["total240_stack_lr"],
            "total240_tree_meta": sources["total240_tree_meta"],
            "highsr_hgb": sources["highsr_default"],
        },
        json.loads(str(report_lock["class_weights_json"])),
        float(report_lock["force_contact_threshold"]),
    )
    sources["report_gate_proba"] = normalize(report_proba)
    sources["report_gate_onehot"] = one_hot(report_pred)
    return {name: normalize(proba) for name, proba in sources.items()}


def final_prediction_paths(output: Path) -> dict[str, Path]:
    root = output / "audio_feature_benchmarks"
    return {
        "total240_grid": root / "audio_tta_grid_hgb_select" / "reports" / "audio_tta_grid_hgb_select_final_test_predictions.csv",
        "total240_ensemble": root
        / "audio_tta_grid_hgb_ensemble_select"
        / "reports"
        / "audio_tta_grid_hgb_ensemble_select_final_test_predictions.csv",
        "total240_stack_lr": root
        / "audio_oof_stacking_select"
        / "reports"
        / "audio_oof_stacking_select_final_test_predictions.csv",
        "total240_tree_meta": root
        / "audio_oof_stacking_tree_meta_select"
        / "reports"
        / "audio_oof_stacking_tree_meta_select_final_test_predictions.csv",
        "total120_grid": root
        / "audio_total120_tta_grid_hgb_select"
        / "reports"
        / "audio_total120_tta_grid_hgb_select_final_test_predictions.csv",
        "mfcc40_grid": root
        / "audio_mfcc40_tta_grid_hgb_select"
        / "reports"
        / "audio_mfcc40_tta_grid_hgb_select_final_test_predictions.csv",
        "highsr_default": root
        / "audio_highsr_temporal_tta_select"
        / "reports"
        / "audio_highsr_temporal_tta_select_final_test_predictions.csv",
        "highsr_regularized": root
        / "audio_highsr_temporal_hgb_regularized_select"
        / "reports"
        / "audio_highsr_temporal_hgb_regularized_select_final_test_predictions.csv",
        "highsr_extratrees": root
        / "audio_highsr_temporal_extratrees_select"
        / "reports"
        / "audio_highsr_temporal_extratrees_select_final_test_predictions.csv",
        "report_gate_proba": root
        / "audio_report_grade_gate_select"
        / "reports"
        / "audio_report_grade_gate_select_final_test_predictions.csv",
        "report_gate_onehot": root
        / "audio_report_grade_gate_select"
        / "reports"
        / "audio_report_grade_gate_select_final_test_predictions.csv",
    }


def load_final_sources(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = {}
    sources = {}
    for name, path in paths.items():
        frame = pd.read_csv(path)
        frames[name] = frame
        if name.endswith("_onehot"):
            sources[name] = one_hot(frame["pred_y"].to_numpy(dtype=np.int64))
        else:
            sources[name] = normalize(frame[PROBA_COLUMNS].to_numpy(dtype=np.float64))
    reference = frames["report_gate_proba"]
    for name, frame in frames.items():
        for column in ["audio_file", "y"]:
            if column in reference and column in frame:
                if not np.array_equal(reference[column].astype(str).to_numpy(), frame[column].astype(str).to_numpy()):
                    raise AssertionError(f"Final source {name} is not aligned on {column}")
    return reference, sources


def entropy(proba: np.ndarray) -> np.ndarray:
    return -np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1, keepdims=True)


def margin(proba: np.ndarray) -> np.ndarray:
    sorted_proba = np.sort(proba, axis=1)
    return (sorted_proba[:, -1] - sorted_proba[:, -2]).reshape(-1, 1)


def source_feature_matrix(sources: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for name in sorted(sources):
        proba = normalize(sources[name])
        pred = np.argmax(proba, axis=1)
        parts.extend(
            [
                proba,
                np.log(np.clip(proba, 1e-8, 1.0)),
                np.sum(proba[:, 1:], axis=1, keepdims=True),
                np.max(proba, axis=1, keepdims=True),
                entropy(proba),
                margin(proba),
                one_hot(pred),
            ]
        )
    return np.hstack(parts).astype(np.float64)


def score_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    macro = locked.fast_macro_f1(y_true, pred, LABELS)
    contact = locked.fast_macro_f1(y_true, pred, CONTACT_LABELS)
    binary = locked.fast_macro_f1((y_true > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
    return {
        "macro_f1": macro,
        "contact_macro_f1": contact,
        "binary_macro_f1": binary,
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
        0.50 * row["macro_f1"]
        + 0.20 * row["contact_macro_f1"]
        + 0.20 * row["worst_fold_hybrid_macro_contact"]
        + 0.10 * row["binary_macro_f1"]
        - 0.5 * row["fold_std_hybrid_macro_contact"]
    )


def predict_proba_aligned(model: object, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    output = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        output[:, int(np.where(LABELS == class_id)[0][0])] = raw[:, column]
    return normalize(output)


def meta_candidates(random_state: int) -> list[dict]:
    rows = []
    for class_weight in [None, "balanced"]:
        for c_value in [0.03, 0.1, 0.3, 1.0]:
            rows.append(
                {
                    "name": f"logreg_C{c_value:g}_cw{class_weight or 'none'}",
                    "model": make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=c_value,
                            class_weight=class_weight,
                            max_iter=3000,
                            random_state=random_state,
                        ),
                    ),
                }
            )
    for leaf in [4, 8, 16, 32]:
        rows.append(
            {
                "name": f"extratrees_leaf{leaf}",
                "model": ExtraTreesClassifier(
                    n_estimators=600,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            }
        )
    for leaf_nodes, min_leaf in [(15, 25), (31, 30), (31, 60)]:
        rows.append(
            {
                "name": f"hgb_leaf{leaf_nodes}_min{min_leaf}",
                "model": HistGradientBoostingClassifier(
                    max_iter=260,
                    learning_rate=0.035,
                    max_leaf_nodes=leaf_nodes,
                    min_samples_leaf=min_leaf,
                    l2_regularization=0.08,
                    class_weight=base.CONFIG["class_weights"],
                    random_state=random_state,
                ),
            }
        )
    return rows


def cv_meta_proba(model: object, X: np.ndarray, y: np.ndarray, fold_assignment: np.ndarray) -> np.ndarray:
    output = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    for fold_id in sorted(set(fold_assignment.tolist())):
        train_mask = fold_assignment != fold_id
        val_mask = fold_assignment == fold_id
        fold_model = clone(model)
        fold_model.fit(X[train_mask], y[train_mask])
        output[val_mask] = predict_proba_aligned(fold_model, X[val_mask])
    return output


def bias_candidates() -> list[np.ndarray]:
    return [
        np.asarray(values, dtype=np.float64)
        for values in [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.2],
            [-0.1, 0.2, 0.0, 0.2],
            [0.0, 0.2, -0.1, 0.2],
            [0.0, 0.25, 0.0, 0.25],
            [-0.1, 0.25, 0.0, 0.25],
        ]
    ]


def evaluate_meta_candidate(
    name: str,
    proba: np.ndarray,
    y: np.ndarray,
    fold_assignment: np.ndarray,
) -> dict:
    best = None
    for gamma in [0.85, 1.0, 1.25, 1.5, 2.0]:
        calibrated = locked.calibrate_proba(proba, gamma)
        for bias in bias_candidates():
            pred = locked.predict_with_bias(calibrated, bias)
            row = score_pred(y, pred)
            row.update(fold_scores(y, pred, fold_assignment))
            row["selection_score"] = selection_score(row)
            if best is None or (
                row["selection_score"],
                row["macro_f1"],
                row["contact_macro_f1"],
                row["worst_fold_hybrid_macro_contact"],
            ) > (
                best["selection_score"],
                best["macro_f1"],
                best["contact_macro_f1"],
                best["worst_fold_hybrid_macro_contact"],
            ):
                best = {
                    "meta_model": name,
                    **row,
                    "probability_gamma": float(gamma),
                    "class_bias_json": json.dumps(bias.tolist()),
                }
    if best is None:
        raise RuntimeError(f"No scoring result for {name}")
    return best


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
    print("Protocol  = audio-only broad OOF meta-stack; robot/test loaded after lock")

    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
        force_rebuild=False,
    )
    y = clean_feat["y"]
    split_path = args.output / "audio_feature_benchmarks" / "audio_tta_grid_hgb_select" / "splits" / "hand_train_full_hgb_tta_grid_folds.csv"
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)

    train_sources = load_train_sources(args.output, y, fold_assignment, args.random_state)
    for name, proba in train_sources.items():
        if proba.shape != (len(y), len(LABELS)):
            raise AssertionError(f"{name}: bad train source shape {proba.shape}")
        print(f"Loaded train OOF source {name}: {proba.shape}", flush=True)
    X_meta = source_feature_matrix(train_sources)
    print("Meta feature shape:", X_meta.shape, flush=True)

    rows = []
    oof_by_meta = {}
    for candidate in meta_candidates(args.random_state):
        print(f"Meta candidate: {candidate['name']}", flush=True)
        proba = cv_meta_proba(candidate["model"], X_meta, y, fold_assignment)
        oof_by_meta[candidate["name"]] = proba
        rows.append(evaluate_meta_candidate(candidate["name"], proba, y, fold_assignment))

    leaderboard = pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "contact_macro_f1", "worst_fold_hybrid_macro_contact"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_oof_meta_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()

    method_card = {
        "protocol": "audio_only_broad_oof_meta_stack_no_test_until_lock",
        "allowed_selection_data": "hand/default train labels and locked audio-only OOF probabilities",
        "forbidden_selection_data": "robot/test labels or robot/test predictions before the selection lock; images; multimodal features",
        "source_names": sorted(train_sources),
        "meta_feature_shape": list(X_meta.shape),
        "split_path": str(split_path.resolve()),
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection_summary = {
        "selection_rule": "best grouped OOF meta-model over locked audio-only OOF sources",
        "selected_without_test": selected,
        "method_card": str(method_card_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_path": str(split_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_frame, final_sources = load_final_sources(final_prediction_paths(args.output))
    X_final = source_feature_matrix(final_sources)
    selected_name = str(selected["meta_model"])
    candidates = {candidate["name"]: candidate["model"] for candidate in meta_candidates(args.random_state)}
    final_meta = clone(candidates[selected_name])
    final_meta.fit(X_meta, y)
    final_proba = predict_proba_aligned(final_meta, X_final)
    final_proba = locked.calibrate_proba(final_proba, float(selected["probability_gamma"]))
    final_bias = np.asarray(json.loads(str(selected["class_bias_json"])), dtype=np.float64)
    final_pred = locked.predict_with_bias(final_proba, final_bias)
    y_test = test_frame["y"].to_numpy(dtype=np.int64)

    final_row = base.make_report_row(
        model_name=f"{selected_name}__audio_broad_oof_meta",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "audio_only_broad_oof_meta_stack",
            "selected_score": selected["selection_score"],
            "selected_oof_macro_f1": selected["macro_f1"],
            "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
            "selected_meta_model": selected["meta_model"],
            "selected_probability_gamma": selected["probability_gamma"],
            "selected_class_bias_json": selected["class_bias_json"],
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_columns = [
        column
        for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"]
        if column in test_frame
    ]
    prediction_frame = test_frame[prediction_columns].copy()
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
            "selected_meta_model": final_meta,
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

    print("\nFinal robot/test result after frozen audio broad OOF meta-stack:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_oof_macro_f1",
                "selected_meta_model",
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
