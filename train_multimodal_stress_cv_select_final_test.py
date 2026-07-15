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
from sklearn.metrics import confusion_matrix, f1_score

import train_cv_select_final_test as cv
import train_multimodal_classical_cv_select_final_test as mm
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_view: str
    train_views: tuple[str, ...]
    factory: Callable[[], object] | None = None
    kind: str = "direct"
    members: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multimodal stress-CV selection. Uses hand/default audio stress features "
            "plus handcrafted image features. Selection maximizes train-only worst-case "
            "stress macro-F1, then robot/test is loaded once for final evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-set", choices=["total240"], default="total240")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--audio-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--audio-stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument(
        "--image-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/multimodal_classical_cv_select/image_features"),
    )
    parser.add_argument("--force-rebuild-stress", action="store_true")
    parser.add_argument("--force-rebuild-image", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def image_compact(X_image: np.ndarray) -> np.ndarray:
    compact_dim = 56 + 30 + 28 + 17
    return X_image[:, :compact_dim]


def build_feature_views(
    X_audio_by_stress: dict[str, np.ndarray],
    X_image: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    X_img = image_compact(X_image)
    views = {
        "audio240": {},
        "audio_image_compact": {},
    }
    for stress_view, X_audio in X_audio_by_stress.items():
        views["audio240"][stress_view] = X_audio.astype(np.float32)
        views["audio_image_compact"][stress_view] = np.hstack([X_audio, X_img]).astype(np.float32)
    return views


def build_train_matrix(
    feature_views: dict[str, dict[str, np.ndarray]],
    y: np.ndarray,
    spec: CandidateSpec,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_parts = [feature_views[spec.feature_view][view][train_idx] for view in spec.train_views]
    y_parts = [y[train_idx] for _ in spec.train_views]
    return np.vstack(X_parts), np.concatenate(y_parts)


def make_candidates(random_state: int) -> dict[str, CandidateSpec]:
    audio_specs = cv.make_candidates(random_state)
    multimodal_specs = mm.make_model_specs(random_state, include_slow_full_image=False)
    specs = [
        CandidateSpec(
            "audio_hgb_clean",
            "audio240",
            ("clean",),
            audio_specs["direct_hgb_regularized"].direct_factory,
        ),
        CandidateSpec(
            "audio_hgb_all_aug",
            "audio240",
            STRESS_VIEWS,
            audio_specs["direct_hgb_regularized"].direct_factory,
        ),
        CandidateSpec(
            "audio_lgbm_all_aug",
            "audio240",
            STRESS_VIEWS,
            audio_specs["direct_lightgbm_regularized"].direct_factory,
        ),
        CandidateSpec(
            "mm_logreg_clean",
            "audio_image_compact",
            ("clean",),
            multimodal_specs["audio_image_compact_logreg"].factory,
        ),
        CandidateSpec(
            "mm_logreg_all_aug",
            "audio_image_compact",
            STRESS_VIEWS,
            multimodal_specs["audio_image_compact_logreg"].factory,
        ),
        CandidateSpec(
            "mm_hgb_all_aug",
            "audio_image_compact",
            STRESS_VIEWS,
            multimodal_specs["audio_image_compact_hgb_regularized"].factory,
        ),
        CandidateSpec(
            "mm_lgbm_all_aug",
            "audio_image_compact",
            STRESS_VIEWS,
            multimodal_specs["audio_image_compact_lgbm_regularized"].factory,
        ),
    ]
    return {spec.name: spec for spec in specs}


def fit_candidate(
    spec: CandidateSpec,
    candidates: dict[str, CandidateSpec],
    feature_views: dict[str, dict[str, np.ndarray]],
    y: np.ndarray,
    train_idx: np.ndarray,
) -> object:
    if spec.kind == "direct":
        X_train, y_train = build_train_matrix(feature_views, y, spec, train_idx)
        model = spec.factory()
        model.fit(X_train, y_train)
        return {"kind": "direct", "feature_view": spec.feature_view, "model": model}

    if spec.kind == "ensemble":
        return {
            "kind": "ensemble",
            "members": {
                member: fit_candidate(candidates[member], candidates, feature_views, y, train_idx)
                for member in spec.members
            },
        }

    raise ValueError(f"Unsupported candidate kind: {spec.kind}")


def predict_artifact(artifact: object, X: np.ndarray) -> np.ndarray:
    if artifact["kind"] == "direct":
        return cv.proba_aligned(artifact["model"], X)
    if artifact["kind"] == "ensemble":
        return np.mean(
            [predict_artifact(member_artifact, X) for member_artifact in artifact["members"].values()],
            axis=0,
        )
    raise ValueError(f"Unsupported artifact kind: {artifact['kind']}")


def report_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_view: str,
    n_features: int,
    train_time_sec: float,
) -> dict:
    row = base.make_report_row(
        model_name=model_name,
        split_name=split_name,
        y_true=y_true,
        y_pred=y_pred,
        train_time_sec=train_time_sec,
        predict_time_sec=0.0,
    )
    row.update(
        {
            "feature_set": feature_view,
            "feature_name": feature_view,
            "n_features": n_features,
        }
    )
    return row


def evaluate_candidate(
    spec: CandidateSpec,
    candidates: dict[str, CandidateSpec],
    feature_views: dict[str, dict[str, np.ndarray]],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    print(f"\nStress multimodal candidate: {spec.name} view={spec.feature_view} train_views={spec.train_views}", flush=True)
    oof_by_view = {view: np.zeros((len(y), len(LABELS)), dtype=np.float64) for view in STRESS_VIEWS}
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        artifact = fit_candidate(spec, candidates, feature_views, y, train_idx)
        train_time = time.perf_counter() - start
        view_scores = {}
        for stress_view in STRESS_VIEWS:
            if spec.kind == "ensemble":
                oof_by_view[stress_view][val_idx] = np.mean(
                    [
                        predict_artifact(
                            artifact["members"][member_name],
                            feature_views[candidates[member_name].feature_view][stress_view][val_idx],
                        )
                        for member_name in spec.members
                    ],
                    axis=0,
                )
            else:
                X_val = feature_views[spec.feature_view][stress_view][val_idx]
                oof_by_view[stress_view][val_idx] = predict_artifact(artifact, X_val)
            pred = oof_by_view[stress_view][val_idx].argmax(axis=1)
            view_scores[f"{stress_view}_macro_f1"] = f1_score(y[val_idx], pred, average="macro")
        fold_row = {
            "candidate": spec.name,
            "fold": fold_id,
            "train_time_sec": train_time,
            "val_samples": int(len(val_idx)),
            **view_scores,
            "fold_worst_macro_f1": min(view_scores.values()),
        }
        fold_rows.append(fold_row)
        print(
            f"  fold {fold_id}: "
            + " | ".join(f"{view}={view_scores[f'{view}_macro_f1']:.4f}" for view in STRESS_VIEWS)
            + f" | worst={fold_row['fold_worst_macro_f1']:.4f} time={train_time:.2f}s",
            flush=True,
        )

    bias, tuned_scores = stress.tune_bias_multiview(oof_by_view, y)
    clean_pred = cv.predict_with_bias(oof_by_view["clean"], bias)
    n_features = int(feature_views[spec.feature_view]["clean"].shape[1]) if spec.kind == "direct" else 0
    row = report_row(
        spec.name,
        "multimodal_stress_oof_cv",
        y,
        clean_pred,
        spec.feature_view,
        n_features,
        sum(item["train_time_sec"] for item in fold_rows),
    )
    row.update(
        {
            "candidate_kind": spec.kind,
            "feature_view": spec.feature_view,
            "train_views_json": json.dumps(list(spec.train_views)),
            "members_json": json.dumps(list(spec.members)),
            "class_bias_json": json.dumps(bias.tolist()),
            **{f"tuned_{view}_macro_f1": tuned_scores[view] for view in STRESS_VIEWS},
            "stress_worst_macro_f1": tuned_scores["stress_worst_macro_f1"],
            "stress_mean_macro_f1": tuned_scores["stress_mean_macro_f1"],
        }
    )
    print(
        "  tuned: "
        + " | ".join(f"{view}={tuned_scores[view]:.4f}" for view in STRESS_VIEWS)
        + f" | worst={tuned_scores['stress_worst_macro_f1']:.4f} bias={np.round(bias, 3).tolist()}",
        flush=True,
    )
    return row, oof_by_view, fold_rows


def build_ensemble_candidates(leaderboard: pd.DataFrame) -> dict[str, CandidateSpec]:
    ranked = leaderboard.sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    )["model"].tolist()
    recipes = {
        "mm_stress_ensemble_top3": tuple(ranked[:3]),
        "mm_stress_ensemble_top5": tuple(ranked[:5]),
    }
    return {
        name: CandidateSpec(name, "ensemble", ("clean",), kind="ensemble", members=members)
        for name, members in recipes.items()
        if len(members) >= 2
    }


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set(args.feature_set)

    root_path = base.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_slug = "multimodal_stress_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir, args.image_feature_dir, args.audio_stress_feature_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH                =", root_path.resolve())
    print("RUN_DIR                  =", run_dir.resolve())
    print("AUDIO_FEATURE_CACHE_DIR  =", args.audio_feature_cache_dir.resolve())
    print("AUDIO_STRESS_FEATURE_DIR =", args.audio_stress_feature_dir.resolve())
    print("IMAGE_FEATURE_DIR        =", args.image_feature_dir.resolve())
    print("Selection                = maximize train-only worst-case multimodal stress-CV macro-F1")

    train_csv = base.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv")
    train_df = mm.attach_image_paths(base.load_manifest(train_csv, "hand_train"), train_dataset_dir)
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)

    clean_audio, clean_audio_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.audio_feature_cache_dir,
        force_rebuild=False,
    )
    audio_payloads = {"clean": clean_audio}
    stress_timing = {"clean": clean_audio_timing}
    for view in STRESS_VIEWS:
        if view == "clean":
            continue
        payload, timing = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.audio_stress_feature_dir,
            force_rebuild=args.force_rebuild_stress,
        )
        audio_payloads[view] = payload
        stress_timing[view] = timing

    train_image, train_image_timing = mm.build_or_load_image_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.image_feature_dir,
        force_rebuild=args.force_rebuild_image,
    )
    y = clean_audio["y"]
    if not np.array_equal(y, train_image["y"]):
        raise AssertionError("Audio/image train labels are not aligned")

    X_audio_by_stress = {view: audio_payloads[view]["X"] for view in STRESS_VIEWS}
    feature_views = build_feature_views(X_audio_by_stress, train_image["X"])
    splits = cv.make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    train_df.assign(cv_fold=fold_assignment).to_csv(
        split_dir / "hand_train_full_multimodal_stress_cv_folds.csv",
        index=False,
    )

    split_summary = {
        "protocol": "hand/default multimodal stress-CV only for selection; robot/test loaded after selection lock",
        "n_train_samples": len(train_df),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": args.n_folds,
        "stress_views": list(STRESS_VIEWS),
        "label_counts": cv.label_counts(y),
        "feature_dims": {name: int(views["clean"].shape[1]) for name, views in feature_views.items()},
        "audio_stress_timing": stress_timing,
        "train_image_timing": train_image_timing,
    }
    split_summary_path = report_dir / f"{run_slug}_split_summary.json"
    write_json(split_summary_path, split_summary)

    candidates = make_candidates(args.random_state)
    leaderboard_rows = []
    fold_rows = []
    for spec in list(candidates.values()):
        row, _, rows = evaluate_candidate(spec, candidates, feature_views, y, splits)
        leaderboard_rows.append(row)
        fold_rows.extend(rows)

    initial_leaderboard = pd.DataFrame(leaderboard_rows)
    candidates.update(build_ensemble_candidates(initial_leaderboard))
    for name, spec in list(candidates.items()):
        if spec.kind != "ensemble":
            continue
        row, _, rows = evaluate_candidate(spec, candidates, feature_views, y, splits)
        leaderboard_rows.append(row)
        fold_rows.extend(rows)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["stress_worst_macro_f1", "stress_mean_macro_f1", "macro_f1_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{run_slug}_oof_stress_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidates[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_kind": selected_spec.kind,
        "selected_feature_view": selected_spec.feature_view,
        "selected_train_views": list(selected_spec.train_views),
        "selected_members": list(selected_spec.members),
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = base.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv")
    test_df = mm.attach_image_paths(base.load_manifest(test_csv, "robot_test"), test_dataset_dir)
    test_audio, test_audio_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=args.audio_feature_cache_dir,
        force_rebuild=False,
    )
    test_image, test_image_timing = mm.build_or_load_image_cache(
        test_df,
        "robot_test",
        feature_dir=args.image_feature_dir,
        force_rebuild=args.force_rebuild_image,
    )
    if not np.array_equal(test_audio["y"], test_image["y"]):
        raise AssertionError("Audio/image test labels are not aligned")
    test_feature_views = build_feature_views({"clean": test_audio["X"]}, test_image["X"])

    full_idx = np.arange(len(y))
    start = time.perf_counter()
    final_artifact = fit_candidate(selected_spec, candidates, feature_views, y, full_idx)
    final_fit_time = time.perf_counter() - start
    if selected_spec.kind == "ensemble":
        final_proba = np.mean(
            [
                predict_artifact(member_artifact, test_feature_views[candidates[member_name].feature_view]["clean"])
                for member_name, member_artifact in final_artifact["members"].items()
            ],
            axis=0,
        )
    else:
        final_proba = predict_artifact(final_artifact, test_feature_views[selected_spec.feature_view]["clean"])
    final_pred = cv.predict_with_bias(final_proba, selected_bias)
    final_row = report_row(
        selected_name,
        "robot_test_final",
        test_audio["y"],
        final_pred,
        selected_spec.feature_view,
        int(selected.get("n_features", 0)),
        final_fit_time,
    )
    final_row.update(
        {
            "selected_by": "hand_default_grouped_multimodal_stress_cv_only",
            "selected_stress_worst_macro_f1": selected["stress_worst_macro_f1"],
            "selected_stress_mean_macro_f1": selected["stress_mean_macro_f1"],
            "selected_clean_oof_macro_f1": selected["tuned_clean_macro_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )

    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "audio_path", "image_path", "label", "y", "group_key", "source"]
    ].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_audio["y"], final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "classical_multimodal_stress_cv_select_no_test_until_final",
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "classical_multimodal_stress_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "audio_feature_cache_dir": str(args.audio_feature_cache_dir.resolve()),
        "audio_stress_feature_dir": str(args.audio_stress_feature_dir.resolve()),
        "image_feature_dir": str(args.image_feature_dir.resolve()),
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "test_audio_timing": test_audio_timing,
        "test_image_timing": test_image_timing,
        "artifacts": {
            "stress_leaderboard": str(leaderboard_path.resolve()),
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

    print("\nFinal robot/test result after frozen multimodal stress-CV selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "feature_set",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_stress_worst_macro_f1",
                "selected_clean_oof_macro_f1",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Stress leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
