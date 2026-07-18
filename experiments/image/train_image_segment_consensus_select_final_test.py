from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

import train_image_handcrafted_ml_select_final_test as img


DECODERS = ("window", "segment_mean_proba", "segment_mean_log_proba", "segment_majority")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only handcrafted-feature + ML with train-only segment consensus "
            "selection. Hand/default is split into train/val; candidate model, class "
            "bias, and segment decoder are selected on val only. Robot/test is loaded "
            "after selected_without_test.json is written."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_segment_consensus_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["segment", "specimen", "row"], default="segment")
    parser.add_argument(
        "--image-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/multimodal_classical_cv_select/image_features"),
    )
    parser.add_argument("--force-rebuild-image", action="store_true")
    parser.add_argument("--profile", choices=["fast", "balanced"], default="fast")
    parser.add_argument(
        "--selection-metric",
        choices=["decoded_val_macro_f1", "decoded_val_contact_macro_f1", "decoded_val_binary_macro_f1"],
        default="decoded_val_macro_f1",
    )
    return parser.parse_args()


def bias_grid() -> list[np.ndarray]:
    grid = np.linspace(-1.2, 1.2, 13)
    biases = [np.zeros(4, dtype=np.float64)]
    for leaf in grid:
        for trunk in grid:
            for twig in grid:
                biases.append(np.asarray([0.0, leaf, trunk, twig], dtype=np.float64))
    return biases


def prepare_group_context(proba: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    log_proba = np.log(np.clip(proba, 1e-12, 1.0))
    sum_proba = np.zeros((len(counts), proba.shape[1]), dtype=np.float64)
    sum_log_proba = np.zeros_like(sum_proba)
    np.add.at(sum_proba, inverse, proba)
    np.add.at(sum_log_proba, inverse, log_proba)
    return {
        "inverse": inverse,
        "counts": counts,
        "log_proba": log_proba,
        "mean_proba": sum_proba / counts[:, None],
        "mean_log_proba": sum_log_proba / counts[:, None],
    }


def decode_predictions(
    proba: np.ndarray,
    groups: np.ndarray,
    decoder: str,
    bias: np.ndarray,
    context: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if decoder == "window":
        return img.predict_with_bias(proba, bias)

    if context is None:
        context = prepare_group_context(proba, groups)

    inverse = context["inverse"].astype(np.int64)
    if decoder == "segment_mean_proba":
        group_score = np.log(np.clip(context["mean_proba"], 1e-12, 1.0)) + bias.reshape(1, -1)
        return group_score.argmax(axis=1)[inverse].astype(np.int64)

    if decoder == "segment_mean_log_proba":
        group_score = context["mean_log_proba"] + bias.reshape(1, -1)
        return group_score.argmax(axis=1)[inverse].astype(np.int64)

    if decoder == "segment_majority":
        row_score = context["log_proba"] + bias.reshape(1, -1)
        window_pred = row_score.argmax(axis=1)
        group_counts = np.zeros((len(context["counts"]), 4), dtype=np.int64)
        np.add.at(group_counts, (inverse, window_pred), 1)
        group_choice = group_counts.argmax(axis=1)
        mean_score = context["mean_log_proba"] + bias.reshape(1, -1)
        tie_rows = np.where((group_counts == group_counts.max(axis=1, keepdims=True)).sum(axis=1) > 1)[0]
        for group_idx in tie_rows:
            tied = np.flatnonzero(group_counts[group_idx] == group_counts[group_idx].max())
            group_choice[group_idx] = int(tied[np.argmax(mean_score[group_idx, tied])])
        return group_choice[inverse].astype(np.int64)

    raise ValueError(f"Unknown decoder: {decoder}")


def score_decoded(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true_binary = (y_true > 0).astype(np.int64)
    pred_binary = (pred > 0).astype(np.int64)
    return {
        "decoded_val_macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "decoded_val_contact_macro_f1": f1_score(
            y_true,
            pred,
            labels=img.CONTACT_LABELS,
            average="macro",
            zero_division=0,
        ),
        "decoded_val_binary_macro_f1": f1_score(
            y_true_binary,
            pred_binary,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ),
    }


def tune_decoder_bias(
    proba: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    decoder: str,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    context = prepare_group_context(proba, groups) if decoder != "window" else None
    best_bias = np.zeros(4, dtype=np.float64)
    best_pred = decode_predictions(proba, groups, decoder, best_bias, context)
    best_scores = score_decoded(y_true, best_pred)
    best_score = best_scores["decoded_val_macro_f1"]
    for bias in bias_grid():
        pred = decode_predictions(proba, groups, decoder, bias, context)
        scores = score_decoded(y_true, pred)
        score = scores["decoded_val_macro_f1"]
        if score > best_score:
            best_score = score
            best_bias = bias
            best_scores = scores
            best_pred = pred
    return best_bias, best_scores, best_pred


def evaluate_candidate_decoders(
    model_name: str,
    view: str,
    val_proba: np.ndarray,
    y_val: np.ndarray,
    val_groups: np.ndarray,
    n_features: int,
    train_time: float,
    predict_time: float,
) -> list[dict]:
    rows = []
    for decoder in DECODERS:
        bias, scores, pred = tune_decoder_bias(val_proba, y_val, val_groups, decoder)
        row = img.make_report_row(
            model_name=model_name,
            split_name="hand_val",
            y_true=y_val,
            y_pred=pred,
            feature_set=view,
            feature_name=f"{view}+{decoder}",
            n_features=n_features,
            train_time_sec=train_time,
            predict_time_sec=predict_time,
        )
        row.update(
            {
                "view": view,
                "decoder": decoder,
                "candidate_kind": "direct",
                "decoded_val_macro_f1": scores["decoded_val_macro_f1"],
                "decoded_val_contact_macro_f1": scores["decoded_val_contact_macro_f1"],
                "decoded_val_binary_macro_f1": scores["decoded_val_binary_macro_f1"],
                "default_window_macro_f1": f1_score(y_val, val_proba.argmax(axis=1), average="macro", zero_division=0),
                "class_bias_json": json.dumps(bias.tolist()),
                "members_json": "[]",
            }
        )
        print(
            f"    {decoder:22s} macro={row['decoded_val_macro_f1']:.4f} "
            f"contact={row['decoded_val_contact_macro_f1']:.4f} "
            f"bias={np.round(bias, 2).tolist()}",
            flush=True,
        )
        rows.append(row)
    return rows


def build_ensemble_rows(
    leaderboard: pd.DataFrame,
    val_probas: dict[str, np.ndarray],
    y_val: np.ndarray,
    val_groups: np.ndarray,
) -> tuple[list[dict], dict[str, img.CandidateSpec], dict[str, np.ndarray]]:
    ranked = leaderboard.sort_values("decoded_val_macro_f1", ascending=False)["model"].tolist()
    recipes = {
        "image_segment_ensemble_top2": tuple(dict.fromkeys(ranked[:2])),
        "image_segment_ensemble_top3": tuple(dict.fromkeys(ranked[:3])),
        "image_segment_ensemble_top5": tuple(dict.fromkeys(ranked[:5])),
    }
    rows = []
    specs = {}
    probas = {}
    for name, members in recipes.items():
        if len(members) < 2:
            continue
        proba = np.mean([val_probas[member] for member in members], axis=0)
        print(f"\nVAL ensemble: {name} members={members}", flush=True)
        for decoder in DECODERS:
            bias, scores, pred = tune_decoder_bias(proba, y_val, val_groups, decoder)
            row = img.make_report_row(
                model_name=name,
                split_name="hand_val",
                y_true=y_val,
                y_pred=pred,
                feature_set="ensemble",
                feature_name=f"mean probability ensemble+{decoder}",
                n_features=0,
                train_time_sec=0.0,
                predict_time_sec=0.0,
            )
            row.update(
                {
                    "view": "ensemble",
                    "decoder": decoder,
                    "candidate_kind": "ensemble",
                    "decoded_val_macro_f1": scores["decoded_val_macro_f1"],
                    "decoded_val_contact_macro_f1": scores["decoded_val_contact_macro_f1"],
                    "decoded_val_binary_macro_f1": scores["decoded_val_binary_macro_f1"],
                    "default_window_macro_f1": f1_score(y_val, proba.argmax(axis=1), average="macro", zero_division=0),
                    "class_bias_json": json.dumps(bias.tolist()),
                    "members_json": json.dumps(list(members)),
                }
            )
            print(
                f"    {decoder:22s} macro={row['decoded_val_macro_f1']:.4f} "
                f"bias={np.round(bias, 2).tolist()}",
                flush=True,
            )
            rows.append(row)
        specs[name] = img.CandidateSpec(name=name, view="ensemble", kind="ensemble", members=members)
        probas[name] = proba
    return rows, specs, probas


def fit_predict_final(
    selected_spec: img.CandidateSpec,
    candidates: dict[str, img.CandidateSpec],
    train_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    y_train: np.ndarray,
) -> tuple[np.ndarray, object, float]:
    start = time.perf_counter()
    if selected_spec.kind == "direct":
        model = clone(selected_spec.factory())
        model.fit(train_views[selected_spec.view], y_train)
        proba = img.proba_aligned(model, test_views[selected_spec.view])
        return proba, {"kind": "direct", "view": selected_spec.view, "model": model}, time.perf_counter() - start

    artifacts = {}
    probas = []
    for member in selected_spec.members:
        member_proba, member_artifact, _ = fit_predict_final(
            candidates[member],
            candidates,
            train_views,
            test_views,
            y_train,
        )
        probas.append(member_proba)
        artifacts[member] = member_artifact
    return np.mean(probas, axis=0), {"kind": "ensemble", "members": artifacts}, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir, args.image_feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH       =", root_path.resolve(), flush=True)
    print("RUN_DIR         =", run_dir.resolve(), flush=True)
    print("Protocol        = image-only train/val segment-consensus selection; robot/test after lock", flush=True)

    train_csv = img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv")
    train_df = img.load_image_manifest(train_csv, train_dataset_dir, "hand_train")
    train_payload, train_timing = img.build_or_load_image_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.image_feature_cache_dir,
        force_rebuild=args.force_rebuild_image,
    )
    y_full = train_payload["y"]
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)
    X_train_inner = train_payload["X"][train_idx]
    y_train_inner = y_full[train_idx]
    X_val = train_payload["X"][val_idx]
    y_val = y_full[val_idx]
    val_groups = train_df.iloc[val_idx]["segment_group"].to_numpy()
    train_views, view_dims = img.build_views(X_train_inner)
    val_views, _ = img.build_views(X_val)

    split_summary = {
        "protocol": "image_only_segment_consensus_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_full_samples": int(len(train_df)),
        "train_full_label_counts": img.label_counts(y_full),
        "view_dims": view_dims,
        "train_image_timing": train_timing,
        "split_paths": split_paths,
        "decoders": list(DECODERS),
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)

    candidates = img.make_candidates(args.random_state, args.profile)
    val_probas = {}
    rows = []
    for spec in candidates.values():
        print(f"\nVAL candidate: {spec.name} view={spec.view}", flush=True)
        model = spec.factory()
        start = time.perf_counter()
        model.fit(train_views[spec.view], y_train_inner)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        proba = img.proba_aligned(model, val_views[spec.view])
        predict_time = time.perf_counter() - start
        val_probas[spec.name] = proba
        rows.extend(
            evaluate_candidate_decoders(
                spec.name,
                spec.view,
                proba,
                y_val,
                val_groups,
                train_views[spec.view].shape[1],
                train_time,
                predict_time,
            )
        )

    initial_leaderboard = pd.DataFrame(rows)
    ensemble_rows, ensemble_specs, ensemble_probas = build_ensemble_rows(
        initial_leaderboard,
        val_probas,
        y_val,
        val_groups,
    )
    rows.extend(ensemble_rows)
    candidates.update(ensemble_specs)
    val_probas.update(ensemble_probas)

    leaderboard = pd.DataFrame(rows).sort_values(
        [args.selection_metric, "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_val_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidates[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selected_decoder = str(selected["decoder"])
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_kind": selected_spec.kind,
        "selected_view": selected_spec.view,
        "selected_members": list(selected_spec.members),
        "selected_decoder": selected_decoder,
        "selected_bias": selected_bias.tolist(),
        "selection_metric": args.selection_metric,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    img.write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_csv = img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv")
    test_df = img.load_image_manifest(test_csv, test_dataset_dir, "robot_test")
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    test_payload, test_timing = img.build_or_load_image_cache(
        test_df,
        "robot_test",
        feature_dir=args.image_feature_cache_dir,
        force_rebuild=args.force_rebuild_image,
    )
    train_full_views, _ = img.build_views(train_payload["X"])
    test_views, _ = img.build_views(test_payload["X"])
    test_groups = test_df["segment_group"].to_numpy()
    final_proba, final_artifact, final_fit_time = fit_predict_final(
        selected_spec,
        candidates,
        train_full_views,
        test_views,
        y_full,
    )
    final_pred = decode_predictions(final_proba, test_groups, selected_decoder, selected_bias)
    final_row = img.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_payload["y"],
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=f"{selected_spec.view}+{selected_decoder}",
        n_features=0 if selected_spec.kind == "ensemble" else train_full_views[selected_spec.view].shape[1],
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_segment_consensus",
            "selected_decoder": selected_decoder,
            "selected_val_macro_f1": selected.get("decoded_val_macro_f1"),
            "selected_val_contact_macro_f1": selected.get("decoded_val_contact_macro_f1"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )

    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(test_payload["y"], final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        [
            "audio_file",
            "image_file",
            "image_path",
            "label",
            "y",
            "segment_group",
            "specimen_group",
            "source",
        ]
    ].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(img.ID2LABEL)
    for class_id, class_name in img.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "image_only_segment_consensus_no_test_until_lock",
            "selected": selection_summary,
            "artifact": final_artifact,
            "selected_bias": selected_bias,
            "selected_decoder": selected_decoder,
            "label_map": img.LABEL_MAP,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "image_only_segment_consensus_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "test_image_timing": test_timing,
        "final_test_report": final_row,
        "artifacts": {
            "val_leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "model_bundle": str(bundle_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    protocol_summary_path = report_dir / f"{args.run_slug}_protocol_summary.json"
    img.write_json(protocol_summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen image-only segment-consensus selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_decoder",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Val leaderboard:", leaderboard_path.resolve(), flush=True)
    print("Selection lock:", selection_path.resolve(), flush=True)
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
