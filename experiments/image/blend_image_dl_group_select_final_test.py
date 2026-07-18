from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_image_dl_finetune_select_final_test as ft
import train_image_group_majority_select_final_test as gm
import train_image_handcrafted_ml_select_final_test as img
import tune_image_dl_checkpoint_bias_tta_select_final_test as tta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend a fine-tuned image DL checkpoint with image-only group-majority "
            "models. Blend weight and class bias are selected only on hand/default "
            "validation; robot/test is loaded after selection lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_dl_group_blend_select")
    parser.add_argument("--dl-source-run-slug", default="image_dl_finetune_resnet18_specimen_select")
    parser.add_argument("--dl-model", choices=["resnet18", "resnet34", "resnet50", "mobilenet_v3_large"], default="resnet18")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-mode", choices=["specimen", "segment"], default="specimen")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbones", default="resnet18,resnet34")
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("outputs/image_deep_features"))
    parser.add_argument(
        "--image-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/multimodal_classical_cv_select/image_features"),
    )
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--force-rebuild-image", action="store_true")
    parser.add_argument("--bias-penalty", type=float, default=0.02)
    return parser.parse_args()


def load_dl_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    source_dir = args.output / "audio_feature_benchmarks" / args.dl_source_run_slug
    checkpoint_path = source_dir / "models" / f"{args.dl_source_run_slug}_best_val_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = ft.make_model(args.dl_model, dropout=0.2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def group_row_proba(spec: gm.CandidateSpec, train_split: gm.GroupSplit, row_split: gm.GroupSplit) -> np.ndarray:
    model = spec.factory()
    model.fit(train_split.X_views[spec.view], train_split.y_group)
    group_proba = gm.aligned_proba(model, row_split.X_views[spec.view])
    return group_proba[row_split.row_to_group]


def evaluate_proba(
    name: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    args: argparse.Namespace,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for bias_mode, bias in [
        ("none", np.zeros(len(img.CLASS_NAMES), dtype=np.float64)),
        ("tuned", img.tune_class_bias(proba, y_true)[0]),
    ]:
        pred = img.predict_with_bias(proba, bias)
        row = img.make_report_row(
            model_name=name,
            split_name="hand_val",
            y_true=y_true,
            y_pred=pred,
            feature_set="dl_group_blend",
            feature_name="probability_blend",
            n_features=0,
            train_time_sec=0.0,
            predict_time_sec=0.0,
        )
        bias_l1 = float(np.abs(bias).sum())
        row.update(
            {
                **metadata,
                "bias_mode": bias_mode,
                "class_bias_json": json.dumps(bias.tolist()),
                "bias_l1": bias_l1,
                "regularized_val_score": float(row["macro_f1_4class"] - args.bias_penalty * bias_l1),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    ft.set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    for directory in [run_dir, report_dir, args.feature_cache_dir, args.image_feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("Protocol  = image-only DL/group blend val selection; robot/test after lock", flush=True)

    train_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_df = gm.add_image_hashes(train_df)
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    dl_model = load_dl_model(args, device)
    y_val, dl_val_proba = tta.predict_proba(
        dl_model,
        val_df,
        "plain",
        args.image_size,
        args.batch_size,
        args.num_workers,
        device,
    )

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    train_views, train_feature_timing = gm.build_all_train_views(train_df, backbones, args, device)
    train_split = gm.make_group_split(train_df, train_idx, train_views)
    val_split = gm.make_group_split(train_df, val_idx, train_views)
    train_full_split = gm.make_group_split(train_df, np.arange(len(train_df)), train_views)
    if not np.array_equal(y_val, val_split.row_y):
        raise AssertionError("DL and group validation labels are not aligned")

    candidate_names = [
        "resnet18_resnet34_group_knn5_distance",
        "resnet18_group_knn5_distance",
        "resnet34_group_knn5_distance",
        "hand_compact_resnet34_group_knn5_distance",
        "hand_compact_group_extra_trees",
        "hand_compact_group_lgbm",
    ]
    group_candidates = gm.make_candidates(args.random_state, train_views)
    rows = []
    rows.extend(
        evaluate_proba(
            "dl_resnet18_plain",
            dl_val_proba,
            y_val,
            args,
            {"candidate_kind": "dl_only", "group_model": "", "dl_weight": 1.0},
        )
    )
    group_val_probas: dict[str, np.ndarray] = {}
    for name in candidate_names:
        if name not in group_candidates:
            continue
        spec = group_candidates[name]
        print(f"VAL group candidate: {name}", flush=True)
        proba = group_row_proba(spec, train_split, val_split)
        group_val_probas[name] = proba
        rows.extend(
            evaluate_proba(
                f"group_{name}",
                proba,
                y_val,
                args,
                {"candidate_kind": "group_only", "group_model": name, "dl_weight": 0.0},
            )
        )
        for dl_weight in np.linspace(0.25, 0.75, 3):
            blend = dl_weight * dl_val_proba + (1.0 - dl_weight) * proba
            rows.extend(
                evaluate_proba(
                    f"blend_dl{dl_weight:.2f}_{name}",
                    blend,
                    y_val,
                    args,
                    {"candidate_kind": "blend", "group_model": name, "dl_weight": float(dl_weight)},
                )
            )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["regularized_val_score", "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_val_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(leaderboard[["model", "candidate_kind", "group_model", "dl_weight", "bias_mode", "macro_f1_4class", "regularized_val_score", "accuracy_4class"]].head(12).to_string(index=False), flush=True)

    selected = leaderboard.iloc[0].to_dict()
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    split_summary = {
        "protocol": "image_only_dl_group_blend_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "dl_source_run_slug": args.dl_source_run_slug,
        "group_candidate_names": candidate_names,
        "train_feature_timing": train_feature_timing,
        "bias_penalty": args.bias_penalty,
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)
    selection_summary = {
        "selected_without_test": selected,
        "selected_bias_mode": selected["bias_mode"],
        "selected_bias": selected_bias.tolist(),
        "selection_metric": "regularized_val_score",
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    img.write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_df = img.load_image_manifest(
        img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv"),
        test_dataset_dir,
        "robot_test",
    )
    test_df = gm.add_image_hashes(test_df)
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    y_test, dl_test_proba = tta.predict_proba(
        dl_model,
        test_df,
        "plain",
        args.image_size,
        args.batch_size,
        args.num_workers,
        device,
    )
    selected_kind = str(selected["candidate_kind"])
    selected_group_name = str(selected.get("group_model", ""))
    selected_dl_weight = float(selected.get("dl_weight", 1.0))
    final_proba = dl_test_proba
    if selected_kind in {"group_only", "blend"}:
        selected_spec = group_candidates[selected_group_name]
        test_views, test_feature_timing = gm.build_selected_test_views(test_df, selected_spec.view, args, device)
        test_split = gm.make_group_split(test_df, np.arange(len(test_df)), test_views)
        if not np.array_equal(y_test, test_split.row_y):
            raise AssertionError("DL and group test labels are not aligned")
        group_model = selected_spec.factory()
        group_model.fit(train_full_split.X_views[selected_spec.view], train_full_split.y_group)
        group_test_proba = gm.aligned_proba(group_model, test_split.X_views[selected_spec.view])[test_split.row_to_group]
        if selected_kind == "group_only":
            final_proba = group_test_proba
        else:
            final_proba = selected_dl_weight * dl_test_proba + (1.0 - selected_dl_weight) * group_test_proba
    final_pred = img.predict_with_bias(final_proba, selected_bias)
    final_row = img.make_report_row(
        model_name=selected["model"],
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        feature_set="dl_group_blend",
        feature_name="probability_blend",
        n_features=0,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_dl_group_blend",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_candidate_kind": selected_kind,
            "selected_group_model": selected_group_name,
            "selected_dl_weight": selected_dl_weight,
            "selected_bias_mode": selected.get("bias_mode"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(y_test, final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "image_path", "label", "y", "segment_group", "specimen_group", "source"]
    ].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(img.ID2LABEL)
    for class_id, class_name in img.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)
    protocol_summary = {
        "protocol": "image_only_dl_group_blend_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "final_test_report": final_row,
        "artifacts": {
            "val_leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    img.write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)
    print("\nFinal robot/test result after frozen DL/group blend selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_regularized_val_score",
                "selected_candidate_kind",
                "selected_group_model",
                "selected_dl_weight",
                "selected_bias_mode",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Val leaderboard:", leaderboard_path.resolve(), flush=True)
    print("Selection lock:", selection_path.resolve(), flush=True)
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
