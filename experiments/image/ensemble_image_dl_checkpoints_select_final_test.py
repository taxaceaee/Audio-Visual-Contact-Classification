from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_image_dl_finetune_select_final_test as ft
import train_image_handcrafted_ml_select_final_test as img
import tune_image_dl_checkpoint_bias_tta_select_final_test as tta


@dataclass(frozen=True)
class MemberSpec:
    name: str
    model_name: str
    source_run_slug: str
    recipe: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only checkpoint probability ensemble. Member weights and class "
            "bias are selected on hand/default validation only; robot/test is loaded "
            "after the selection lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_dl_checkpoint_ensemble_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-mode", choices=["specimen", "segment", "row"], default="specimen")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument("--bias-penalty", type=float, default=0.02)
    return parser.parse_args()


def default_members() -> list[MemberSpec]:
    return [
        MemberSpec("resnet18_plain", "resnet18", "image_dl_finetune_resnet18_specimen_select", "plain"),
        MemberSpec("resnet34_plain", "resnet34", "image_dl_finetune_resnet34_specimen_select", "plain"),
        MemberSpec("mobilenetv3_plain", "mobilenet_v3_large", "image_dl_finetune_mobilenetv3_specimen_select", "plain"),
        MemberSpec("mobilenetv3_all4", "mobilenet_v3_large", "image_dl_finetune_mobilenetv3_specimen_select", "all4"),
    ]


def load_member_model(member: MemberSpec, output: Path, device: torch.device) -> torch.nn.Module:
    source_dir = output / "audio_feature_benchmarks" / member.source_run_slug
    checkpoint_path = source_dir / "models" / f"{member.source_run_slug}_best_val_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = ft.make_model(member.model_name, dropout=0.2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def weight_grid(n_members: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    rows = []
    for parts in itertools.product(range(units + 1), repeat=n_members):
        if sum(parts) != units:
            continue
        weight = np.asarray(parts, dtype=np.float64) / units
        if np.count_nonzero(weight) == 0:
            continue
        rows.append(weight)
    return rows


def weighted_proba(member_probas: dict[str, np.ndarray], member_names: list[str], weights: np.ndarray) -> np.ndarray:
    first_active = next(name for name, weight in zip(member_names, weights) if weight > 0)
    proba = np.zeros_like(member_probas[first_active], dtype=np.float64)
    for name, weight in zip(member_names, weights):
        if weight <= 0:
            continue
        proba += float(weight) * member_probas[name]
    return proba


def main() -> None:
    args = parse_args()
    ft.set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    for directory in [run_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("Protocol  = image-only checkpoint ensemble val selection; robot/test after lock", flush=True)

    train_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    members = default_members()
    member_probas: dict[str, np.ndarray] = {}
    y_val: np.ndarray | None = None
    member_timing = {}
    for member in members:
        print(f"VAL member: {member.name}", flush=True)
        model = load_member_model(member, args.output, device)
        start = time.perf_counter()
        y_current, proba = tta.predict_proba(
            model,
            val_df,
            member.recipe,
            args.image_size,
            args.batch_size,
            args.num_workers,
            device,
        )
        member_timing[member.name] = time.perf_counter() - start
        if y_val is None:
            y_val = y_current
        elif not np.array_equal(y_val, y_current):
            raise AssertionError("Validation labels are not aligned across members")
        member_probas[member.name] = proba
    if y_val is None:
        raise RuntimeError("No validation predictions")

    rows = []
    member_names = [member.name for member in members]
    for weights in weight_grid(len(member_names), args.weight_step):
        proba = weighted_proba(member_probas, member_names, weights)
        active = [name for name, weight in zip(member_names, weights) if weight > 0]
        name = "ensemble_" + "_".join(f"{member}:{weight:.2f}" for member, weight in zip(member_names, weights) if weight > 0)
        bias_options = [
            ("none", np.zeros(len(img.CLASS_NAMES), dtype=np.float64)),
            ("tuned", img.tune_class_bias(proba, y_val)[0]),
        ]
        for bias_mode, bias in bias_options:
            pred = img.predict_with_bias(proba, bias)
            row = img.make_report_row(
                model_name=name,
                split_name="hand_val",
                y_true=y_val,
                y_pred=pred,
                feature_set="checkpoint_ensemble",
                feature_name="weighted_probability_ensemble",
                n_features=0,
                train_time_sec=0.0,
                predict_time_sec=0.0,
            )
            bias_l1 = float(np.abs(bias).sum())
            row.update(
                {
                    "member_names_json": json.dumps(member_names),
                    "active_members_json": json.dumps(active),
                    "weights_json": json.dumps(weights.tolist()),
                    "bias_mode": bias_mode,
                    "class_bias_json": json.dumps(bias.tolist()),
                    "bias_l1": bias_l1,
                    "regularized_val_score": float(row["macro_f1_4class"] - args.bias_penalty * bias_l1),
                }
            )
            rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values(
        ["regularized_val_score", "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_val_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(leaderboard[["model", "bias_mode", "macro_f1_4class", "regularized_val_score", "accuracy_4class"]].head(10).to_string(index=False), flush=True)

    selected = leaderboard.iloc[0].to_dict()
    selected_weights = np.asarray(json.loads(selected["weights_json"]), dtype=np.float64)
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    split_summary = {
        "protocol": "image_only_checkpoint_ensemble_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "members": [member.__dict__ for member in members],
        "member_val_predict_time_sec": member_timing,
        "weight_step": args.weight_step,
        "bias_penalty": args.bias_penalty,
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)
    selection_summary = {
        "selected_without_test": selected,
        "selected_weights": selected_weights.tolist(),
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
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    test_member_probas: dict[str, np.ndarray] = {}
    y_test: np.ndarray | None = None
    for member in members:
        if selected_weights[member_names.index(member.name)] <= 0.0:
            continue
        print(f"TEST member: {member.name}", flush=True)
        model = load_member_model(member, args.output, device)
        y_current, proba = tta.predict_proba(
            model,
            test_df,
            member.recipe,
            args.image_size,
            args.batch_size,
            args.num_workers,
            device,
        )
        if y_test is None:
            y_test = y_current
        elif not np.array_equal(y_test, y_current):
            raise AssertionError("Test labels are not aligned across members")
        test_member_probas[member.name] = proba
    if y_test is None:
        raise RuntimeError("No test predictions")
    test_proba = weighted_proba(test_member_probas, member_names, selected_weights)
    final_pred = img.predict_with_bias(test_proba, selected_bias)
    final_row = img.make_report_row(
        model_name=selected["model"],
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        feature_set="checkpoint_ensemble",
        feature_name="weighted_probability_ensemble",
        n_features=0,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_checkpoint_ensemble",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_weights_json": json.dumps(selected_weights.tolist()),
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
        prediction_frame[f"proba_{class_name}"] = test_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)
    protocol_summary = {
        "protocol": "image_only_checkpoint_ensemble_no_test_until_lock",
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
    print("\nFinal robot/test result after frozen checkpoint ensemble selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_regularized_val_score",
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
