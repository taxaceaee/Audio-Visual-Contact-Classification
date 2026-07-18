from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

import train_image_handcrafted_ml_select_final_test as img
import train_image_timm_finetune_select_final_test as ft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a previously selected image-only timm fine-tune checkpoint. "
            "The selection lock is written from checkpoint validation metadata before "
            "robot/test is loaded."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def checkpoint_args(checkpoint: dict[str, object]) -> argparse.Namespace:
    saved = dict(checkpoint["args"])
    saved["root"] = Path(saved["root"]) if saved.get("root") else None
    saved["output"] = Path(saved.get("output", "outputs"))
    return argparse.Namespace(**saved)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint_args(checkpoint)
    if args.root is not None:
        saved_args.root = args.root
    if args.run_slug is not None:
        saved_args.run_slug = args.run_slug
    saved_args.output = args.output
    if args.batch_size is not None:
        saved_args.batch_size = args.batch_size
    if args.num_workers is not None:
        saved_args.num_workers = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(saved_args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = saved_args.output / "audio_feature_benchmarks" / saved_args.run_slug
    report_dir = run_dir / "reports"
    for directory in [run_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_full_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_idx, val_idx, split_info = img.split_train_val(
        train_full_df,
        val_size=float(saved_args.val_size),
        random_state=int(saved_args.random_state),
        split_mode=saved_args.split_mode,
    )
    train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_full_df.iloc[val_idx].reset_index(drop=True)
    split_paths = img.save_split_manifests(run_dir, train_full_df, None, train_idx, val_idx)

    best_row = checkpoint["best_row"]
    config = checkpoint["data_config"]
    history_path = report_dir / f"{saved_args.run_slug}_val_history_from_checkpoint.csv"
    pd.DataFrame([best_row]).to_csv(history_path, index=False)
    split_summary = {
        "protocol": "image_only_timm_finetune_checkpoint_eval_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "train_label_counts": img.label_counts(train_df["y"].to_numpy()),
        "val_label_counts": img.label_counts(val_df["y"].to_numpy()),
        "data_config": config,
        "args": ft.jsonable_args(saved_args),
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{saved_args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)
    selection_summary = {
        "selected_without_test": best_row,
        "selected_model": saved_args.model,
        "selected_epoch": int(best_row["epoch"]),
        "selection_metric": "val_macro_f1_4class",
        "best_checkpoint": str(args.checkpoint.resolve()),
        "history_path": str(history_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{saved_args.run_slug}_selected_without_test.json"
    img.write_json(selection_path, selection_summary)
    print("Selection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    model = ft.make_model(saved_args.model, float(saved_args.drop_rate)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    test_df = img.load_image_manifest(
        img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv"),
        test_dataset_dir,
        "robot_test",
    )
    split_paths = img.save_split_manifests(run_dir, train_full_df, test_df, train_idx, val_idx)
    test_loader = ft.make_eval_loader(
        test_df,
        config,
        int(saved_args.batch_size),
        int(saved_args.num_workers),
    )
    y_test, test_logits = ft.evaluate(model, test_loader, device)
    final_row = ft.report_row(saved_args.model, y_test, test_logits, time.perf_counter() - start)
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_timm_finetune_checkpoint",
            "selected_epoch": int(best_row["epoch"]),
            "selected_val_macro_f1": float(best_row["macro_f1_4class"]),
        }
    )
    final_report_path = report_dir / f"{saved_args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    final_pred = test_logits.argmax(axis=1)
    confusion_path = report_dir / f"{saved_args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(y_test, final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{saved_args.run_slug}_final_test_predictions.csv"
    pred_frame = test_df[
        ["audio_file", "image_file", "image_path", "label", "y", "segment_group", "specimen_group", "source"]
    ].copy()
    pred_frame["pred_y"] = final_pred
    pred_frame["pred_label"] = pred_frame["pred_y"].map(img.ID2LABEL)
    probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    for class_id, class_name in img.ID2LABEL.items():
        pred_frame[f"proba_{class_name}"] = probs[:, class_id]
    pred_frame.to_csv(predictions_path, index=False)
    protocol_summary = {
        "protocol": "image_only_timm_finetune_checkpoint_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "final_test_report": final_row,
        "artifacts": {
            "history": str(history_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "best_checkpoint": str(args.checkpoint.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    img.write_json(report_dir / f"{saved_args.run_slug}_protocol_summary.json", protocol_summary)
    print("\nFinal robot/test result after frozen checkpoint selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_epoch",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Selection lock:", selection_path.resolve(), flush=True)
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
