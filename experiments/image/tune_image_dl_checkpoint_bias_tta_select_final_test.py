from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import train_image_dl_finetune_select_final_test as ft
import train_image_handcrafted_ml_select_final_test as img


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = Path(row.image_path)
        if path.exists():
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (640, 480), color=(0, 0, 0))
        else:
            image = Image.new("RGB", (640, 480), color=(0, 0, 0))
        return self.transform(image), int(row.y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune deterministic TTA and class bias for an already selected image-only "
            "DL checkpoint using hand/default validation only; load robot/test after "
            "writing the selection lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--source-run-slug", required=True)
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--model", choices=["resnet18", "resnet34", "resnet50", "mobilenet_v3_large"], required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-mode", choices=["specimen", "segment", "row"], default="specimen")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bias-penalty", type=float, default=0.02)
    return parser.parse_args()


def make_transform(image_size: int, variant: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if variant == "plain":
        ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
    elif variant == "hflip":
        ops = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda image: ImageOps.mirror(image)),
        ]
    elif variant == "center256":
        ops = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    elif variant == "center320":
        ops = [
            transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    else:
        raise ValueError(variant)
    return transforms.Compose([*ops, transforms.ToTensor(), normalize])


def recipe_variants(recipe: str) -> list[str]:
    recipes = {
        "plain": ["plain"],
        "plain_hflip": ["plain", "hflip"],
        "plain_center256": ["plain", "center256"],
        "plain_hflip_center256": ["plain", "hflip", "center256"],
        "all4": ["plain", "hflip", "center256", "center320"],
    }
    return recipes[recipe]


@torch.inference_mode()
def predict_proba(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    recipe: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    y_reference: np.ndarray | None = None
    proba_sum: np.ndarray | None = None
    for variant in recipe_variants(recipe):
        loader = DataLoader(
            ImageDataset(frame, make_transform(image_size, variant)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        labels = []
        probas = []
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probas.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.extend(np.asarray(y, dtype=np.int64).tolist())
        y_array = np.asarray(labels, dtype=np.int64)
        proba = np.concatenate(probas, axis=0)
        if y_reference is None:
            y_reference = y_array
            proba_sum = proba
        else:
            if not np.array_equal(y_reference, y_array):
                raise AssertionError("TTA label alignment mismatch")
            proba_sum = proba_sum + proba
    if y_reference is None or proba_sum is None:
        raise RuntimeError("No predictions produced")
    return y_reference, proba_sum / len(recipe_variants(recipe))


def main() -> None:
    args = parse_args()
    ft.set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_dir = args.output / "audio_feature_benchmarks" / args.source_run_slug
    checkpoint_path = source_dir / "models" / f"{args.source_run_slug}_best_val_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    print("ROOT_PATH   =", root_path.resolve(), flush=True)
    print("RUN_DIR     =", run_dir.resolve(), flush=True)
    print("CHECKPOINT  =", checkpoint_path.resolve(), flush=True)
    print("Protocol    = image-only TTA/bias val selection; robot/test after lock", flush=True)

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

    model = ft.make_model(args.model, dropout=0.2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    val_probas: dict[str, np.ndarray] = {}
    recipes = ["plain", "plain_hflip", "plain_center256", "plain_hflip_center256", "all4"]
    for recipe in recipes:
        start = time.perf_counter()
        y_val, proba = predict_proba(
            model,
            val_df,
            recipe,
            args.image_size,
            args.batch_size,
            args.num_workers,
            device,
        )
        predict_time = time.perf_counter() - start
        val_probas[recipe] = proba
        bias_options = [
            ("none", np.zeros(len(img.CLASS_NAMES), dtype=np.float64)),
            ("tuned", img.tune_class_bias(proba, y_val)[0]),
        ]
        for bias_mode, bias in bias_options:
            pred = img.predict_with_bias(proba, bias)
            row = img.make_report_row(
                model_name=f"{args.model}_{recipe}_{bias_mode}",
                split_name="hand_val",
                y_true=y_val,
                y_pred=pred,
                feature_set=args.model,
                feature_name=f"{args.model}_checkpoint_{recipe}",
                n_features=0,
                train_time_sec=0.0,
                predict_time_sec=predict_time,
            )
            bias_l1 = float(np.abs(bias).sum())
            row.update(
                {
                    "recipe": recipe,
                    "bias_mode": bias_mode,
                    "class_bias_json": json.dumps(bias.tolist()),
                    "bias_l1": bias_l1,
                    "regularized_val_score": float(row["macro_f1_4class"] - args.bias_penalty * bias_l1),
                }
            )
            print(
                f"{recipe:24s} {bias_mode:5s} macro={row['macro_f1_4class']:.4f} "
                f"reg={row['regularized_val_score']:.4f} acc={row['accuracy_4class']:.4f}",
                flush=True,
            )
            rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values(
        ["regularized_val_score", "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_val_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    selected_recipe = str(selected["recipe"])
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    split_summary = {
        "protocol": "image_only_checkpoint_tta_bias_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "source_run_slug": args.source_run_slug,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "recipes": recipes,
        "bias_penalty": args.bias_penalty,
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)
    selection_summary = {
        "selected_without_test": selected,
        "selected_recipe": selected_recipe,
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
    y_test, test_proba = predict_proba(
        model,
        test_df,
        selected_recipe,
        args.image_size,
        args.batch_size,
        args.num_workers,
        device,
    )
    final_pred = img.predict_with_bias(test_proba, selected_bias)
    final_row = img.make_report_row(
        model_name=f"{args.model}_{selected_recipe}_{selected['bias_mode']}",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        feature_set=args.model,
        feature_name=f"{args.model}_checkpoint_{selected_recipe}",
        n_features=0,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_checkpoint_tta_bias",
            "source_run_slug": args.source_run_slug,
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_recipe": selected_recipe,
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
        "protocol": "image_only_checkpoint_tta_bias_no_test_until_lock",
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
    print("\nFinal robot/test result after frozen checkpoint TTA/bias selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_regularized_val_score",
                "selected_recipe",
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
