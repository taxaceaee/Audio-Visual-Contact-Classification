from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

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
            "Image-only timm classifier fine-tuning. Hand/default is split into "
            "train/val; best epoch is selected on val Macro F1 and locked before "
            "robot/test is loaded."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_timm_finetune_convnext_tiny_specimen_select")
    parser.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-mode", choices=["specimen", "segment", "row"], default="specimen")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=8e-4)
    parser.add_argument("--lr-backbone", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--sampler", choices=["weighted", "shuffle"], default="weighted")
    parser.add_argument("--loss", choices=["weighted_ce", "ce"], default="weighted_ce")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def model_config(model: nn.Module) -> dict[str, object]:
    config = timm.data.resolve_model_data_config(model)
    return dict(config)


def train_transform(config: dict[str, object]) -> transforms.Compose:
    input_size = config.get("input_size", (3, 224, 224))
    size = int(input_size[-1])
    mean = tuple(config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(config.get("std", (0.229, 0.224, 0.225)))
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.28, 0.28, 0.22, 0.04)], p=0.8),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )


def eval_transform(config: dict[str, object]) -> Callable:
    return timm.data.create_transform(**config, is_training=False)


def make_model(name: str, drop_rate: float) -> nn.Module:
    return timm.create_model(name, pretrained=True, num_classes=len(img.CLASS_NAMES), drop_rate=drop_rate)


def is_head_parameter(name: str) -> bool:
    return (
        name.startswith("head.")
        or name.startswith("classifier.")
        or name.startswith("fc.")
        or ".head." in name
        or ".classifier." in name
        or name.endswith(".head.weight")
        or name.endswith(".head.bias")
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or is_head_parameter(name)


def make_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_head_parameter(name):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def make_train_loader(
    train_df: pd.DataFrame,
    config: dict[str, object],
    batch_size: int,
    num_workers: int,
    sampler_mode: str,
) -> DataLoader:
    dataset = ImageDataset(train_df, train_transform(config))
    if sampler_mode == "weighted":
        y = train_df["y"].to_numpy(dtype=np.int64)
        class_count = np.bincount(y, minlength=len(img.CLASS_NAMES))
        sample_weight = np.asarray([1.0 / max(class_count[label], 1) for label in y], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_eval_loader(frame: pd.DataFrame, config: dict[str, object], batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ImageDataset(frame, eval_transform(config)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_criterion(frame: pd.DataFrame, loss_name: str, device: torch.device) -> nn.Module:
    class_weights = None
    if loss_name == "weighted_ce":
        counts = np.bincount(frame["y"].to_numpy(dtype=np.int64), minlength=len(img.CLASS_NAMES))
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=class_weights)


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(axis=1)
    y_true_binary = (y_true > 0).astype(np.int64)
    pred_binary = (pred > 0).astype(np.int64)
    return {
        "macro_f1_4class": f1_score(y_true, pred, average="macro", zero_division=0),
        "contact_macro_f1": f1_score(y_true, pred, labels=img.CONTACT_LABELS, average="macro", zero_division=0),
        "binary_macro_f1": f1_score(y_true_binary, pred_binary, average="macro", zero_division=0),
        "accuracy_4class": float((pred == y_true).mean()),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * len(labels)
        total += len(labels)
    return total_loss / max(total, 1)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_rows = []
    labels = []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_rows.append(logits.detach().cpu().float().numpy())
        labels.extend(np.asarray(y, dtype=np.int64).tolist())
    return np.asarray(labels, dtype=np.int64), np.concatenate(logits_rows, axis=0)


def report_row(model_name: str, y_true: np.ndarray, logits: np.ndarray, train_time: float) -> dict[str, object]:
    pred = logits.argmax(axis=1)
    return img.make_report_row(
        model_name=model_name,
        split_name="robot_test_final",
        y_true=y_true,
        y_pred=pred,
        feature_set=model_name,
        feature_name=f"{model_name}_timm_finetuned_classifier",
        n_features=0,
        train_time_sec=train_time,
        predict_time_sec=0.0,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("DEVICE    =", device, flush=True)
    print("Protocol  = image-only timm fine-tune; best val epoch locked before robot/test", flush=True)

    train_full_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_idx, val_idx, split_info = img.split_train_val(
        train_full_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_full_df, None, train_idx, val_idx)
    train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_full_df.iloc[val_idx].reset_index(drop=True)

    model = make_model(args.model, args.drop_rate).to(device)
    config = model_config(model)
    train_loader = make_train_loader(train_df, config, args.batch_size, args.num_workers, args.sampler)
    val_loader = make_eval_loader(val_df, config, args.batch_size, args.num_workers)
    criterion = make_criterion(train_df, args.loss, device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_row: dict[str, object] | None = None
    best_state_path = model_dir / f"{args.run_slug}_best_val_model.pt"
    history_rows = []
    stale_epochs = 0
    start_all = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_backbone = epoch > args.warmup_epochs
        set_backbone_trainable(model, train_backbone)
        optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        y_val, val_logits = evaluate(model, val_loader, device)
        metrics = metrics_from_logits(y_val, val_logits)
        elapsed = time.perf_counter() - start_all
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "train_backbone": train_backbone,
            "elapsed_sec": elapsed,
            **metrics,
        }
        history_rows.append(row)
        print(
            f"epoch={epoch:02d} loss={loss:.4f} val_macro={metrics['macro_f1_4class']:.4f} "
            f"contact={metrics['contact_macro_f1']:.4f} acc={metrics['accuracy_4class']:.4f} "
            f"backbone_train={train_backbone}",
            flush=True,
        )
        if best_row is None or metrics["macro_f1_4class"] > float(best_row["macro_f1_4class"]):
            best_row = row
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": jsonable_args(args),
                    "data_config": config,
                    "best_row": best_row,
                    "label_map": img.LABEL_MAP,
                },
                best_state_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} stale epochs.", flush=True)
            break

    if best_row is None:
        raise RuntimeError("No training epoch completed")
    history_path = report_dir / f"{args.run_slug}_val_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    split_summary = {
        "protocol": "image_only_timm_finetune_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "train_label_counts": img.label_counts(train_df["y"].to_numpy()),
        "val_label_counts": img.label_counts(val_df["y"].to_numpy()),
        "data_config": config,
        "args": jsonable_args(args),
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)
    selection_summary = {
        "selected_without_test": best_row,
        "selected_model": args.model,
        "selected_epoch": int(best_row["epoch"]),
        "selection_metric": "val_macro_f1_4class",
        "best_checkpoint": str(best_state_path.resolve()),
        "history_path": str(history_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    img.write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    checkpoint = torch.load(best_state_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_df = img.load_image_manifest(
        img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv"),
        test_dataset_dir,
        "robot_test",
    )
    split_paths = img.save_split_manifests(run_dir, train_full_df, test_df, train_idx, val_idx)
    test_loader = make_eval_loader(test_df, config, args.batch_size, args.num_workers)
    y_test, test_logits = evaluate(model, test_loader, device)
    final_row = report_row(args.model, y_test, test_logits, time.perf_counter() - start_all)
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_timm_finetune",
            "selected_epoch": int(best_row["epoch"]),
            "selected_val_macro_f1": float(best_row["macro_f1_4class"]),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    final_pred = test_logits.argmax(axis=1)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(y_test, final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
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
        "protocol": "image_only_timm_finetune_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "final_test_report": final_row,
        "artifacts": {
            "history": str(history_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "best_checkpoint": str(best_state_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    img.write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)
    print("\nFinal robot/test result after frozen timm fine-tuned image classifier:", flush=True)
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
    print("History:", history_path.resolve(), flush=True)
    print("Selection lock:", selection_path.resolve(), flush=True)
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
