from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
    resnet18,
    resnet34,
    resnet50,
)
from tqdm.auto import tqdm

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
            "Image-only deep-learning classifier fine-tuning. Hand/default is split "
            "into train/val; best epoch is selected by val macro F1 and locked before "
            "robot/test is loaded."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_dl_finetune_resnet18_specimen_select")
    parser.add_argument("--model", choices=["resnet18", "resnet34", "resnet50", "mobilenet_v3_large"], default="resnet18")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-mode", choices=["specimen", "segment", "row"], default="specimen")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sampler", choices=["weighted", "shuffle"], default="weighted")
    parser.add_argument("--loss", choices=["weighted_ce", "ce"], default="weighted_ce")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--resume-finalize",
        action="store_true",
        help="Reuse an existing val history and best checkpoint to write the lock and run final test.",
    )
    parser.add_argument(
        "--final-train-mode",
        choices=["best_checkpoint", "retrain_full_selected_epoch"],
        default="best_checkpoint",
        help=(
            "best_checkpoint evaluates the selected train-inner checkpoint. "
            "retrain_full_selected_epoch locks the best epoch on val, retrains from "
            "ImageNet initialization on all hand/default rows for that many epochs, "
            "then loads robot/test."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    output = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.20, 0.04)], p=0.8),
            transforms.RandomGrayscale(p=0.08),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def make_model(name: str, dropout: float) -> nn.Module:
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 4))
        return model
    if name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 4))
        return model
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 4))
        return model
    if name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 4)
        if dropout != 0.2 and isinstance(model.classifier[2], nn.Dropout):
            model.classifier[2].p = dropout
        return model
    raise ValueError(name)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        is_head = name.startswith("fc.") or name.startswith("classifier.")
        parameter.requires_grad = trainable or is_head


def make_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc.") or name.startswith("classifier."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    sampler_mode: str,
) -> tuple[DataLoader, DataLoader]:
    train_ds = ImageDataset(train_df, train_transform(image_size))
    val_ds = ImageDataset(val_df, eval_transform(image_size))
    if sampler_mode == "weighted":
        y = train_df["y"].to_numpy(dtype=np.int64)
        class_count = np.bincount(y, minlength=4)
        sample_weight = np.asarray([1.0 / max(class_count[label], 1) for label in y], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def make_train_loader(
    train_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    sampler_mode: str,
) -> DataLoader:
    train_ds = ImageDataset(train_df, train_transform(image_size))
    if sampler_mode == "weighted":
        y = train_df["y"].to_numpy(dtype=np.int64)
        class_count = np.bincount(y, minlength=4)
        sample_weight = np.asarray([1.0 / max(class_count[label], 1) for label in y], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float | list[list[int]]]:
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


def save_report_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    logits: np.ndarray,
    train_time_sec: float,
) -> dict[str, object]:
    pred = logits.argmax(axis=1)
    return img.make_report_row(
        model_name=model_name,
        split_name=split_name,
        y_true=y_true,
        y_pred=pred,
        feature_set=model_name,
        feature_name=f"{model_name}_finetuned_classifier",
        n_features=0,
        train_time_sec=train_time_sec,
        predict_time_sec=0.0,
    )


def make_criterion(frame: pd.DataFrame, loss_name: str, device: torch.device) -> nn.Module:
    class_weights = None
    if loss_name == "weighted_ce":
        counts = np.bincount(frame["y"].to_numpy(dtype=np.int64), minlength=4)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=class_weights)


def retrain_full_selected_epoch(
    args: argparse.Namespace,
    train_full_df: pd.DataFrame,
    selected_epoch: int,
    device: torch.device,
) -> tuple[nn.Module, float, Path]:
    model = make_model(args.model, args.dropout).to(device)
    loader = make_train_loader(
        train_full_df,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.sampler,
    )
    criterion = make_criterion(train_full_df, args.loss, device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start = time.perf_counter()
    for epoch in range(1, selected_epoch + 1):
        train_backbone = epoch > args.warmup_epochs
        set_backbone_trainable(model, train_backbone)
        optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
        loss = train_one_epoch(model, loader, criterion, optimizer, scaler, device)
        print(
            f"final_full_epoch={epoch:02d}/{selected_epoch:02d} loss={loss:.4f} "
            f"backbone_train={train_backbone}",
            flush=True,
        )
    train_time = time.perf_counter() - start
    return model, train_time, Path()


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
    print("Protocol  = image-only DL fine-tune; best val epoch locked before robot/test", flush=True)

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
    train_loader, val_loader = make_loaders(
        train_df,
        val_df,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.sampler,
    )

    criterion = make_criterion(train_df, args.loss, device)
    model = make_model(args.model, args.dropout).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_row: dict[str, object] | None = None
    best_state_path = model_dir / f"{args.run_slug}_best_val_model.pt"
    history_path = report_dir / f"{args.run_slug}_val_history.csv"
    history_rows = []
    start_all = time.perf_counter()
    stale_epochs = 0
    optimizer = None
    if args.resume_finalize:
        if not history_path.exists() or not best_state_path.exists():
            raise FileNotFoundError(
                f"Cannot --resume-finalize without {history_path} and {best_state_path}"
            )
        history = pd.read_csv(history_path)
        best_row = history.sort_values("macro_f1_4class", ascending=False).iloc[0].to_dict()
        print(
            f"Resuming from existing best checkpoint: epoch={int(best_row['epoch'])} "
            f"val_macro={float(best_row['macro_f1_4class']):.4f}",
            flush=True,
        )
    else:
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

        history = pd.DataFrame(history_rows)
        history.to_csv(history_path, index=False)
    if best_row is None:
        raise RuntimeError("No training epoch completed")

    split_summary = {
        "protocol": "image_only_dl_finetune_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "train_label_counts": img.label_counts(train_df["y"].to_numpy()),
        "val_label_counts": img.label_counts(val_df["y"].to_numpy()),
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

    final_model_source = "selected_train_inner_checkpoint"
    final_training_time = time.perf_counter() - start_all
    final_full_model_path = None
    if args.final_train_mode == "retrain_full_selected_epoch":
        print(
            "\nRetraining final model on all hand/default rows after selection lock "
            f"for selected_epoch={int(best_row['epoch'])}.",
            flush=True,
        )
        model, final_retrain_time, _ = retrain_full_selected_epoch(
            args,
            train_full_df,
            int(best_row["epoch"]),
            device,
        )
        final_model_source = "retrained_full_hand_default_selected_epoch"
        final_training_time = final_retrain_time
        final_full_model_path = model_dir / f"{args.run_slug}_final_full_selected_epoch_model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": jsonable_args(args),
                "selected_epoch": int(best_row["epoch"]),
                "selected_without_test": best_row,
                "label_map": img.LABEL_MAP,
            },
            final_full_model_path,
        )
    else:
        checkpoint = torch.load(best_state_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    test_df = img.load_image_manifest(
        img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv"),
        test_dataset_dir,
        "robot_test",
    )
    split_paths = img.save_split_manifests(run_dir, train_full_df, test_df, train_idx, val_idx)
    test_loader = DataLoader(
        ImageDataset(test_df, eval_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    y_test, test_logits = evaluate(model, test_loader, device)
    final_row = save_report_row(args.model, "robot_test_final", y_test, test_logits, final_training_time)
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_dl_finetune",
            "final_model_source": final_model_source,
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
        "protocol": "image_only_dl_finetune_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "final_test_report": final_row,
        "artifacts": {
            "history": str(history_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "best_checkpoint": str(best_state_path.resolve()),
            "final_full_model": str(final_full_model_path.resolve()) if final_full_model_path is not None else None,
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    img.write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)
    print("\nFinal robot/test result after frozen fine-tuned image classifier:", flush=True)
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
