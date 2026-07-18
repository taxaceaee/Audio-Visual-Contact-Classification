#!/usr/bin/env python3
"""
Standard Audio Spectrogram Transformer (AST) fine-tuning
for bee-queen 3-class audio classification (bee / noqueen / nobee).

Proper split policy (leakage-safe):
  - Group by recording session (parent clip / time window), NOT random files.
  - Many 2s wavs share the same hive+date+time prefix; random split would leak.
  - Stratified group assignment: ~70% / 15% / 15% of clips, all 3 classes in each split.

Model: MIT/ast-finetuned-audioset-10-10-0.4593
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    get_cosine_schedule_with_warmup,
)

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

import soundfile as sf


CLASSES = ["bee", "noqueen", "nobee"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
TARGET_SR = 16000
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


@dataclass
class TrainConfig:
    data_dir: str = "/root/bee_queen/nuhive_processed"
    output_dir: str = "/root/bee_queen/runs/ast_group_split"
    model_name: str = MODEL_NAME
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: int = 8
    batch_size: int = 8
    eval_batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 4
    max_length: int = 1024
    amp: bool = True
    max_train_samples: int = 0
    patience: int = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recording_group_key(path: str | Path) -> str:
    """
    Group key = recording session / parent window.

    Examples:
      CF003 - Active - Day - (214)-0-0.wav
        -> CF003 - Active - Day - (214)
      Hive1_12_06_2018_QueenBee_H1_audio___15_00_00-8-14-upsampled.wav
        -> Hive1_12_06_2018_QueenBee_H1_audio___15_00_00
      GH001 - Active - Day - 141022_0659_0751-...
        -> GH001 - Active - Day - 141022_0659_0751
      CF001 - Missing Queen - Day --0-0.wav
        -> CF001 - Missing Queen - Day
    """
    name = Path(path).name
    s = re.sub(r"-upsampled\.wav$", ".wav", name, flags=re.I)
    s = re.sub(r"\.wav$", "", s, flags=re.I)

    # CF / CJ style with parenthetical recording id: "... (214)-seg-win"
    m = re.match(r"(.+\(\d+\))", s)
    if m:
        return m.group(1).strip()

    # Hive NuHive style: "...___HH_MM_SS-seg-win"
    m = re.match(r"(.+?___\d{2}_\d{2}_\d{2})", s)
    if m:
        return m.group(1).strip()

    # GH / long continuous session names with trailing -a-b indices
    m = re.match(r"(.+?\d{6}_\d{4}_\d{4})", s)
    if m:
        return m.group(1).strip()

    # Fallback: drop trailing -N or -N-M segment indices
    s2 = re.sub(r"-\d+-\d+$", "", s)
    s2 = re.sub(r"-\d+$", "", s2)
    # collapse trailing double dashes from "Day --0-0"
    s2 = re.sub(r"[-_\s]+$", "", s2)
    return s2.strip() or s


def collect_samples(data_dir: Path) -> list[tuple[str, int, str]]:
    """Return list of (path, label, group_key)."""
    samples: list[tuple[str, int, str]] = []
    for cls in CLASSES:
        folder = data_dir / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.glob("*.wav")):
            g = recording_group_key(path)
            samples.append((str(path), LABEL2ID[cls], g))
    if not samples:
        raise RuntimeError(f"No wav files found under {data_dir}")
    return samples


def stratified_group_split(
    samples: list[tuple[str, int, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list, dict]:
    """
    Assign whole recording groups to train/val/test.

    Strategy:
      1. Each group gets a majority-class label for stratification.
      2. Within each majority class, shuffle groups (seeded).
      3. Greedy fill test then val then train targeting sample ratios,
         preferring groups that keep class clip counts balanced.
    """
    rng = random.Random(seed)

    by_group: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for item in samples:
        by_group[item[2]].append(item)

    group_info = []
    for g, items in by_group.items():
        labels = [y for _, y, _ in items]
        counts = Counter(labels)
        majority = max(counts.keys(), key=lambda k: (counts[k], -k))
        group_info.append(
            {
                "group": g,
                "n": len(items),
                "majority": majority,
                "class_counts": {ID2LABEL[k]: v for k, v in counts.items()},
                "items": items,
            }
        )

    # Bucket groups by majority class, shuffle within bucket
    buckets: dict[int, list[dict]] = defaultdict(list)
    for info in group_info:
        buckets[info["majority"]].append(info)
    for k in buckets:
        rng.shuffle(buckets[k])

    # Round-robin merge so classes interleave
    ordered: list[dict] = []
    pointers = {k: 0 for k in buckets}
    while True:
        progressed = False
        for cls_id in range(len(CLASSES)):
            b = buckets.get(cls_id, [])
            i = pointers[cls_id]
            if i < len(b):
                ordered.append(b[i])
                pointers[cls_id] = i + 1
                progressed = True
        if not progressed:
            break

    n_total = len(samples)
    n_test_target = int(round(n_total * test_ratio))
    n_val_target = int(round(n_total * val_ratio))

    # Target class counts roughly proportional
    global_cls = Counter(y for _, y, _ in samples)
    test_cls_target = {c: int(round(global_cls[c] * test_ratio)) for c in range(len(CLASSES))}
    val_cls_target = {c: int(round(global_cls[c] * val_ratio)) for c in range(len(CLASSES))}

    test_groups, val_groups, train_groups = [], [], []
    test_n = val_n = 0
    test_cls = Counter()
    val_cls = Counter()

    def class_need(target, current, items):
        # lower is better: how much overshoot after adding items
        add = Counter(y for _, y, _ in items)
        score = 0.0
        for c in range(len(CLASSES)):
            after = current[c] + add[c]
            # prefer filling under-target classes
            score += abs(after - target[c]) - abs(current[c] - target[c])
        return score

    remaining = ordered[:]

    # Fill test
    while remaining and test_n < n_test_target:
        # pick group that best matches class needs among next candidates
        best_i = 0
        best_score = None
        # look at first K remaining to preserve some order randomness
        K = min(12, len(remaining))
        for i in range(K):
            sc = class_need(test_cls_target, test_cls, remaining[i]["items"])
            # slight preference for groups that don't overshoot total too much
            sc += 0.01 * max(0, test_n + remaining[i]["n"] - n_test_target)
            if best_score is None or sc < best_score:
                best_score = sc
                best_i = i
        g = remaining.pop(best_i)
        # ensure we don't leave zero of a class for later splits if possible
        test_groups.append(g)
        test_n += g["n"]
        for _, y, _ in g["items"]:
            test_cls[y] += 1

    # Fill val
    while remaining and val_n < n_val_target:
        best_i = 0
        best_score = None
        K = min(12, len(remaining))
        for i in range(K):
            sc = class_need(val_cls_target, val_cls, remaining[i]["items"])
            sc += 0.01 * max(0, val_n + remaining[i]["n"] - n_val_target)
            if best_score is None or sc < best_score:
                best_score = sc
                best_i = i
        g = remaining.pop(best_i)
        val_groups.append(g)
        val_n += g["n"]
        for _, y, _ in g["items"]:
            val_cls[y] += 1

    train_groups = remaining

    def flatten(groups):
        out = []
        for g in groups:
            out.extend(g["items"])
        return out

    train_s, val_s, test_s = flatten(train_groups), flatten(val_groups), flatten(test_groups)

    # Safety: every split must contain all classes; if not, rebalance by moving groups
    def labels_present(split):
        return {y for _, y, _ in split}

    def rebalance_missing():
        nonlocal train_s, val_s, test_s, train_groups, val_groups, test_groups
        for split_name, groups in [("val", val_groups), ("test", test_groups)]:
            present = labels_present(flatten(groups))
            missing = set(range(len(CLASSES))) - present
            if not missing:
                continue
            # steal a small group from train that has missing class
            for miss in list(missing):
                donor_i = None
                for i, g in enumerate(train_groups):
                    if any(y == miss for _, y, _ in g["items"]):
                        donor_i = i
                        break
                if donor_i is None:
                    continue
                g = train_groups.pop(donor_i)
                if split_name == "val":
                    val_groups.append(g)
                else:
                    test_groups.append(g)
        train_s = flatten(train_groups)
        val_s = flatten(val_groups)
        test_s = flatten(test_groups)

    rebalance_missing()

    # No group overlap
    g_train = {g for *_, g in train_s}
    g_val = {g for *_, g in val_s}
    g_test = {g for *_, g in test_s}
    assert g_train.isdisjoint(g_val) and g_train.isdisjoint(g_test) and g_val.isdisjoint(g_test)

    def cls_counts(split):
        c = Counter(y for _, y, _ in split)
        return {ID2LABEL[i]: c[i] for i in range(len(CLASSES))}

    meta = {
        "policy": "stratified_group_by_recording_session",
        "group_definition": (
            "recording session / parent window "
            "(Hive...___HH_MM_SS or CF...(id) or GH continuous session)"
        ),
        "n_groups_total": len(by_group),
        "n_groups_train": len(g_train),
        "n_groups_val": len(g_val),
        "n_groups_test": len(g_test),
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
        "class_counts_train": cls_counts(train_s),
        "class_counts_val": cls_counts(val_s),
        "class_counts_test": cls_counts(test_s),
        "ratios": {
            "train": len(train_s) / n_total,
            "val": len(val_s) / n_total,
            "test": len(test_s) / n_total,
        },
        "group_overlap_check": {
            "train_val": len(g_train & g_val),
            "train_test": len(g_train & g_test),
            "val_test": len(g_val & g_test),
        },
        "groups_val": sorted(g_val),
        "groups_test": sorted(g_test),
    }
    return train_s, val_s, test_s, meta


def load_mono_16k(path: str) -> np.ndarray:
    if torchaudio is not None:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        return wav.numpy().astype(np.float32)

    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        duration = len(audio) / float(sr)
        n = int(round(duration * TARGET_SR))
        if n <= 1:
            return audio.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio.astype(np.float32)


class BeeQueenDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int, str]],
        feature_extractor: ASTFeatureExtractor,
        max_length: int,
        train: bool = False,
    ):
        self.samples = samples
        self.fe = feature_extractor
        self.max_length = max_length
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label, _group = self.samples[idx]
        audio = load_mono_16k(path)

        if self.train and len(audio) > 1:
            if random.random() < 0.5:
                gain = random.uniform(0.8, 1.2)
                audio = np.clip(audio * gain, -1.0, 1.0)
            if random.random() < 0.5:
                shift = random.randint(-len(audio) // 10, len(audio) // 10)
                audio = np.roll(audio, shift)

        feats = self.fe(
            audio,
            sampling_rate=TARGET_SR,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_values = feats["input_values"].squeeze(0)
        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


def collate_fn(batch: list[dict]) -> dict:
    input_values = torch.stack([b["input_values"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_values": input_values, "labels": labels}


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        input_values = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(input_values=input_values)
            loss = loss_fn(out.logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
        preds = out.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": classification_report(
            y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    samples = collect_samples(data_dir)
    print(f"Total samples: {len(samples)}")
    counts = Counter(y for _, y, _ in samples)
    print("Class counts:", {ID2LABEL[i]: counts[i] for i in range(len(CLASSES))})
    n_groups = len({g for *_, g in samples})
    print(f"Recording groups: {n_groups}")

    train_s, val_s, test_s, split_meta = stratified_group_split(
        samples, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )
    if cfg.max_train_samples and cfg.max_train_samples < len(train_s):
        rng = random.Random(cfg.seed)
        train_s = rng.sample(train_s, cfg.max_train_samples)

    print("=== GROUP-SAFE SPLIT ===")
    print(json.dumps({k: v for k, v in split_meta.items() if not k.startswith("groups_")}, indent=2))
    for name, split in [("train", train_s), ("val", val_s), ("test", test_s)]:
        present = sorted({ID2LABEL[y] for _, y, _ in split})
        if len(present) < len(CLASSES):
            raise RuntimeError(f"Split {name} missing classes: {present}")

    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    for name, split in [("train", train_s), ("val", val_s), ("test", test_s)]:
        with open(out_dir / f"{name}_files.txt", "w") as f:
            for p, y, g in split:
                f.write(f"{y}\t{ID2LABEL[y]}\t{g}\t{p}\n")

    feature_extractor = ASTFeatureExtractor.from_pretrained(cfg.model_name)
    model = ASTForAudioClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(CLASSES),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_ds = BeeQueenDataset(train_s, feature_extractor, cfg.max_length, train=True)
    val_ds = BeeQueenDataset(val_s, feature_extractor, cfg.max_length, train=False)
    test_ds = BeeQueenDataset(test_s, feature_extractor, cfg.max_length, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    train_labels = np.array([y for _, y, _ in train_s])
    class_count = np.bincount(train_labels, minlength=len(CLASSES)).astype(np.float64)
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    weight_t = torch.tensor(class_weight, dtype=torch.float32, device=device)
    print("Class weights:", {ID2LABEL[i]: float(class_weight[i]) for i in range(len(CLASSES))})
    loss_fn = nn.CrossEntropyLoss(weight=weight_t)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    history = []
    best_f1 = -1.0
    bad_epochs = 0
    best_path = out_dir / "checkpoints" / "best"

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=True)
        for batch in pbar:
            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                out = model(input_values=input_values)
                loss = loss_fn(out.logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = labels.size(0)
            running += float(loss.item()) * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.4f}")

        train_loss = running / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device, cfg.amp)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        print(val_metrics["report"])

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["macro_f1"] > best_f1 + 1e-4:
            best_f1 = val_metrics["macro_f1"]
            bad_epochs = 0
            model.save_pretrained(best_path)
            feature_extractor.save_pretrained(best_path)
            with open(out_dir / "best_val.json", "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_weighted_f1": val_metrics["weighted_f1"],
                        "confusion_matrix": val_metrics["confusion_matrix"],
                        "report": val_metrics["report"],
                    },
                    f,
                    indent=2,
                )
            print(f"  -> saved best checkpoint (macro_f1={best_f1:.4f})")
        else:
            bad_epochs += 1
            print(f"  -> no improve ({bad_epochs}/{cfg.patience})")
            if bad_epochs >= cfg.patience:
                print("Early stopping.")
                break

    print("Loading best checkpoint for test evaluation...")
    model = ASTForAudioClassification.from_pretrained(best_path)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device, cfg.amp)
    result = {
        "config": asdict(cfg),
        "split": split_meta,
        "best_val_macro_f1": best_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_loss": test_metrics["loss"],
        "test_report": test_metrics["report"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "note": (
            "Metrics use group-safe split by recording session; "
            "do not compare naively to random-file split (leaky)."
        ),
    }
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "test_report.txt", "w") as f:
        f.write("GROUP-SAFE SPLIT (by recording session)\n")
        f.write(test_metrics["report"])
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(np.array(test_metrics["confusion_matrix"])))
        f.write("\n")

    print("=" * 60)
    print("TEST RESULTS (group-safe split)")
    print(f"accuracy     : {test_metrics['accuracy']:.4f}")
    print(f"macro_f1     : {test_metrics['macro_f1']:.4f}")
    print(f"weighted_f1  : {test_metrics['weighted_f1']:.4f}")
    print(test_metrics["report"])
    print("Confusion matrix:\n", np.array(test_metrics["confusion_matrix"]))
    print(f"Saved to: {out_dir}")
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="AST fine-tune with group-safe split")
    p.add_argument("--data_dir", default=TrainConfig.data_dir)
    p.add_argument("--output_dir", default=TrainConfig.output_dir)
    p.add_argument("--model_name", default=TrainConfig.model_name)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--eval_batch_size", type=int, default=TrainConfig.eval_batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        patience=args.patience,
        amp=not args.no_amp,
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    train(parse_args())
