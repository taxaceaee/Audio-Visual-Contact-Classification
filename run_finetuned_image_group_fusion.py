from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

import run_multimodal_val_locked_suite as suite


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "finetuned_image_group_fusion"
NAMES = ["ambient", "leaf", "trunk", "twig"]
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrainDataset(Dataset):
    def __init__(self, frame, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.85, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.25, 0.25, 0.2, 0.04),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]) if train else transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self): return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        try:
            image = Image.open(str(row.image_path)).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        return self.transform(image), int(row.y)


def make_model(device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 4))
    return model.to(device)


def predict(model, frame, device):
    loader = DataLoader(TrainDataset(frame, False), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    rows = []
    model.eval()
    with torch.inference_mode():
        for x, _ in loader:
            rows.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return suite.normalize(np.concatenate(rows))


def train_epochs(model, frame, epochs, device, val_frame=None, val_y=None, audio_views=None):
    y = frame.y.to_numpy(dtype=np.int64)
    counts = np.bincount(y, minlength=4)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    loader = DataLoader(TrainDataset(frame, True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
        if val_frame is not None:
            pi = predict(model, val_frame, device)
            for alpha in np.linspace(0.0, 1.0, 21):
                scores = []
                for pa in audio_views.values():
                    p = suite.normalize(np.exp(alpha * np.log(suite.normalize(pa)) + (1 - alpha) * np.log(suite.normalize(pi))))
                    scores.append(float(f1_score(val_y, p.argmax(axis=1), average="macro", zero_division=0)))
                row = {"epoch": epoch, "audio_weight": float(alpha), "worst_view_macro_f1": min(scores), "mean_view_macro_f1": float(np.mean(scores))}
                if best is None or (row["worst_view_macro_f1"], row["mean_view_macro_f1"]) > (best["worst_view_macro_f1"], best["mean_view_macro_f1"]):
                    best = {**row, "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            print(f"epoch={epoch} best_worst={best['worst_view_macro_f1']:.4f} alpha={best['audio_weight']}", flush=True)
    return best


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full, train_idx, val_idx = suite.build_group_split(ROOT, REPORT)
    _, pa_full = suite.load_audio_oof(ROOT, OUT)
    # Use train-only stress OOF views for validation selection.
    run = OUT / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    lock_audio = json.loads((run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    cand = lock_audio["selected_candidate"]
    pair = suite.normalize(np.load(OUT / "audio_feature_benchmarks" / "audio_group_consistency_pair_blend_select" / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"))
    audio_views = {}
    for view in ["clean", "robot_mix", "bandlimit"]:
        high = np.load(run / "oof_proba" / cand / f"{view}_oof_proba.npy")
        audio_views[view] = suite.segment_lift(full.audio_file.iloc[val_idx], suite.normalize(0.8 * high[val_idx] + 0.2 * pair[val_idx]))
    train_frame = full.iloc[train_idx].copy()
    val_frame = full.iloc[val_idx].copy()
    model = make_model(device)
    best = train_epochs(model, train_frame, 8, device, val_frame, val_frame.y.to_numpy(dtype=np.int64), audio_views)
    lock = {"protocol": "finetuned_image_group_fusion_val_only", "group_column": "specimen_group", "group_overlap": 0, "test_loaded": False, "selected_epoch": best["epoch"], "selected_audio_weight": best["audio_weight"], "selected_worst_stress_macro_f1": best["worst_view_macro_f1"]}
    torch.save({"model_state": best["model_state"], "lock": lock}, REPORT / "selected_group_val_model.pt")
    (REPORT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, default=float))


if __name__ == "__main__":
    main()
