from __future__ import annotations

import copy
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


ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT = Path('outputs')
REPORT = OUT / 'audio_feature_benchmarks/domain_augmented_resnet_group_selection'
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFrame(Dataset):
    def __init__(self, frame: pd.DataFrame, profile: str, train: bool):
        self.frame = frame.reset_index(drop=True)
        if not train:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        elif profile == 'camera_heavy':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.75, 1.35)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.45, 0.45, 0.35, 0.08)], p=0.85),
                transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.0))], p=0.45),
                transforms.RandomGrayscale(p=0.12),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.RandomErasing(p=0.18, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.85, 1.2)), transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.25, 0.25, 0.2, 0.04), transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self): return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        try: image = Image.open(str(row.image_path)).convert('RGB')
        except Exception: image = Image.new('RGB', (224, 224))
        return self.transform(image), int(row.y)


def make_model(device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.fc.in_features, 4))
    return model.to(device)


def predict(model, frame, device):
    loader = DataLoader(ImageFrame(frame, 'eval', False), batch_size=96, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    out = []
    model.eval()
    with torch.inference_mode():
        for x, _ in loader: out.append(torch.softmax(model(x.to(device, non_blocking=True)), 1).cpu().numpy())
    return suite.normalize(np.concatenate(out))


def train_profile(profile, train_frame, val_frame, audio_views, device):
    model = make_model(device)
    y = train_frame.y.to_numpy(dtype=np.int64)
    counts = np.bincount(y, minlength=4)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device); weights /= weights.mean()
    loader = DataLoader(ImageFrame(train_frame, profile, True), batch_size=96, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best = None
    for epoch in range(1, 9):
        model.train()
        for x, target in loader:
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(x.to(device, non_blocking=True)), target.to(device, non_blocking=True)); loss.backward(); optimizer.step()
        pi = predict(model, val_frame, device)
        for alpha in np.linspace(0, 1, 21):
            scores = []
            for pa in audio_views.values():
                p = suite.normalize(np.exp(alpha*np.log(suite.normalize(pa)) + (1-alpha)*np.log(suite.normalize(pi))))
                scores.append(float(f1_score(val_frame.y, p.argmax(1), average='macro', zero_division=0)))
            row = {'profile': profile, 'epoch': epoch, 'audio_weight': float(alpha), 'worst_view_macro_f1': min(scores), 'mean_view_macro_f1': float(np.mean(scores))}
            if best is None or (row['worst_view_macro_f1'], row['mean_view_macro_f1']) > (best['worst_view_macro_f1'], best['mean_view_macro_f1']):
                best = {**row, 'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print(profile, epoch, best['worst_view_macro_f1'], best['audio_weight'], flush=True)
    return best


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full, train_idx, val_idx = suite.build_group_split(ROOT, REPORT)
    run = OUT / 'audio_feature_benchmarks/audio_highsr_temporal_tta_select'
    audio_lock = json.loads((run / 'reports/audio_highsr_temporal_tta_select_selected_without_test.json').read_text())
    cand = audio_lock['selected_candidate']; pair = suite.normalize(np.load(OUT / 'audio_feature_benchmarks/audio_group_consistency_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy'))
    audio_views = {}
    for view in ['clean', 'robot_mix', 'bandlimit']:
        high = np.load(run / 'oof_proba' / cand / f'{view}_oof_proba.npy')
        audio_views[view] = suite.segment_lift(full.audio_file.iloc[val_idx], suite.normalize(0.8*high[val_idx] + 0.2*pair[val_idx]))
    train_frame, val_frame = full.iloc[train_idx].copy(), full.iloc[val_idx].copy()
    results = [train_profile(profile, train_frame, val_frame, audio_views, device) for profile in ['standard', 'camera_heavy']]
    best = max(results, key=lambda x: (x['worst_view_macro_f1'], x['mean_view_macro_f1']))
    torch.save({'model_state': best['state'], 'lock': {k: v for k, v in best.items() if k != 'state'}}, REPORT / 'selected_group_val_model.pt')
    lock = {'protocol': 'domain_augmented_resnet_specimen_group_hand_only', 'group_column': 'specimen_group', 'group_overlap': 0, 'test_loaded': False, 'selected_profile': best['profile'], 'selected_epoch': best['epoch'], 'selected_audio_weight': best['audio_weight'], 'selected_worst_stress_macro_f1': best['worst_view_macro_f1']}
    (REPORT / 'selection_lock.json').write_text(json.dumps(lock, indent=2, default=float))
    pd.DataFrame([{k: v for k, v in x.items() if k != 'state'} for x in results]).to_csv(REPORT / 'hand_group_domain_aug_leaderboard.csv', index=False)
    print(json.dumps(lock, indent=2, default=float))


if __name__ == '__main__': main()
