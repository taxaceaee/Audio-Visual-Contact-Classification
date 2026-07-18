from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT = Path('outputs/audio_feature_benchmarks/avr_mlp_group_selection')
IMG = Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy')


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.10), nn.Linear(hidden, 3))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.body(x)


def fit(x, logits, y, idx, device, hidden=128):
    ci = idx[y[idx] > 0]
    mean = x[ci].mean(0, keepdims=True).astype(np.float32)
    std = (x[ci].std(0, keepdims=True) + 1e-5).astype(np.float32)
    xt = torch.from_numpy(((x[ci] - mean) / std).astype(np.float32)).to(device)
    base = torch.from_numpy(logits[ci].astype(np.float32)).to(device)
    target = torch.from_numpy((y[ci] - 1).astype(np.int64)).to(device)
    counts = torch.bincount(target, minlength=3).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0); weights /= weights.mean()
    model = ResidualMLP(x.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=2e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    model.train()
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(base + model(xt), target)
        loss.backward(); opt.step()
    return model.eval(), mean, std


def predict(model, mean, std, x, device):
    with torch.inference_mode():
        return model(torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device)).cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set('total240')
    af = audio_base.load_manifest(ROOT / 'audio_visual_dataset_default/dataset.csv', 'hand_train')
    frame = suite.load_manifest(ROOT / 'audio_visual_dataset_default/dataset.csv', ROOT / 'audio_visual_dataset_default', 'hand_train')
    y = frame.y.to_numpy(np.int64); groups = suite.specimen_group(frame.audio_file).to_numpy()
    image = np.load(IMG).astype(np.float32)
    if not np.array_equal(y, np.load(IMG.parent / 'y.npy').astype(np.int64)): raise AssertionError('image alignment')
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))
    assignment = np.full(len(y), -1, np.int64)
    for k, (_, va) in enumerate(folds): assignment[va] = k
    high = group_audio.load_highsr_oof(OUT.parent.parent)
    pair = specimen.load_pairwise_oof(OUT.parent.parent / 'audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')
    anchor = lift.anchor_lift_proba(af, high, pair)
    sources = broad.load_train_sources(OUT.parent.parent, y, assignment, 42)
    audio = lift.postprocess(af, lift.normalize(.95 * anchor + .05 * sources['report_gate_onehot']), 'segment_lift')
    p_contact, logits = avr.avr_inputs(audio)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof = np.zeros((len(y), 3), np.float32)
    for k, (tr, va) in enumerate(folds):
        model, mean, std = fit(image, logits, y, tr, device)
        oof[va] = predict(model, mean, std, image[va], device)
        print(f'fold {k+1}/5 done', flush=True)
    rows = []
    for alpha in [0.0, .05, .1, .2, .4, .6, 1.0, 1.5, 2.0]:
        mat = suite.normalize(np.exp(logits + alpha * oof))
        ca, ma = avr.aggregate_avr(frame.audio_file, p_contact, mat)
        for th in np.linspace(.30, .70, 17):
            pred = np.where(ca >= th, ma.argmax(1) + 1, 0)
            rows.append({'alpha': float(alpha), 'contact_threshold': float(th), 'macro_f1_4class': float(f1_score(y, pred, average='macro', zero_division=0)), 'binary_macro_f1': float(f1_score(y > 0, pred > 0, average='macro', zero_division=0))})
    board = pd.DataFrame(rows).sort_values(['macro_f1_4class', 'binary_macro_f1'], ascending=False).reset_index(drop=True)
    board.to_csv(OUT / 'hand_avr_mlp_leaderboard.csv', index=False)
    best = board.iloc[0].to_dict()
    lock = {'protocol': 'AVR_audio_anchor_visual_MLP_residual_contact_only_specimen_group_OOF', 'audio_source': 'locked audio anchor + report_gate_onehot@0.05, group-aware OOF', 'image_source': 'CLIP embedding', 'residual_model': '2-layer MLP hidden=128', 'test_loaded': False, 'selected_candidate': best}
    (OUT / 'selection_lock.json').write_text(json.dumps(lock, indent=2, default=float), encoding='utf-8')
    print(json.dumps(lock, indent=2, default=float))


if __name__ == '__main__': main()
