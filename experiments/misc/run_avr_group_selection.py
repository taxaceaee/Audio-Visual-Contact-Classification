from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base


ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT = Path('outputs')
MODE = os.environ.get('AVR_AUDIO_MODE', 'anchor')
REPORT = OUT / 'audio_feature_benchmarks' / ('avr_locked_audio_group_selection' if MODE == 'locked' else 'avr_group_selection')
IMAGE_CACHE = OUT / 'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy'


def segment_keys(files: pd.Series) -> pd.Series:
    return files.astype(str).str.replace(r'_window_\d+.*$', '', regex=True).str.replace(r'^audio/', '', regex=True)


def avr_inputs(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = suite.normalize(p)
    contact = 1.0 - p[:, 0]
    material = suite.normalize(p[:, 1:4])
    logits = np.log(np.clip(material, 1e-8, 1.0))
    return contact, logits


def aggregate_avr(files: pd.Series, p_contact: np.ndarray, p_material: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = segment_keys(files)
    codes, uniques = pd.factorize(keys, sort=False)
    n = len(uniques)
    contact_seg = np.bincount(codes, weights=p_contact, minlength=n) / np.maximum(np.bincount(codes, minlength=n), 1)
    material_seg = np.vstack([np.bincount(codes, weights=p_material[:, c], minlength=n) for c in range(3)]).T
    material_seg = suite.normalize(material_seg)
    return contact_seg[codes], material_seg[codes]


class ResidualLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.residual = nn.Linear(dim, 3)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        return self.residual(x)


def fit_residual(x: np.ndarray, audio_logits: np.ndarray, y: np.ndarray, train_idx: np.ndarray, device: torch.device) -> tuple[ResidualLinear, np.ndarray, np.ndarray]:
    contact_idx = train_idx[y[train_idx] > 0]
    mean = x[contact_idx].mean(axis=0, keepdims=True)
    std = x[contact_idx].std(axis=0, keepdims=True) + 1e-5
    xs = ((x[contact_idx] - mean) / std).astype(np.float32)
    base = torch.from_numpy(audio_logits[contact_idx].astype(np.float32)).to(device)
    xt = torch.from_numpy(xs).to(device)
    target = torch.from_numpy((y[contact_idx] - 1).astype(np.int64)).to(device)
    counts = torch.bincount(target, minlength=3).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    model = ResidualLinear(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(base + model(xt), target)
        loss.backward()
        optimizer.step()
    return model.eval(), mean.astype(np.float32), std.astype(np.float32)


def residual_predict(model, mean, std, x: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        return model(torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device)).cpu().numpy()


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set('total240')
    audio_frame = audio_base.load_manifest(ROOT / 'audio_visual_dataset_default/dataset.csv', 'hand_train')
    frame = suite.load_manifest(ROOT / 'audio_visual_dataset_default/dataset.csv', ROOT / 'audio_visual_dataset_default', 'hand_train')
    frame['specimen_group'] = suite.specimen_group(frame.audio_file)
    y = frame.y.to_numpy(dtype=np.int64)
    if not np.array_equal(y, audio_frame.y.to_numpy(dtype=np.int64)):
        raise AssertionError('manifest alignment')
    image = np.load(IMAGE_CACHE).astype(np.float32)
    if not np.array_equal(y, np.load(IMAGE_CACHE.parent / 'y.npy').astype(np.int64)):
        raise AssertionError('image alignment')

    high = group_audio.load_highsr_oof(OUT)
    pair = specimen.load_pairwise_oof(OUT / 'audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')
    groups = frame.specimen_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))
    fold_assignment = np.full(len(y), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(folds):
        fold_assignment[val_idx] = fold_id
    audio_anchor = lift.anchor_lift_proba(audio_frame, high, pair)
    if MODE == 'locked':
        locked_sources = broad.load_train_sources(OUT, y, fold_assignment, 42)
        audio_p = lift.postprocess(audio_frame, lift.normalize(0.95 * audio_anchor + 0.05 * locked_sources['report_gate_onehot']), 'segment_lift')
        audio_description = 'group-aware OOF audio_only_lock anchor+report_gate_onehot@0.05'
    else:
        audio_p = audio_anchor
        audio_description = 'group-aware OOF highsr+pairwise anchor_lift'
    p_contact_audio, audio_logits = avr_inputs(audio_p)
    fold_summary = []
    oof_residual = np.zeros((len(y), 3), dtype=np.float32)
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        if overlap:
            raise AssertionError('specimen overlap')
        model, mean, std = fit_residual(image, audio_logits, y, train_idx, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        oof_residual[val_idx] = residual_predict(model, mean, std, image[val_idx], torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        fold_summary.append({'fold': fold_id, 'train_rows': len(train_idx), 'val_rows': len(val_idx), 'train_groups': len(set(groups[train_idx])), 'val_groups': len(set(groups[val_idx])), 'overlap': len(overlap)})
    (REPORT / 'fold_summary.json').write_text(json.dumps(fold_summary, indent=2), encoding='utf-8')

    rows = []
    for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        material = suite.normalize(np.exp(audio_logits + alpha * oof_residual))
        contact_agg, material_agg = aggregate_avr(frame.audio_file, p_contact_audio, material)
        for threshold in np.linspace(0.30, 0.70, 17):
            pred = np.where(contact_agg >= threshold, material_agg.argmax(axis=1) + 1, 0)
            rows.append({'alpha': float(alpha), 'contact_threshold': float(threshold), 'macro_f1_4class': float(f1_score(y, pred, average='macro', zero_division=0)), 'binary_macro_f1': float(f1_score(y > 0, pred > 0, average='macro', zero_division=0)), 'contact_recall': float(f1_score(y[y > 0], pred[y > 0], average='macro', zero_division=0))})
    leaderboard = pd.DataFrame(rows).sort_values(['macro_f1_4class', 'binary_macro_f1'], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(REPORT / 'hand_avr_leaderboard.csv', index=False)
    best = leaderboard.iloc[0].to_dict()
    lock = {'protocol': 'AVR_audio_anchor_visual_residual_contact_only_specimen_group_OOF', 'group_column': 'specimen_group', 'audio_source': audio_description, 'image_source': 'CLIP embedding', 'residual_loss': 'contact samples only; audio logits as fixed offset', 'downstream': 'segment-wise p_contact mean and conditional material mean; no four-class argmax', 'test_loaded': False, 'selected_candidate': best}
    (REPORT / 'selection_lock.json').write_text(json.dumps(lock, indent=2, default=float), encoding='utf-8')
    print(json.dumps(lock, indent=2, default=float))


if __name__ == '__main__':
    main()
