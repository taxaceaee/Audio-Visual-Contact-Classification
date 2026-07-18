from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base


ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT = Path('outputs')
REPORT = OUT / 'audio_feature_benchmarks/avr_group_selection'
MODE = os.environ.get('AVR_AUDIO_MODE', 'anchor')
REPORT = OUT / 'audio_feature_benchmarks' / ('avr_locked_audio_group_selection' if MODE == 'locked' else 'avr_group_selection')
NAMES = ['ambient', 'leaf', 'trunk', 'twig']
LABELS = np.arange(4)


def main():
    lock = json.loads((REPORT / 'selection_lock.json').read_text())
    if lock.get('test_loaded'):
        raise AssertionError('selection lock already used test')
    alpha = float(lock['selected_candidate']['alpha'])
    threshold = float(lock['selected_candidate']['contact_threshold'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_base.configure_feature_set('total240')

    # Train the residual on all hand rows, using only group-aware audio OOF.
    hand_audio = audio_base.load_manifest(ROOT / 'audio_visual_dataset_default/dataset.csv', 'hand_train')
    hand = suite.load_manifest(ROOT / 'audio_visual_dataset_default/dataset.csv', ROOT / 'audio_visual_dataset_default', 'hand_train')
    y_hand = hand.y.to_numpy(dtype=np.int64)
    image_hand = np.load(OUT / 'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy').astype(np.float32)
    if not np.array_equal(y_hand, np.load(OUT / 'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/y.npy').astype(np.int64)):
        raise AssertionError('hand image alignment')
    high_oof = group_audio.load_highsr_oof(OUT)
    pair_oof = specimen.load_pairwise_oof(OUT / 'audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')
    if MODE == 'locked':
        groups = suite.specimen_group(hand_audio.audio_file).to_numpy()
        folds = list(__import__('sklearn.model_selection', fromlist=['StratifiedGroupKFold']).StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y_hand)), y_hand, groups))
        assignment = np.full(len(y_hand), -1, dtype=np.int64)
        for fold_id, (_, val_idx) in enumerate(folds): assignment[val_idx] = fold_id
        sources = __import__('train_audio_broad_oof_meta_select_final_test', fromlist=['load_train_sources']).load_train_sources(OUT, y_hand, assignment, 42)
        audio_oof = lift.postprocess(hand_audio, lift.normalize(0.95 * lift.anchor_lift_proba(hand_audio, high_oof, pair_oof) + 0.05 * sources['report_gate_onehot']), 'segment_lift')
    else:
        audio_oof = lift.anchor_lift_proba(hand_audio, high_oof, pair_oof)
    _, audio_logits = avr.avr_inputs(audio_oof)
    train_idx = np.arange(len(y_hand))
    residual, mean, std = avr.fit_residual(image_hand, audio_logits, y_hand, train_idx, device)

    # Test is opened only after the AVR selection lock.
    test_audio = audio_base.load_manifest(ROOT / 'audio_visual_dataset_robo_default/dataset.csv', 'robot_test')
    raw_test = pd.read_csv(ROOT / 'audio_visual_dataset_robo_default/dataset.csv')
    image_test = np.load(OUT / 'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/X.npy').astype(np.float32)
    high_test = pd.read_csv(OUT / 'audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv')
    pair_test = pd.read_csv(OUT / 'audio_feature_benchmarks/audio_group_consistency_pair_blend_select/reports/audio_group_consistency_pair_blend_select_final_test_predictions.csv')
    for name, df in [('highsr', high_test), ('pairwise', pair_test)]:
        if not np.array_equal(df.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
            raise AssertionError(f'{name} test alignment')
    if MODE == 'locked':
        audio_frame = pd.read_csv(OUT / 'audio_feature_benchmarks/audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv')
        if not np.array_equal(audio_frame.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
            raise AssertionError('locked audio final alignment')
        audio_test = suite.normalize(audio_frame[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64))
    else:
        audio_test = lift.anchor_lift_proba(test_audio, high_test[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64), pair_test[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64))
    p_contact_audio, logits_audio = avr.avr_inputs(audio_test)
    residual_test = avr.residual_predict(residual, mean, std, image_test, device)
    p_material_window = suite.normalize(np.exp(logits_audio + alpha * residual_test))

    # AVR downstream: aggregate contact and conditional material independently at segment level.
    p_contact, p_material = avr.aggregate_avr(test_audio.audio_file, p_contact_audio, p_material_window)
    pred = np.where(p_contact >= threshold, p_material.argmax(axis=1) + 1, 0)
    y = test_audio.y.to_numpy(dtype=np.int64)
    binary_y, binary_p = (y > 0).astype(int), (pred > 0).astype(int)
    result = {
        'split': 'robot_test_final',
        'protocol': lock['protocol'],
        'n': int(len(y)),
        'locked_candidate': lock['selected_candidate'],
        'avr_invariants': {
            'audio_controls_contact': True,
            'residual_trained_contact_only': True,
            'audio_source_oof_group_aware': True,
            'downstream_separate_contact_material_aggregation': True,
            'four_class_argmax_used': False,
        },
        'accuracy_4class': float(accuracy_score(y, pred)),
        'macro_precision_4class': float(precision_score(y, pred, average='macro', zero_division=0)),
        'macro_recall_4class': float(recall_score(y, pred, average='macro', zero_division=0)),
        'macro_f1_4class': float(f1_score(y, pred, average='macro', zero_division=0)),
        'weighted_f1_4class': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'binary_accuracy_ambient_noambient': float(accuracy_score(binary_y, binary_p)),
        'binary_macro_precision_ambient_noambient': float(precision_score(binary_y, binary_p, average='macro', zero_division=0)),
        'binary_macro_recall_ambient_noambient': float(recall_score(binary_y, binary_p, average='macro', zero_division=0)),
        'binary_macro_f1_ambient_noambient': float(f1_score(binary_y, binary_p, average='macro', zero_division=0)),
        'per_class_4class': classification_report(y, pred, labels=LABELS, target_names=NAMES, output_dict=True, zero_division=0),
        'per_class_binary': classification_report(binary_y, binary_p, labels=[0, 1], target_names=['ambient', 'noambient'], output_dict=True, zero_division=0),
        'confusion_matrix_4class': confusion_matrix(y, pred, labels=LABELS).tolist(),
        'confusion_matrix_binary': confusion_matrix(binary_y, binary_p, labels=[0, 1]).tolist(),
    }
    (REPORT / 'avr_final_test_metrics.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(REPORT / 'avr_final_test_report.csv', index=False)
    pd.DataFrame(result['confusion_matrix_4class'], index=NAMES, columns=NAMES).to_csv(REPORT / 'avr_final_test_confusion_matrix_4class.csv')
    pd.DataFrame(result['confusion_matrix_binary'], index=['ambient', 'noambient'], columns=['ambient', 'noambient']).to_csv(REPORT / 'avr_final_test_confusion_matrix_binary.csv')
    print(json.dumps(result, indent=2, default=float))


if __name__ == '__main__':
    main()
