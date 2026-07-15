from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

import run_domain_augmented_resnet_group_selection as impl
import run_multimodal_val_locked_suite as suite


ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT = Path('outputs')
REPORT = OUT / 'audio_feature_benchmarks/domain_augmented_resnet_group_selection'
NAMES = ['ambient', 'leaf', 'trunk', 'twig']
LABELS = np.arange(4)


def main():
    lock = json.loads((REPORT / 'selection_lock.json').read_text())
    if lock.get('test_loaded'):
        raise AssertionError('selection lock already opened test')
    full, _, _ = suite.build_group_split(ROOT, REPORT)
    model = impl.make_model(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    saved = torch.load(REPORT / 'selected_group_val_model.pt', map_location='cpu')
    model.load_state_dict(saved['model_state'])
    device = next(model.parameters()).device

    # Test is opened only after the hand-only lock.
    test = suite.load_manifest(ROOT / 'audio_visual_dataset_robo_default/dataset.csv', ROOT / 'audio_visual_dataset_robo_default', 'robot_test')
    raw_test = pd.read_csv(ROOT / 'audio_visual_dataset_robo_default/dataset.csv')
    p_img = impl.predict(model, test, device)

    high = pd.read_csv(OUT / 'audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv')
    pair = pd.read_csv(OUT / 'audio_feature_benchmarks/audio_group_consistency_pair_blend_select/reports/audio_group_consistency_pair_blend_select_final_test_predictions.csv')
    for name, frame in [('highsr', high), ('pairwise', pair)]:
        if not np.array_equal(frame.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
            raise AssertionError(f'{name} audio alignment')
    p_audio = suite.segment_lift(test.audio_file, suite.normalize(0.8 * high[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64) + 0.2 * pair[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64)))
    alpha = float(lock['selected_audio_weight'])
    p = suite.normalize(np.exp(alpha * np.log(suite.normalize(p_audio)) + (1.0 - alpha) * np.log(suite.normalize(p_img))))
    y = test.y.to_numpy(dtype=np.int64)
    pred = p.argmax(axis=1)
    binary_y, binary_p = (y > 0).astype(int), (pred > 0).astype(int)
    result = {
        'split': 'robot_test_final',
        'protocol': 'domain_augmented_resnet_after_specimen_group_hand_only_selection',
        'n': int(len(y)),
        'locked_candidate': lock,
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
    (REPORT / 'domain_augmented_final_test_metrics.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    print(json.dumps(result, indent=2, default=float))


if __name__ == '__main__':
    main()
