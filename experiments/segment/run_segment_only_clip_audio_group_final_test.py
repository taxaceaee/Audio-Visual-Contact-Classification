from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite

ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT = Path('outputs')
REPORT = OUT / 'audio_feature_benchmarks/segment_only_clip_audio_group_selection'
NAMES = ['ambient', 'leaf', 'trunk', 'twig']
LABELS = np.arange(4)


def main():
    lock = json.loads((REPORT / 'selection_lock.json').read_text())
    cand = lock['selected_candidate']
    hand = suite.load_manifest(ROOT / 'audio_visual_dataset_default/dataset.csv', ROOT / 'audio_visual_dataset_default', 'hand_train')
    Xh = np.load(OUT / 'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy')
    yh = hand.y.to_numpy(dtype=np.int64)
    model = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1200, random_state=42))]).fit(Xh, yh)
    raw_test = pd.read_csv(ROOT / 'audio_visual_dataset_robo_default/dataset.csv')
    test = suite.load_manifest(ROOT / 'audio_visual_dataset_robo_default/dataset.csv', ROOT / 'audio_visual_dataset_robo_default', 'robot_test')
    Xt = np.load(OUT / 'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/X.npy')
    y = test.y.to_numpy(dtype=np.int64)
    raw = model.predict_proba(Xt)
    pi = np.zeros((len(Xt), 4), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        pi[:, int(cls)] = raw[:, col]
    pi = suite.normalize(pi)
    high = pd.read_csv(OUT / 'audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv')
    pair = pd.read_csv(OUT / 'audio_feature_benchmarks/audio_group_consistency_pair_blend_select/reports/audio_group_consistency_pair_blend_select_final_test_predictions.csv')
    if not np.array_equal(high.audio_file.astype(str).to_numpy(), raw_test.audio_file.astype(str).to_numpy()):
        raise AssertionError('high alignment')
    p_audio = suite.segment_lift(test.audio_file, suite.normalize(0.8 * high[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64) + 0.2 * pair[suite.PROBA_COLUMNS].to_numpy(dtype=np.float64)))
    iw = float(cand['image_weight'])
    p = suite.normalize(np.exp((1 - iw) * np.log(p_audio) + iw * np.log(pi)))
    pred = p.argmax(axis=1)
    by, bp = (y > 0).astype(int), (pred > 0).astype(int)
    result = {
        'split': 'robot_test_final', 'protocol': 'segment_only_clip_audio_after_5fold_group_selection', 'n': int(len(y)), 'locked_candidate': cand,
        'accuracy_4class': float(accuracy_score(y, pred)), 'macro_precision_4class': float(precision_score(y, pred, average='macro', zero_division=0)), 'macro_recall_4class': float(recall_score(y, pred, average='macro', zero_division=0)), 'macro_f1_4class': float(f1_score(y, pred, average='macro', zero_division=0)), 'weighted_f1_4class': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'binary_accuracy_ambient_noambient': float(accuracy_score(by, bp)), 'binary_macro_precision_ambient_noambient': float(precision_score(by, bp, average='macro', zero_division=0)), 'binary_macro_recall_ambient_noambient': float(recall_score(by, bp, average='macro', zero_division=0)), 'binary_macro_f1_ambient_noambient': float(f1_score(by, bp, average='macro', zero_division=0)),
        'per_class_4class': classification_report(y, pred, labels=LABELS, target_names=NAMES, output_dict=True, zero_division=0), 'per_class_binary': classification_report(by, bp, labels=[0, 1], target_names=['ambient', 'noambient'], output_dict=True, zero_division=0), 'confusion_matrix_4class': confusion_matrix(y, pred, labels=LABELS).tolist(), 'confusion_matrix_binary': confusion_matrix(by, bp, labels=[0, 1]).tolist(),
    }
    (REPORT / 'segment_only_clip_final_test_metrics.json').write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == '__main__':
    main()
