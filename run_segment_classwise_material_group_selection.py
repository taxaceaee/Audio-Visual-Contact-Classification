from __future__ import annotations
import json, os
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT = Path(os.environ.get('SEG_OUT', 'outputs/audio_feature_benchmarks/segment_classwise_material_group_selection'))
IMG = Path(os.environ.get('SEG_IMG_ROOT', 'outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'))

def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$', '', regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$', '', regex=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set('total240')
    fr = suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv', ROOT/'audio_visual_dataset_default', 'hand_train')
    y = fr.y.to_numpy(np.int64); files = fr.audio_file.astype(str)
    x = np.load(IMG/'hand_train_full/X.npy').astype(np.float32)
    iy = np.load(IMG/'hand_train_full/y.npy').astype(np.int64)
    assert np.array_equal(y, iy), 'image alignment'
    sk = seg(files).to_numpy(); sg = spec(files).to_numpy()
    useg, codes = np.unique(sk, return_inverse=True)
    seg_y = np.array([y[codes == i][0] for i in range(len(useg))])
    seg_spec = np.array([sg[codes == i][0] for i in range(len(useg))])
    seg_x = np.vstack([x[codes == i].mean(0) for i in range(len(useg))])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(seg_y)), seg_y, seg_spec))

    # Locked audio OOF source; no robot rows are loaded here.
    af = audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv', 'hand_train')
    high = group_audio.load_highsr_oof(Path('outputs'))
    pair = specimen.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy'))
    anchor = lift.anchor_lift_proba(af, high, pair)
    assignment = np.full(len(y), -1, np.int64)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.arange(len(y)), y, sg)):
        assignment[va] = k
    src = broad.load_train_sources(Path('outputs'), y, assignment, 42)
    audio = lift.postprocess(af, lift.normalize(.95*anchor + .05*src['report_gate_onehot']), 'segment_lift')
    pc, lg = avr.avr_inputs(audio)
    seg_pc = np.array([pc[codes == i].mean() for i in range(len(useg))])
    seg_audio = np.vstack([suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0] for i in range(len(useg))])

    img_oof = np.zeros((len(seg_y), 3), np.float32)
    for fold, (tr, va) in enumerate(folds):
        contact = tr[seg_y[tr] > 0]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=1000, class_weight='balanced', random_state=42))
        model.fit(seg_x[contact], seg_y[contact]-1)
        img_oof[va] = model.predict_proba(seg_x[va])
        print(f'fold {fold+1}/5 done', flush=True)

    # Class-wise image weights for [leaf, trunk, twig], selected only on grouped OOF.
    rows = []
    for ws in product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], repeat=3):
        w = np.asarray(ws, dtype=np.float64)
        mat = suite.normalize(seg_audio * (1.0-w) + img_oof * w)
        for th in np.linspace(0.35, 0.70, 15):
            pred = np.where(seg_pc >= th, mat.argmax(1)+1, 0)
            rows.append({'leaf_weight': ws[0], 'trunk_weight': ws[1], 'twig_weight': ws[2],
                         'contact_threshold': float(th),
                         'macro_f1_4class': float(f1_score(seg_y, pred, average='macro', zero_division=0)),
                         'binary_macro_f1': float(f1_score(seg_y > 0, pred > 0, average='macro', zero_division=0))})
    board = pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'], ascending=False).reset_index(drop=True)
    board.to_csv(OUT/'hand_segment_classwise_material_leaderboard.csv', index=False)
    best = board.iloc[0].to_dict()
    lock = {'protocol':'segment_classwise_audio_anchor_visual_material_group_OOF', 'selection_data':'hand/default only',
            'group_column':'specimen_group', 'audio_source':'locked group-aware OOF audio', 'image_source':'CLIP mean embedding per segment',
            'image_role':'material only; class-wise weights; never contact', 'test_loaded':False, 'selected_candidate':best}
    (OUT/'selection_lock.json').write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))

if __name__ == '__main__': main()
