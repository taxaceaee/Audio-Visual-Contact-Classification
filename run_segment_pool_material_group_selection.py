from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_audio_specimen_contact_lift_select_final_test as spec_lift
import train_val_select_final_test as audio_base

ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/segment_pool_material_group_selection'))
IMG_ROOT=Path(os.environ.get('SEG_IMG_ROOT','outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'))

def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240')
    af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train')
    fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str)
    x=np.load(IMG_ROOT/'hand_train_full/X.npy').astype(np.float32); img_y=np.load(IMG_ROOT/'hand_train_full/y.npy').astype(np.int64)
    if not np.array_equal(y,img_y): raise AssertionError('image alignment')
    sg=spec(files).to_numpy(); sk=seg(files).to_numpy(); unique_seg, codes=np.unique(sk,return_inverse=True)
    seg_y=np.array([y[codes==i][0] for i in range(len(unique_seg))]); seg_spec=np.array([sg[codes==i][0] for i in range(len(unique_seg))]); seg_x=np.vstack([x[codes==i].mean(0) for i in range(len(unique_seg))])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(seg_y)),seg_y,seg_spec))
    # Group-aware OOF audio window predictions, then separate segment contact/material means.
    row_groups=spec(files).to_numpy(); assignment=np.full(len(y),-1,np.int64)
    row_folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,row_groups))
    for k,(_,va) in enumerate(row_folds): assignment[va]=k
    high=group_audio.load_highsr_oof(Path('outputs')); pair=specimen.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); src=broad.load_train_sources(Path('outputs'),y,assignment,42)
    audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift')
    if os.environ.get('SEG_AUDIO_SOURCE','locked')=='spec_lift':
        raw_audio=spec_lift.normalize(.8*high+.2*pair); seg_audio,w2s=specimen.segment_proba_from_window(af,raw_audio); codes_audio=specimen.specimen_codes_for_segments(af,w2s); locked=json.load(open('outputs/audio_feature_benchmarks/audio_specimen_contact_lift_select/reports/audio_specimen_contact_lift_select_selected_without_test.json'))['selected_without_test']; seg_audio=spec_lift.consensus_and_lift(seg_audio,codes_audio,float(locked['consensus_threshold']),int(locked['min_contact_segments']),locked['lift_min_mass'],float(locked['lift_floor']),float(locked['lift_confidence'])); audio=seg_audio[w2s]
    pc,lg=avr.avr_inputs(audio)
    seg_pc=np.array([pc[codes==i].mean() for i in range(len(unique_seg))]); seg_mat=np.vstack([suite.normalize(np.exp(lg[codes==i].mean(0,keepdims=True)))[0] for i in range(len(unique_seg))])
    oof_img={f'C{c}':np.zeros((len(seg_y),3),np.float32) for c in [0.03,0.1,0.3,1.0,3.0]}
    for fold,(tr,va) in enumerate(folds):
        contact=tr[seg_y[tr]>0];
        for c in [0.03,0.1,0.3,1.0,3.0]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,class_weight='balanced',random_state=42))
            model.fit(seg_x[contact],seg_y[contact]-1); oof_img[f'C{c}'][va]=model.predict_proba(seg_x[va])
        print(f'fold {fold+1}/5 done',flush=True)
    rows=[]
    for c in [0.03,0.1,0.3,1.0,3.0]:
      for iw in np.linspace(0,1,21):
       mat=suite.normalize((1-iw)*seg_mat+iw*oof_img[f'C{c}'])
       for th in np.linspace(.35,.70,15):
        pred=np.where(seg_pc>=th,mat.argmax(1)+1,0); rows.append({'C':c,'image_material_weight':float(iw),'contact_threshold':float(th),'macro_f1_4class':float(f1_score(seg_y,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(seg_y>0,pred>0,average='macro',zero_division=0))})
    board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_segment_pool_material_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_pool_audio_anchor_visual_material_classifier_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','audio_source':'locked group-aware OOF audio; contact/material aggregated per segment','image_source':'CLIP mean embedding per segment','image_role':'material only; never contact','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
