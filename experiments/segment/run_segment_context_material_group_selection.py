from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_specimen_contact_lift_select_final_test as sl
import train_audio_specimen_contact_consensus_select_final_test as sp
import train_audio_group_consistency_pair_blend_select_final_test as ga
import train_audio_lift_source_blend_select_final_test as lift
import train_val_select_final_test as audio_base

ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/segment_context_material_group_selection'); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def specimen(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def audio_oof(frame,y):
    groups=specimen(frame.audio_file).to_numpy(); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,groups)); ass=np.full(len(y),-1,np.int64)
    for k,(_,va) in enumerate(folds): ass[va]=k
    high=ga.load_highsr_oof(Path('outputs')); pair=sp.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy'))
    raw=sl.normalize(.8*high+.2*pair); segp,w2s=sp.segment_proba_from_window(frame,raw); codes=sp.specimen_codes_for_segments(frame,w2s); locked=json.load(open('outputs/audio_feature_benchmarks/audio_specimen_contact_lift_select/reports/audio_specimen_contact_lift_select_selected_without_test.json'))['selected_without_test']
    segp=sl.consensus_and_lift(segp,codes,float(locked['consensus_threshold']),int(locked['min_contact_segments']),locked['lift_min_mass'],float(locked['lift_floor']),float(locked['lift_confidence']))
    return segp[w2s]
def main():
    OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240')
    af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str); sk=seg(files).to_numpy(); ss=specimen(files).to_numpy(); su,sc=np.unique(sk,return_inverse=True); ug,seg_spec=np.unique(ss[sc==np.arange(len(su))],return_inverse=True) if False else (None,None)
    x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); seg_x=np.vstack([x[sc==i].mean(0) for i in range(len(su))]); seg_y=np.array([y[sc==i][0] for i in range(len(su))]); seg_spec_names=specimen(pd.Series(su)).to_numpy(); spec_names,seg_spec_idx=np.unique(seg_spec_names,return_inverse=True); context=np.vstack([seg_x[seg_spec_idx==i].mean(0) for i in range(len(spec_names))]); context_x=context[seg_spec_idx]
    purity=[len(set(y[sc==i])) for i in range(len(su))];
    if max(purity)>1: raise AssertionError('segment is not label-pure')
    audio=audio_oof(af,y); pc,lg=avr.avr_inputs(audio); seg_pc=np.array([pc[sc==i].mean() for i in range(len(su))]); seg_mat=np.vstack([suite.normalize(np.exp(lg[sc==i].mean(0,keepdims=True)))[0] for i in range(len(su))]); feats=np.hstack([seg_x,context_x]); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(seg_y)),seg_y,spec_names[seg_spec_idx])); oof={str(c):np.zeros((len(su),3),np.float32) for c in [.003,.01,.03,.1,.3]}
    for k,(tr,va) in enumerate(folds):
        ci=tr[seg_y[tr]>0]
        for c in [.003,.01,.03,.1,.3]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1200,class_weight='balanced',random_state=42)); model.fit(feats[ci],seg_y[ci]-1); oof[str(c)][va]=model.predict_proba(feats[va])
        print(f'fold {k+1}/5 done',flush=True)
    rows=[]
    for c in [.003,.01,.03,.1,.3]:
      for iw in np.linspace(0,.7,15):
       mat=suite.normalize((1-iw)*seg_mat+iw*oof[str(c)])
       for th in np.linspace(.35,.65,13):
        pred_seg=np.where(seg_pc>=th,mat.argmax(1)+1,0); pred=pred_seg[sc]; rows.append({'C':c,'context_image_material_weight':float(iw),'contact_threshold':float(th),'macro_f1_4class':float(f1_score(y,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(y>0,pred>0,average='macro',zero_division=0))})
    board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_segment_context_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_label_pure_audio_specimen_lift_context_visual_material_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','segment_label_purity_checked':True,'image_input':'segment mean CLIP + unlabeled specimen context mean','image_role':'material only','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
