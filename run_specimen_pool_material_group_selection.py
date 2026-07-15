from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as ga
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as sp
import train_audio_specimen_contact_lift_select_final_test as spec_lift
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SPEC_OUT','outputs/audio_feature_benchmarks/specimen_pool_material_group_selection')); IMG=Path(os.environ.get('SPEC_IMG','outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy'))
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def specimen(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def fast_macro(y,p):
    vals=[]
    for c in range(4):
        tp=np.sum((y==c)&(p==c)); fp=np.sum((y!=c)&(p==c)); fn=np.sum((y==c)&(p!=c)); vals.append(0. if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    return float(np.mean(vals))
def main():
 OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str); x=np.load(IMG).astype(np.float32); sk=seg(files).to_numpy(); keys=specimen(files).to_numpy(); u,c=np.unique(keys,return_inverse=True); sx=np.vstack([x[c==i].mean(0) for i in range(len(u))]); sy=np.array([y[c==i][0] for i in range(len(u))]);
 # group-aware OOF audio windows
 groups=keys; folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,groups)); ass=np.full(len(y),-1,np.int64)
 for k,(_,va) in enumerate(folds): ass[va]=k
 high=ga.load_highsr_oof(Path('outputs')); pair=sp.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); src=broad.load_train_sources(Path('outputs'),y,ass,42); audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift')
 if os.environ.get('SPEC_AUDIO_SOURCE','locked')=='spec_lift':
  raw_audio=spec_lift.normalize(.8*high+.2*pair); seg_audio,w2s=sp.segment_proba_from_window(af,raw_audio); codes_audio=sp.specimen_codes_for_segments(af,w2s); locked=json.load(open('outputs/audio_feature_benchmarks/audio_specimen_contact_lift_select/reports/audio_specimen_contact_lift_select_selected_without_test.json'))['selected_without_test']; seg_audio=spec_lift.consensus_and_lift(seg_audio,codes_audio,float(locked['consensus_threshold']),int(locked['min_contact_segments']),locked['lift_min_mass'],float(locked['lift_floor']),float(locked['lift_confidence'])); audio=seg_audio[w2s]
 pc,lg=avr.avr_inputs(audio); # segment-wise audio, then specimen image material
 segkeys=sk; su,sc=np.unique(segkeys,return_inverse=True); seg_pc=np.array([pc[sc==i].mean() for i in range(len(su))]); seg_mat=np.vstack([suite.normalize(np.exp(lg[sc==i].mean(0,keepdims=True)))[0] for i in range(len(su))]); seg_spec=specimen(pd.Series(su)).to_numpy(); smap={k:i for i,k in enumerate(u)}; seg_spec_idx=np.array([smap[k] for k in seg_spec]); spec_pc=np.array([seg_pc[seg_spec_idx==i].mean() for i in range(len(u))]); spec_audio_mat=np.vstack([suite.normalize(seg_mat[seg_spec_idx==i].mean(0,keepdims=True))[0] for i in range(len(u))]);
 oof={str(C):np.zeros((len(u),3),np.float32) for C in [.03,.1,.3,1.,3.]}; s_folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,u))
 for k,(tr,va) in enumerate(s_folds):
  ci=tr[sy[tr]>0]
  for C in [.03,.1,.3,1.,3.]:
   m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1200,class_weight='balanced',random_state=42)); m.fit(sx[ci],sy[ci]-1); oof[str(C)][va]=m.predict_proba(sx[va])
  print(f'fold {k+1}/5 done',flush=True)
 rows=[]
 for C in [.03,.1,.3,1.,3.]:
  for iw in np.linspace(0,1,21):
   im=oof[str(C)]; specmat=suite.normalize((1-iw)*(seg_mat if os.environ.get('SPEC_AUDIO_LEVEL','specimen')=='segment' else spec_audio_mat)[seg_spec_idx]+iw*im[seg_spec_idx]); contact_scores=seg_pc if os.environ.get('SPEC_CONTACT_LEVEL','segment')=='segment' else spec_pc[seg_spec_idx]; # audio remains the sole contact source
   bias_values=[-.3,0.,.3,.6] if os.environ.get('SPEC_USE_BIAS','0')=='1' else [0.]
   for tb in bias_values:
    for wb in bias_values:
     biased=suite.normalize(specmat*np.exp(np.array([0.,tb,wb])[None,:]))
     for th in np.linspace(.40,.68,8):
      pred=np.where(contact_scores>=th,biased.argmax(1)+1,0); yy=np.array([y[sc==i][0] for i in range(len(su))]); rows.append({'C':C,'image_material_weight':float(iw),'trunk_bias':float(tb),'twig_bias':float(wb),'contact_threshold':float(th),'macro_f1_4class':fast_macro(yy,pred),'binary_macro_f1':f1_score(yy>0,pred>0,average='macro',zero_division=0)})
 board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_specimen_pool_material_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'specimen_pool_audio_anchor_visual_material_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','audio_source':'locked group-aware OOF audio; contact/material segment aggregation','image_source':'CLIP mean embedding per specimen','image_role':'material only','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
