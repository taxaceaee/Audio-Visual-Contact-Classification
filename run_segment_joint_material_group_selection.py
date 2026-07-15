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
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/segment_joint_material_group_selection'); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str); x=np.load(IMG).astype(np.float32); keys=seg(files).to_numpy(); sg=spec(files).to_numpy(); u,c=np.unique(keys,return_inverse=True); sx=np.vstack([x[c==i].mean(0) for i in range(len(u))]); sy=np.array([y[c==i][0] for i in range(len(u))]); ssg=np.array([sg[c==i][0] for i in range(len(u))])
 rf=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,sg)); ass=np.full(len(y),-1,np.int64)
 for k,(_,va) in enumerate(rf): ass[va]=k
 high=group_audio.load_highsr_oof(Path('outputs')); pair=specimen.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); src=broad.load_train_sources(Path('outputs'),y,ass,42); audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); pc,lg=avr.avr_inputs(audio); spc=np.array([pc[c==i].mean() for i in range(len(u))]); slg=np.vstack([lg[c==i].mean(0) for i in range(len(u))]);
 # Standardized joint model: the audio logits are explicit features, and the image is material-only.
 feats=np.hstack([slg,sx]); oof=np.zeros((len(sy),3),np.float32); Cs=[.003,.01,.03,.1,.3,1.0]
 for k,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,ssg)):
  ci=tr[sy[tr]>0]
  for C in Cs:
   m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1500,class_weight='balanced',random_state=42)); m.fit(feats[ci],sy[ci]-1); oof[va]=m.predict_proba(feats[va]) if C==.1 else oof[va]
  print(f'fold {k+1}/5 done',flush=True)
 # Store all C predictions by repeating folds to preserve candidate selection.
 allp={str(C):np.zeros((len(sy),3),np.float32) for C in Cs}
 for k,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,ssg)):
  ci=tr[sy[tr]>0]
  for C in Cs:
   m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1500,class_weight='balanced',random_state=42)); m.fit(feats[ci],sy[ci]-1); allp[str(C)][va]=m.predict_proba(feats[va])
 rows=[]
 for C in Cs:
  for th in np.linspace(.35,.70,15):
   pred=np.where(spc>=th,allp[str(C)].argmax(1)+1,0); rows.append({'C':C,'contact_threshold':float(th),'macro_f1_4class':float(f1_score(sy,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(sy>0,pred>0,average='macro',zero_division=0))})
 board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_joint_material_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_joint_audio_logits_clip_material_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','audio_contact_gate':'locked group-aware OOF audio','material_model':'standardized logistic regression on segment audio logits + pooled CLIP','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
