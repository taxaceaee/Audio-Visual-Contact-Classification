from __future__ import annotations
import json,numpy as np,pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as ga
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as sp
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/segment_meta_oof_group_selection'); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def visual_fit(x,y,idx):
    ci=idx[y[idx]>0]; m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1200,class_weight='balanced',random_state=42)); m.fit(x[ci],y[ci]-1); return m
def main():
 OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); yrow=fr.y.to_numpy(np.int64); f=fr.audio_file.astype(str); sk=seg(f).to_numpy(); su,sc=np.unique(sk,return_inverse=True); sg=spec(f).to_numpy(); xrow=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); x=np.vstack([xrow[sc==i].mean(0) for i in range(len(su))]); y=np.array([yrow[sc==i][0] for i in range(len(su))]); groups=np.array([sg[sc==i][0] for i in range(len(su))]);
 if any(len(set(yrow[sc==i]))>1 for i in range(len(su))): raise AssertionError('segment purity')
 assign=np.full(len(yrow),-1,np.int64)
 for k,(_,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(yrow)),yrow,sg)): assign[va]=k
 high=ga.load_highsr_oof(Path('outputs')); pair=sp.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); src=broad.load_train_sources(Path('outputs'),yrow,assign,42); audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); pc,lg=avr.avr_inputs(audio); pcseg=np.array([pc[sc==i].mean() for i in range(len(su))]); pmat=np.vstack([suite.normalize(np.exp(lg[sc==i].mean(0,keepdims=True)))[0] for i in range(len(su))]);
 outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,groups)); oof_meta=np.zeros((len(y),3))
 for fold,(tr,va) in enumerate(outer):
  inner_oof=np.zeros((len(tr),3)); inner_groups=groups[tr];
  for it,(itr,iva) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=100+fold).split(np.arange(len(tr)),y[tr],inner_groups)):
   vm=visual_fit(x[tr],y[tr],itr); inner_oof[iva]=vm.predict_proba(x[tr[iva]])
  meta_x=np.hstack([pmat[tr],inner_oof]); meta_y=y[tr]-1; ci=y[tr]>0; meta=LogisticRegression(C=.1,max_iter=1000,class_weight='balanced',random_state=42); meta.fit(meta_x[ci],meta_y[ci]); vm=visual_fit(x[tr],y[tr],np.arange(len(tr))); oof_meta[va]=meta.predict_proba(np.hstack([pmat[va],vm.predict_proba(x[va])]))
  print(f'outer fold {fold+1}/5 done',flush=True)
 rows=[]
 for th in np.linspace(.35,.70,15):
  pred=np.where(pcseg>=th,oof_meta.argmax(1)+1,0)[sc]; rows.append({'meta_model':'logistic_C0.1','contact_threshold':float(th),'macro_f1_4class':float(f1_score(yrow,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(yrow>0,pred>0,average='macro',zero_division=0))})
 board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_meta_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_label_pure_nested_visual_oof_audio_meta_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','nested_visual_oof':True,'audio_source':'locked group-aware OOF','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
