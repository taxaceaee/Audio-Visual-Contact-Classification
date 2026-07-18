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
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/specimen_soft_material_group_selection'); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def specimen(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def audio_oof(frame):
 high=ga.load_highsr_oof(Path('outputs')); pair=sp.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); raw=sl.normalize(.8*high+.2*pair); segp,w2s=sp.segment_proba_from_window(frame,raw); codes=sp.specimen_codes_for_segments(frame,w2s); l=json.load(open('outputs/audio_feature_benchmarks/audio_specimen_contact_lift_select/reports/audio_specimen_contact_lift_select_selected_without_test.json'))['selected_without_test']; segp=sl.consensus_and_lift(segp,codes,float(l['consensus_threshold']),int(l['min_contact_segments']),l['lift_min_mass'],float(l['lift_floor']),float(l['lift_confidence'])); return segp[w2s]
def fit_soft(x,target,c):
    keep=target.sum(1)>0; xx=np.repeat(x[keep],3,axis=0); yy=np.tile(np.arange(3),keep.sum()); ww=target[keep].reshape(-1); m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,random_state=42)); m.fit(xx,yy,logisticregression__sample_weight=ww); return m
def main():
 OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str); sk=seg(files).to_numpy(); su,sc=np.unique(sk,return_inverse=True); spk=specimen(pd.Series(su)).to_numpy(); names,si=np.unique(spk,return_inverse=True); x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); segx=np.vstack([x[sc==i].mean(0) for i in range(len(su))]); specx=np.vstack([segx[si==i].mean(0) for i in range(len(names))]); sy=np.array([y[sc==i][0] for i in range(len(su))]); target=np.zeros((len(names),3),float)
 for i in range(len(su)):
  vals=sy[si==i]; vals=vals[vals>0];
  if len(vals): target[i]=np.bincount(vals-1,minlength=3)/len(vals)
 audio=audio_oof(af); pc,lg=avr.avr_inputs(audio); segpc=np.array([pc[sc==i].mean() for i in range(len(su))]); segmat=np.vstack([suite.normalize(np.exp(lg[sc==i].mean(0,keepdims=True)))[0] for i in range(len(su))]); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(names)),target.argmax(1),names)); oof={str(c):np.zeros((len(names),3),float) for c in [.003,.01,.03,.1,.3]}
 for k,(tr,va) in enumerate(folds):
  for c in [.003,.01,.03,.1,.3]: oof[str(c)][va]=fit_soft(specx[tr],target[tr],c).predict_proba(specx[va])
  print(f'fold {k+1}/5 done',flush=True)
 rows=[]
 for c in [.003,.01,.03,.1,.3]:
  for iw in np.linspace(0,.7,15):
   mat=suite.normalize((1-iw)*segmat+iw*oof[str(c)][si])
   for th in np.linspace(.35,.65,13):
    pred=np.where(segpc>=th,mat.argmax(1)+1,0)[sc]; rows.append({'C':c,'image_material_weight':float(iw),'contact_threshold':float(th),'macro_f1_4class':float(f1_score(y,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(y>0,pred>0,average='macro',zero_division=0))})
 board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_soft_material_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'mixed_label_specimen_soft_material_audio_lift_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','target':'contact material distribution per specimen; no first-row label','image_role':'material prior only','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
