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
import train_audio_group_consistency_pair_blend_select_final_test as ga
import train_audio_specimen_contact_consensus_select_final_test as sp
import train_audio_lift_source_blend_select_final_test as lift
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/segment_clip_eff_group_selection'); CLIP=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'); EFF=Path('outputs/image_timm_features/efficientnet_b3.ra2_in1k')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); f=fr.audio_file.astype(str); sk=seg(f).to_numpy(); su,sc=np.unique(sk,return_inverse=True); sg=spec(f).to_numpy(); x1=np.load(CLIP/'hand_train_full/X.npy').astype(np.float32); x2=np.load(EFF/'hand_train_full/X.npy').astype(np.float32); a=np.vstack([x1[sc==i].mean(0) for i in range(len(su))]); b=np.vstack([x2[sc==i].mean(0) for i in range(len(su))]); yy=np.array([y[sc==i][0] for i in range(len(su))]); groups=np.array([sg[sc==i][0] for i in range(len(su))]); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(yy)),yy,groups)); assign=np.full(len(y),-1,np.int64);
 for k,(_,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,sg)): assign[va]=k
 high=ga.load_highsr_oof(Path('outputs')); pair=sp.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); import train_audio_broad_oof_meta_select_final_test as broad; src=broad.load_train_sources(Path('outputs'),y,assign,42); audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); pc,lg=avr.avr_inputs(audio); spc=np.array([pc[sc==i].mean() for i in range(len(su))]); smat=np.vstack([suite.normalize(np.exp(lg[sc==i].mean(0,keepdims=True)))[0] for i in range(len(su))]); out1={str(c):np.zeros((len(yy),3)) for c in [.03,.1,.3]}; out2={str(c):np.zeros((len(yy),3)) for c in [.03,.1,.3]}
 for k,(tr,va) in enumerate(folds):
  ci=tr[yy[tr]>0]
  for c in [.03,.1,.3]:
   m1=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1200,class_weight='balanced',random_state=42)); m2=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1200,class_weight='balanced',random_state=42)); m1.fit(a[ci],yy[ci]-1); m2.fit(b[ci],yy[ci]-1); out1[str(c)][va]=m1.predict_proba(a[va]); out2[str(c)][va]=m2.predict_proba(b[va])
  print(f'fold {k+1}/5 done',flush=True)
 rows=[]
 for c in [.03,.1,.3]:
  for cw in np.linspace(0,1,11):
   img=suite.normalize(cw*out1[str(c)]+(1-cw)*out2[str(c)])
   for iw in np.linspace(0,.7,15):
    mat=suite.normalize((1-iw)*smat+iw*img)
    for th in np.linspace(.35,.70,15):
     pred=np.where(spc>=th,mat.argmax(1)+1,0)[sc]; rows.append({'C':c,'clip_weight':float(cw),'image_material_weight':float(iw),'contact_threshold':float(th),'macro_f1_4class':float(f1_score(y,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(y>0,pred>0,average='macro',zero_division=0))})
 board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_clip_eff_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_label_pure_audio_anchor_clip_eff_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
