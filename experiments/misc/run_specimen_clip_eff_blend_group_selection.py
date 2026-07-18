from pathlib import Path
import json,numpy as np,pandas as pd
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
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/specimen_clip_eff_blend_group_selection'); CLIP=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'); EFF=Path('outputs/image_timm_features/efficientnet_b3.ra2_in1k')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def specimen(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); sf=specimen(fr.audio_file).to_numpy(); _,row_spec=np.unique(sf,return_inverse=True); u=np.unique(sf); xc=np.load(CLIP/'hand_train_full/X.npy').astype(np.float32); xe=np.load(EFF/'hand_train_full/X.npy').astype(np.float32); xc=np.vstack([xc[row_spec==i].mean(0) for i in range(len(u))]); xe=np.vstack([xe[row_spec==i].mean(0) for i in range(len(u))]); ys=np.array([y[row_spec==i][0] for i in range(len(u))]);
 folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,sf)); ass=np.full(len(y),-1,np.int64)
 for k,(_,va) in enumerate(folds): ass[va]=k
 high=ga.load_highsr_oof(Path('outputs')); pair=sp.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); src=broad.load_train_sources(Path('outputs'),y,ass,42); ap=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); pc,lg=avr.avr_inputs(ap); sk=seg(fr.audio_file).to_numpy(); su,sc=np.unique(sk,return_inverse=True); segpc=np.array([pc[sc==i].mean() for i in range(len(su))]); segmat=np.vstack([suite.normalize(np.exp(lg[sc==i].mean(0,keepdims=True)))[0] for i in range(len(su))]); ssi=np.array([np.where(u==specimen(pd.Series([k])).iloc[0])[0][0] for k in su]); audio_spec=np.vstack([suite.normalize(segmat[ssi==i].mean(0,keepdims=True))[0] for i in range(len(u))]);
 predc=np.zeros((len(u),3)); prede=np.zeros((len(u),3)); Cs=[.03,.1,.3,1.,3.]
 for k,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(ys)),ys,u)):
  ci=tr[ys[tr]>0]
  for C in Cs:
   mc=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1200,class_weight='balanced',random_state=42)); me=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1200,class_weight='balanced',random_state=42)); mc.fit(xc[ci],ys[ci]-1); me.fit(xe[ci],ys[ci]-1); predc[va]=mc.predict_proba(xc[va]); prede[va]=me.predict_proba(xe[va])
  print(f'fold {k+1}/5 done',flush=True)
 rows=[]
 for iw in np.linspace(0,1,11):
  for th in np.linspace(.45,.68,10):
   mat_img=suite.normalize(iw*predc+(1-iw)*prede); mat_spec=suite.normalize(.65*audio_spec+.35*mat_img); mat=mat_spec[ssi]; pred=np.where(segpc>=th,mat.argmax(1)+1,0); yy=np.array([y[sc==i][0] for i in range(len(su))]); rows.append({'clip_weight':float(iw),'efficientnet_weight':float(1-iw),'image_material_weight':.35,'contact_threshold':float(th),'macro_f1_4class':float(f1_score(yy,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(yy>0,pred>0,average='macro',zero_division=0))})
 board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_specimen_clip_eff_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'specimen_clip_efficientnet_audio_material_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
