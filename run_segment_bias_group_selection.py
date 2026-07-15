from pathlib import Path
import json,numpy as np,pandas as pd
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
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/segment_bias_group_selection'); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); f=fr.audio_file.astype(str); sk=seg(f).to_numpy(); su,sc=np.unique(sk,return_inverse=True); sg=f.str.replace(r'_segment_.*$','',regex=True).to_numpy(); x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); sx=np.vstack([x[sc==i].mean(0) for i in range(len(su))]); sy=np.array([y[sc==i][0] for i in range(len(su))]); groups=np.array([sg[sc==i][0] for i in range(len(su))]); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,groups)); ass=np.full(len(y),-1,np.int64)
 for k,(_,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,sg)): ass[va]=k
 high=ga.load_highsr_oof(Path('outputs')); pair=sp.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); import train_audio_broad_oof_meta_select_final_test as broad; src=broad.load_train_sources(Path('outputs'),y,ass,42); audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); pc,lg=avr.avr_inputs(audio); spc=np.array([pc[sc==i].mean() for i in range(len(su))]); smat=np.vstack([suite.normalize(np.exp(lg[sc==i].mean(0,keepdims=True)))[0] for i in range(len(su))]); oof=np.zeros((len(sy),3))
 for tr,va in folds:
  ci=tr[sy[tr]>0]; m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1200,class_weight='balanced',random_state=42)); m.fit(sx[ci],sy[ci]-1); oof[va]=m.predict_proba(sx[va])
 rows=[]
 for iw in [.2,.3,.4,.5,.6]:
  base=suite.normalize((1-iw)*smat+iw*oof)
  for tb in [-.8,-.4,0,.4,.8]:
   for wb in [-.8,-.4,0,.4,.8]:
    p=suite.normalize(base*np.exp(np.array([0.,tb,wb])[None,:]))
    for th in [.45,.5,.55,.6,.65]:
     pred=np.where(spc>=th,p.argmax(1)+1,0)[sc]; rows.append({'image_weight':iw,'trunk_bias':tb,'twig_bias':wb,'contact_threshold':th,'macro_f1_4class':f1_score(y,pred,average='macro',zero_division=0),'binary_macro_f1':f1_score(y>0,pred>0,average='macro',zero_division=0)})
 board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_segment_bias_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_label_pure_audio_clip_material_class_bias_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
