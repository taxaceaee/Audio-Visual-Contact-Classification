from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import run_multimodal_val_locked_suite as suite
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

OUT=Path('outputs'); ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
def aligned(m,X):
 raw=m.predict_proba(X); p=np.zeros((len(X),4))
 for col,cls in enumerate(m.classes_): p[:,int(cls)]=raw[:,col]
 return suite.normalize(p)
def main():
 report=OUT/'audio_feature_benchmarks/exact_anchor_clip_mlp_group_selection'; report.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240')
 af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); frame=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); frame['specimen_group']=suite.specimen_group(frame.audio_file); y=frame.y.to_numpy(int); groups=frame.specimen_group.to_numpy(); folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.arange(len(y)),y,groups))
 high=group_audio.load_highsr_oof(OUT); pair=specimen.load_pairwise_oof(OUT/'audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy'); pa=lift.anchor_lift_proba(af,high,pair); img=np.load(OUT/'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy').astype(np.float32); Z=np.hstack([img,np.log(suite.normalize(pa)),pa[:,1:].sum(1,keepdims=True)]).astype(np.float32); rows=[]
 specs=[((64,),.0001),((128,),.0001),((128,64),.0001),((128,64),.001)]
 for hidden,alpha in specs:
  po=np.zeros((len(y),4))
  for tr,va in folds:
   m=Pipeline([('scale',StandardScaler()),('model',MLPClassifier(hidden_layer_sizes=hidden,alpha=alpha,batch_size=128,learning_rate_init=.001,max_iter=120,early_stopping=True,validation_fraction=.15,n_iter_no_change=12,random_state=42))]).fit(Z[tr],y[tr]); po[va]=aligned(m,Z[va])
  for tb in [-.4,0,.4]:
   for wb in [0,.4,.8]:
    p=suite.normalize(np.exp(np.log(po)+np.array([0.,0.,tb,wb])[None,:])); fs=[f1_score(y[va],p[va].argmax(1),average='macro',zero_division=0) for _,va in folds]; rows.append({'hidden':str(hidden),'alpha':alpha,'trunk_bias':tb,'twig_bias':wb,'mean_fold_macro_f1':float(np.mean(fs)),'worst_fold_macro_f1':float(np.min(fs)),'std_fold_macro_f1':float(np.std(fs))})
 lb=pd.DataFrame(rows).sort_values(['worst_fold_macro_f1','mean_fold_macro_f1'],ascending=False).reset_index(drop=True); lb.to_csv(report/'hand_exact_anchor_clip_mlp_leaderboard.csv',index=False); best=lb.iloc[0].to_dict(); lock={'protocol':'exact_anchor_clip_mlp_specimen_group_hand_only','group_column':'specimen_group','n_folds':5,'test_loaded':False,'selected_candidate':best}; (report/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
