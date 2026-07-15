from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import run_multimodal_val_locked_suite as suite
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

OUT=Path('outputs'); ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
def segment_only(files,p):
 codes,_=suite.audio_keys(files); n=int(codes.max())+1; lp=np.log(suite.normalize(p)); sums=np.vstack([np.bincount(codes,weights=lp[:,c],minlength=n) for c in range(4)]).T; seg=suite.normalize(np.exp(sums-sums.max(1,keepdims=True))); return seg[codes]
def main():
 report=OUT/'audio_feature_benchmarks/segment_only_clip_audio_group_selection'; report.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240')
 af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); frame=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); frame['specimen_group']=suite.specimen_group(frame.audio_file); y=frame.y.to_numpy(int); groups=frame.specimen_group.to_numpy(); folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.arange(len(y)),y,groups))
 run=OUT/'audio_feature_benchmarks/audio_highsr_temporal_tta_select'; al=json.loads((run/'reports/audio_highsr_temporal_tta_select_selected_without_test.json').read_text()); cand=al['selected_candidate']; high=np.load(run/'oof_proba'/cand/'clean_oof_proba.npy'); pair=specimen.load_pairwise_oof(OUT/'audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy'); raw=suite.normalize(.8*high+.2*pair); audio_variants={'window':raw,'segment_only':segment_only(frame.audio_file,raw),'suite_segment_lift':suite.segment_lift(frame.audio_file,raw)}
 X=np.load(OUT/'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy'); pi=np.zeros((len(y),4))
 for tr,va in folds:
  m=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.1,class_weight='balanced',max_iter=900,random_state=42))]).fit(X[tr],y[tr]); rawp=m.predict_proba(X[va]);
  for col,cls in enumerate(m.classes_): pi[va,int(cls)]=rawp[:,col]
 pi=suite.normalize(pi); rows=[]
 for an,pa in audio_variants.items():
  for iw in np.linspace(0,.6,25):
   p=suite.normalize(np.exp((1-iw)*np.log(suite.normalize(pa))+iw*np.log(suite.normalize(pi)))); fs=[f1_score(y[va],p[va].argmax(1),average='macro',zero_division=0) for _,va in folds]; rows.append({'audio_aggregation':an,'image_weight':float(iw),'mean_fold_macro_f1':float(np.mean(fs)),'worst_fold_macro_f1':float(np.min(fs)),'std_fold_macro_f1':float(np.std(fs))})
 lb=pd.DataFrame(rows).sort_values(['worst_fold_macro_f1','mean_fold_macro_f1'],ascending=False).reset_index(drop=True); lb.to_csv(report/'hand_segment_only_clip_audio_leaderboard.csv',index=False); best=lb.iloc[0].to_dict(); lock={'protocol':'segment_only_clip_audio_specimen_group_hand_only','group_column':'specimen_group','n_folds':5,'test_loaded':False,'selected_candidate':best}; (report/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
