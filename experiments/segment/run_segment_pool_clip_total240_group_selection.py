from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import run_multimodal_val_locked_suite as suite

OUT=Path('outputs'); ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
def seg_key(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True).str.replace(r'^audio/','',regex=True)
def pooled(frame, img, aud):
    keys=seg_key(frame.audio_file); codes=pd.factorize(keys,sort=False)[0]; n=int(codes.max())+1; y=np.zeros(n,dtype=int); rows=[]
    for g in range(n):
        ix=np.where(codes==g)[0]; labels=frame.y.to_numpy(int)[ix];
        if len(set(labels.tolist()))!=1: raise AssertionError('segment not pure')
        y[g]=labels[0]; rows.append((g,ix))
    means_i=np.vstack([img[ix].mean(0) for _,ix in rows]); std_i=np.vstack([img[ix].std(0) for _,ix in rows]); means_a=np.vstack([aud[ix].mean(0) for _,ix in rows]); std_a=np.vstack([aud[ix].std(0) for _,ix in rows]); return keys,codes,y,means_i,std_i,means_a,std_a
def aligned(m,X):
 raw=m.predict_proba(X); p=np.zeros((len(X),4))
 for col,cls in enumerate(m.classes_): p[:,int(cls)]=raw[:,col]
 return suite.normalize(p)
def main():
 report=OUT/'audio_feature_benchmarks/segment_pool_clip_total240_group_selection'; report.mkdir(parents=True,exist_ok=True)
 frame=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); frame['specimen_group']=suite.specimen_group(frame.audio_file); img=np.load(OUT/'image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy').astype(np.float32); aud=np.load(OUT/'audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy').astype(np.float32); _,codes,y,mi,si,ma,sa=pooled(frame,img,aud); seg_groups=frame.specimen_group.to_numpy()[np.unique(codes,return_index=True)[1]]
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.arange(len(y)),y,seg_groups)); rows=[]
 feature_sets={'img_mean':mi,'img_mean_std':np.hstack([mi,si]),'audio_mean':ma,'audio_mean_std':np.hstack([ma,sa]),'both_mean':np.hstack([mi,ma]),'both_mean_std':np.hstack([mi,si,ma,sa])}
 specs={'log':LogisticRegression(C=.1,class_weight='balanced',max_iter=1500,random_state=42),'lgbm':LGBMClassifier(n_estimators=250,learning_rate=.03,num_leaves=23,min_child_samples=20,reg_lambda=1.,class_weight='balanced',verbosity=-1,random_state=42,n_jobs=8)}
 for fs,X in feature_sets.items():
  for name,s in specs.items():
   po=np.zeros((len(y),4))
   for tr,va in folds:
    model=Pipeline([('scale',StandardScaler()),('model',clone(s))]) if name=='log' else clone(s); model.fit(X[tr],y[tr]); pseg=aligned(model,X[va]); po[va]=pseg
   fscores=[f1_score(y[va],po[va].argmax(1),average='macro',zero_division=0) for _,va in folds]; rows.append({'feature_set':fs,'model':name,'mean_fold_macro_f1':float(np.mean(fscores)),'worst_fold_macro_f1':float(np.min(fscores)),'std_fold_macro_f1':float(np.std(fscores))})
 lb=pd.DataFrame(rows).sort_values(['worst_fold_macro_f1','mean_fold_macro_f1'],ascending=False).reset_index(drop=True); lb.to_csv(report/'hand_segment_pool_leaderboard.csv',index=False); best=lb.iloc[0].to_dict(); lock={'protocol':'segment_pool_clip_total240_specimen_group_hand_only','group_column':'specimen_group','segment_purity_hand':True,'n_folds':5,'test_loaded':False,'selected_candidate':best}; (report/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
