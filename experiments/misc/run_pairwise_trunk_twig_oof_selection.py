"""Hand grouped-OOF pairwise trunk/twig head on existing material OOF heads."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures');BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection');ARR=Path('outputs/audio_feature_benchmarks/segment_hier_contact_rescue_group_selection/hand_oof_arrays.npz');OUT=Path('outputs/audio_feature_benchmarks/pairwise_trunk_twig_group_selection')
def macro(y,p):
 v=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();v.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(v))
def main():
 OUT.mkdir(parents=True,exist_ok=True);a=np.load(ARR,allow_pickle=True);y=a['sy'].astype(int);h=np.load(BASE/'hand_oof_hier.npy');m=np.load(BASE/'hand_oof_meta.npy');base=np.where(h.argmax(1)==2,2,m.argmax(1));fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv');key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True);sp=key.str.replace(r'_segment_.*$','',regex=True);u,c=np.unique(key.to_numpy(),return_inverse=True);g=np.array([sp.to_numpy()[c==i][0] for i in range(len(u))]);
 heads=[a['seg_audio_mat'],a['img_mat_clip'],a['img_mat_eff'],a['img_mat_cat'],a['trunk_oof'][:,None]];z=np.hstack([np.asarray(x) if np.asarray(x).ndim==2 else np.asarray(x)[:,None] for x in heads]).astype(np.float32);z=np.hstack([z,np.log(np.clip(z,1e-6,1))]).astype(np.float32);folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g));rows=[]
 for C in (.003,.01,.03,.1):
  for cw in (0.7,1.,1.3,1.7,2.2):
   pp=base.copy();conf=np.zeros(len(y));
   for tr,va in folds:
    train=tr[np.isin(y[tr],[2,3])]; model=make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight={2:1.,3:cw},max_iter=2000,random_state=42));model.fit(z[train],y[train]); pr=model.predict_proba(z[va]); classes=model[-1].classes_; pred=np.array([classes[i] for i in pr.argmax(1)]); pp[va]=pred;conf[va]=pr.max(1)
   for th in (.5,.6,.7,.8,.9):
    p=base.copy(); mask=(base>0)&(conf>=th);p[mask]=pp[mask];scores=[macro(y[va],p[va]) for _,va in folds];rows.append({'C':C,'twig_class_weight':cw,'confidence_threshold':th,'mean_cv_macro_f1':float(np.mean(scores)),'worst_cv_macro_f1':float(np.min(scores)),'base_macro_f1':macro(y,base)})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True);board.to_csv(OUT/'hand_pairwise_leaderboard.csv',index=False);best=board.iloc[0].to_dict();lock={'protocol':'pairwise_trunk_twig_grouped_hand_oof','selection_data':'hand/default only','group_column':'specimen_group','rule':'override only nonambient base predictions above locked confidence','candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.head(15).to_string(index=False))
if __name__=='__main__':main()
