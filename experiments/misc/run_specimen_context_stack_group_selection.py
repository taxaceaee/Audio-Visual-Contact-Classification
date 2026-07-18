"""Hand-only grouped OOF context stack for mixed-label specimens.

Unlike majority pooling, this keeps each segment label and uses the other
segments in the same specimen only as context features.
"""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection')
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
OUT=Path('outputs/audio_feature_benchmarks/specimen_context_stack_group_selection')
warnings.filterwarnings('ignore')

def norm(x):
 x=np.clip(np.asarray(x,dtype=float),1e-12,None);return x/x.sum(1,keepdims=True)
def macro(y,p):
 vals=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(vals))
def context(p,g):
 out=[]
 for i in range(len(p)):
  ix=np.where(g==g[i])[0]; other=ix[ix!=i]
  if len(other)==0: mean=np.zeros(4); mx=np.zeros(4); n=0
  else: mean=p[other].mean(0); mx=p[other].max(0); n=len(other)
  out.append(np.r_[mean,mx,np.log1p(n)])
 return np.asarray(out)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 p1=np.load(BASE/'hand_oof_hier.npy');p2=np.load(BASE/'hand_oof_meta.npy');y=np.load(BASE/'hand_stack_oof_y.npy').astype(int)
 fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv'); key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True); spec=key.str.replace(r'_segment_.*$','',regex=True)
 u,code=np.unique(key.to_numpy(),return_inverse=True); g=np.array([spec.to_numpy()[code==i][0] for i in range(len(u))]);
 # Use both base distributions and leave-one-segment-out context.
 ctx1=context(norm(p1),g);ctx2=context(norm(p2),g)
 z=np.hstack([np.log(norm(p1)),np.log(norm(p2)),norm(p1),norm(p2),ctx1,ctx2]).astype(np.float32)
 folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g))
 rows=[]
 for C in (.01,.1):
  for tw in (1.0,1.5):
   for gw in (1.0,1.5):
    scores=[]
    for tr,va in folds:
     m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,class_weight={0:1.,1:1.,2:tw,3:gw},multi_class='multinomial',random_state=42));m.fit(z[tr],y[tr]);pred=m.predict(z[va]);scores.append(macro(y[va],pred))
    rows.append({'C':C,'trunk_class_weight':tw,'twig_class_weight':gw,'mean_cv_macro_f1':float(np.mean(scores)),'worst_cv_macro_f1':float(np.min(scores))})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True);board.to_csv(OUT/'hand_context_leaderboard.csv',index=False);best=board.iloc[0].to_dict();np.save(OUT/'hand_context_features.npy',z);np.save(OUT/'hand_context_y.npy',y);np.save(OUT/'hand_specimen_ids.npy',g)
 lock={'protocol':'mixed_label_specimen_context_stack_group_oof','selection_data':'hand/default only','group_column':'specimen_group','target':'segment label','context_excludes_self':True,'candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.head(12).to_string(index=False))
if __name__=='__main__':main()
