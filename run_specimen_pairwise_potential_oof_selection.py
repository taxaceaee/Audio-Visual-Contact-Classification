"""Hand-only OOF pairwise specimen potential; keeps mixed labels per segment."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings('ignore')
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures');BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection');OUT=Path('outputs/audio_feature_benchmarks/specimen_pairwise_potential_group_selection')
def norm(x):
 x=np.clip(np.asarray(x,float),1e-12,None);return x/x.sum(1,keepdims=True)
def macro(y,p):
 v=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();v.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(v))
def main():
 OUT.mkdir(parents=True,exist_ok=True);h=np.load(BASE/'hand_oof_hier.npy');m=np.load(BASE/'hand_oof_meta.npy');y=np.load(BASE/'hand_stack_oof_y.npy').astype(int);fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv');key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True);sp=key.str.replace(r'_segment_.*$','',regex=True);u,c=np.unique(key.to_numpy(),return_inverse=True);g=np.array([sp.to_numpy()[c==i][0] for i in range(len(u))]);
 # Co-occurrence estimated from hand segment labels only.
 co=np.ones((4,4),float)
 for sid in np.unique(g):
  yy=y[g==sid]
  for a in yy:
   for b in yy:
    if a!=b: co[a,b]+=1
 cond=co/co.sum(1,keepdims=True); logcond=np.log(cond)
 base=np.where(h.argmax(1)==2,2,m.argmax(1));p0=norm(.6*h+.4*m);folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g));rows=[]
 for lam in (0.0,.02,.05,.1,.2,.35,.5, .8,1.2):
  for it in (1,2,3):
   q=p0.copy()
   for _ in range(it):
    nxt=q.copy()
    for sid in np.unique(g):
     ix=np.where(g==sid)[0]; ctx=q[ix].sum(0)-q[ix]; scores=np.log(np.clip(q[ix],1e-8,1))+lam*(ctx@logcond.T);nxt[ix]=norm(np.exp(scores))
    q=nxt
   pred=np.where(base==2,2,q.argmax(1));ss=[macro(y[va],pred[va]) for _,va in folds];rows.append({'lambda':lam,'iterations':it,'mean_cv_macro_f1':float(np.mean(ss)),'worst_cv_macro_f1':float(np.min(ss)),'base_macro_f1':macro(y,base)})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True);board.to_csv(OUT/'hand_potential_leaderboard.csv',index=False);best=board.iloc[0].to_dict();lock={'protocol':'specimen_pairwise_potential_grouped_hand_oof','selection_data':'hand/default only','group_column':'specimen_group','mixed_label_safe':True,'candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));np.save(OUT/'cooccurrence.npy',co);print(json.dumps(lock,indent=2,default=float));print(board.head(15).to_string(index=False))
if __name__=='__main__':main()
