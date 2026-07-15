"""Hand-only OOF selection of a conservative trunk rescue for ambient errors."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings('ignore')
BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection')
ARR=Path('outputs/audio_feature_benchmarks/segment_hier_contact_rescue_group_selection/hand_oof_arrays.npz')
OUT=Path('outputs/audio_feature_benchmarks/trunk_ambient_rescue_group_selection')
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
def macro(y,p):
 v=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();v.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(v))
def main():
 OUT.mkdir(parents=True,exist_ok=True);h=np.load(BASE/'hand_oof_hier.npy');m=np.load(BASE/'hand_oof_meta.npy');a=np.load(ARR,allow_pickle=True);y=a['sy'].astype(int);tr=a['trunk_oof'].astype(float)
 fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv');key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True);sp=key.str.replace(r'_segment_.*$','',regex=True);u,c=np.unique(key.to_numpy(),return_inverse=True);g=np.array([sp.to_numpy()[c==i][0] for i in range(len(u))]);folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g))
 base=np.where(h.argmax(1)==2,2,m.argmax(1)); rows=[]
 for th in np.linspace(.05,.95,37):
  for min_conf in [.0,.1,.2]:
   p=base.copy(); mask=(p==0)&(tr>=th)&(tr>=min_conf);p[mask]=2;ss=[macro(y[va],p[va]) for _,va in folds];rows.append({'threshold':float(th),'min_conf':float(min_conf),'rescued':int(mask.sum()),'mean_cv_macro_f1':float(np.mean(ss)),'worst_cv_macro_f1':float(np.min(ss))})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True);board.to_csv(OUT/'hand_rescue_leaderboard.csv',index=False);best=board.iloc[0].to_dict();lock={'protocol':'trunk_ambient_rescue_grouped_hand_oof','selection_data':'hand/default only','group_column':'specimen_group','rule':'only base ambient and trunk_oof above locked threshold','candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.head(15).to_string(index=False))
if __name__=='__main__':main()
