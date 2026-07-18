"""Hand-only search of a small interpretable rule table over v2 OOF heads."""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures');BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection');OUT=Path('outputs/audio_feature_benchmarks/rule_table_v2_group_selection')
def f1(y,p):
 v=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();v.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(v))
def main():
 OUT.mkdir(parents=True,exist_ok=True);h=np.load(BASE/'hand_oof_hier.npy');m=np.load(BASE/'hand_oof_meta.npy');y=np.load(BASE/'hand_stack_oof_y.npy').astype(int);fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv');k=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True);sp=k.str.replace(r'_segment_.*$','',regex=True);u,c=np.unique(k.to_numpy(),return_inverse=True);g=np.array([sp.to_numpy()[c==i][0] for i in range(len(u))]);folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g));ha=h.argmax(1);ma=m.argmax(1);rows=[]
 for tw in (.25,.35,.45,.55,.65):
  for gap in (-.05,.0,.05,.1,.15):
   for amb_contact in (None,.35,.45,.55,.65):
    p=np.where(ha==2,2,ma);mask=(ma==3)&(m[:,3]>=tw)&(m[:,3]>=h[:,3]+gap)&(ha!=2);p[mask]=3
    if amb_contact is not None:
     mass=1-h[:,0];p[(ha==0)&(mass>=amb_contact)&(ma!=0)]=ma[(ha==0)&(mass>=amb_contact)&(ma!=0)]
    ss=[f1(y[va],p[va]) for _,va in folds];rows.append({'twig_conf':tw,'twig_gap':gap,'ambient_contact':amb_contact if amb_contact is not None else -1,'mean_cv_macro_f1':float(np.mean(ss)),'worst_cv_macro_f1':float(np.min(ss))})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True);board.to_csv(OUT/'hand_rule_table_leaderboard.csv',index=False);best=board.iloc[0].to_dict();lock={'protocol':'interpretable_rule_table_grouped_hand_oof','selection_data':'hand/default only','group_column':'specimen_group','candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.head(12).to_string(index=False))
if __name__=='__main__':main()
