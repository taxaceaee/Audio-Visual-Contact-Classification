"""Hand-only OOF raw CLIP + leave-one-segment-out visual context head."""
from pathlib import Path
import json,warnings
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures');BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection');IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k');OUT=Path('outputs/audio_feature_benchmarks/raw_visual_context_group_selection')
def macro(y,p):
 v=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();v.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(v))
def main():
 OUT.mkdir(parents=True,exist_ok=True);fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv');yr=pd.Categorical(fr.category,categories=['ambient','leaf','trunk','twig']).codes;key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True);sp=key.str.replace(r'_segment_.*$','',regex=True);u,c=np.unique(key.to_numpy(),return_inverse=True);y=np.array([yr[c==i][0] for i in range(len(u))]);g=np.array([sp.to_numpy()[c==i][0] for i in range(len(u))]);x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32);sx=np.vstack([x[c==i].mean(0) for i in range(len(u))]);ctx=[]
 for i in range(len(u)):
  ix=np.where(g==g[i])[0];o=ix[ix!=i];ctx.append(np.zeros(x.shape[1]) if len(o)==0 else sx[o].mean(0))
 ctx=np.asarray(ctx);p1=np.load(BASE/'hand_oof_hier.npy');p2=np.load(BASE/'hand_oof_meta.npy');z=np.hstack([sx,ctx,p1,p2]).astype(np.float32);folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g));rows=[]
 for C in (.001,.003,.01,.03):
  pp=np.zeros((len(y),4))
  for tr,va in folds:
   m=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='liblinear',multi_class='ovr',class_weight='balanced',max_iter=600,random_state=42));m.fit(z[tr],y[tr]);pp[va]=m.predict_proba(z[va])
  scores=[macro(y[va],pp[va].argmax(1)) for _,va in folds];rows.append({'C':C,'mean_cv_macro_f1':float(np.mean(scores)),'worst_cv_macro_f1':float(np.min(scores))});np.save(OUT/f'hand_oof_C{C}.npy',pp)
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False);board.to_csv(OUT/'hand_visual_context_leaderboard.csv',index=False);best=board.iloc[0].to_dict();lock={'protocol':'raw_visual_leave_one_segment_context_group_oof','selection_data':'hand/default only','group_column':'specimen_group','context_excludes_self':True,'candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.to_string(index=False))
if __name__=='__main__':main()
