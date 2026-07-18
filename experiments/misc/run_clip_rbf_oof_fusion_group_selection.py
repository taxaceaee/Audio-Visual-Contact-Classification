"""Hand-only grouped OOF RBF-SVM over pooled CLIP embeddings."""
from pathlib import Path
import json,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures');IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k');BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection');OUT=Path('outputs/audio_feature_benchmarks/clip_rbf_fusion_group_selection')
def norm(p):
 p=np.clip(np.asarray(p,dtype=float),1e-12,None);return p/p.sum(1,keepdims=True)
def macro(y,p):
 v=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();v.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(v))
def main():
 OUT.mkdir(parents=True,exist_ok=True);fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv');yr=pd.Categorical(fr.category,categories=['ambient','leaf','trunk','twig']).codes;key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True);sp=key.str.replace(r'_segment_.*$','',regex=True);u,c=np.unique(key.to_numpy(),return_inverse=True);y=np.array([yr[c==i][0] for i in range(len(u))]);g=np.array([sp.to_numpy()[c==i][0] for i in range(len(u))]);x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32);sx=np.vstack([x[c==i].mean(0) for i in range(len(u))]);p1=np.load(BASE/'hand_oof_hier.npy');p2=np.load(BASE/'hand_oof_meta.npy');folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g));rows=[]
 for C in (.3,1.,3.,10.):
  pp=np.zeros((len(y),4))
  for tr,va in folds:
   m=make_pipeline(StandardScaler(),SVC(C=C,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,cache_size=2048,random_state=42));m.fit(sx[tr],y[tr]);raw=m.predict_proba(sx[va]);out=np.zeros_like(raw);out[:,m[-1].classes_.astype(int)]=raw;pp[va]=norm(out)
  np.save(OUT/f'hand_oof_C{C}.npy',pp); scores=[macro(y[va],pp[va].argmax(1)) for _,va in folds];rows.append({'C':C,'mean_cv_macro_f1':float(np.mean(scores)),'worst_cv_macro_f1':float(np.min(scores))});print(C,macro(y,pp.argmax(1)),flush=True)
 # choose a fixed/simple blend with existing stack only if OOF supports it.
 for C in (.3,1.,3.,10.):
  pp=np.load(OUT/f'hand_oof_C{C}.npy')
  for w in np.linspace(0,.6,7):
   q=norm(w*pp+(1-w)*(.4*p2+.6*p1));scores=[macro(y[va],q[va].argmax(1)) for _,va in folds];rows.append({'C':C,'blend_weight':float(w),'mean_cv_macro_f1':float(np.mean(scores)),'worst_cv_macro_f1':float(np.min(scores))})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True);board.to_csv(OUT/'hand_oof_leaderboard.csv',index=False);best=board.iloc[0].to_dict();lock={'protocol':'clip_rbf_svm_grouped_hand_oof','selection_data':'hand/default only','group_column':'specimen_group','candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.head(12).to_string(index=False))
if __name__=='__main__':main()
