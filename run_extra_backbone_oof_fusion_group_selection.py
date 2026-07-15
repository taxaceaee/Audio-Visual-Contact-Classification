"""Hand-only OOF fusion with DINOv2 and ConvNeXt image backbones."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection')
OUT=Path('outputs/audio_feature_benchmarks/extra_backbone_fusion_group_selection')
MODELS={'dino':'outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m','convnext':'outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k'}
def macro(y,p):
 vals=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(vals))
def norm(p):
 p=np.clip(np.asarray(p,dtype=float),1e-12,None);return p/p.sum(1,keepdims=True)
def pool(x,c): return np.vstack([x[c==i].mean(0) for i in range(int(c.max())+1)])
def main():
 OUT.mkdir(parents=True,exist_ok=True);fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv'); yrow=pd.Categorical(fr.category,categories=['ambient','leaf','trunk','twig']).codes; key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True); sp=key.str.replace(r'_segment_.*$','',regex=True);u,c=np.unique(key.to_numpy(),return_inverse=True);y=np.array([yrow[c==i][0] for i in range(len(u))]);g=np.array([sp.to_numpy()[c==i][0] for i in range(len(u))]);folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,g));
 probs={}
 for name,path in MODELS.items():
  x=np.load(Path(path)/'hand_train_full/X.npy').astype(np.float32);sx=pool(x,c); p=np.zeros((len(y),4))
  for tr,va in folds:
   m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,solver='liblinear',multi_class='ovr',class_weight='balanced',max_iter=800,random_state=42));m.fit(sx[tr],y[tr]);p[va]=norm(m.predict_proba(sx[va]))
  probs[name]=p;np.save(OUT/f'hand_oof_{name}.npy',p);print(name,'done',macro(y,p.argmax(1)),flush=True)
 pm=np.load(BASE/'hand_oof_meta.npy');ph=np.load(BASE/'hand_oof_hier.npy');pdino=probs['dino'];pconv=probs['convnext'];rows=[]
 # fixed small simplex grid; score by worst grouped fold.
 for wd in np.linspace(0,0.6,7):
  for wc in np.linspace(0,0.6,7):
   for wh in np.linspace(0,1-wd-wc,6) if wd+wc<=1 else []:
    wm=1-wd-wc-wh
    p=norm(wm*pm+wh*ph+wd*pdino+wc*pconv); scores=[macro(y[va],p[va].argmax(1)) for _,va in folds];rows.append({'meta_weight':wm,'hier_weight':wh,'dino_weight':wd,'convnext_weight':wc,'mean_cv_macro_f1':float(np.mean(scores)),'worst_cv_macro_f1':float(np.min(scores))})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True);board.to_csv(OUT/'hand_oof_leaderboard.csv',index=False);best=board.iloc[0].to_dict();lock={'protocol':'extra_image_backbones_grouped_hand_oof','selection_data':'hand/default only','group_column':'specimen_group','candidate':best,'test_loaded':False};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.head(12).to_string(index=False))
if __name__=='__main__':main()
