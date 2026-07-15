"""Fit one fixed v2 specimen fold and export test image-head probabilities."""
from pathlib import Path
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures');OUT=Path('outputs/audio_feature_benchmarks/v2_fold_ensemble_group_selection');IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k');EFF=Path('outputs/image_timm_features/efficientnet_b3.ra2_in1k')
FOLD=int(os.environ['V2_FOLD'])
def seg(s):return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def pool(keys,x):
 u,c=np.unique(keys,return_inverse=True);return u,np.vstack([x[c==i].mean(0) for i in range(len(u))])
def norm(x):x=np.clip(np.asarray(x,float),1e-12,None);return x/x.sum(1,keepdims=True)
def main():
 h=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv'); y=pd.Categorical(h.category,categories=['ambient','leaf','trunk','twig']).codes; hk=seg(h.audio_file).to_numpy();hg=np.array([x.split('_segment_')[0] for x in hk]);hu,hc=np.unique(hk,return_inverse=True);xc=np.load(IMG/'hand_train_full/X.npy').astype(np.float32);xe=np.load(EFF/'hand_train_full/X.npy').astype(np.float32);hx_c=pool(hk,xc)[1];hx_e=pool(hk,xe)[1];hy=np.array([y[hc==i][0] for i in range(len(hu))]);groups=np.array([hg[hc==i][0] for i in range(len(hu))]);folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(hy)),hy,groups));tr,va=folds[FOLD]
 t=pd.read_csv(ROOT/'audio_visual_dataset_robo_default/dataset.csv');tk=seg(t.audio_file).to_numpy();tu,tc=np.unique(tk,return_inverse=True);tx_c=np.load(IMG/'robot_test/X.npy').astype(np.float32);tx_e=np.load(EFF/'robot_test/X.npy').astype(np.float32);tx_c=pool(tk,tx_c)[1];tx_e=pool(tk,tx_e)[1];tx_cat=np.hstack([tx_c,tx_e]);hx_cat=np.hstack([hx_c,hx_e])
 m4=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500,class_weight='balanced',multi_class='multinomial',random_state=42)).fit(hx_c[tr],hy[tr]);mb=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500,class_weight='balanced',random_state=42)).fit(hx_c[tr],(hy[tr]>0).astype(int));mm=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500,class_weight='balanced',multi_class='multinomial',random_state=42)).fit(hx_cat[tr][hy[tr]>0],hy[tr][hy[tr]>0]-1);mt=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1200,class_weight='balanced',random_state=42)).fit(hx_c[tr],(hy[tr]==2).astype(int))
 np.savez(OUT/f'fold_{FOLD}.npz',p4=norm(m4.predict_proba(tx_c)),pb=mb.predict_proba(tx_c)[:,1],pmat=mm.predict_proba(tx_cat),pt=mt.predict_proba(tx_c)[:,1],segment_ids=tu);print('saved',FOLD,len(tu),flush=True)
if __name__=='__main__':main()
