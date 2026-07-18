from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_multisource_meta_group_selection')); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k')
NAMES=['highsr_default','highsr_regularized','highsr_extratrees','total240_ensemble','total240_stack_lr','report_gate_proba']
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def pool(codes,x,n): return np.vstack([x[codes==i].mean(0) for i in range(n)])
def feat(srcs,img):
    parts=[]
    for x in srcs: parts += [x,np.log(np.clip(x,1e-6,1)),x[:,1:].sum(1,keepdims=True),x.max(1,keepdims=True)]
    parts += [img,np.log(np.clip(img,1e-6,1)),img[:,1:].sum(1,keepdims=True),img.max(1,keepdims=True)]
    return np.hstack(parts).astype(np.float32)
def main():
    OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str); sk=seg(files).to_numpy(); sg=spec(files).to_numpy(); useg,codes=np.unique(sk,return_inverse=True); sy=np.array([y[codes==i][0] for i in range(len(useg))]); ss=np.array([sg[codes==i][0] for i in range(len(useg))]); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,ss)); assignment=np.full(len(y),-1,np.int64); [assignment.__setitem__(va,k) for k,(_,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,sg))]
    src=broad.load_train_sources(Path('outputs'),y,assignment,42); src_seg=[pool(codes,src[n],len(useg)) for n in NAMES]
    x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); xseg=pool(codes,x,len(useg)); image_oof=np.zeros((len(sy),4),np.float32)
    for k,(tr,va) in enumerate(folds):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=2000,class_weight='balanced',multi_class='multinomial',random_state=42)); m.fit(xseg[tr],sy[tr]); image_oof[va]=suite.normalize(m.predict_proba(xseg[va])); print(f'fold {k+1}/5 done',flush=True)
    z=feat(src_seg,image_oof); np.save(OUT/'hand_multisource_oof_features.npy',z); np.save(OUT/'hand_multisource_oof_y.npy',sy); rows=[]
    for cval in [.03,.1,.3,1.0]:
      for tw in [1.0,1.2,1.5,2.0]:
       for gw in [1.0,1.2,1.5,2.0]:
        cw={0:1.0,1:1.0,2:tw,3:gw}; fs=[]
        for tr,va in folds:
            m=make_pipeline(StandardScaler(),LogisticRegression(C=cval,max_iter=4000,class_weight=cw,multi_class='multinomial',random_state=42)); m.fit(z[tr],sy[tr]); fs.append(f1_score(sy[va],m.predict(z[va]),average='macro',zero_division=0))
        rows.append({'C':cval,'trunk_class_weight':tw,'twig_class_weight':gw,'mean_cv_macro_f1':float(np.mean(fs)),'worst_cv_macro_f1':float(np.min(fs))})
    board=pd.DataFrame(rows).sort_values(['mean_cv_macro_f1','worst_cv_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_multisource_meta_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'multisource_audio_image_meta_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','audio_sources':NAMES,'image_source':'CLIP segment embedding','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
