from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/segment_rule_stack_resnet_fusion_group_selection'))
def features(pm,ph,pr): return np.hstack([pm,ph,pr,np.log(np.clip(pm,1e-6,1)),np.log(np.clip(ph,1e-6,1)),np.log(np.clip(pr,1e-6,1)),pm.max(1,keepdims=True),ph.max(1,keepdims=True),pr.max(1,keepdims=True)]).astype(np.float32)
def main():
    OUT.mkdir(parents=True,exist_ok=True); d=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); files=d.audio_file.astype(str); seg=sel.seg(files).to_numpy(); spr=sel.spec(files).to_numpy(); useg,codes=np.unique(seg,return_inverse=True); sy=np.array([d.y.to_numpy(np.int64)[codes==i][0] for i in range(len(useg))]); sp=np.array([spr[codes==i][0] for i in range(len(useg))]); pm=np.load('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection/hand_oof_meta.npy'); ph=np.load('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection/hand_oof_hier.npy'); pr=np.load('outputs/audio_feature_benchmarks/segment_rule_stack_ft_resnet_group_selection/hand_oof_ft_mat.npy'); z=features(pm,ph,pr); np.save(OUT/'hand_oof_features.npy',z); np.save(OUT/'hand_oof_y.npy',sy); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,sp)); rows=[]
    for c in [.01,.03,.1,.3,1.0]:
      for tw in [1.0,1.2,1.5,2.0]:
       for gw in [1.0,1.2,1.5,2.0]:
        cw={0:1,1:1,2:tw,3:gw}; fs=[]
        for tr,va in folds:
         m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=4000,class_weight=cw,multi_class='multinomial',random_state=42)); m.fit(z[tr],sy[tr]); fs.append(f1_score(sy[va],m.predict(z[va]),average='macro',zero_division=0))
        rows.append({'C':c,'trunk_class_weight':tw,'twig_class_weight':gw,'mean_cv_macro_f1':float(np.mean(fs)),'worst_cv_macro_f1':float(np.min(fs))})
    board=pd.DataFrame(rows).sort_values(['mean_cv_macro_f1','worst_cv_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_resnet_fusion_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_rule_stack_resnet_ooo_fusion_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','base_sources':'v2 meta+hier OOF plus domain-augmented ResNet material OOF','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
