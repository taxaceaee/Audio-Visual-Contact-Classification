from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite
from run_multibackbone_image_audio_group_selection import BACKBONES, load_backbone


def make_lr(c=.1):return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=c,class_weight="balanced",max_iter=1000,random_state=42))])


def hierarchical_fit_predict(Xtr,ytr,Xv):
    binary=make_lr().fit(Xtr,(ytr>0).astype(int)); contact_idx=ytr>0; subtype=make_lr().fit(Xtr[contact_idx],ytr[contact_idx]); pb=binary.predict_proba(Xv); pc_raw=subtype.predict_proba(Xv); pc=np.zeros((len(Xv),3))
    for col,cls in enumerate(subtype.classes_):pc[:,int(cls)-1]=pc_raw[:,col]
    p=np.zeros((len(Xv),4));contact=pb[:,list(binary.classes_).index(1)];p[:,0]=1-contact;p[:,1:]=contact[:,None]*suite.normalize(pc);return suite.normalize(p)


def main():
    out=Path("outputs");root=Path("/home/ttung05/Desktop/tree_base/tree_structures");report=out/"audio_feature_benchmarks"/"hierarchical_image_audio_group_selection";report.mkdir(parents=True,exist_ok=True);full,tr,va=suite.build_group_split(root,report);y=full.y.to_numpy(int)
    run=out/"audio_feature_benchmarks"/"audio_highsr_temporal_tta_select";lock=json.loads((run/"reports"/"audio_highsr_temporal_tta_select_selected_without_test.json").read_text());pair=suite.normalize(np.load(out/"audio_feature_benchmarks"/"audio_group_consistency_pair_blend_select"/"oof_sources"/"pairwise_selected_clean_oof_proba.npy"))[va];views={v:suite.segment_lift(full.audio_file.iloc[va],suite.normalize(.8*np.load(run/"oof_proba"/lock["selected_candidate"]/f"{v}_oof_proba.npy")[va]+.2*pair)) for v in ["clean","robot_mix","bandlimit"]}
    rows=[]
    for backbone in ["clip_vit_base","resnet18","efficientnet_b3","vit_b16"]:
        X,yc=load_backbone(backbone,"hand_train_full");pi=hierarchical_fit_predict(X[tr],y[tr],X[va])
        for a in np.linspace(0,1,21):
            for kind in ["linear","log"]:
                scores={}
                for v,pa in views.items():
                    p=suite.normalize(a*pa+(1-a)*pi) if kind=="linear" else suite.normalize(np.exp(a*np.log(pa)+(1-a)*np.log(pi)));scores[v]=float(f1_score(y[va],p.argmax(1),average="macro",zero_division=0))
                rows.append({"backbone":backbone,"kind":kind,"audio_weight":float(a),**{f"macro_f1_{v}":s for v,s in scores.items()},"worst_view_macro_f1":min(scores.values()),"mean_view_macro_f1":float(np.mean(list(scores.values())))})
    lb=pd.DataFrame(rows).sort_values(["worst_view_macro_f1","mean_view_macro_f1"],ascending=False).reset_index(drop=True);lb.to_csv(report/"hand_group_hierarchical_leaderboard.csv",index=False);best=lb.iloc[0].to_dict();locked={"protocol":"hierarchical_image_audio_specimen_group_val_only","group_overlap":0,"test_loaded":False,"selected_candidate":best};(report/"selection_lock.json").write_text(json.dumps(locked,indent=2,default=float),encoding="utf-8");print(json.dumps(locked,indent=2,default=float))


if __name__=="__main__":main()
