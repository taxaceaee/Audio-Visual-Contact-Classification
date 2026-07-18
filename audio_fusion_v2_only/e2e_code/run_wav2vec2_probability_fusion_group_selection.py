from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


def proba(model, X):
    raw=model.predict_proba(X);p=np.zeros((len(X),4))
    for col,cls in enumerate(model.classes_):p[:,int(cls)]=raw[:,col]
    return suite.normalize(p)


def main():
    out=Path("outputs");root=Path("/home/ttung05/Desktop/tree_base/tree_structures");report=out/"audio_feature_benchmarks"/"wav2vec2_probability_fusion_group_selection";report.mkdir(parents=True,exist_ok=True);full,tr,va=suite.build_group_split(root,report);y=full.y.to_numpy(int);audio=np.load(out/"audio_wav2vec2_features"/"hand_train_full_X.npy");image=np.load(out/"image_deep_features"/"resnet18_224"/"hand_train_full"/"X.npy");run=out/"audio_feature_benchmarks"/"audio_highsr_temporal_tta_select";lock=json.loads((run/"reports"/"audio_highsr_temporal_tta_select_selected_without_test.json").read_text());pair=suite.normalize(np.load(out/"audio_feature_benchmarks"/"audio_group_consistency_pair_blend_select"/"oof_sources"/"pairwise_selected_clean_oof_proba.npy"))[va];views={v:suite.segment_lift(full.audio_file.iloc[va],suite.normalize(.8*np.load(run/"oof_proba"/lock["selected_candidate"]/f"{v}_oof_proba.npy")[va]+.2*pair)) for v in ["clean","robot_mix","bandlimit"]}
    model=Pipeline([("scale",StandardScaler()),("model",SGDClassifier(loss="log_loss",alpha=.001,class_weight="balanced",max_iter=120,tol=1e-3,random_state=42,n_jobs=-1))]).fit(np.hstack([audio[tr],image[tr]]),y[tr]);pw=proba(model,np.hstack([audio[va],image[va]]));rows=[]
    for a in np.linspace(0,1,21):
        scores={}
        for v,pa in views.items():
            p=suite.normalize(np.exp(a*np.log(pa)+(1-a)*np.log(pw)));scores[v]=float(f1_score(y[va],p.argmax(1),average="macro",zero_division=0))
        rows.append({"wav2vec_weight":float(1-a),"old_audio_weight":float(a),**{f"macro_f1_{v}":s for v,s in scores.items()},"worst_view_macro_f1":min(scores.values()),"mean_view_macro_f1":float(np.mean(list(scores.values())))})
    lb=pd.DataFrame(rows).sort_values(["worst_view_macro_f1","mean_view_macro_f1"],ascending=False).reset_index(drop=True);lb.to_csv(report/"hand_group_wav2vec_fusion_leaderboard.csv",index=False);best=lb.iloc[0].to_dict();locked={"protocol":"wav2vec2_old_audio_probability_group_val_only","group_overlap":0,"test_loaded":False,"selected_candidate":best};(report/"selection_lock.json").write_text(json.dumps(locked,indent=2,default=float),encoding="utf-8");print(json.dumps(locked,indent=2,default=float))


if __name__=="__main__":main()
