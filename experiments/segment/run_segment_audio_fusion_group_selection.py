from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite


def segment_only(files: pd.Series, p: np.ndarray) -> np.ndarray:
    stem = files.astype(str).map(lambda x: re.sub(r"_window_\d+.*$", "", Path(x).stem))
    codes, _ = pd.factorize(stem, sort=False)
    n = int(codes.max()) + 1
    logp = np.log(suite.normalize(p))
    sums = np.vstack([np.bincount(codes, weights=logp[:, c], minlength=n) for c in range(4)]).T
    group_p = suite.normalize(np.exp(sums - sums.max(axis=1, keepdims=True)))
    return group_p[codes]


def main():
    out=Path("outputs"); root=Path("/home/ttung05/Desktop/tree_base/tree_structures"); report=out/"audio_feature_benchmarks"/"segment_audio_fusion_group_selection"; report.mkdir(parents=True,exist_ok=True)
    full,train_idx,val_idx=suite.build_group_split(root,report); X,y_img=suite.image_cache(out,"hand_train_full"); y=full.y.to_numpy(int)
    image=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.1,class_weight="balanced",max_iter=800,random_state=42))]).fit(X[train_idx],y[train_idx]); raw=image.predict_proba(X[val_idx]); pi=np.zeros((len(val_idx),4));
    for col,cls in enumerate(image.classes_): pi[:,int(cls)]=raw[:,col]
    pi=suite.apply_bias(pi)
    run=out/"audio_feature_benchmarks"/"audio_highsr_temporal_tta_select"; al=json.loads((run/"reports"/"audio_highsr_temporal_tta_select_selected_without_test.json").read_text()); pair=suite.normalize(np.load(out/"audio_feature_benchmarks"/"audio_group_consistency_pair_blend_select"/"oof_sources"/"pairwise_selected_clean_oof_proba.npy"))[val_idx]
    views={}
    for v in ["clean","robot_mix","bandlimit"]:
        h=np.load(run/"oof_proba"/al["selected_candidate"]/f"{v}_oof_proba.npy")[val_idx]; views[v]=segment_only(full.audio_file.iloc[val_idx],suite.normalize(.8*h+.2*pair))
    rows=[]
    for a in np.linspace(0,1,21):
        for kind in ["linear","log"]:
            scores={}
            for v,pa in views.items():
                p=suite.normalize(a*pa+(1-a)*pi) if kind=="linear" else suite.normalize(np.exp(a*np.log(pa)+(1-a)*np.log(pi)))
                scores[v]=float(f1_score(y[val_idx],p.argmax(1),average="macro",zero_division=0))
            rows.append({"kind":kind,"audio_weight":float(a),**{f"macro_f1_{v}":s for v,s in scores.items()},"worst_view_macro_f1":min(scores.values()),"mean_view_macro_f1":float(np.mean(list(scores.values())))})
    lb=pd.DataFrame(rows).sort_values(["worst_view_macro_f1","mean_view_macro_f1"],ascending=False).reset_index(drop=True); lb.to_csv(report/"hand_group_segment_audio_leaderboard.csv",index=False); best=lb.iloc[0].to_dict(); lock={"protocol":"segment_audio_fusion_specimen_group_val_only","split_group":"specimen_group","inference_group":"segment_group","group_overlap":0,"test_loaded":False,"selected_candidate":best}; (report/"selection_lock.json").write_text(json.dumps(lock,indent=2,default=float),encoding="utf-8"); print(json.dumps(lock,indent=2,default=float))


if __name__=="__main__":main()
