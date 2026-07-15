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


BACKBONES = {
    "resnet18": Path("outputs/image_deep_features/resnet18_224"),
    "resnet34": Path("outputs/image_deep_features/resnet34_224"),
    "vit_b16": Path("outputs/image_deep_features/vit_b16_224"),
    "clip_vit_base": Path("outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"),
    "efficientnet_b3": Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k"),
}

COMBOS = {
    "clip_resnet18": ["clip_vit_base", "resnet18"],
    "clip_efficientnet": ["clip_vit_base", "efficientnet_b3"],
    "clip_resnet18_efficientnet": ["clip_vit_base", "resnet18", "efficientnet_b3"],
    "all_visual": ["clip_vit_base", "resnet18", "resnet34", "vit_b16", "efficientnet_b3"],
}


def load_backbone(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    members = COMBOS.get(name, [name])
    arrays = [np.load(BACKBONES[member] / split / "X.npy") for member in members]
    y = np.load(BACKBONES[members[0]] / split / "y.npy").astype(int)
    return np.hstack(arrays), y


def aligned_proba(model, X):
    raw=model.predict_proba(X); p=np.zeros((len(X),4))
    for col,cls in enumerate(model.classes_):p[:,int(cls)]=raw[:,col]
    return suite.normalize(p)


def main():
    out=Path("outputs");root=Path("/home/ttung05/Desktop/tree_base/tree_structures");report=out/"audio_feature_benchmarks"/"multibackbone_image_audio_group_selection";report.mkdir(parents=True,exist_ok=True)
    full,train_idx,val_idx=suite.build_group_split(root,report);y=full.y.to_numpy(int)
    run=out/"audio_feature_benchmarks"/"audio_highsr_temporal_tta_select";al=json.loads((run/"reports"/"audio_highsr_temporal_tta_select_selected_without_test.json").read_text());pair=suite.normalize(np.load(out/"audio_feature_benchmarks"/"audio_group_consistency_pair_blend_select"/"oof_sources"/"pairwise_selected_clean_oof_proba.npy"))[val_idx]
    audio_views={v:suite.segment_lift(full.audio_file.iloc[val_idx],suite.normalize(.8*np.load(run/"oof_proba"/al["selected_candidate"]/f"{v}_oof_proba.npy")[val_idx]+.2*pair)) for v in ["clean","robot_mix","bandlimit"]}
    rows=[]
    for name in [*BACKBONES, *COMBOS]:
        X,yc=load_backbone(name,"hand_train_full")
        if not np.array_equal(y,yc):raise AssertionError(f"{name} labels misaligned")
        model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.1,class_weight="balanced",max_iter=1000,random_state=42))]).fit(X[train_idx],y[train_idx]);pi=aligned_proba(model,X[val_idx])
        for alpha in np.linspace(0,1,21):
            for kind in ["linear","log"]:
                scores={}
                for v,pa in audio_views.items():
                    p=suite.normalize(alpha*pa+(1-alpha)*pi) if kind=="linear" else suite.normalize(np.exp(alpha*np.log(pa)+(1-alpha)*np.log(pi)))
                    scores[v]=float(f1_score(y[val_idx],p.argmax(1),average="macro",zero_division=0))
                rows.append({"backbone":name,"kind":kind,"audio_weight":float(alpha),**{f"macro_f1_{v}":s for v,s in scores.items()},"worst_view_macro_f1":min(scores.values()),"mean_view_macro_f1":float(np.mean(list(scores.values())))})
    lb=pd.DataFrame(rows).sort_values(["worst_view_macro_f1","mean_view_macro_f1"],ascending=False).reset_index(drop=True);lb.to_csv(report/"hand_group_multibackbone_leaderboard.csv",index=False);best=lb.iloc[0].to_dict();lock={"protocol":"multibackbone_image_audio_specimen_group_val_only","group_overlap":0,"test_loaded":False,"selected_candidate":best};(report/"selection_lock.json").write_text(json.dumps(lock,indent=2,default=float),encoding="utf-8");print(json.dumps(lock,indent=2,default=float))


if __name__=="__main__":main()
