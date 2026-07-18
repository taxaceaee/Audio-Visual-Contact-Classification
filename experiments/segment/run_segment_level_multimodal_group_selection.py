from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs")
REPORT = OUT / "audio_feature_benchmarks" / "segment_level_multimodal_group_selection"


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "audio_visual_dataset_default" / "dataset.csv")
    labels = {"ambient": 0, "leaf": 1, "trunk": 2, "twig": 3}
    frame = pd.DataFrame({"audio_file": raw.audio_file.astype(str), "y": raw.category.astype(str).str.lower().map(labels).astype(int)})
    frame["segment"] = frame.audio_file.map(lambda x: re.sub(r"_window_\d+.*$", "", Path(x).stem))
    frame["specimen"] = frame.audio_file.map(lambda x: re.sub(r"_segment_.*$", "", Path(x).stem))
    split = OUT / "audio_feature_benchmarks" / "multimodal_val_locked_suite" / "splits"
    train_files = set(pd.read_csv(split / "hand_group_train.csv").audio_file.astype(str))
    val_files = set(pd.read_csv(split / "hand_group_val.csv").audio_file.astype(str))
    train_idx = frame.index[frame.audio_file.isin(train_files)].to_numpy(); val_idx = frame.index[frame.audio_file.isin(val_files)].to_numpy()
    a = np.load(OUT / "audio_feature_benchmarks" / "total240_trainval_select" / "features" / "hand_train_full" / "X.npy")
    i = np.load(OUT / "image_deep_features" / "resnet18_224" / "hand_train_full" / "X.npy")
    X = np.hstack([a, i])
    rows=[]
    for name, idx in [("train", train_idx), ("val", val_idx)]:
        sub=frame.iloc[idx].copy(); sub["row_idx"]=idx
        g=sub.groupby("segment",sort=False)
        seg_names=list(g.groups)
        seg_idx=[g.groups[s].to_numpy() for s in seg_names]
        Xg=np.vstack([X[sub.loc[pos, "row_idx"].to_numpy()].mean(axis=0) for pos in seg_idx])
        yg=np.asarray([sub.loc[pos, "y"].mode().iloc[0] for pos in seg_idx],dtype=int)
        if name=="train": Xtr, ytr, train_seg = Xg, yg, seg_names
        else:
            local = {int(orig): i for i, orig in enumerate(idx.tolist())}
            Xva, yva, val_seg = Xg, yg, seg_names
            val_pos = [np.asarray([local[int(orig)] for orig in pos], dtype=np.int64) for pos in seg_idx]
    for c in [0.03,0.1,0.3,1.0]:
        model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=c,class_weight="balanced",max_iter=1000,random_state=42))]).fit(Xtr,ytr)
        pred_seg=model.predict(Xva)
        pred=np.empty(len(val_idx),dtype=int)
        for p, pos in zip(pred_seg,val_pos): pred[pos]=p
        y_window=frame.iloc[val_idx].y.to_numpy()
        rows.append({"model":f"segment_logreg_c{c:g}","n_train_segments":len(train_seg),"n_val_segments":len(val_seg),"accuracy_4class":accuracy_score(y_window,pred),"macro_precision_4class":precision_score(y_window,pred,average="macro",zero_division=0),"macro_recall_4class":recall_score(y_window,pred,average="macro",zero_division=0),"macro_f1_4class":f1_score(y_window,pred,average="macro",zero_division=0),"binary_macro_f1":f1_score(y_window>0,pred>0,average="macro",zero_division=0)})
    lb=pd.DataFrame(rows).sort_values("macro_f1_4class",ascending=False).reset_index(drop=True); lb.to_csv(REPORT/"hand_group_segment_leaderboard.csv",index=False)
    lock={"protocol":"segment_level_multimodal_specimen_group_val_only","group_column":"specimen_group","segment_column":"segment","group_overlap":0,"test_loaded":False,"best_val_candidate":lb.iloc[0].to_dict()}
    (REPORT/"selection_lock.json").write_text(json.dumps(lock,indent=2,default=float),encoding="utf-8")
    print(json.dumps(lock,indent=2,default=float))


if __name__=="__main__": main()
