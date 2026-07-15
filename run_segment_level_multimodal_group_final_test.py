from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT=Path("/home/ttung05/Desktop/tree_base/tree_structures"); OUT=Path("outputs"); REPORT=OUT/"audio_feature_benchmarks"/"segment_level_multimodal_group_selection"; NAMES=["ambient","leaf","trunk","twig"]; LABELS=np.arange(4)


def main():
    lock=json.loads((REPORT/"selection_lock.json").read_text()); raw_tr=pd.read_csv(ROOT/"audio_visual_dataset_default"/"dataset.csv"); raw_te=pd.read_csv(ROOT/"audio_visual_dataset_robo_default"/"dataset.csv"); labels={n:i for i,n in enumerate(NAMES)}
    def prep(raw):
        f=pd.DataFrame({"audio_file":raw.audio_file.astype(str),"y":raw.category.astype(str).str.lower().map(labels).astype(int)})
        f["segment"]=f.audio_file.map(lambda x:re.sub(r"_window_\d+.*$","",Path(x).stem)); return f
    tr=prep(raw_tr); te=prep(raw_te); y=te.y.to_numpy(dtype=int)
    a_tr=np.load(OUT/"audio_feature_benchmarks"/"total240_trainval_select"/"features"/"hand_train_full"/"X.npy"); a_te=np.load(OUT/"audio_feature_benchmarks"/"total240_trainval_select"/"features"/"robot_test"/"X.npy"); i_tr=np.load(OUT/"image_deep_features"/"resnet18_224"/"hand_train_full"/"X.npy"); i_te=np.load(OUT/"image_deep_features"/"resnet18_224"/"robot_test"/"X.npy"); Xtr=np.hstack([a_tr,i_tr]); Xte=np.hstack([a_te,i_te])
    def aggregate(frame,X):
        groups=frame.groupby("segment",sort=False); names=list(groups.groups); Xg=np.vstack([X[groups.groups[g].to_numpy()].mean(axis=0) for g in names]); yg=np.asarray([frame.loc[groups.groups[g],"y"].mode().iloc[0] for g in names],int); return names,Xg,yg
    gtr,Xg,yg=aggregate(tr,Xtr); gte,Xtg,yte=aggregate(te,Xte); model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.03,class_weight="balanced",max_iter=1000,random_state=42))]).fit(Xg,yg); pred_seg=model.predict(Xtg); mapping=dict(zip(gte,pred_seg)); pred=te.segment.map(mapping).to_numpy(int); by,bp=(y>0).astype(int),(pred>0).astype(int)
    result={"split":"robot_test_final","n":int(len(y)),"locked_candidate":lock["best_val_candidate"],"accuracy_4class":float(accuracy_score(y,pred)),"macro_precision_4class":float(precision_score(y,pred,average="macro",zero_division=0)),"macro_recall_4class":float(recall_score(y,pred,average="macro",zero_division=0)),"macro_f1_4class":float(f1_score(y,pred,average="macro",zero_division=0)),"weighted_f1_4class":float(f1_score(y,pred,average="weighted",zero_division=0)),"binary_accuracy_ambient_noambient":float(accuracy_score(by,bp)),"binary_macro_precision_ambient_noambient":float(precision_score(by,bp,average="macro",zero_division=0)),"binary_macro_recall_ambient_noambient":float(recall_score(by,bp,average="macro",zero_division=0)),"binary_macro_f1_ambient_noambient":float(f1_score(by,bp,average="macro",zero_division=0)),"per_class_4class":classification_report(y,pred,labels=LABELS,target_names=NAMES,output_dict=True,zero_division=0),"per_class_binary":classification_report(by,bp,labels=[0,1],target_names=["ambient","noambient"],output_dict=True,zero_division=0),"confusion_matrix_4class":confusion_matrix(y,pred,labels=LABELS).tolist(),"confusion_matrix_binary":confusion_matrix(by,bp,labels=[0,1]).tolist()}
    (REPORT/"segment_final_test_metrics.json").write_text(json.dumps(result,indent=2,default=float),encoding="utf-8"); pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,(dict,list))}]).to_csv(REPORT/"segment_final_test_report.csv",index=False); print(json.dumps(result,indent=2,default=float))


if __name__=="__main__": main()
