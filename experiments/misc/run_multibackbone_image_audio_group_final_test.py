from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_multimodal_val_locked_suite as suite
from run_multibackbone_image_audio_group_selection import aligned_proba, load_backbone


def main():
    out=Path("outputs");root=Path("/home/ttung05/Desktop/tree_base/tree_structures");report=out/"audio_feature_benchmarks"/"multibackbone_image_audio_group_selection";lock=json.loads((report/"selection_lock.json").read_text());cand=lock["selected_candidate"]
    full,_,_=suite.build_group_split(root,report);test=suite.load_manifest(root/"audio_visual_dataset_robo_default"/"dataset.csv",root/"audio_visual_dataset_robo_default","robot_test");X,y=load_backbone(cand["backbone"],"hand_train_full");Xt,yt=load_backbone(cand["backbone"],"robot_test")
    model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.1,class_weight="balanced",max_iter=1000,random_state=42))]).fit(X,y);pi=aligned_proba(model,Xt)
    audio=pd.read_csv(out/"audio_feature_benchmarks"/"audio_lift_source_blend_select"/"reports"/"audio_lift_source_blend_select_final_test_predictions.csv");pa=suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy());a=float(cand["audio_weight"]);p=suite.normalize(np.exp(a*np.log(pa)+(1-a)*np.log(pi)));pred=p.argmax(1);by,bp=(yt>0).astype(int),(pred>0).astype(int);names=["ambient","leaf","trunk","twig"]
    result={"split":"robot_test_final","n":int(len(yt)),"locked_candidate":cand,"accuracy_4class":float(accuracy_score(yt,pred)),"macro_precision_4class":float(precision_score(yt,pred,average="macro",zero_division=0)),"macro_recall_4class":float(recall_score(yt,pred,average="macro",zero_division=0)),"macro_f1_4class":float(f1_score(yt,pred,average="macro",zero_division=0)),"weighted_f1_4class":float(f1_score(yt,pred,average="weighted",zero_division=0)),"binary_accuracy_ambient_noambient":float(accuracy_score(by,bp)),"binary_macro_precision_ambient_noambient":float(precision_score(by,bp,average="macro",zero_division=0)),"binary_macro_recall_ambient_noambient":float(recall_score(by,bp,average="macro",zero_division=0)),"binary_macro_f1_ambient_noambient":float(f1_score(by,bp,average="macro",zero_division=0)),"per_class_4class":classification_report(yt,pred,labels=np.arange(4),target_names=names,output_dict=True,zero_division=0),"per_class_binary":classification_report(by,bp,labels=[0,1],target_names=["ambient","noambient"],output_dict=True,zero_division=0),"confusion_matrix_4class":confusion_matrix(yt,pred,labels=np.arange(4)).tolist(),"confusion_matrix_binary":confusion_matrix(by,bp,labels=[0,1]).tolist()};(report/"multibackbone_final_test_metrics.json").write_text(json.dumps(result,indent=2,default=float),encoding="utf-8");pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,(dict,list))}]).to_csv(report/"multibackbone_final_test_report.csv",index=False);print(json.dumps(result,indent=2,default=float))


if __name__=="__main__":main()
