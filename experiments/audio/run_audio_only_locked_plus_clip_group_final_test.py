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


ROOT=Path("/home/ttung05/Desktop/tree_base/tree_structures"); OUT=Path("outputs"); REPORT=OUT/"audio_feature_benchmarks/audio_only_locked_plus_clip_group_selection"; NAMES=["ambient","leaf","trunk","twig"]; LABELS=np.arange(4)

def main():
    lock=json.loads((REPORT/"selection_lock.json").read_text()); c=lock["selected_candidate"]; iw=float(c["image_weight"])
    hand=suite.load_manifest(ROOT/"audio_visual_dataset_default/dataset.csv",ROOT/"audio_visual_dataset_default","hand_train"); yh=hand.y.to_numpy(int); Xh=np.load(OUT/"image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy"); m=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.1,class_weight="balanced",max_iter=1200,random_state=42))]).fit(Xh,yh)
    rawtest=pd.read_csv(ROOT/"audio_visual_dataset_robo_default/dataset.csv"); test=suite.load_manifest(ROOT/"audio_visual_dataset_robo_default/dataset.csv",ROOT/"audio_visual_dataset_robo_default","robot_test"); Xt=np.load(OUT/"image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/X.npy"); y=test.y.to_numpy(int); raw=m.predict_proba(Xt); pi=np.zeros((len(Xt),4))
    for col,cls in enumerate(m.classes_): pi[:,int(cls)]=raw[:,col]
    pi=suite.apply_bias(pi)
    audio=pd.read_csv(OUT/"audio_feature_benchmarks/audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv")
    if not np.array_equal(audio.audio_file.astype(str),rawtest.audio_file.astype(str)): raise AssertionError("audio alignment")
    pa=suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy(float)); p=suite.normalize(np.exp((1-iw)*np.log(pa)+iw*np.log(pi))); pred=p.argmax(1); by,bp=(y>0).astype(int),(pred>0).astype(int)
    result={"split":"robot_test_final","protocol":"audio_only_locked_plus_clip_after_5fold_specimen_group_selection","n":int(len(y)),"locked_candidate":c,"accuracy_4class":float(accuracy_score(y,pred)),"macro_precision_4class":float(precision_score(y,pred,average="macro",zero_division=0)),"macro_recall_4class":float(recall_score(y,pred,average="macro",zero_division=0)),"macro_f1_4class":float(f1_score(y,pred,average="macro",zero_division=0)),"weighted_f1_4class":float(f1_score(y,pred,average="weighted",zero_division=0)),"binary_accuracy_ambient_noambient":float(accuracy_score(by,bp)),"binary_macro_precision_ambient_noambient":float(precision_score(by,bp,average="macro",zero_division=0)),"binary_macro_recall_ambient_noambient":float(recall_score(by,bp,average="macro",zero_division=0)),"binary_macro_f1_ambient_noambient":float(f1_score(by,bp,average="macro",zero_division=0)),"per_class_4class":classification_report(y,pred,labels=LABELS,target_names=NAMES,output_dict=True,zero_division=0),"per_class_binary":classification_report(by,bp,labels=[0,1],target_names=["ambient","noambient"],output_dict=True,zero_division=0),"confusion_matrix_4class":confusion_matrix(y,pred,labels=LABELS).tolist(),"confusion_matrix_binary":confusion_matrix(by,bp,labels=[0,1]).tolist()}
    (REPORT/"audio_only_locked_plus_clip_final_test_metrics.json").write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))

if __name__=="__main__": main()
