from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

import run_multimodal_val_locked_suite as suite
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base


ROOT=Path("/home/ttung05/Desktop/tree_base/tree_structures"); OUT=Path("outputs"); REPORT=OUT/"audio_feature_benchmarks/exact_anchor_clip_knn_svm_group_selection"; NAMES=["ambient","leaf","trunk","twig"]; LABELS=np.arange(4)

def aligned(model,X):
    raw=model.predict_proba(X); p=np.zeros((len(X),4))
    for col,cls in enumerate(model.classes_): p[:,int(cls)]=raw[:,col]
    return suite.normalize(p)

def main():
    lock=json.loads((REPORT/"selection_lock.json").read_text()); c=lock["selected_candidate"]
    if lock.get("test_loaded"): raise AssertionError("test used")
    Xh=np.load(OUT/"image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy").astype(np.float32); hand=audio_base.load_manifest(ROOT/"audio_visual_dataset_default/dataset.csv","hand_train"); yh=hand.y.to_numpy(int)
    m=Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=15,weights="distance",p=2,n_jobs=8))]).fit(Xh,yh)
    test=audio_base.load_manifest(ROOT/"audio_visual_dataset_robo_default/dataset.csv","robot_test"); rawtest=pd.read_csv(ROOT/"audio_visual_dataset_robo_default/dataset.csv"); Xt=np.load(OUT/"image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/X.npy").astype(np.float32); y=test.y.to_numpy(int)
    pi=aligned(m,Xt); high=pd.read_csv(OUT/"audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv"); pair=pd.read_csv(OUT/"audio_feature_benchmarks/audio_group_consistency_pair_blend_select/reports/audio_group_consistency_pair_blend_select_final_test_predictions.csv")
    if not np.array_equal(high.audio_file.astype(str),rawtest.audio_file.astype(str)) or not np.array_equal(pair.audio_file.astype(str),rawtest.audio_file.astype(str)): raise AssertionError("audio alignment")
    pa=lift.anchor_lift_proba(test,high[suite.PROBA_COLUMNS].to_numpy(float),pair[suite.PROBA_COLUMNS].to_numpy(float)); iw=float(c["image_weight"]); bias=np.array([0.,0.,float(c["trunk_bias"]),float(c["twig_bias"])])
    p=suite.normalize(np.exp((1-iw)*np.log(suite.normalize(pa))+iw*np.log(suite.normalize(pi))+bias[None,:])); pred=p.argmax(1); by,bp=(y>0).astype(int),(pred>0).astype(int)
    result={"split":"robot_test_final","protocol":"locked_after_5fold_specimen_group_hand_only","n":int(len(y)),"locked_candidate":c,"macro_f1_4class":float(f1_score(y,pred,average="macro",zero_division=0)),"accuracy_4class":float(accuracy_score(y,pred)),"macro_precision_4class":float(precision_score(y,pred,average="macro",zero_division=0)),"macro_recall_4class":float(recall_score(y,pred,average="macro",zero_division=0)),"weighted_f1_4class":float(f1_score(y,pred,average="weighted",zero_division=0)),"binary_accuracy_ambient_noambient":float(accuracy_score(by,bp)),"binary_macro_precision_ambient_noambient":float(precision_score(by,bp,average="macro",zero_division=0)),"binary_macro_recall_ambient_noambient":float(recall_score(by,bp,average="macro",zero_division=0)),"binary_macro_f1_ambient_noambient":float(f1_score(by,bp,average="macro",zero_division=0)),"per_class_4class":classification_report(y,pred,labels=LABELS,target_names=NAMES,output_dict=True,zero_division=0),"per_class_binary":classification_report(by,bp,labels=[0,1],target_names=["ambient","noambient"],output_dict=True,zero_division=0),"confusion_matrix_4class":confusion_matrix(y,pred,labels=LABELS).tolist(),"confusion_matrix_binary":confusion_matrix(by,bp,labels=[0,1]).tolist()}
    (REPORT/"exact_anchor_clip_knn_final_test_metrics.json").write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))

if __name__=="__main__": main()
