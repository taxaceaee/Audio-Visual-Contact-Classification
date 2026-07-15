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


def main():
    out = Path("outputs")
    report = out / "audio_feature_benchmarks" / "multimodal_contact_bias_group_selection"
    lock = json.loads((report / "selection_lock.json").read_text())
    root = Path("/home/ttung05/Desktop/tree_base/tree_structures")
    full, _, _ = suite.build_group_split(root, report)
    X, y = suite.image_cache(out, "hand_train_full")
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=800, random_state=42))]).fit(X, y)
    test = suite.load_manifest(root / "audio_visual_dataset_robo_default" / "dataset.csv", root / "audio_visual_dataset_robo_default", "robot_test")
    Xt, yt = suite.image_cache(out, "robot_test")
    raw = model.predict_proba(Xt)
    pi = np.zeros((len(Xt), 4))
    for col, cls in enumerate(model.classes_): pi[:, int(cls)] = raw[:, col]
    pi = suite.apply_bias(pi)
    audio = pd.read_csv(out / "audio_feature_benchmarks" / "audio_lift_source_blend_select" / "reports" / "audio_lift_source_blend_select_final_test_predictions.csv")
    pa = suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy())
    cand = lock["selected_candidate"]
    bias = np.asarray([-0.05, 0.0, float(cand["trunk_bias"]), float(cand["twig_bias"])])
    p = suite.normalize(np.exp(float(cand["audio_weight"]) * np.log(pa) + (1-float(cand["audio_weight"])) * np.log(pi) + bias[None, :]))
    y = test.y.to_numpy(dtype=np.int64); pred = p.argmax(axis=1); by, bp = (y > 0).astype(int), (pred > 0).astype(int)
    result = {"split":"robot_test_final","n":int(len(y)),"locked_candidate":cand,"accuracy_4class":float(accuracy_score(y,pred)),"macro_precision_4class":float(precision_score(y,pred,average="macro",zero_division=0)),"macro_recall_4class":float(recall_score(y,pred,average="macro",zero_division=0)),"macro_f1_4class":float(f1_score(y,pred,average="macro",zero_division=0)),"weighted_f1_4class":float(f1_score(y,pred,average="weighted",zero_division=0)),"binary_accuracy_ambient_noambient":float(accuracy_score(by,bp)),"binary_macro_precision_ambient_noambient":float(precision_score(by,bp,average="macro",zero_division=0)),"binary_macro_recall_ambient_noambient":float(recall_score(by,bp,average="macro",zero_division=0)),"binary_macro_f1_ambient_noambient":float(f1_score(by,bp,average="macro",zero_division=0)),"per_class_4class":classification_report(y,pred,labels=np.arange(4),target_names=["ambient","leaf","trunk","twig"],output_dict=True,zero_division=0),"per_class_binary":classification_report(by,bp,labels=[0,1],target_names=["ambient","noambient"],output_dict=True,zero_division=0),"confusion_matrix_4class":confusion_matrix(y,pred,labels=np.arange(4)).tolist(),"confusion_matrix_binary":confusion_matrix(by,bp,labels=[0,1]).tolist()}
    (report / "contact_bias_final_test_metrics.json").write_text(json.dumps(result,indent=2,default=float),encoding="utf-8")
    pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,(dict,list))}]).to_csv(report/"contact_bias_final_test_report.csv",index=False)
    print(json.dumps(result,indent=2,default=float))


if __name__ == "__main__": main()
