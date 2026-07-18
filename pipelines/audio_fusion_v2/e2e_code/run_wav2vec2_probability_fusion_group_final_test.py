from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torchaudio
import run_multimodal_val_locked_suite as suite

ROOT=Path("/home/ttung05/Desktop/tree_base/tree_structures");OUT=Path("outputs");REPORT=OUT/"audio_feature_benchmarks"/"wav2vec2_probability_fusion_group_selection";CACHE=OUT/"audio_wav2vec2_features"/"robot_test_X.npy"

def load(path,sr):
    old,x=wavfile.read(path); integer=np.issubdtype(x.dtype,np.integer); x=x.astype(np.float32); x=x.mean(1) if x.ndim==2 else x; x=x/32768 if integer else x; x=resample_poly(x,sr,old) if old!=sr else x; x=np.pad(x,(0,max(0,sr-len(x))))[:sr]; return (x/(np.max(np.abs(x))+1e-8)).astype(np.float32)
def extract(frame,device):
    if CACHE.exists():return np.load(CACHE)
    model=torchaudio.pipelines.WAV2VEC2_BASE.get_model().to(device).eval();sr=16000;rows=[];batch=[]
    for i,f in enumerate(frame.audio_file.astype(str)):
        batch.append(load(ROOT/"audio_visual_dataset_robo_default"/f,sr))
        if len(batch)==32 or i==len(frame)-1:
            with torch.inference_mode():
                h=model.extract_features(torch.from_numpy(np.stack(batch)).to(device))[0][-1];rows.append(torch.cat([h.mean(1),h.std(1)],1).cpu().numpy().astype(np.float32))
            batch=[];print(i+1,len(frame),flush=True)
    X=np.concatenate(rows);np.save(CACHE,X);return X
def aligned(m,X):
    raw=m.predict_proba(X);p=np.zeros((len(X),4))
    for col,cls in enumerate(m.classes_):p[:,int(cls)]=raw[:,col]
    return suite.normalize(p)
def main():
    lock=json.loads((REPORT/"selection_lock.json").read_text());c=lock["selected_candidate"];hand=suite.load_manifest(ROOT/"audio_visual_dataset_default"/"dataset.csv",ROOT/"audio_visual_dataset_default","hand_train");test=suite.load_manifest(ROOT/"audio_visual_dataset_robo_default"/"dataset.csv",ROOT/"audio_visual_dataset_robo_default","robot_test");audio=np.load(OUT/"audio_wav2vec2_features"/"hand_train_full_X.npy");img=np.load(OUT/"image_deep_features"/"resnet18_224"/"hand_train_full"/"X.npy");it=np.load(OUT/"image_deep_features"/"resnet18_224"/"robot_test"/"X.npy");device=torch.device("cuda" if torch.cuda.is_available() else "cpu");at=extract(test,device);model=Pipeline([("scale",StandardScaler()),("model",SGDClassifier(loss="log_loss",alpha=.001,class_weight="balanced",max_iter=120,tol=1e-3,random_state=42,n_jobs=-1))]).fit(np.hstack([audio,img]),hand.y.to_numpy(int));pw=aligned(model,np.hstack([at,it]));old=pd.read_csv(OUT/"audio_feature_benchmarks"/"audio_lift_source_blend_select"/"reports"/"audio_lift_source_blend_select_final_test_predictions.csv");pa=suite.normalize(old[suite.PROBA_COLUMNS].to_numpy());a=float(c["old_audio_weight"]);p=suite.normalize(np.exp(a*np.log(pa)+(1-a)*np.log(pw)));y=test.y.to_numpy(int);pred=p.argmax(1);by,bp=(y>0).astype(int),(pred>0).astype(int);names=["ambient","leaf","trunk","twig"];result={"split":"robot_test_final","n":int(len(y)),"locked_candidate":c,"accuracy_4class":float(accuracy_score(y,pred)),"macro_precision_4class":float(precision_score(y,pred,average="macro",zero_division=0)),"macro_recall_4class":float(recall_score(y,pred,average="macro",zero_division=0)),"macro_f1_4class":float(f1_score(y,pred,average="macro",zero_division=0)),"weighted_f1_4class":float(f1_score(y,pred,average="weighted",zero_division=0)),"binary_accuracy_ambient_noambient":float(accuracy_score(by,bp)),"binary_macro_precision_ambient_noambient":float(precision_score(by,bp,average="macro",zero_division=0)),"binary_macro_recall_ambient_noambient":float(recall_score(by,bp,average="macro",zero_division=0)),"binary_macro_f1_ambient_noambient":float(f1_score(by,bp,average="macro",zero_division=0)),"per_class_4class":classification_report(y,pred,labels=np.arange(4),target_names=names,output_dict=True,zero_division=0),"per_class_binary":classification_report(by,bp,labels=[0,1],target_names=["ambient","noambient"],output_dict=True,zero_division=0),"confusion_matrix_4class":confusion_matrix(y,pred,labels=np.arange(4)).tolist(),"confusion_matrix_binary":confusion_matrix(by,bp,labels=[0,1]).tolist()};(REPORT/"wav2vec2_final_test_metrics.json").write_text(json.dumps(result,indent=2,default=float),encoding="utf-8");print(json.dumps(result,indent=2,default=float))
if __name__=="__main__":main()
