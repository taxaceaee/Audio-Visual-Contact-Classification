from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_lift_source_blend_select_final_test as lift
import train_val_select_final_test as audio_base

ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/segment_pool_material_group_selection')); IMG=Path(os.environ.get('SEG_IMG_ROOT','outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'))
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def pool(files,x,y=None):
    keys=seg(files).to_numpy(); u,c=np.unique(keys,return_inverse=True); px=np.vstack([x[c==i].mean(0) for i in range(len(u))]); py=None if y is None else np.array([y[c==i][0] for i in range(len(u))]); return u,c,px,py
def main():
    lock=json.loads((OUT/'selection_lock.json').read_text()); assert not lock.get('test_loaded'); b=lock['selected_candidate']; C=float(b['C']); iw=float(b['image_material_weight']); th=float(b['contact_threshold']); audio_base.configure_feature_set('total240')
    hand=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); hfr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); yh=hfr.y.to_numpy(np.int64); xh=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); hu,hc,hx,hy=pool(hfr.audio_file,xh,yh)
    af=pd.read_csv('outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv')
    # Use the locked final audio output on hand only via the already selected OOF source for fitting.
    # For material classifier, labels and image segment features are sufficient; audio is combined at test.
    model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000,class_weight='balanced',random_state=42)); contact=hy>0; model.fit(hx[contact],hy[contact]-1)
    test=audio_base.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv','robot_test'); tfr=suite.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv',ROOT/'audio_visual_dataset_robo_default','robot_test'); yt=tfr.y.to_numpy(np.int64); xt=np.load(IMG/'robot_test/X.npy').astype(np.float32); tu,tc,tx,ty=pool(tfr.audio_file,xt,yt)
    raw=pd.read_csv(ROOT/'audio_visual_dataset_robo_default/dataset.csv'); ap=pd.read_csv('outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv'); assert np.array_equal(ap.audio_file.astype(str).to_numpy(),raw.audio_file.astype(str).to_numpy())
    audio=suite.normalize(ap[suite.PROBA_COLUMNS].to_numpy(np.float64)); pc,lg=avr.avr_inputs(audio); sk=seg(test.audio_file).to_numpy(); # align audio to pooled test keys
    order={k:i for i,k in enumerate(tu)}; seg_pc=np.array([pc[sk==k].mean() for k in tu]); seg_mat=np.vstack([suite.normalize(np.exp(lg[sk==k].mean(0,keepdims=True)))[0] for k in tu]); img_mat=model.predict_proba(tx); mat=suite.normalize((1-iw)*seg_mat+iw*img_mat); pred_seg=np.where(seg_pc>=th,mat.argmax(1)+1,0); pred=np.array([pred_seg[order[k]] for k in sk]); by=(yt>0).astype(int); bp=(pred>0).astype(int); result={'split':'robot_test_final','protocol':lock['protocol'],'n':int(len(yt)),'locked_candidate':b,'metrics':{'accuracy_4class':float(accuracy_score(yt,pred)),'macro_precision_4class':float(precision_score(yt,pred,average='macro',zero_division=0)),'macro_recall_4class':float(recall_score(yt,pred,average='macro',zero_division=0)),'macro_f1_4class':float(f1_score(yt,pred,average='macro',zero_division=0)),'weighted_f1_4class':float(f1_score(yt,pred,average='weighted',zero_division=0)),'binary_accuracy':float(accuracy_score(by,bp)),'binary_macro_precision':float(precision_score(by,bp,average='macro',zero_division=0)),'binary_macro_recall':float(recall_score(by,bp,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(by,bp,average='macro',zero_division=0))},'per_class_4class':classification_report(yt,pred,labels=[0,1,2,3],target_names=['ambient','leaf','trunk','twig'],output_dict=True,zero_division=0),'confusion_matrix_4class':confusion_matrix(yt,pred,labels=[0,1,2,3]).tolist(),'confusion_matrix_binary':confusion_matrix(by,bp,labels=[0,1]).tolist(),'invariants':{'test_loaded_after_lock':True,'image_controls_contact':False,'audio_controls_contact':True,'material_classifier_fit_contact_only':True,'separate_segment_aggregation':True,'raw_four_class_audio_argmax':False}}
    (OUT/'segment_pool_material_final_test_metrics.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))
if __name__=='__main__': main()
