from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/segment_rowprob_material_group_selection')); IMG=Path(os.environ.get('SEG_IMG_ROOT','outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'))
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def pool(files,x,y=None):
    keys=seg(files).to_numpy(); u,c=np.unique(keys,return_inverse=True); px=np.vstack([x[c==i].mean(0) for i in range(len(u))]); py=None if y is None else np.array([y[c==i][0] for i in range(len(u))]); return u,c,px,py
def main():
    lock=json.loads((OUT/'selection_lock.json').read_text()); assert not lock.get('test_loaded'); b=lock['selected_candidate']; w=np.array([float(b['leaf_weight']),float(b['trunk_weight']),float(b['twig_weight'])]); th=float(b['contact_threshold']); audio_base.configure_feature_set('total240')
    hfr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); yh=hfr.y.to_numpy(np.int64); xh=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); _,hc,_,_=pool(hfr.audio_file,xh,yh); skh=seg(hfr.audio_file).to_numpy(); sfh=np.unique(skh,return_inverse=True)[1]
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1000,class_weight='balanced',random_state=42)); m.fit(xh[yh>0],yh[yh>0]-1)
    tfr=suite.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv',ROOT/'audio_visual_dataset_robo_default','robot_test'); yt=tfr.y.to_numpy(np.int64); xt=np.load(IMG/'robot_test/X.npy').astype(np.float32); tu,tc,_,_=pool(tfr.audio_file,xt,yt); row_img=m.predict_proba(xt); img_seg=np.vstack([row_img[tc==i].mean(0) for i in range(len(tu))])
    audio=pd.read_csv('outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv'); raw=pd.read_csv(ROOT/'audio_visual_dataset_robo_default/dataset.csv'); assert np.array_equal(audio.audio_file.astype(str).to_numpy(),raw.audio_file.astype(str).to_numpy()); proba=suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy(np.float64)); pc,lg=avr.avr_inputs(proba); sk=seg(tfr.audio_file).to_numpy(); spc=np.array([pc[sk==k].mean() for k in tu]); sa=np.vstack([suite.normalize(np.exp(lg[sk==k].mean(0,keepdims=True)))[0] for k in tu]); mat=suite.normalize(sa*(1-w)+img_seg*w); pred_seg=np.where(spc>=th,mat.argmax(1)+1,0); order={k:i for i,k in enumerate(tu)}; pred=np.array([pred_seg[order[k]] for k in sk]); by=(yt>0).astype(int); bp=(pred>0).astype(int)
    result={'split':'robot_test_final','protocol':lock['protocol'],'n':int(len(yt)),'locked_candidate':b,'metrics':{'accuracy_4class':float(accuracy_score(yt,pred)),'macro_precision_4class':float(precision_score(yt,pred,average='macro',zero_division=0)),'macro_recall_4class':float(recall_score(yt,pred,average='macro',zero_division=0)),'macro_f1_4class':float(f1_score(yt,pred,average='macro',zero_division=0)),'weighted_f1_4class':float(f1_score(yt,pred,average='weighted',zero_division=0)),'binary_accuracy':float(accuracy_score(by,bp)),'binary_macro_precision':float(precision_score(by,bp,average='macro',zero_division=0)),'binary_macro_recall':float(recall_score(by,bp,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(by,bp,average='macro',zero_division=0))},'per_class_4class':classification_report(yt,pred,labels=[0,1,2,3],target_names=['ambient','leaf','trunk','twig'],output_dict=True,zero_division=0),'confusion_matrix_4class':confusion_matrix(yt,pred,labels=[0,1,2,3]).tolist(),'confusion_matrix_binary':confusion_matrix(by,bp,labels=[0,1]).tolist(),'invariants':{'test_loaded_after_lock':True,'selection_used_hand_only':True,'no_new_split':True,'image_controls_contact':False,'audio_controls_contact':True}}
    (OUT/'segment_rowprob_material_final_test_metrics.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))
if __name__=='__main__': main()
