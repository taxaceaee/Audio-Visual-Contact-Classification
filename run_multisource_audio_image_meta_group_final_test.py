from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_multisource_meta_group_selection')); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'); NAMES=['highsr_default','highsr_regularized','highsr_extratrees','total240_ensemble','total240_stack_lr','report_gate_proba']
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def pool(codes,x,n): return np.vstack([x[codes==i].mean(0) for i in range(n)])
def feat(srcs,img):
    p=[]
    for x in srcs: p += [x,np.log(np.clip(x,1e-6,1)),x[:,1:].sum(1,keepdims=True),x.max(1,keepdims=True)]
    p += [img,np.log(np.clip(img,1e-6,1)),img[:,1:].sum(1,keepdims=True),img.max(1,keepdims=True)]
    return np.hstack(p).astype(np.float32)
def main():
    lock=json.load(open(OUT/'selection_lock.json')); assert not lock.get('test_loaded'); b=lock['selected_candidate']; c=float(b['C']); cw={0:1.0,1:1.0,2:float(b['trunk_class_weight']),3:float(b['twig_class_weight'])}; hx=np.load(OUT/'hand_multisource_oof_features.npy'); hy=np.load(OUT/'hand_multisource_oof_y.npy'); audio_base.configure_feature_set('total240')
    hfr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); yh=hfr.y.to_numpy(np.int64); xh=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); hseg=seg(hfr.audio_file).to_numpy(); hu,hcodes=np.unique(hseg,return_inverse=True); hxseg=pool(hcodes,xh,len(hu)); hsy=np.array([yh[hcodes==i][0] for i in range(len(hu))]); im=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=2000,class_weight='balanced',multi_class='multinomial',random_state=42)); im.fit(hxseg,hsy)
    paths=broad.final_prediction_paths(Path('outputs')); _, final_sources=broad.load_final_sources(paths); tfr=suite.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv',ROOT/'audio_visual_dataset_robo_default','robot_test'); yt=tfr.y.to_numpy(np.int64); tx=np.load(IMG/'robot_test/X.npy').astype(np.float32); tseg=seg(tfr.audio_file).to_numpy(); tu,tc=np.unique(tseg,return_inverse=True); txseg=pool(tc,tx,len(tu)); img=suite.normalize(im.predict_proba(txseg)); src_seg=[pool(tc,final_sources[n],len(tu)) for n in NAMES]; z=feat(src_seg,img); meta=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=4000,class_weight=cw,multi_class='multinomial',random_state=42)); meta.fit(hx,hy); pred_seg=meta.predict(z); order={k:i for i,k in enumerate(tu)}; pred=np.array([pred_seg[order[k]] for k in tseg]); by=(yt>0).astype(int); bp=(pred>0).astype(int)
    result={'split':'robot_test_final','protocol':lock['protocol'],'n':int(len(yt)),'locked_candidate':b,'metrics':{'accuracy_4class':float(accuracy_score(yt,pred)),'macro_precision_4class':float(precision_score(yt,pred,average='macro',zero_division=0)),'macro_recall_4class':float(recall_score(yt,pred,average='macro',zero_division=0)),'macro_f1_4class':float(f1_score(yt,pred,average='macro',zero_division=0)),'weighted_f1_4class':float(f1_score(yt,pred,average='weighted',zero_division=0)),'binary_accuracy':float(accuracy_score(by,bp)),'binary_macro_precision':float(precision_score(by,bp,average='macro',zero_division=0)),'binary_macro_recall':float(recall_score(by,bp,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(by,bp,average='macro',zero_division=0))},'per_class_4class':classification_report(yt,pred,labels=[0,1,2,3],target_names=['ambient','leaf','trunk','twig'],output_dict=True,zero_division=0),'confusion_matrix_4class':confusion_matrix(yt,pred,labels=[0,1,2,3]).tolist(),'confusion_matrix_binary':confusion_matrix(by,bp,labels=[0,1]).tolist(),'invariants':{'selection_used_hand_only':True,'test_loaded_after_lock':True,'no_new_split':True}}
    (OUT/'multisource_audio_image_meta_final_test_metrics.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))
if __name__=='__main__': main()
