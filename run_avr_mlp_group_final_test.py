from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import run_avr_group_selection as avr
import run_avr_mlp_group_selection as mlp
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT = Path('outputs/audio_feature_benchmarks/avr_mlp_group_selection'); NAMES=['ambient','leaf','trunk','twig']

def main():
    lock = json.loads((OUT/'selection_lock.json').read_text())
    if lock.get('test_loaded'): raise AssertionError('selection lock already used test')
    alpha=float(lock['selected_candidate']['alpha']); threshold=float(lock['selected_candidate']['contact_threshold'])
    audio_base.configure_feature_set('total240')
    hand_audio=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train')
    hand=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); yh=hand.y.to_numpy(np.int64)
    xh=np.load('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/hand_train_full/X.npy').astype(np.float32)
    groups=suite.specimen_group(hand_audio.audio_file).to_numpy(); from sklearn.model_selection import StratifiedGroupKFold
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(yh)),yh,groups)); ass=np.full(len(yh),-1,np.int64)
    for k,(_,va) in enumerate(folds): ass[va]=k
    high=group_audio.load_highsr_oof(Path('outputs')); pair=specimen.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy'))
    anchor=lift.anchor_lift_proba(hand_audio,high,pair); src=broad.load_train_sources(Path('outputs'),yh,ass,42)
    audio_oof=lift.postprocess(hand_audio,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); _,logits=avr.avr_inputs(audio_oof)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,mean,std=mlp.fit(xh,logits,yh,np.arange(len(yh)),device)
    test_audio=audio_base.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv','robot_test'); raw=pd.read_csv(ROOT/'audio_visual_dataset_robo_default/dataset.csv')
    xt=np.load('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/X.npy').astype(np.float32)
    af=pd.read_csv('outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv')
    if not np.array_equal(af.audio_file.astype(str).to_numpy(),raw.audio_file.astype(str).to_numpy()): raise AssertionError('audio test alignment')
    audio=suite.normalize(af[suite.PROBA_COLUMNS].to_numpy(np.float64)); pc,lg=avr.avr_inputs(audio); res=mlp.predict(model,mean,std,xt,device); mat=suite.normalize(np.exp(lg+alpha*res)); pc,mat=avr.aggregate_avr(test_audio.audio_file,pc,mat); pred=np.where(pc>=threshold,mat.argmax(1)+1,0); y=test_audio.y.to_numpy(np.int64); by=(y>0).astype(int); bp=(pred>0).astype(int)
    result={'split':'robot_test_final','protocol':lock['protocol'],'n':int(len(y)),'locked_candidate':lock['selected_candidate'],'avr_invariants':{'audio_controls_contact':True,'residual_trained_contact_only':True,'audio_source_oof_group_aware':True,'downstream_separate_contact_material_aggregation':True,'four_class_argmax_used':False},'accuracy_4class':float(accuracy_score(y,pred)),'macro_precision_4class':float(precision_score(y,pred,average='macro',zero_division=0)),'macro_recall_4class':float(recall_score(y,pred,average='macro',zero_division=0)),'macro_f1_4class':float(f1_score(y,pred,average='macro',zero_division=0)),'weighted_f1_4class':float(f1_score(y,pred,average='weighted',zero_division=0)),'binary_accuracy_ambient_noambient':float(accuracy_score(by,bp)),'binary_macro_precision_ambient_noambient':float(precision_score(by,bp,average='macro',zero_division=0)),'binary_macro_recall_ambient_noambient':float(recall_score(by,bp,average='macro',zero_division=0)),'binary_macro_f1_ambient_noambient':float(f1_score(by,bp,average='macro',zero_division=0)),'per_class_4class':classification_report(y,pred,labels=[0,1,2,3],target_names=NAMES,output_dict=True,zero_division=0),'per_class_binary':classification_report(by,bp,labels=[0,1],target_names=['ambient','noambient'],output_dict=True,zero_division=0),'confusion_matrix_4class':confusion_matrix(y,pred,labels=[0,1,2,3]).tolist(),'confusion_matrix_binary':confusion_matrix(by,bp,labels=[0,1]).tolist()}
    (OUT/'avr_mlp_final_test_metrics.json').write_text(json.dumps(result,indent=2,default=float)); pd.DataFrame(result['confusion_matrix_4class'],index=NAMES,columns=NAMES).to_csv(OUT/'avr_mlp_final_test_confusion_matrix_4class.csv'); print(json.dumps(result,indent=2,default=float))

if __name__=='__main__': main()
