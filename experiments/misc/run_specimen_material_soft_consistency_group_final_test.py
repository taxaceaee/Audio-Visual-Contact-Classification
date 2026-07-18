from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/specimen_material_soft_consistency_group_selection')); ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
def main():
    lock=json.load(open(OUT/'selection_lock.json')); assert not lock.get('test_loaded'); b=lock['selected_candidate']; z=np.load('outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/segment_rule_stack_v2_base_segment_outputs.npz',allow_pickle=True); pm=z['p_meta']; ph=z['p_h']; base=np.where(z['pred_h']==2,2,z['pred_m']); conf=np.maximum(pm.max(1),ph.max(1)); sp=z['specimen_ids'].astype(str); score=ph; cutoff=float(b['cutoff']); strength=float(b['strength']); pred=base.copy()
    for s in np.unique(sp):
        ix=np.flatnonzero(sp==s); material=int(np.argmax(score[ix,1:].sum(0))+1); target=ix[(base[ix]>0)&(conf[ix]<cutoff)];
        if len(target):
            cur=score[target,base[target]]; dom=score[target,material]; pred[target[dom-cur>=strength*.1]]=material
    row_seg=z['row_segment_ids'].astype(str); order={str(k):i for i,k in enumerate(z['segment_ids'].astype(str))}; row_pred=np.array([pred[order[k]] for k in row_seg]); test=pd.read_csv(ROOT/'audio_visual_dataset_robo_default/dataset.csv'); labels=test.category.map({'ambient':0,'leaf':1,'trunk':2,'twig':3}).to_numpy(np.int64); by=(labels>0).astype(int); bp=(row_pred>0).astype(int)
    result={'split':'robot_test_final','protocol':lock['protocol'],'n':int(len(labels)),'locked_candidate':b,'metrics':{'accuracy_4class':float(accuracy_score(labels,row_pred)),'macro_precision_4class':float(precision_score(labels,row_pred,average='macro',zero_division=0)),'macro_recall_4class':float(recall_score(labels,row_pred,average='macro',zero_division=0)),'macro_f1_4class':float(f1_score(labels,row_pred,average='macro',zero_division=0)),'weighted_f1_4class':float(f1_score(labels,row_pred,average='weighted',zero_division=0)),'binary_accuracy':float(accuracy_score(by,bp)),'binary_macro_precision':float(precision_score(by,bp,average='macro',zero_division=0)),'binary_macro_recall':float(recall_score(by,bp,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(by,bp,average='macro',zero_division=0))},'per_class_4class':classification_report(labels,row_pred,labels=[0,1,2,3],target_names=['ambient','leaf','trunk','twig'],output_dict=True,zero_division=0),'confusion_matrix_4class':confusion_matrix(labels,row_pred,labels=[0,1,2,3]).tolist(),'confusion_matrix_binary':confusion_matrix(by,bp,labels=[0,1]).tolist(),'invariants':{'selection_used_hand_only':True,'test_loaded_after_lock':True,'no_new_split':True}}
    (OUT/'specimen_material_soft_consistency_final_test_metrics.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))
if __name__=='__main__': main()
