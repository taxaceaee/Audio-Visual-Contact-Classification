from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/specimen_material_consistency_group_selection'))
def main():
    OUT.mkdir(parents=True,exist_ok=True); d=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); files=d.audio_file.astype(str); seg=sel.seg(files).to_numpy(); sp_row=sel.spec(files).to_numpy(); useg,codes=np.unique(seg,return_inverse=True); sp=np.array([sp_row[codes==i][0] for i in range(len(useg))]); sy=np.array([d.y.to_numpy(np.int64)[codes==i][0] for i in range(len(useg))]); pmeta=np.load('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection/hand_oof_meta.npy'); ph=np.load('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection/hand_oof_hier.npy'); lock=json.load(open('outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/selection_lock.json')); b=lock['selected_candidate']; base=np.where(ph.argmax(1)==2,2,pmeta.argmax(1)); rows=[]
    for src in ['meta','hier','mean']:
      for mode in ['all_nonambient','threshold_contact']:
       for th in [0.0,.35,.45,.55,.65,.75]:
        pred=base.copy()
        for s in np.unique(sp):
          ix=np.flatnonzero(sp==s); eligible=ix if mode=='all_nonambient' else ix[((ph[ix,1:].sum(1))>=th)]
          if len(eligible)==0: continue
          probs=pmeta if src=='meta' else ph if src=='hier' else (pmeta+ph)/2
          material=int(np.argmax(probs[eligible,1:].sum(0))+1)
          target=ix[pred[ix]>0] if mode=='all_nonambient' else eligible[pred[eligible]>0]
          pred[target]=material
        rows.append({'source':src,'mode':mode,'threshold':th,'macro_f1_4class':float(f1_score(sy,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(sy>0,pred>0,average='macro',zero_division=0))})
    board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_specimen_material_consistency_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); out={'protocol':'specimen_material_consistency_decoder_group_OOF','selection_data':'hand/default OOF only','group_column':'specimen_group','base_rule':'segment_rule_stack_v2 hier_trunk_meta_else','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
