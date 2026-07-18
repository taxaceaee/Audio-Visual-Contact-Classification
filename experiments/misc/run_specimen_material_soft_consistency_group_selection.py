from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/specimen_material_soft_consistency_group_selection'))
def main():
    OUT.mkdir(parents=True,exist_ok=True); d=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); files=d.audio_file.astype(str); seg=sel.seg(files).to_numpy(); spr=sel.spec(files).to_numpy(); useg,codes=np.unique(seg,return_inverse=True); sp=np.array([spr[codes==i][0] for i in range(len(useg))]); sy=np.array([d.y.to_numpy(np.int64)[codes==i][0] for i in range(len(useg))]); pm=np.load('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection/hand_oof_meta.npy'); ph=np.load('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection/hand_oof_hier.npy'); base=np.where(ph.argmax(1)==2,2,pm.argmax(1)); conf=np.maximum(pm.max(1),ph.max(1)); rows=[]
    for source in ['meta','hier','mean']:
      score=pm if source=='meta' else ph if source=='hier' else (pm+ph)/2
      for mode in ['low_conf','leaf_twig','all_contact']:
       for cutoff in [0.0,.35,.50,.65,.80,.90]:
        for strength in [0.25,.5,.75,1.0]:
         pred=base.copy()
         for s in np.unique(sp):
          ix=np.flatnonzero(sp==s); material=int(np.argmax(score[ix,1:].sum(0))+1); target=ix[base[ix]>0]
          if mode=='low_conf': target=target[conf[target]<cutoff]
          elif mode=='leaf_twig': target=target[np.isin(base[target],[1,3]) & (conf[target]<cutoff)]
          pred[target]=np.where(np.random.default_rng(0).random(len(target))<strength,material,pred[target]) if False else pred[target]
          # deterministic soft rule: only apply when material score beats current class by strength margin
          if len(target):
           cur=score[target,base[target]]; dom=score[target,material]; pred[target[dom-cur>=strength*.1]]=material
         rows.append({'source':source,'mode':mode,'cutoff':cutoff,'strength':strength,'macro_f1_4class':float(f1_score(sy,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(sy>0,pred>0,average='macro',zero_division=0))})
    board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_specimen_material_soft_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); out={'protocol':'specimen_material_soft_consistency_group_OOF','selection_data':'hand/default OOF only','group_column':'specimen_group','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
