from __future__ import annotations
import json, os
from pathlib import Path
from itertools import product
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/segment_image4_material_group_selection')); IMG=Path(os.environ.get('SEG_IMG_ROOT','outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'))
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def main():
    OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str); x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); sk=seg(files).to_numpy(); sg=spec(files).to_numpy(); useg,codes=np.unique(sk,return_inverse=True); sy=np.array([y[codes==i][0] for i in range(len(useg))]); ss=np.array([sg[codes==i][0] for i in range(len(useg))]); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,ss))
    af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); high=group_audio.load_highsr_oof(Path('outputs')); pair=specimen.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); rg=spec(files).to_numpy(); assignment=np.full(len(y),-1,np.int64); [assignment.__setitem__(va,k) for k,(_,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,rg))]; src=broad.load_train_sources(Path('outputs'),y,assignment,42); audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); pc,lg=avr.avr_inputs(audio); spc=np.array([pc[codes==i].mean() for i in range(len(useg))]); sa=np.vstack([suite.normalize(np.exp(lg[codes==i].mean(0,keepdims=True)))[0] for i in range(len(useg))]); sio=np.zeros((len(sy),3),np.float32)
    for k,(tr,va) in enumerate(folds):
        c_tr=np.concatenate([np.flatnonzero(codes==i) for i in tr]); c_va=np.concatenate([np.flatnonzero(codes==i) for i in va]); m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1000,class_weight='balanced',random_state=42)); m.fit(np.vstack([x[codes==i].mean(0) for i in tr]),sy[tr]); p=m.predict_proba(np.vstack([x[codes==i].mean(0) for i in va])); sio[va]=suite.normalize(p[:,1:]); print(f'fold {k+1}/5 done',flush=True)
    rows=[]
    for ws in product([0,.2,.4,.6,.8,1.0],repeat=3):
      w=np.asarray(ws); mat=suite.normalize(sa*(1-w)+sio*w)
      for th in [.45,.50,.55,.60,.65]:
       pred=np.where(spc>=th,mat.argmax(1)+1,0); rows.append({'leaf_weight':ws[0],'trunk_weight':ws[1],'twig_weight':ws[2],'contact_threshold':th,'macro_f1_4class':float(f1_score(sy,pred,average='macro',zero_division=0)),'binary_macro_f1':float(f1_score(sy>0,pred>0,average='macro',zero_division=0))})
    board=pd.DataFrame(rows).sort_values(['macro_f1_4class','binary_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_segment_image4_material_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_image4class_visual_material_audio_anchor_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','audio_source':'locked group-aware OOF audio','image_source':'4-class CLIP material probabilities from segment mean embedding','image_role':'material only; ambient column discarded after training','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
