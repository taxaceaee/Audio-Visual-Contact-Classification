from __future__ import annotations
import json, os
from pathlib import Path
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
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path(os.environ.get('SEG_OUT','outputs/audio_feature_benchmarks/segment_robust_material_group_selection')); IMG=Path(os.environ.get('SEG_IMG_ROOT','outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k'))
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def spec(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def main():
    OUT.mkdir(parents=True,exist_ok=True); audio_base.configure_feature_set('total240'); fr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); y=fr.y.to_numpy(np.int64); files=fr.audio_file.astype(str); x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); assert np.array_equal(y,np.load(IMG/'hand_train_full/y.npy').astype(np.int64)); sk=seg(files).to_numpy(); sg=spec(files).to_numpy(); useg,codes=np.unique(sk,return_inverse=True); sy=np.array([y[codes==i][0] for i in range(len(useg))]); ss=np.array([sg[codes==i][0] for i in range(len(useg))]); sx=np.vstack([x[codes==i].mean(0) for i in range(len(useg))]); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(sy)),sy,ss)); fold_id=np.full(len(sy),-1); [fold_id.__setitem__(va,k) for k,(_,va) in enumerate(folds)]
    af=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); high=group_audio.load_highsr_oof(Path('outputs')); pair=specimen.load_pairwise_oof(Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')); anchor=lift.anchor_lift_proba(af,high,pair); row_groups=spec(files).to_numpy(); assignment=np.full(len(y),-1,np.int64); [assignment.__setitem__(va,k) for k,(_,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,row_groups))]; src=broad.load_train_sources(Path('outputs'),y,assignment,42); audio=lift.postprocess(af,lift.normalize(.95*anchor+.05*src['report_gate_onehot']),'segment_lift'); pc,lg=avr.avr_inputs(audio); spc=np.array([pc[codes==i].mean() for i in range(len(useg))]); sa=np.vstack([suite.normalize(np.exp(lg[codes==i].mean(0,keepdims=True)))[0] for i in range(len(useg))])
    sio=np.zeros((len(sy),3),np.float32)
    for k,(tr,va) in enumerate(folds):
        c=tr[sy[tr]>0]; m=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=1000,class_weight='balanced',random_state=42)); m.fit(sx[c],sy[c]-1); sio[va]=m.predict_proba(sx[va]); print(f'fold {k+1}/5 done',flush=True)
    rows=[]
    for w in np.linspace(0,0.6,13):
      mat=suite.normalize(sa*(1-w)+sio*w)
      for th in np.linspace(.35,.70,15):
        pred=np.where(spc>=th,mat.argmax(1)+1,0); fs=[f1_score(sy[fold_id==k],pred[fold_id==k],average='macro',zero_division=0) for k in range(5)]; mean=float(np.mean(fs)); worst=float(np.min(fs)); rows.append({'image_weight':float(w),'contact_threshold':float(th),'mean_fold_macro_f1':mean,'worst_fold_macro_f1':worst,'std_fold_macro_f1':float(np.std(fs)),'robust_score':float(.6*worst+.4*mean),'full_oof_macro_f1':float(f1_score(sy,pred,average='macro',zero_division=0))})
    board=pd.DataFrame(rows).sort_values(['robust_score','mean_fold_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_robust_material_leaderboard.csv',index=False); best=board.iloc[0].to_dict(); lock={'protocol':'segment_robust_global_audio_visual_material_group_OOF','selection_data':'hand/default only','group_column':'specimen_group','audio_source':'locked group-aware OOF audio','image_source':'CLIP mean embedding per segment','image_role':'material only; global conservative blend','selection_rule':'max 0.6*worst_fold_macro_f1 + 0.4*mean_fold_macro_f1','test_loaded':False,'selected_candidate':best}; (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float)); print(json.dumps(lock,indent=2,default=float))
if __name__=='__main__': main()
