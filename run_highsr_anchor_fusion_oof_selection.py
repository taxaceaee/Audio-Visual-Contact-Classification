"""Hand-only OOF screen for a locked high-SR anchor as a third fusion source."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import train_audio_specimen_contact_consensus_select_final_test as spec
import train_audio_specimen_contact_lift_select_final_test as lift

ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures')
BASE=Path('outputs/audio_feature_benchmarks/segment_stack_meta_hier_group_selection')
HS=Path('outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/oof_proba/highsr_hgb_default__all_aug')
PAIR=Path('outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/oof_sources/pairwise_selected_clean_oof_proba.npy')
OUT=Path('outputs/audio_feature_benchmarks/highsr_anchor_fusion_group_selection')

def norm(x):
 x=np.clip(np.asarray(x,dtype=float),1e-12,None);return x/x.sum(1,keepdims=True)
def macro(y,p):
 vals=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(vals))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 fr=pd.read_csv(ROOT/'audio_visual_dataset_default/dataset.csv'); yrow=pd.Categorical(fr.category,categories=['ambient','leaf','trunk','twig']).codes
 key=fr.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True); group=key.str.replace(r'_segment_.*$','',regex=True)
 u,c=np.unique(key.to_numpy(),return_inverse=True); y=np.array([yrow[c==i][0] for i in range(len(u))]); groups=np.array([group.to_numpy()[c==i][0] for i in range(len(u))])
 pm=np.load(BASE/'hand_oof_meta.npy'); ph=np.load(BASE/'hand_oof_hier.npy');
 clean=np.load(HS/'clean_oof_proba.npy'); mix=np.load(HS/'robot_mix_oof_proba.npy'); pair=np.load(PAIR)
 high=norm(np.exp(.6*np.log(norm(clean))+.4*np.log(norm(mix))+np.array([.2,0,0,0])))
 window=norm(.8*high+.2*norm(pair))
 # Recreate the locked anchor lift on hand only.
 frame=pd.DataFrame({'audio_file':fr.audio_file.astype(str),'group_key':key.astype(str)})
 segp,w2s=spec.segment_proba_from_window(frame,window); sc=spec.specimen_codes_for_segments(frame,w2s)
 segp=lift.consensus_and_lift(segp,sc,consensus_threshold=.45,min_contact_segments=1,lift_min_mass=.35,lift_floor=.58,lift_confidence=.45)
 # factorize order differs from np.unique; remap to sorted segment order.
 sf=frame.groupby('group_key',sort=False).first().reset_index(); row_order={k:i for i,k in enumerate(sf.group_key.astype(str))}
 anchor=np.vstack([segp[row_order[str(k)]] for k in u])
 folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(y)),y,groups))
 rows=[]
 for wa in np.linspace(0,1,21):
  for wm in np.linspace(0,1,21):
   q=norm(wa*anchor+(1-wa)*(wm*pm+(1-wm)*ph))
   p=np.where(ph.argmax(1)==2,2,q.argmax(1)); ss=[macro(y[va],p[va]) for _,va in folds]
   rows.append({'anchor_weight':float(wa),'meta_weight':float(wm),'mean_cv_macro_f1':float(np.mean(ss)),'worst_cv_macro_f1':float(np.min(ss))})
 board=pd.DataFrame(rows).sort_values(['worst_cv_macro_f1','mean_cv_macro_f1'],ascending=False).reset_index(drop=True); board.to_csv(OUT/'hand_oof_leaderboard.csv',index=False)
 best=board.iloc[0].to_dict(); np.save(OUT/'hand_oof_anchor.npy',anchor)
 lock={'protocol':'highsr_anchor_third_source_grouped_hand_oof','selection_data':'hand/default only','group_column':'specimen_group','candidate':best,'test_loaded':False}
 (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2,default=float));print(json.dumps(lock,indent=2,default=float));print(board.head(12).to_string(index=False))
if __name__=='__main__':main()
