import json,numpy as np,pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_specimen_contact_lift_select_final_test as sl
import train_audio_specimen_contact_consensus_select_final_test as sp
import train_audio_group_consistency_pair_blend_select_final_test as ga
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/specimen_soft_material_group_selection'); IMG=Path('outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k')
def seg(s): return s.astype(str).str.replace(r'_window_\d+.*$','',regex=True)
def specimen(s): return seg(s).str.replace(r'_segment_.*$','',regex=True)
def audio_final():
 hi=pd.read_csv('outputs/audio_feature_benchmarks/audio_highsr_temporal_tta_select/reports/audio_highsr_temporal_tta_select_final_test_predictions.csv'); pa=pd.read_csv('outputs/audio_feature_benchmarks/audio_pairwise_contact_stress_cv_select/reports/audio_pairwise_contact_stress_cv_select_final_test_predictions.csv'); assert np.array_equal(hi.audio_file.astype(str),pa.audio_file.astype(str)); raw=sl.normalize(.8*hi[sl.PROBA_COLUMNS].to_numpy()+.2*pa[sl.PROBA_COLUMNS].to_numpy()); segp,w2s=sp.segment_proba_from_window(hi,raw); codes=sp.specimen_codes_for_segments(hi,w2s); l=json.load(open('outputs/audio_feature_benchmarks/audio_specimen_contact_lift_select/reports/audio_specimen_contact_lift_select_selected_without_test.json'))['selected_without_test']; segp=sl.consensus_and_lift(segp,codes,float(l['consensus_threshold']),int(l['min_contact_segments']),l['lift_min_mass'],float(l['lift_floor']),float(l['lift_confidence'])); return segp[w2s]
def fit_soft(x,target,c):
 keep=target.sum(1)>0; xx=np.repeat(x[keep],3,axis=0); yy=np.tile(np.arange(3),keep.sum()); ww=target[keep].reshape(-1); m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,random_state=42)); m.fit(xx,yy,logisticregression__sample_weight=ww); return m
def main():
 lock=json.load(open(OUT/'selection_lock.json')); b=lock['selected_candidate']; C=float(b['C']); iw=float(b['image_material_weight']); th=float(b['contact_threshold']); audio_base.configure_feature_set('total240')
 h=audio_base.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv','hand_train'); hf=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train'); yh=hf.y.to_numpy(np.int64); hk=seg(hf.audio_file).to_numpy(); su,sc=np.unique(hk,return_inverse=True); hsp=specimen(pd.Series(su)).to_numpy(); names,si=np.unique(hsp,return_inverse=True); x=np.load(IMG/'hand_train_full/X.npy').astype(np.float32); sx=np.vstack([x[sc==i].mean(0) for i in range(len(su))]); specx=np.vstack([sx[si==i].mean(0) for i in range(len(names))]); sy=np.array([yh[sc==i][0] for i in range(len(su))]); target=np.zeros((len(names),3),float)
 for i in range(len(su)):
  v=sy[si==i]; v=v[v>0];
  if len(v): target[i]=np.bincount(v-1,minlength=3)/len(v)
 model=fit_soft(specx,target,C)
 test=audio_base.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv','robot_test'); tf=suite.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv',ROOT/'audio_visual_dataset_robo_default','robot_test'); y=tf.y.to_numpy(np.int64); tk=seg(tf.audio_file).to_numpy(); tu,tc=np.unique(tk,return_inverse=True); tsp=specimen(pd.Series(tu)).to_numpy(); tnames,tsi=np.unique(tsp,return_inverse=True); xt=np.load(IMG/'robot_test/X.npy').astype(np.float32); tx=np.vstack([xt[tc==i].mean(0) for i in range(len(tu))]); tsx=np.vstack([tx[tsi==i].mean(0) for i in range(len(tnames))]); im=model.predict_proba(tsx)[tsi]; ap=audio_final(); pc,lg=avr.avr_inputs(ap); tpc=np.array([pc[tc==i].mean() for i in range(len(tu))]); tmat=np.vstack([suite.normalize(np.exp(lg[tc==i].mean(0,keepdims=True)))[0] for i in range(len(tu))]); mat=suite.normalize((1-iw)*tmat+iw*im); predseg=np.where(tpc>=th,mat.argmax(1)+1,0); pred=predseg[tc]; by=(y>0).astype(int); bp=(pred>0).astype(int); result={'split':'robot_test_final','protocol':lock['protocol'],'locked_candidate':b,'n':len(y),'accuracy_4class':accuracy_score(y,pred),'macro_precision_4class':precision_score(y,pred,average='macro',zero_division=0),'macro_recall_4class':recall_score(y,pred,average='macro',zero_division=0),'macro_f1_4class':f1_score(y,pred,average='macro',zero_division=0),'weighted_f1_4class':f1_score(y,pred,average='weighted',zero_division=0),'binary_macro_f1':f1_score(by,bp,average='macro',zero_division=0),'per_class_4class':classification_report(y,pred,labels=[0,1,2,3],target_names=['ambient','leaf','trunk','twig'],output_dict=True,zero_division=0),'confusion_matrix_4class':confusion_matrix(y,pred,labels=[0,1,2,3]).tolist(),'no_test_leakage':True,'soft_label_target_used':True}; (OUT/'specimen_soft_material_final_test_metrics.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))
if __name__=='__main__': main()
