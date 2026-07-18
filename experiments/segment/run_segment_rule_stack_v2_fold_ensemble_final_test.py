"""Final-only evaluator: locked v2 heads averaged over specimen-disjoint hand folds."""
from pathlib import Path
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import run_segment_stack_meta_hier_group_selection as sel
import train_val_select_final_test as audio_base
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures');OUT=Path('outputs/audio_feature_benchmarks/segment_rule_stack_v2_fold_ensemble_group_selection');BASE=Path('outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection')
def pool(files,x,y=None):
 k=sel.seg(files).to_numpy();u,c=np.unique(k,return_inverse=True);px=np.vstack([x[c==i].mean(0) for i in range(len(u))]);py=None if y is None else np.array([y[c==i][0] for i in range(len(u))]);return u,c,px,py
def main():
 print('start',flush=True);OUT.mkdir(parents=True,exist_ok=True);lock=json.loads((BASE/'selection_lock.json').read_text());assert not lock.get('test_loaded');b=lock['selected_candidate'];alpha=float(b['alpha']);tb=float(b['tb']);th=float(b['th']);audio_base.configure_feature_set('total240')
 hfr=suite.load_manifest(ROOT/'audio_visual_dataset_default/dataset.csv',ROOT/'audio_visual_dataset_default','hand_train');yh=hfr.y.to_numpy(np.int64);xh_c=np.load(sel.IMG_CLIP/'hand_train_full/X.npy').astype(np.float32);xh_e=np.load(sel.IMG_EFF/'hand_train_full/X.npy').astype(np.float32);hu,hc,hx_c,hy=pool(hfr.audio_file,xh_c,yh);_,_,hx_e,_=pool(hfr.audio_file,xh_e,yh);hx_cat=np.hstack([hx_c,hx_e]);groups=sel.spec(pd.Series(hu)).to_numpy();folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(hy)),hy,groups));print('hand',len(hy),flush=True)
 def m4(): return make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500,class_weight='balanced',multi_class='multinomial',random_state=42))
 def mb(): return make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500,class_weight='balanced',random_state=42))
 def mm(): return make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500,class_weight='balanced',multi_class='multinomial',random_state=42))
 def mt(): return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1200,class_weight='balanced',random_state=42))
 tfr=suite.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv',ROOT/'audio_visual_dataset_robo_default','robot_test');yt=tfr.y.to_numpy(np.int64);xt_c=np.load(sel.IMG_CLIP/'robot_test/X.npy').astype(np.float32);xt_e=np.load(sel.IMG_EFF/'robot_test/X.npy').astype(np.float32);tu,tc,tx_c,_=pool(tfr.audio_file,xt_c,yt);_,_,tx_e,_=pool(tfr.audio_file,xt_e,yt);tx_cat=np.hstack([tx_c,tx_e])
 p4=np.zeros((len(tu),4));pb=np.zeros(len(tu));pmat=np.zeros((len(tu),3));pt=np.zeros(len(tu)); print('test features',len(tu),flush=True)
 for tr,va in folds:
  a=m4().fit(hx_c[tr],hy[tr]);bb=mb().fit(hx_c[tr],(hy[tr]>0).astype(int));mmod=mm().fit(hx_cat[tr][hy[tr]>0],hy[tr][hy[tr]>0]-1);tt=mt().fit(hx_c[tr],(hy[tr]==2).astype(int));p4+=suite.normalize(a.predict_proba(tx_c));pb+=bb.predict_proba(tx_c)[:,1];pmat+=mmod.predict_proba(tx_cat);pt+=tt.predict_proba(tx_c)[:,1]
 p4/=len(folds);pb/=len(folds);pmat=suite.normalize(pmat/len(folds));pt/=len(folds)
 audio=pd.read_csv('outputs/audio_feature_benchmarks/audio_lift_source_blend_select/reports/audio_lift_source_blend_select_final_test_predictions.csv');raw=pd.read_csv(ROOT/'audio_visual_dataset_robo_default/dataset.csv');assert np.array_equal(audio.audio_file.astype(str),raw.audio_file.astype(str));proba=suite.normalize(audio[suite.PROBA_COLUMNS].to_numpy(float));pc,lg=avr.avr_inputs(proba);sk=sel.seg(tfr.audio_file).to_numpy();sa_mat=np.vstack([suite.normalize(np.exp(lg[sk==k].mean(0,keepdims=True)))[0] for k in tu]);spc=np.array([pc[sk==k].mean() for k in tu]);img_mat3=suite.normalize(p4[:,1:]);zmeta=sel.feat_meta(sa_mat,img_mat3,spc);hm_x=np.load('outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_features.npy');hm_y=np.load('outputs/audio_feature_meta_weighted_group_selection/hand_meta_oof_y.npy');cw={0:1.,1:1.,2:1.2,3:1.2};meta=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=4000,class_weight=cw,multi_class='multinomial',random_state=42)).fit(hm_x,hm_y);pmeta=suite.normalize(meta.predict_proba(zmeta));contact=alpha*spc+(1-alpha)*pb;mat=suite.normalize(sa_mat*(1-np.array([.4,.55,.4]))+pmat*np.array([.4,.55,.4]));logits=np.log(np.clip(mat,1e-8,1));logits[:,1]+=tb*pt;mat=suite.normalize(np.exp(logits));predh=np.where(contact>=th,mat.argmax(1)+1,0);predm=pmeta.argmax(1);predseg=np.where(predh==2,predh,predm);order={k:i for i,k in enumerate(tu)};pred=np.array([predseg[order[k]] for k in sk]);by=yt>0;bp=pred>0;res={'split':'robot_test_final','protocol':'v2_locked_specimen_fold_ensemble','locked_candidate':b,'n':len(yt),'metrics':{'accuracy_4class':float(accuracy_score(yt,pred)),'macro_precision_4class':float(precision_score(yt,pred,average='macro',zero_division=0)),'macro_recall_4class':float(recall_score(yt,pred,average='macro',zero_division=0)),'macro_f1_4class':float(f1_score(yt,pred,average='macro',zero_division=0)),'weighted_f1_4class':float(f1_score(yt,pred,average='weighted',zero_division=0)),'binary_macro_f1':float(f1_score(by,bp,average='macro',zero_division=0))},'per_class_4class':classification_report(yt,pred,labels=[0,1,2,3],target_names=['ambient','leaf','trunk','twig'],output_dict=True,zero_division=0),'confusion_matrix_4class':confusion_matrix(yt,pred,labels=[0,1,2,3]).tolist(),'invariants':{'selection_used_hand_only':True,'test_loaded_after_lock':True,'fold_ensemble':True}}
 (OUT/'v2_fold_ensemble_final_test_metrics.json').write_text(json.dumps(res,indent=2,default=float));print(json.dumps(res,indent=2,default=float))
if __name__=='__main__':main()
