from pathlib import Path
import numpy as np, pandas as pd, torch
import run_domain_augmented_resnet_group_selection as impl
import run_multimodal_val_locked_suite as suite
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/segment_rule_stack_resnet_fusion_group_selection'); REPORT=Path('outputs/audio_feature_benchmarks/domain_augmented_resnet_group_selection')
def main():
    lock=__import__('json').load(open(REPORT/'selection_lock.json')); assert not lock.get('test_loaded'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=impl.make_model(device); model.load_state_dict(torch.load(REPORT/'selected_group_val_model.pt',map_location=device)['model_state']); test=suite.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv',ROOT/'audio_visual_dataset_robo_default','robot_test'); p=impl.predict(model,test,device); keys=test.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True).to_numpy(); u,c=np.unique(keys,return_inverse=True); segp=np.vstack([p[c==i].mean(0) for i in range(len(u))]); np.save(OUT/'robot_resnet_segment_proba.npy',suite.normalize(segp)); np.save(OUT/'robot_resnet_segment_ids.npy',u); print('saved',segp.shape)
if __name__=='__main__': main()
