from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torch import nn
import run_domain_augmented_resnet_group_selection as impl
import run_multimodal_val_locked_suite as suite
ROOT=Path('/home/ttung05/Desktop/tree_base/tree_structures'); OUT=Path('outputs/audio_feature_benchmarks/segment_rule_stack_resnet_fusion_group_selection'); REPORT=Path('outputs/audio_feature_benchmarks/domain_augmented_resnet_group_selection')
def main():
    print('start', flush=True)
    lock=json.load(open(REPORT/'selection_lock.json')); assert not lock.get('test_loaded'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device,flush=True); full,_,_=suite.build_group_split(ROOT,REPORT); print('frame',len(full),flush=True); model=impl.make_model(device); print('model',flush=True); 
    for name, param in model.named_parameters(): param.requires_grad = name.startswith('fc')
    y=full.y.to_numpy(np.int64); counts=np.bincount(y,minlength=4); weights=torch.tensor(counts.sum()/np.maximum(counts,1),dtype=torch.float32,device=device); weights/=weights.mean(); loader=DataLoader(impl.ImageFrame(full,lock['selected_profile'],True),batch_size=32,shuffle=True,num_workers=0,pin_memory=False); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=2e-4); crit=nn.CrossEntropyLoss(weight=weights)
    for epoch in range(1,int(lock['selected_epoch'])+1):
        model.train()
        for xb,target in loader:
            opt.zero_grad(set_to_none=True); loss=crit(model(xb.to(device)),target.to(device)); loss.backward(); opt.step()
        print('epoch',epoch,flush=True)
    test=suite.load_manifest(ROOT/'audio_visual_dataset_robo_default/dataset.csv',ROOT/'audio_visual_dataset_robo_default','robot_test'); p=impl.predict(model,test,device); keys=test.audio_file.astype(str).str.replace(r'_window_\d+.*$','',regex=True).to_numpy(); u,c=np.unique(keys,return_inverse=True); segp=np.vstack([p[c==i].mean(0) for i in range(len(u))]); np.save(OUT/'robot_resnet_fullhand_segment_proba.npy',suite.normalize(segp)); np.save(OUT/'robot_resnet_fullhand_segment_ids.npy',u); print('saved',segp.shape)
if __name__=='__main__': main()
