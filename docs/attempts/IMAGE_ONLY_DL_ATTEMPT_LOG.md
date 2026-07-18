# Image-only Macro F1 Attempts

Date: 2026-07-06

## Protocol

- Input modality: images only.
- Hand/default dataset split into train/validation by specimen group.
- Robot dataset kept as fixed final test.
- Each script writes `selected_without_test.json` before loading robot/test.
- No audio features, no multimodal features, no filename/text features as model input.

## Best Result So Far

Best robot/test Macro F1 is `0.450474`.

| Run | Selected model | Val Macro F1 | Robot Macro F1 | Robot Acc | Report |
|---|---:|---:|---:|---:|---|
| `image_dl_resnet18_tta_bias_specimen_select` | ResNet18 checkpoint + validation-tuned class bias | 0.576328 | 0.450474 | 0.488058 | `outputs/audio_feature_benchmarks/image_dl_resnet18_tta_bias_specimen_select/reports/image_dl_resnet18_tta_bias_specimen_select_final_test_report.csv` |
| `image_dl_checkpoint_ensemble_select` | Same ResNet18 member selected by validation ensemble search | 0.576328 | 0.450474 | 0.488058 | `outputs/audio_feature_benchmarks/image_dl_checkpoint_ensemble_select/reports/image_dl_checkpoint_ensemble_select_final_test_report.csv` |
| `image_deep_transfer_resnet50_vit_specimen_select` | ViT-B16 embeddings + ExtraTrees + validation bias | 0.583618 | 0.421930 | 0.496620 | `outputs/audio_feature_benchmarks/image_deep_transfer_resnet50_vit_specimen_select/reports/image_deep_transfer_resnet50_vit_specimen_select_final_test_report.csv` |
| `image_dl_finetune_resnet18_specimen_select` | ResNet18 fine-tune | 0.500096 | 0.421853 | 0.407841 | `outputs/audio_feature_benchmarks/image_dl_finetune_resnet18_specimen_select/reports/image_dl_finetune_resnet18_specimen_select_final_test_report.csv` |
| `image_dl_resnet34_tta_bias_specimen_select` | ResNet34 checkpoint + validation bias | 0.529064 | 0.421309 | 0.401082 | `outputs/audio_feature_benchmarks/image_dl_resnet34_tta_bias_specimen_select/reports/image_dl_resnet34_tta_bias_specimen_select_final_test_report.csv` |
| `image_dl_finetune_mobilenetv3_specimen_select` | MobileNetV3-Large fine-tune | 0.438087 | 0.399431 | 0.383957 | `outputs/audio_feature_benchmarks/image_dl_finetune_mobilenetv3_specimen_select/reports/image_dl_finetune_mobilenetv3_specimen_select_final_test_report.csv` |
| `image_dl_finetune_resnet34_specimen_select` | ResNet34 fine-tune | 0.453996 | 0.386365 | 0.383055 | `outputs/audio_feature_benchmarks/image_dl_finetune_resnet34_specimen_select/reports/image_dl_finetune_resnet34_specimen_select_final_test_report.csv` |
| `image_timm_transfer_convnext_eff_swin_specimen_select` | EfficientNet-B3 timm embedding + RandomForest + validation bias | 0.556770 | 0.319875 | 0.368635 | `outputs/audio_feature_benchmarks/image_timm_transfer_convnext_eff_swin_specimen_select/reports/image_timm_transfer_convnext_eff_swin_specimen_select_final_test_report.csv` |
| `image_timm_finetune_convnext_tiny_specimen_select` | ConvNeXt-Tiny timm fine-tune checkpoint selected by validation | 0.484214 | 0.301334 | 0.288869 | `outputs/audio_feature_benchmarks/image_timm_finetune_convnext_tiny_specimen_select/reports/image_timm_finetune_convnext_tiny_specimen_select_final_test_report.csv` |
| `image_dl_group_blend_select` | ResNet18 + group-majority KNN blend | 0.634885 | 0.346171 | 0.471834 | `outputs/audio_feature_benchmarks/image_dl_group_blend_select/reports/image_dl_group_blend_select_final_test_report.csv` |
| `image_group_majority_deep_select` | Unique-image group-majority KNN | 0.555292 | 0.305211 | 0.469581 | `outputs/audio_feature_benchmarks/image_group_majority_deep_select/reports/image_group_majority_deep_select_final_test_report.csv` |
| `image_timm_dedup_group_specimen_select` | Exact-image dedup + DINOv2/CLIP embeddings + train-only label policy selection | 0.598263 | 0.326138 | 0.432177 | `outputs/audio_feature_benchmarks/image_timm_dedup_group_specimen_select/reports/image_timm_dedup_group_specimen_select_final_test_report.csv` |

## Feasibility Notes

Exact image duplication creates a hard image-only ambiguity:

- Hand/default: `10676` rows but only `235` unique exact images. `202` exact-image groups have more than one label, covering `10132` rows.
- Robot/test: `2219` rows but only `45` unique exact images. `43` exact-image groups have more than one label, covering `2204` rows.
- Robot/test one-label-per-exact-image upper bound is not zero: exact-hash majority oracle reaches about `0.653288` macro F1 and macro-F1-optimized exact-hash assignment reaches about `0.658364`. This means the target is information-theoretically possible only if the method can infer the right per-image majority assignment without using robot/test labels.

This means the same pixels often appear with both ambient and contact labels. A static image-only model cannot infer the audio contact state for those rows. The best validation models often overfit hand/default label ratios and fail under the robot domain shift.

The 2026-07-08 exact-image dedup run confirms this: the selected DINOv2+CLIP model reached `0.598263` hand validation macro F1 before the robot/test lock, but transferred to only `0.326138` robot/test macro F1. The failure mode is severe robot-domain label-prior shift, especially leaf recall (`0.075085`) and trunk recall (`0.197397`).

## Current Conclusion

The target `robot/test Macro F1 > 0.5` or `> 0.6` has not been reached under the clean image-only protocol. The strongest clean result is `0.450474`. Reaching `>0.5` appears unlikely without changing the problem information, for example adding a non-image signal, using robot labels for calibration, or allowing filename/protocol metadata, all of which would violate the current constraints.
