# Bảng so sánh thống nhất (đầy đủ metrics)

**Robot/test n=2219** · mọi số lấy từ artifacts (nested prec/rec suy ra từ confusion matrix khi report stage chưa lưu).

CSV: `outputs/audio_feature_benchmarks/macro_f1_nested_contribution_breakdown/reports/unified_macro_f1_comparison_table_full.csv`

## Bảng chính — Acc / Macro-P / Macro-R / Macro-F1 / Contact / Δ

| # | Family | Stage | Dim | Views | Model | Acc | Macro-P | Macro-R | Macro-F1 | Contact F1 | Binary F1 | ΔF1 prev | ΔF1 base | ΔAcc prev | Valid | Rank |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | ❌ rename | `anchor_blend` | 1440 | sr16_clean,sr16_robot_mix,sr16_bandlimit,high… | HistGB | 0.7102 | 0.5777 | 0.5573 | **0.5583** | 0.4276 | 0.9465 | — | 0 | — | no | 10 |
| 2 | ❌ rename | `report_gate` | 1440 | sr16_clean,sr16_robot_mix,sr16_bandlimit,high… | HistGB | 0.7102 | 0.5777 | 0.5573 | **0.5583** | 0.4276 | 0.9465 | 0 | 0 | 0 | no | 11 |
| 3 | ❌ rename | `final_source_blend` | 1440 | sr16_clean,sr16_robot_mix,sr16_bandlimit,high… | HistGB | 0.7102 | 0.5777 | 0.5573 | **0.5583** | 0.4276 | 0.9465 | 0 | 0 | 0 | no | 12 |
| 4 | ❌ rename | `segment_aggregation` | 1440 | sr16_clean,sr16_robot_mix,sr16_bandlimit,high… | HistGB | 0.7102 | 0.5777 | 0.5573 | **0.5583** | 0.4276 | 0.9465 | 0 | 0 | 0 | no | 13 |
| 5 | ❌ rename | `segment_consensus` | 1440 | sr16_clean,sr16_robot_mix,sr16_bandlimit,high… | HistGB | 0.7102 | 0.5777 | 0.5573 | **0.5583** | 0.4276 | 0.9465 | 0 | 0 | 0 | no | 14 |
| 6 | ❌ rename | `contact_mass` | 1440 | sr16_clean,sr16_robot_mix,sr16_bandlimit,high… | HistGB | 0.7102 | 0.5777 | 0.5573 | **0.5583** | 0.4276 | 0.9465 | 0 | 0 | 0 | no | 15 |
| 7 | ✅ nested | `highsr_3view_720` | 720 | highsr_clean,highsr_robot_mix,highsr_bandlimit | HistGradientBoosting | 0.7044 | 0.5603 | 0.5443 | **0.5434** | 0.4083 | 0.9443 | — | 0 | — | yes | 17 |
| 8 | ✅ nested | `sr16_3view_720` | 720 | sr16_clean,sr16_robot_mix,sr16_bandlimit | HistGradientBoosting | 0.7098 | 0.5836 | 0.5600 | **0.5636** | 0.4422 | 0.9199 | +0.0202 | +0.0202 | +0.0054 | yes | 9 |
| 9 | ✅ nested | `sixview_1440` | 1440 | sr16_clean,sr16_robot_mix,sr16_bandlimit,high… | HistGradientBoosting | 0.7102 | 0.5777 | 0.5573 | **0.5583** | 0.4276 | 0.9465 | -0.0053 | +0.0150 | +0.0005 | yes | 16 |
| 10 | ✅ nested | `multimodal_1952` | 1952 | sixview_audio + resnet18_224 | ExtraTrees | 0.7972 | 0.7630 | 0.7057 | **0.7053** | 0.6280 | 0.9310 | +0.1470 | +0.1619 | +0.0870 | yes | 1 |
| 11 | ✅ proba | `01_highsr_raw` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7116 | 0.5995 | 0.5705 | **0.5731** | 0.4598 | 0.9018 | — | 0 | — | yes | 8 |
| 12 | ✅ proba | `02_pairwise_raw` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7003 | 0.5955 | 0.5711 | **0.5377** | 0.4026 | 0.9378 | -0.0354 | -0.0354 | -0.0113 | yes | 18 |
| 13 | ✅ proba | `03_highsr80_pairwise20_raw` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7147 | 0.6065 | 0.5786 | **0.5764** | 0.4627 | 0.9069 | +0.0387 | +0.0033 | +0.0144 | yes | 7 |
| 14 | ✅ proba | `04_segment_pooling` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7553 | 0.6741 | 0.6466 | **0.6393** | 0.5409 | 0.9277 | +0.0629 | +0.0662 | +0.0406 | yes | 6 |
| 15 | ✅ proba | `05_specimen_consensus_no_lift` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7891 | 0.7608 | 0.7083 | **0.6907** | 0.6098 | 0.9264 | +0.0514 | +0.1175 | +0.0338 | yes | 5 |
| 16 | ✅ proba | `06_anchor_consensus_lift` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7918 | 0.7630 | 0.7134 | **0.6938** | 0.6133 | 0.9291 | +0.0032 | +0.1207 | +0.0027 | yes | 4 |
| 17 | ✅ proba | `07_anchor_plus_report_gate_raw` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7950 | 0.7658 | 0.7187 | **0.7000** | 0.6214 | 0.9291 | +0.0061 | +0.1268 | +0.0032 | yes | 3 |
| 18 | ✅ proba | `08_final_anchor_source_segment_lift` | — | highsr_source ± pairwise_source (probability … | proba_stack | 0.7968 | 0.7708 | 0.7217 | **0.7027** | 0.6250 | 0.9291 | +0.0027 | +0.1295 | +0.0018 | yes | 2 |

## Per-class F1 (ambient / leaf / trunk / twig)

| # | Stage | Amb | Leaf | Trunk | Twig | Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `anchor_blend` | 0.9505 | 0.3916 | 0.4575 | 0.4338 | 0.5583 |
| 2 | `report_gate` | 0.9505 | 0.3916 | 0.4575 | 0.4338 | 0.5583 |
| 3 | `final_source_blend` | 0.9505 | 0.3916 | 0.4575 | 0.4338 | 0.5583 |
| 4 | `segment_aggregation` | 0.9505 | 0.3916 | 0.4575 | 0.4338 | 0.5583 |
| 5 | `segment_consensus` | 0.9505 | 0.3916 | 0.4575 | 0.4338 | 0.5583 |
| 6 | `contact_mass` | 0.9505 | 0.3916 | 0.4575 | 0.4338 | 0.5583 |
| 7 | `highsr_3view_720` | 0.9485 | 0.3187 | 0.4615 | 0.4447 | 0.5434 |
| 8 | `sr16_3view_720` | 0.9279 | 0.4255 | 0.4433 | 0.4578 | 0.5636 |
| 9 | `sixview_1440` | 0.9505 | 0.3916 | 0.4575 | 0.4338 | 0.5583 |
| 10 | `multimodal_1952` | 0.9371 | 0.6038 | 0.6036 | 0.6767 | 0.7053 |
| 11 | `01_highsr_raw` | 0.9133 | 0.5112 | 0.4090 | 0.4590 | 0.5731 |
| 12 | `02_pairwise_raw` | 0.9429 | 0.4778 | 0.2509 | 0.4793 | 0.5377 |
| 13 | `03_highsr80_pairwise20_raw` | 0.9173 | 0.5139 | 0.3994 | 0.4749 | 0.5764 |
| 14 | `04_segment_pooling` | 0.9344 | 0.6545 | 0.4808 | 0.4874 | 0.6393 |
| 15 | `05_specimen_consensus_no_lift` | 0.9332 | 0.7538 | 0.5215 | 0.5542 | 0.6907 |
| 16 | `06_anchor_consensus_lift` | 0.9355 | 0.7641 | 0.5215 | 0.5542 | 0.6938 |
| 17 | `07_anchor_plus_report_gate_raw` | 0.9355 | 0.7803 | 0.5198 | 0.5643 | 0.7000 |
| 18 | `08_final_anchor_source_segment_lift` | 0.9355 | 0.7759 | 0.5215 | 0.5778 | 0.7027 |

## Δ đầy đủ (F1 / Acc / Macro-P / Macro-R) trong từng family

| # | Stage | Δ Macro-F1 | Δ Acc | Δ Macro-P | Δ Macro-R | ΔF1 vs family base |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `anchor_blend` | — | — | — | — | 0 |
| 2 | `report_gate` | 0 | 0 | 0 | 0 | 0 |
| 3 | `final_source_blend` | 0 | 0 | 0 | 0 | 0 |
| 4 | `segment_aggregation` | 0 | 0 | 0 | 0 | 0 |
| 5 | `segment_consensus` | 0 | 0 | 0 | 0 | 0 |
| 6 | `contact_mass` | 0 | 0 | 0 | 0 | 0 |
| 7 | `highsr_3view_720` | — | — | — | — | 0 |
| 8 | `sr16_3view_720` | +0.0202 | +0.0054 | +0.0233 | +0.0157 | +0.0202 |
| 9 | `sixview_1440` | -0.0053 | +0.0005 | -0.0059 | -0.0027 | +0.0150 |
| 10 | `multimodal_1952` | +0.1470 | +0.0870 | +0.1853 | +0.1484 | +0.1619 |
| 11 | `01_highsr_raw` | — | — | — | — | 0 |
| 12 | `02_pairwise_raw` | -0.0354 | -0.0113 | -0.0040 | +0.0005 | -0.0354 |
| 13 | `03_highsr80_pairwise20_raw` | +0.0387 | +0.0144 | +0.0109 | +0.0075 | +0.0033 |
| 14 | `04_segment_pooling` | +0.0629 | +0.0406 | +0.0676 | +0.0680 | +0.0662 |
| 15 | `05_specimen_consensus_no_lift` | +0.0514 | +0.0338 | +0.0867 | +0.0617 | +0.1175 |
| 16 | `06_anchor_consensus_lift` | +0.0032 | +0.0027 | +0.0023 | +0.0051 | +0.1207 |
| 17 | `07_anchor_plus_report_gate_raw` | +0.0061 | +0.0032 | +0.0027 | +0.0053 | +0.1268 |
| 18 | `08_final_anchor_source_segment_lift` | +0.0027 | +0.0018 | +0.0050 | +0.0030 | +0.1295 |

## Views chi tiết (không cắt)

| # | Stage | n_views | views |
|---:|---|---:|---|
| 1 | `anchor_blend` | 6 | `sr16_clean,sr16_robot_mix,sr16_bandlimit,highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 2 | `report_gate` | 6 | `sr16_clean,sr16_robot_mix,sr16_bandlimit,highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 3 | `final_source_blend` | 6 | `sr16_clean,sr16_robot_mix,sr16_bandlimit,highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 4 | `segment_aggregation` | 6 | `sr16_clean,sr16_robot_mix,sr16_bandlimit,highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 5 | `segment_consensus` | 6 | `sr16_clean,sr16_robot_mix,sr16_bandlimit,highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 6 | `contact_mass` | 6 | `sr16_clean,sr16_robot_mix,sr16_bandlimit,highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 7 | `highsr_3view_720` | 3 | `highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 8 | `sr16_3view_720` | 3 | `sr16_clean,sr16_robot_mix,sr16_bandlimit` |
| 9 | `sixview_1440` | 6 | `sr16_clean,sr16_robot_mix,sr16_bandlimit,highsr_clean,highsr_robot_mix,highsr_bandlimit` |
| 10 | `multimodal_1952` | 6+img | `sixview_audio + resnet18_224` |
| 11 | `01_highsr_raw` |  | `highsr_source ± pairwise_source (probability ops)` |
| 12 | `02_pairwise_raw` |  | `highsr_source ± pairwise_source (probability ops)` |
| 13 | `03_highsr80_pairwise20_raw` |  | `highsr_source ± pairwise_source (probability ops)` |
| 14 | `04_segment_pooling` |  | `highsr_source ± pairwise_source (probability ops)` |
| 15 | `05_specimen_consensus_no_lift` |  | `highsr_source ± pairwise_source (probability ops)` |
| 16 | `06_anchor_consensus_lift` |  | `highsr_source ± pairwise_source (probability ops)` |
| 17 | `07_anchor_plus_report_gate_raw` |  | `highsr_source ± pairwise_source (probability ops)` |
| 18 | `08_final_anchor_source_segment_lift` |  | `highsr_source ± pairwise_source (probability ops)` |

## Check field thiếu

| Field | Status |
|---|---|
| `macro_f1` | ✅ full |
| `macro_precision` | ✅ full |
| `macro_recall` | ✅ full |
| `accuracy_4class` | ✅ full |
| `views` | ✅ full |
| `delta_prev_macro_f1` | ✅ full |
| `contact_macro_f1` | ✅ full |
| `binary_macro_f1` | ✅ full |
| `ambient_f1` | ✅ full |
| `leaf_f1` | ✅ full |
| `trunk_f1` | ✅ full |
| `twig_f1` | ✅ full |

**Ghi chú:** nested #7–#10 `macro_precision` / `macro_recall` / per-class F1 được **tính lại từ confusion_matrix** trong `stage_*_report.json` (pipeline cũ chỉ dump macro_f1 + acc + contact + binary). Invalid #1–#6 và proba #11–#18 lấy trực tiếp từ report/CSV gốc.
