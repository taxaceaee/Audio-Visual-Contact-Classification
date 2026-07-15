# Audio-Only 0.75 Macro-F1 Attempt Log

Date: 2026-07-06

Target: push `macro_f1_4class` on robot/test above `0.75` without using robot/test for training or validation, without image features, and without label leakage.

## Current Paper-Clean Anchor

- Checkpoint: `checkpoints/audio_only_paper_safe_current_0702672_20260706`
- Script: `train_audio_lift_source_blend_select_final_test.py`
- Final robot/test `macro_f1_4class`: `0.7026719927789447`
- Final robot/test `contact_macro_f1`: `0.625050260344378`

Confusion matrix:

```text
ambient -> ambient 1132
leaf    -> ambient 2, leaf 277, twig 14
trunk   -> ambient 126, leaf 38, trunk 164, twig 133
twig    -> ambient 28, leaf 106, trunk 4, twig 195
```

Main remaining errors are `trunk -> ambient/twig` and `twig -> leaf`.

## Clean Script Added

New script:

- `train_audio_anchor_hsrc_guard_select_final_test.py`

Protocol:

- Uses the paper-clean anchor OOF probabilities.
- Adds the locked `audio_highsr_contact_source_consensus_select` source.
- Selects convex blend or guarded trunk rescue using hand/default OOF only.
- Writes `selected_without_test.json` before loading robot/test predictions.
- Does not use image features.
- Does not parse class/contact words from filenames.

Result:

- Selected candidate: `anchor_only`
- OOF macro F1: `0.9353447727801658`
- Final robot/test `macro_f1_4class`: `0.7026719927789447`

Artifacts:

- `outputs/audio_feature_benchmarks/audio_anchor_hsrc_guard_select/reports/audio_anchor_hsrc_guard_select_selected_without_test.json`
- `outputs/audio_feature_benchmarks/audio_anchor_hsrc_guard_select/reports/audio_anchor_hsrc_guard_select_final_test_report.csv`

Conclusion: under clean OOF selection, the guarded high-SR contact source is rejected because it lowers validation performance.

## Probes Tried

These probes were exploratory and are not paper-clean selection artifacts by themselves.

### High-SR Contact Trunk Rescue

Best simple trunk rescue using current anchor plus `audio_highsr_contact_source_consensus_select` reached approximately:

- robot/test macro F1: `0.7184733989253463`

This was still below `0.75`.

### Binary Contact Rescue

- robot/test macro F1: `0.7154780924381106`

### Combined total240 + high-SR Contact Classifier

Supervised contact-subclass probes using `total240 + highSR1966` features did not improve the anchor. Logistic variants over-predicted trunk and damaged leaf performance; ExtraTrees only matched or slightly degraded the anchor at small blend weights.

### MiniRocket Probe

Installed `aeon` and probed MiniRocket on 4k downsampled raw waveform.

- MiniRocket 1000 kernels + RidgeClassifierCV robot/test macro F1: `0.4753374654951879`

Conclusion: raw random-convolution features did not solve the domain shift.

## Trunk Rescue + Heavy Augmentation + Domain-Holdout Worst-View Selection (2026-07-09)

New script:

- `train_audio_trunk_rescue_domain_robust_select_final_test.py`
- Run slug: `audio_trunk_rescue_domain_robust_select`

Combines three clean directions:

1. Heavy robot-like train-time augmentation views: `robot_heavy` (synthetic IR reverb + strong bandlimit + motor-hum noise + tanh drive + roll) and `pitch_speed` (speed/pitch perturb + bandlimit + noise). Cached under the run dir.
2. Selection on the hand/default domain-holdout split using WORST macro-F1 across stress views (`clean`, `robot_mix`, `bandlimit`, `robot_heavy`) with a leaf-F1 guard, instead of mean OOF.
3. Trunk-vs-rest binary detector (HGB/ExtraTrees over up to 5 augmentation views) applied as a guarded rescue on the locked 0.7027 anchor: predicted ambient/twig windows flip to trunk when segment-mean trunk probability crosses a locked threshold.

Result:

- Baseline holdout worst-view macro F1: `0.9173`.
- All `ambient` rescue rules were no-ops on the holdout (zero flips fired), so they only tied the baseline.
- All `twig` and `ambient_twig` rescue rules REDUCED holdout worst-view macro F1.
- Clean selection therefore locked `anchor_baseline`; final robot/test macro F1 unchanged: `0.702672`, `n_rescued = 0`.

Diagnosis: the trunk->ambient failure mode does not exist on any hand/default holdout, even under heavy synthetic robot-like stress. The anchor simply never mispredicts trunk windows as ambient on hand data, so no clean hand-only validation can certify a rescue threshold. The synthetic views shift features but do not reproduce the specific robot acquisition artifact that collapses trunk windows into ambient.

Untried remaining idea: surrogate-anchor selection. Train a cheap 4-class proxy model on holdout-train, predict holdout-val under `robot_heavy` to force proxy trunk->ambient errors, select rescue rules that fix the proxy, then lock and apply to the real anchor. Still train-only, but proxy/anchor mismatch risk is real and unverifiable without test.

## Current Conclusion

With the current non-leaky audio-only sources and group-consistency protocol, I have not found a clean path to `macro_f1_4class >= 0.75`.

The visible ceiling from source/rule combinations is around `0.72`. Reaching `0.75` likely needs one of:

- a genuinely better supervised audio feature/model family that improves trunk/twig under robot domain shift,
- a new untouched robot-like validation set from the same acquisition condition,
- or metadata that identifies contact/class status, which should not be used because it would be leakage if derived from filenames such as `_contact`, `leaf`, `trunk`, or `twig`.
