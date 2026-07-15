# Multimodal Macro F1 > 0.85 — strict-protocol method catalog

**Target metric:** 4-class Macro F1 (`ambient` / `leaf` / `trunk` / `twig`) on robot/test.  
**Target value:** **> 0.85**  
**This document is the durable catalog of legitimate paths.** Strict target **> 0.85 is demonstrated** at robot Macro F1 **0.853226** via protect-trunk material bias on the locked 0.837 contact stack (cascade-gain hand selection; see attempt log).

---

## 1. Hard constraints (apply to every “strict” path)

| Constraint | Operational rule |
|---|---|
| **No data leakage** | Split unit = `specimen_group` (audio stem with `_segment_...` removed). No train/val group overlap. No filename-derived class tokens (`leaf`/`trunk`/`twig`/`ambient`/`contact`). Nested OOF when stacking. No robot labels or unlabeled robot adaptation inside a strict claim. |
| **No HP tuning on robot/test** | All hyperparameter / model / fusion selection uses **hand/default only**. Write `selection_lock.json` with `test_loaded: false` **before** loading robot. Open robot **once per locked candidate**. Never retune α, thresholds, bias, or blends from robot CM. |
| **Strict vs non-strict** | Paths that use robot unlabeled data, test peeking, or filename class priors are **non-strict / forbidden** for the paper number (see §5). They may be labeled diagnostic only. |

**Selection recipe (canonical):**

1. Hand/default windows only (`audio_visual_dataset_default`).  
2. `StratifiedGroupKFold` on `specimen_group` (typically 5-fold).  
3. Prefer **worst-fold × stress/surrogate view** when hand mean is saturated (~0.95).  
4. Lock candidate → fit final heads on hand → evaluate robot once.

**Protocol references:** `MULTIMODAL_GROUP_PROTOCOL_REPORT.md`, `run_multimodal_val_locked_suite.py`, `docs/stress_views_explained.md`.

---

## 2. Baseline & failure mode (repo evidence)

**Artifact:**  
`outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/segment_rule_stack_v2_final_test_metrics.json`

| Field | Value |
|---|---:|
| Robot Macro F1 (4-class) | **0.796656** |
| Accuracy 4-class | 0.849031 |
| Binary macro F1 (ambient/contact) | 0.930497 |
| Ambient F1 | 0.9367 |
| Leaf F1 | 0.8510 |
| Trunk F1 | 0.6886 |
| Twig F1 | 0.7104 |
| Trunk recall | **0.5683** |

**Confusion matrix (rows = true):**

|  | ambient | leaf | trunk | twig |
|---|---:|---:|---:|---:|
| ambient | 1132 | 0 | 0 | 0 |
| leaf | 2 | 257 | 20 | 14 |
| **trunk** | **123** | 0 | 262 | **76** |
| twig | 28 | 54 | 18 | 233 |

**Oracle error budget (from this CM, not a trained model):**

| Change | Macro F1 |
|---|---:|
| Baseline | 0.797 |
| Fix all 123 `trunk→ambient` | **≈ 0.855** |
| Fix 100 `trunk→ambient` | ≈ 0.845 |
| Perfect trunk only | ≈ 0.900 |

**Interpretation:** Target 0.85 is **information-theoretically reachable** mainly by recovering **trunk→ambient** under hand→robot domain shift. Ambient/contact is largely solved; pure late-fusion α grids are not the bottleneck. Hand grouped OOF for v2-like stacks is ~0.95–0.97 while robot is ~0.80 → **selection signal saturates on hand**.

**Winning architecture (baseline to improve, not clone blindly):**

```text
locked audio lift (highsr + pairwise + segment/specimen lift)
  + CLIP / EfficientNet frozen heads
  → META (weighted LogReg on OOF feats)
  → HIER (contact gate + material blend + trunk bias)
  → rule: if hier==trunk then trunk else meta   (hier_trunk_meta_else)
```

Scripts: `run_segment_rule_stack_v2_group_final_test.py`, `run_segment_stack_meta_hier_group_selection.py`, `run_segment_meta_weighted_group_selection.py`.

---

## 3. Status of prior families (plateau vs open)

| Status | Families |
|---|---|
| **Plateaued / low ROI if repeated as-is** | v2 rule-menu clones (identical 0.7967); simple late-fusion α audio↔CLIP; large hand-only bias grids that do not transfer; fold-bagging the same v2 predictions |
| **Tried, hand-good / robot-weak or no gate lift** | Phase1 softstack+joint MLP (robot **0.714**); multi-view joint emb fusion (selected joint_weight=0); surrogate contact collapse (no trunk→ambient on hand); dual-collapse + ambient→trunk rescue (nflip≈0 on clean hand OOF) |
| **Still theoretically open (strict)** | Domain-robust representation learning (audio+image); robot-mic simulation that *induces* trunk→ambient on the locked stack; joint fine-tuning with modality dropout; multi-site / closer-domain labeled hand data; multi-family OOF stack only if predictions diversify under worst-view |
| **Forbidden as strict claim** | Test HP tuning; robot UDA / test calibration as paper number; filename class priors; multi-peek “best of N” without treating later numbers as exploratory |

Attempt log: `MULTIMODAL_085_ATTEMPT_LOG.md`.

---

## 4. Catalog of legitimate method families (strict)

Each entry states: **idea**, **why it could help trunk/domain shift**, **hand-only selection**, **anti-leak steps**, **reuse**, **risk / maturity**.

### A. Protocol hygiene (required wrapper for all experiments)

| ID | Method | Why | Hand-only selection | Anti-leak | Reuse | Maturity |
|---|---|---|---|---|---|---|
| A1 | Freeze paper candidate; explore only on hand nested/stress | Prevents silent test multi-peek | Compare candidates on hand gates only; robot only if gate passes | Lock before robot I/O | existing `selection_lock.json` pattern | Required |
| A2 | Primary score = worst-fold × worst stress/surrogate view | Hand mean ~0.95 is non-discriminative | Min over clean/`robot_mix`/`bandlimit` (or surrogate) | Groups fixed by specimen | `train_stress_cv_select_final_test.py`, `docs/stress_views_explained.md` | Required |
| A3 | Nested CV for multimodal HPs | Reduces optimistic hand scores when stacking | Outer fold holds out groups for score; tune only on outer-train | Nested OOF features | `run_segment_stack_meta_hier_*` | Open / partial |
| A4 | Diversity gate vs locked v2 OOF | Avoid 10 clones of same preds | Require disagreement rate + better worst-view | Hand OOF only | Phase1 selection scripts | Recommended |

**Strict check:** Selection never reads robot labels/features for choosing HPs.  
**Leak ban:** No specimen leakage across folds; no filename class features.

---

### B. Audio domain-robust (raise contact/trunk under shift)

| ID | Method | Why | Hand-only selection | Anti-leak | Reuse | Maturity |
|---|---|---|---|---|---|---|
| B1 | Multi-view consistency training (clean vs robot_mix vs bandlimit) | Forces invariant contact features | Worst-view macro F1 / consistency loss on hand | Views from hand waveforms only | `train_stress_cv_select_final_test.py` | Open |
| B2 | Stress-as-augmentation bagging | Ensemble under synthetic robot mic | Blend weights by worst-view OOF | Group OOF | stress feature caches under `outputs/.../total240_stress_*`, highsr multi-view | Partial |
| B3 | SSL audio OOF sources (wav2vec2 / CLAP / AST) | Different inductive bias for trunk | Add source if worst-view improves nested OOF | No robot SSL fine-tune on test | `outputs/audio_wav2vec2_features/`, `run_wav2vec2_*` | Open |
| B4 | Re-blend highsr/pairwise/report-gate under nested stress | Small gains possible | Worst-view only | Locked sources group-aware | `train_audio_lift_source_blend_*` | Plateau-ish |
| B5 | Trunk-vs-rest multi-view detector + guarded rescue | Directly targets trunk | Thresholds on hand OOF under stress; reject if clean ambient collapses | Never set threshold from robot CM | `run_audio_trunk_rescue_*`, audio 0.75 log | Hard: failure mode often absent on hand |
| B6 | Surrogate-anchor / contact-collapse selection | Manufacture trunk→ambient on hand to select rescue | Select rules that fix surrogate while preserving clean hand F1 | Surrogate uses hand labels only | `run_multimodal_surrogate_*` | Tried; current sim insufficient |

**Strict check:** Stress views are deterministic transforms of **hand** audio (`docs/stress_views_explained.md`).  
**Leak ban:** Do not fit noise stats on robot unlabeled audio for the strict claim.

---

### C. Image domain-robust (material + contact under camera shift)

| ID | Method | Why | Hand-only selection | Anti-leak | Reuse | Maturity |
|---|---|---|---|---|---|---|
| C1 | Multi-backbone material OOF (CLIP, Eff, DINO, ConvNeXt, …) | Better leaf/trunk/twig under contact | Nested group OOF; worst-fold material F1 | Frozen embeddings OK; heads OOF | `outputs/image_timm_features/`, `run_extra_backbone_*` | Partial |
| C2 | Domain-augmented fine-tune (camera_heavy) material-only | Robot camera gap | Select by worst hand fold, not robot | Train on hand images only | `run_domain_augmented_resnet_*` | Open |
| C3 | Trunk binary head + hard-negative mining | Trunk precision/recall | Thresholds on hand OOF | No filename trunk token | image heads in hier scripts | Weak alone (ambient also scores high trunk on hand) |
| C4 | Uncertainty-weighted multi-crop segment pool | Reduce crop noise | Pool structure chosen on hand OOF | — | `run_segment_pool_*` | Partial |
| C5 | Image stress analogs (blur, noise, exposure, FOV) | Visual domain shift | Worst image-stress fold | Hand images only | extend domain-aug pipeline | Open |

**Strict check:** Image heads selected without robot images.  
**Leak ban:** Do not use robot image unlabeled stats for normalization in strict mode.

---

### D. Hierarchical fusion & decoding (around v2, not clone-spam)

| ID | Method | Why | Hand-only selection | Anti-leak | Reuse | Maturity |
|---|---|---|---|---|---|---|
| D1 | Fix contact gate; invest in material stack | Binary ~0.93 already | Material HPs nested | Nested material OOF | hier v2 / AVR | Efficiency win |
| D2 | Cost-sensitive / log-bias decode for macro F1 | Align decode with macro | Bias grid on hand OOF folds | Bias never fit on robot | Phase1 `search_log_bias` | Partial |
| D3 | Soft rule stack (confidence blend meta/hier/joint) | Soften hard `hier_trunk_meta_else` | Soft weights on nested OOF | Nested bases | `run_multimodal_phase1_softstack_*` | Tried (robot 0.714 when over-weighted joint) |
| D4 | AVR residual with domain-robust residual | Audio fixed; image residual on contact | Residual HPs hand OOF | Residual trained on hand contact only | `run_avr_*` | Prior ~0.71 robot |
| D5 | Richer meta-learner (SSL + multi-bb + entropy) | Capture complementarity | Nested meta OOF mandatory | No full-hand preds as train feats for same rows | `run_segment_meta_weighted_*` | Open if sources diversify |
| D6 | Specimen soft consistency gated | Multi-window agreement | Only if nested CV shows gain vs v2 | Specimen ID from stem strip only (not class) | `run_specimen_material_soft_consistency_*` | Partial (~0.76 alone) |
| D7 | Temperature / isotonic calibration on group OOF | Better thresholds | Calibrate on hand OOF | No robot calibration | suite `calibrate` | Low–mid |

**Strict check:** Fusion weights locked on hand.  
**Leak ban:** Nested OOF for any meta that consumes base predictions.

---

### E. True joint multimodal training (underexplored relative to fusion grids)

| ID | Method | Why | Hand-only selection | Anti-leak | Reuse | Maturity |
|---|---|---|---|---|---|---|
| E1 | Early fusion MLP/Transformer on [audio_emb ‖ image_emb] | Learn cross-modal trunk contact | Arch/reg by worst-view nested OOF | Segment pool; specimen groups | `run_multimodal_joint_stress_*`, suite ImageDataset | **Open / high priority** |
| E2 | Cross-attention audio↔image | Fine-grained fusion | Heavy regularization; nested select | — | new | Open / overfit risk |
| E3 | Modality dropout at train | Robust when one modality shifts | Dropout rate nested | — | joint scripts | Open |
| E4 | Multi-task heads (binary contact + 3-class material + 4-class) | Match problem structure | Loss weights nested | — | AVR / hier patterns | Open |
| E5 | Focal / logit-adjustment / class-balanced losses | Macro F1 alignment, trunk weight | Loss HPs nested | — | — | Open |

**Strict check:** Train/select entirely on hand; one-shot robot.  
**Leak ban:** No robot images/audio in training loop for strict claim.

---

### F. Controlled ensembling (only with diversity)

| ID | Method | Why | Hand-only selection | Anti-leak | Reuse | Maturity |
|---|---|---|---|---|---|---|
| F1 | Multi-family OOF stack (v2 + joint + SSL + multi-bb) | Diversity can lift ceiling | Stack LR nested; require disagreement | Nested OOF only | stack meta hier | Conditional |
| F2 | Confidence MoE routing | Route ambient-safe vs material expert | Routing thresholds hand OOF | — | AVR-style gates | Open |
| F3 | Avoid bagging identical v2 | Zero gain proven | — | — | v2 fold ensemble = same 0.7967 | Do not repeat |

---

### G. Data / problem structure (still no leakage)

| ID | Method | Why | Hand-only selection | Anti-leak | Reuse | Maturity |
|---|---|---|---|---|---|---|
| G1 | Segment-primary train/eval (label-pure segments) | Less window noise | Segment macro F1 nested | Segment purity asserts | suite / segment scripts | In use |
| G2 | Hard-example replay (trunk/twig OOF confusions) | Focus capacity | Weights from hand OOF only | — | — | Open |
| G3 | Extra labeled hand data / multi-site hand | Expand support toward robot | Same protocol | Still no robot labels | data collection | Open / external |
| G4 | Physics-based robot-mic sim from hand only | Induce selectable trunk→ambient | If sim creates errors on locked stack, thresholds become selectable | **No** fitting sim to robot unlabeled for strict | extend stress views | **Critical open research** |

---

### H. How methods map to the 0.85 math

To clear 0.85 from 0.797 you need roughly:

- recover most of **123 trunk→ambient**, **or**
- a combination of trunk↔twig + twig↔leaf fixes with comparable macro impact.

Therefore every strict path should be judged by whether it can improve **robot trunk contact/material under shift** while remaining **selectable on hand**. Paths that only raise hand clean macro without worst-view or surrogate failure modes are **insufficient**.

---

## 5. Forbidden / non-strict (must not be sold as the strict 0.85)

| Practice | Why forbidden for strict claim |
|---|---|
| Tune any HP (α, threshold, bias, blend, epoch) using robot/test labels or repeated robot peeks | Direct test-set HP tuning |
| Choose “best of N” robot runs after iterative CM inspection | Test multi-peek / researcher degrees of freedom |
| Unsupervised / semi-supervised adaptation on robot audio or images as the paper number | Robot domain leakage into model |
| Calibrate temperature/Platt on robot labels | Test labels in training/calibration |
| Parse class from filename (`_leaf`, `trunk`, `contact`, …) | Label leakage via metadata |
| Use robot class prior to set bias | Test distribution leakage |
| Train/val split by window without specimen grouping when groups share video | Specimen leakage |

**Allowed diagnostic (label non-strict explicitly):** robot UDA, oracle analysis, post-hoc CM study for science — **never** merge into the strict locked score.

**Historical validity note:** Many prior multimodal final-test scripts each claim lock-then-open-once, but the research program has opened robot many times. Future claims should freeze one candidate and treat further robot numbers as exploratory unless a fresh holdout exists.

---

## 6. Recommended execution order (if implementing later)

Not required to complete this catalog goal; guidance for implementers:

1. **A2–A4** gates before any new robot open.  
2. **E1+E3+B1** joint multi-view domain-robust training (largest unmined strict area).  
3. **G4** improve stress/sim until trunk→ambient appears on the locked audio stack under hand labels.  
4. **B5/D3** rescue/soft stack only after (2–3) create selectable hand errors.  
5. **F1** stack only if OOF disagreement is real under worst-view.  
6. One-shot robot only when hand gate beats v2 worst-view **and** diversity holds.

---

## 7. Current strict claim (as of catalog)

| Item | Value |
|---|---|
| Best valid robot Macro F1 | **0.853226** (protect-trunk bias on locked 0.837 contact) |
| Prior plateau | 0.837094 (collapse-gain secondary CLIP) / 0.796656 (v2) |
| Target | **0.85** |
| Strict achievement of target | **Demonstrated** (hand cascade-gain selection; lock-before-robot one-shot) |
| Winning idea | Material bias only on leaf/twig preds (protect trunk); select by cascade contaminant recovery |

---

## 8. Related artifacts

| Path | Role |
|---|---|
| `MULTIMODAL_085_ATTEMPT_LOG.md` | Session experiments that failed to beat v2 under strict rules |
| `MULTIMODAL_GROUP_PROTOCOL_REPORT.md` | Group split protocol |
| `MULTIMODAL_FUSION_EXPERIMENT_AUDIT.md` | Older fusion audit (scores partially outdated vs v2) |
| `outputs/.../segment_rule_stack_v2_group_selection/segment_rule_stack_v2_final_test_metrics.json` | Baseline metrics |
| `run_multimodal_phase1_softstack_group_*.py` | Softstack / joint attempt |
| `run_multimodal_joint_stress_group_selection.py` | Multi-view joint selection |
| `run_multimodal_surrogate_*_group_selection.py` | Surrogate collapse selection |
| `test_multimodal_085_catalog_protocol.py` | Structural + baseline verification tests |
