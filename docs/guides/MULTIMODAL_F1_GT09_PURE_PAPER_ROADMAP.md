# Pure-paper roadmap: robot Macro F1 > 0.9

**Goal kind:** analysis / research plan only.  
**No new sealed robot run was executed for this document.**  
**Baseline (existing seals only):**

| Claim path | Robot Macro F1 4-class | Source |
|---|---:|---|
| paper_claim_v2 / clear_pipeline | **0.850781** | `outputs/paper_claim_v2/final_test_metrics.json`, `CLAIM_SEAL.json` |
| paper_claim_v1 / BEST_STRICT protect-trunk | **0.853226** | `outputs/paper_claim_v1/`, `outputs/.../BEST_STRICT_final_test_metrics.json` |
| segment_rule_stack_v2 base | 0.796656 | v2 final_test metrics |
| Target | **> 0.900** | this roadmap |

**In-repo durable copy:** `docs/MULTIMODAL_F1_GT09_PURE_PAPER_ROADMAP.md`  
**Grounding docs:** `MULTIMODAL_085_ATTEMPT_LOG.md`, `MULTIMODAL_085_STRICT_CATALOG.md`, `PAPER_CLAIM_V2_FUSION_GUIDE.md`, `paper_claim/ARCHITECTURE_CLEAR.md`, sealed CM in `outputs/checkpoints/paper_claim_v2_clear_LATEST/evaluation_report.json`.

---

## 0. Pure-paper invariants (hard requirements on every primary path)

Every primary path below **must** keep all of:

1. **Hand/default-only selection** — all HPs, architectures, thresholds, biases chosen on `audio_visual_dataset_default` only.  
2. **Lock before robot** — write `selection_lock.json` with `test_loaded: false` before any robot I/O.  
3. **Single robot open for claim** — one-shot final test + `CLAIM_SEAL.json`; no multi-peek “best of N” as the paper number.  
4. **No robot UDA / unlabeled adaptation** — no robot feature stats, no SSL on robot, no test-time training.  
5. **No filename class features** — no parsing `leaf`/`trunk`/`twig`/`ambient` tokens from paths as predictors.  
6. **No test HP tuning** — no thresholds/biases/epochs from robot labels or robot CM.  
7. **Specimen/group leakage controls** — `StratifiedGroupKFold` on `specimen_group`; nested OOF for any stack that consumes base predictions.

**Non-pure / diagnostic-only** ideas are marked §D and **must not** be sold as the paper number.

---

## 1. Residual budget (why >0.9 is a different problem than >0.85)

### 1.1 Sealed v2 confusion matrix (robot, N=2219 windows)

From `evaluation_report.json` / `final_test_metrics.json` (rows = true):

| true \ pred | ambient | leaf | trunk | twig | support |
|---|---:|---:|---:|---:|---:|
| ambient | 1131 | 0 | 1 | 0 | 1132 |
| leaf | 2 | 220 | 20 | **51** | 293 |
| trunk | **45** | 0 | 340 | **76** | 461 |
| twig | **28** | 0 | 18 | 287 | 333 |

**Macro F1 = 0.8508.** Total errors = **241**.

### 1.2 Per-class (sealed v2)

| Class | P | R | F1 | Role |
|---|---:|---:|---:|---|
| ambient | 0.938 | 0.999 | **0.967** | Solved |
| leaf | 1.000 | 0.751 | 0.858 | Precision max; recall hurt by leaf-bias + leaf→twig |
| trunk | 0.897 | **0.738** | 0.810 | Contact residual + trunk→twig |
| twig | **0.693** | 0.862 | **0.768** | Absorbs leaf+trunk false positives |

### 1.3 Oracle ladders on the **sealed v2 CM** (information bounds, not models)

Computed by correcting off-diagonal cells back to the true class (no new model):

| Intervention | Errors fixed | Macro F1 | ≥0.9? |
|---|---:|---:|:---:|
| Baseline sealed v2 | 0 | **0.8508** | no |
| Fix all **trunk→ambient** (45) | 45 | 0.8707 | no |
| Fix all **twig→ambient** (28) | 28 | 0.8648 | no |
| Fix **all contact residual** (t/tw/l→amb + amb→t) | 76 | **0.8865** | **no** |
| Fix all **trunk→twig** (76) | 76 | **0.8972** | almost |
| Fix all **leaf→twig** (51) | 51 | 0.8907 | no |
| Fix **all wood/material residual** (t↔tw, l→tw, l→t) | 165 | **0.9669** | **yes** |
| Fix top-3 modes (t→tw + t→a + l→tw) | 172 | 0.9582 | yes |
| Greedy: full t→tw then full l→tw | 127 | **0.9406** | yes |
| Cheapest combo of (t→tw + l→tw) only | **66** (e.g. 15 t→tw + 51 l→tw) | ~0.9004 | yes |

### 1.4 Central conclusion for >0.9

1. **Contact-only perfect recovery is not enough** (ceiling **0.8865 < 0.9** on this CM).  
2. The path that historically unlocked 0.85 (amb-lift + secondary trunk rescue) is **necessary but saturated** for a 0.9 claim.  
3. **>0.9 is primarily a material / wood+leaf problem:** `trunk→twig` (76) and `leaf→twig` (51) dominate the gap; together with related wood confusions they admit Macro F1 **~0.97** if solved.  
4. Historical **material logit-bias oracle on the older 0.837 contact base ≈ 0.879** (`MULTIMODAL_085_ATTEMPT_LOG.md`) shows that **pure reweighting of frozen soft scores is unlikely to clear 0.9**. Paths that only search biases/thresholds on today’s towers should be treated as **sub-0.9 ceiling** unless representations change.  
5. Therefore primary pure-paper paths must **change features / training support / domain-sim**, then hand-select, then one-shot robot — not more rule clones on frozen meta.

---

## 2. Why prior pure attempts plateau (failure-mode map)

| Observation | Evidence | Implication for >0.9 |
|---|---|---|
| Hand OOF ~0.95–0.97, robot ~0.80–0.85 | Attempt log, catalog §2 | Clean hand macro is non-discriminative; select on **worst-fold × stress/surrogate** |
| Robot trunk→ambient scarce on hand | Surrogate collapse trials; nflip≈0 | Contact rescue **cannot** be further hand-tuned without better sim |
| Leaf-bias hand-selected via cascade_gain | v2 `b_leaf=-1.3` | Trades leaf recall for twig; **creates** part of leaf→twig cost |
| protect-trunk forbids material flip of trunk | Architecture clear | Correct for contact; **blocks** material repair of trunk if hard pred is already trunk; trunk→twig errors are preds labeled twig (material can touch) but soft trunk mass is weak |
| Joint softstack / multitask / CLAP wood / domain-aug ResNet | robot 0.71–0.80 | Wrong objective (clean hand) or weak material transfer |
| v2 rule-menu clones | identical 0.7967 | Zero diversity — do not repeat |

---

## 3. Ranked pure-paper paths to Macro F1 > 0.9

Ranking axes: **(1) honesty of pure-paper fit**, **(2) residual modes attacked**, **(3) ability to exceed frozen-soft oracle (~0.879 class)**, **(4) hand-selectability**.

For each path: **(a)** mechanism, **(b)** hand selection without robot peek, **(c)** residual classes, **(d)** feasibility vs oracle ceiling, **(e)** kill criteria.

---

### P0 — Primary stack (recommended program)

#### Path P0.1 — Material-first representation rebuild (image + audio material heads)

**Rank: #1 pure-paper honesty × expected lift for >0.9**

| Field | Content |
|---|---|
| **(a) Mechanism** | Freeze or lightly adapt contact stack; **rebuild 3-class material** (leaf/trunk/twig) with domain-augmented image FT + multi-backbone OOF + optional wood-contrastive / hard-negative mining on hand confusions. Decode material **only when contact=positive**, with protect-ambient. Goal: cut **trunk→twig** and **leaf→twig** (~66+ combined errors to clear 0.9 in oracle combo). |
| **(b) Hand selection** | Nested `specimen_group` OOF; primary score = **worst-fold material macro F1** on contact-positive segments + clean ambient preservation; optional image-stress (blur/exposure/FOV) min score. Lock material recipe before robot. |
| **(c) Residuals attacked** | trunk→twig (76), leaf→twig (51), leaf→trunk (20), twig→trunk (18). Contact modes secondary. |
| **(d) Feasibility** | Material residual oracle **0.967** → information room exists. Frozen-soft bias oracle ~0.879 does **not** apply if embeddings/heads change. Plausible pure-paper path to >0.9 **if** new material features reduce wood confusions under camera shift. Not guaranteed. |
| **(e) Kill criteria** | Hand worst-fold material F1 on contact segments fails to beat locked meta material by ≥0.02; or ambient clean F1 drops >0.01; or OOF predictions agree >98% with current meta (no diversity) → abandon before robot. |

**Reuse:** `run_domain_augmented_resnet_*`, `outputs/image_timm_features/*`, `run_extra_backbone_*`, hier material blend patterns.  
**Do not:** full 4-class redecode that destroys trunk (history: 0.70).

---

#### Path P0.2 — Calibrated robot-mic / camera sim that induces **selectable** hand failures

**Rank: #2 (enabler for any further contact rescue + robust selection)**

| Field | Content |
|---|---|
| **(a) Mechanism** | Design **hand-only** stress views (audio physics + image corruptions) until the **locked claim stack** produces non-trivial **trunk→ambient** and **trunk↔twig** on hand OOF. Then train multi-view invariant models (consistency loss clean↔stress) and/or select rescue thresholds on those induced errors. |
| **(b) Hand selection** | Surrogate utility: rescue that **fixes induced errors** while clean hand macro ≥0.94 and false trunk ≤ budget (same spirit as cascade_gain gates). Never fit noise stats on robot unlabeled audio/images. |
| **(c) Residuals attacked** | Enables further cuts to trunk→ambient (45) and twig→ambient (28); improves selection signal for material under shift. Alone (contact fix) caps at **0.8865** — **must pair with P0.1**. |
| **(d) Feasibility** | Catalog marks this **critical open research** (G4). Prior surrogate collapse **failed** to create trunk→ambient. Success would unlock pure-paper contact headroom (~+0.02–0.04) **and** trustworthy gates. Not sufficient alone for 0.9. |
| **(e) Kill criteria** | After iterative hand-only sim design, induced trunk→ambient count on locked stack remains ≈0–4 segments; or any sim that needs robot unlabeled calibration → demote to diagnostic. |

**Reuse:** `docs/stress_views_explained.md`, `total240_stress_*`, `run_multimodal_surrogate_*` (as negative results).

---

#### Path P0.3 — Joint multimodal training with modality dropout + multi-task heads

**Rank: #3 (largest unmined representation space; higher overfit risk)**

| Field | Content |
|---|---|
| **(a) Mechanism** | Train early/mid fusion on segment-pooled `[audio_emb ‖ image_emb]` (or cross-attn light) with: multi-task losses (binary contact + 3-class material + 4-class), **modality dropout**, multi-view audio stress batching, class-balanced / focal loss for macro alignment. Replace frozen cascade heads with learned joint logits; keep protocol pure. |
| **(b) Hand selection** | Nested group OOF; score = **min(clean, robot_mix, bandlimit, image-stress) macro F1**; diversity gate vs locked v2/clear predictions (disagreement rate floor); reject if only clean improves. |
| **(c) Residuals attacked** | All four major modes if representation moves; priority wood+contact jointly. |
| **(d) Feasibility** | Can exceed frozen-soft oracle. Prior joint attempts (Phase1, multitask) hit **0.71–0.72 robot** when hand selected clean/soft clones — **only viable with P0.2-style worst-view selection and strong regularization**. High variance; pure-paper fit OK if no robot peek. |
| **(e) Kill criteria** | Nested worst-view does not beat clear baseline OOF proxy by ≥0.015; or disagreement with locked claim preds <2%; or clean≫worst-view gap >0.08 (overfit signature) → do not open robot. |

**Reuse:** `run_multimodal_joint_stress_*`, `run_multimodal_085_joint_multitask_*` (as cautionary baselines), suite ImageDataset.

---

### P1 — Secondary pure paths (support / stack after P0)

#### Path P1.1 — Nested multi-family OOF stack **only after** diversity under worst-view

| Field | Content |
|---|---|
| **(a)** | Stack logistic/meta on OOF from: new material tower (P0.1), joint (P0.3), locked clear cascade, SSL audio. |
| **(b)** | Nested OOF only; require pairwise disagreement on hand stress views. |
| **(c)** | Residual modes where experts specialize (contact expert vs wood expert). |
| **(d)** | Can push past single-expert plateau **if** P0 creates real diversity; bagging identical v2 is proven zero-gain (0.7967). |
| **(e)** | Kill if stack weight concentrates >0.9 on one family or nested gain <0.005. |

#### Path P1.2 — Contact-positive material MoE (routing, not global leaf bias)

| Field | Content |
|---|---|
| **(a)** | Replace global `b_leaf=-1.3` with **gated experts**: wood-vs-leaf head when contact high; ambient-safe path when contact low. Protect ambient hard. |
| **(b)** | Gate thresholds + expert HPs on hand nested OOF with cascade surrogate (trunk→twig / twig→leaf soft contaminants) — same family as v2 selection but multi-expert. |
| **(c)** | leaf→twig, trunk→twig without further punishing all leaf. |
| **(d)** | Still limited by soft feature quality; treat as **decode layer on P0.1 features**, not standalone >0.9 plan. Frozen meta + richer bias alone likely stays **≤ ~0.88–0.89** (history). |
| **(e)** | Kill if hand selects near-identity or single global bias again; or clean leaf recall collapses >0.05 with no worst-view wood gain. |

#### Path P1.3 — Extra labeled hand domains closer to robot (multi-site hand)

| Field | Content |
|---|---|
| **(a)** | Collect/use additional **labeled** hand-side data nearer robot mic/camera (still not robot test). Expand train support. |
| **(b)** | Same group protocol; select on multi-domain worst site. |
| **(c)** | All shift-driven residuals. |
| **(d)** | Highest external cost; highest honesty; can break the hand≠robot failure-mode mismatch. |
| **(e)** | Kill only if new domains do not increase worst-domain OOF difficulty (still saturated). |

---

### P2 — Low priority / likely sub-0.9 under pure paper (do not lead with these)

| Path | Why demoted |
|---|---|
| Larger bias/threshold grids on **frozen** clear stack | Material oracle ~0.879 class; contact-only CM oracle 0.8865; **no representation change** |
| More v2 rule-menu clones / fold bagging same preds | Proven identical robot F1 |
| Specimen soft consistency alone | Prior robot ~0.77; over-smooths wood |
| CLAP wood-only experts as sole lift | Prior transfer fail (~0.76) |
| Phase1 softstack without new features | Robot 0.714 |

These may be **ablation controls**, not primary >0.9 programs.

---

## 4. Explicit non-pure / diagnostic-only (out of claim)

| Idea | Why forbidden as paper number |
|---|---|
| Robot-label oracle bias / CM-tuned thresholds | Test HP tuning (history oracle ~0.879 on 0.837 base) |
| Multi-peek robot, keep best run | Researcher degrees of freedom |
| Robot UDA / test-time adapt / unlabeled robot SSL | Robot domain leakage |
| Filename class priors | Label leakage |
| Calibrate temperature on robot labels | Test labels in calibration |
| Pseudo-label robot then train | Semi-supervised on test domain |

**Allowed:** offline oracle studies, failure analysis, upper bounds — labeled **non-claim**.

---

## 5. Recommended execution order (pure claim program)

```text
Phase A (no robot): protocol gates A2–A4
  worst-view scoring, diversity vs locked clear, nested OOF

Phase B (no robot): P0.1 material representation rebuild
  kill if no hand worst-view material lift

Phase C (no robot): P0.2 stress/sim until failure modes appear on hand
  kill if still nflip≈0 for target modes

Phase D (no robot): P0.3 joint train only if B/C create signal
  kill on overfit signature

Phase E (no robot): P1.1 stack if disagreement real
  lock single candidate → selection_lock.json

Phase F (once): one-shot robot final test + CLAIM_SEAL
  success criterion: Macro F1 > 0.9
  failure: publish exploratory note; do not retune on robot
```

**Parallel honesty rule:** Never open robot “to see if sim is good.” Sim quality is judged only by **hand-induced errors + clean preservation**.

---

## 6. Quantitative success bars (hand gates before any robot open)

| Gate | Suggested bar (hand-only) |
|---|---|
| Worst-view 4-class macro vs locked clear OOF proxy | **+≥0.015** absolute |
| Contact-positive material macro (leaf/trunk/twig) | **+≥0.02** vs locked meta material |
| Clean ambient F1 | **≥ 0.99** (do not regress ambient) |
| Disagreement rate vs locked claim hard preds | **≥ 5%** of segments (diversity) |
| Induced trunk→ambient (if P0.2 claims contact work) | **≥ 15** hand segments under stress with clean ambient false-trunk ≤ budget |
| Nested stack gain (if stacking) | **≥ 0.005** worst-view |

If gates fail → **no robot open** (pure paper discipline).

---

## 7. What would count as “done” for a future >0.9 claim

1. New recipe JSON + SHA, hand leaderboard, `selection_lock.json` (`test_loaded: false`).  
2. Single `final_test_metrics.json` with `macro_f1_4class > 0.9`, `n=2219`, protocol flags true.  
3. `CLAIM_SEAL.json` with same invariants as v2 seal.  
4. Clear architecture doc + ablation table (contact vs material modules).  
5. Explicit statement that multi-peek exploratory dirs are non-claim.

**This analysis goal does not produce that seal** — only the pure-paper path design.

---

## 8. Bottom line

| Question | Answer |
|---|---|
| Is >0.9 possible under pure paper? | **Plausible, not guaranteed.** CM oracle shows **material residual alone → 0.967**; contact alone → **0.886 < 0.9**. |
| Is more cascade bias on frozen towers enough? | **Almost certainly no** (history ~0.879 material-oracle class; contact ceiling 0.886). |
| What must change? | **Material representations (+ optional joint train)** and/or **hand domain-sim that induces real errors**, then strict hand gates, then **one** robot open. |
| Fastest honest portfolio | **P0.1 → P0.2 → (optional P0.3) → P1.1 → one-shot claim** |

---

## 9. Source citations (open these to audit)

| Path | Use |
|---|---|
| `outputs/paper_claim_v2/final_test_metrics.json` | Sealed Macro F1 0.850781 + CM |
| `outputs/paper_claim_v2/CLAIM_SEAL.json` | Pure single-shot invariants |
| `outputs/checkpoints/paper_claim_v2_clear_LATEST/evaluation_report.json` | Major errors list |
| `outputs/paper_claim_v1/final_test_metrics.json` | Best absolute 0.853226 |
| `MULTIMODAL_085_ATTEMPT_LOG.md` | Failed paths, material oracle ~0.879, residual narrative |
| `MULTIMODAL_085_STRICT_CATALOG.md` | Strict method families A–G, forbidden practices |
| `PAPER_CLAIM_V2_FUSION_GUIDE.md` | Architecture, ablations (−amb/−sec/−mat/−w2v) |
| `paper_claim/ARCHITECTURE_CLEAR.md` | Clear 4-step diagram |
| `paper_claim_protocol.py` / `multimodal_085_protocol.py` | Protocol helpers |
