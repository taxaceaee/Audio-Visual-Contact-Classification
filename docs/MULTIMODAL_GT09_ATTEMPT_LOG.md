# Pure-paper attempts toward robot Macro F1 > 0.9

**Protocol:** hand-only selection → lock → one-shot robot → seal. No robot UDA, no filename class features, no test HP tuning.  
**Environment constraint:** `tree_structures` raw dataset path is **offline** (broken symlink to `/home/ttung05/Desktop/tree_base/...`). Only precomputed `paths/y/X` feature caches are available — **no raw wav/jpg for new representation learning**.

## Baseline (pre-existing seals; not reopened)

| Claim | Robot Macro F1 |
|---|---:|
| paper_claim_v2 / clear | **0.850781** |
| paper_claim_v1 BEST_STRICT | **0.853226** |
| Target | **> 0.900** |

## Residual math (sealed v2 CM, N=2219)

| Oracle intervention | Macro F1 |
|---|---:|
| Contact residual only perfect | **0.8865 (< 0.9)** |
| Material/wood residual perfect | **0.9669** |
| Cheapest combo ~15 trunk→twig + 51 leaf→twig | ~0.900 |

→ >0.9 is **material-dominated**, not contact-only.

## Attempt A — `outputs/paper_claim_gt09` (sealed)

| Field | Value |
|---|---|
| Recipe | `paper_claim/RECIPE_gt09_material_multibb.json` |
| Hand selection | Multi-bb (CLIP+Eff+DINO+ConvNeXt) ‖ total240 ‖ wav2vec2 group-OOF soft blend + protect-trunk biases; cascade-gain eligible |
| Locked candidate | α=0.5, b_leaf=-1.0, b_trunk=-0.25, b_twig=0.5, T=0.75; hand clean≈0.948, casc_gain≈0.025 |
| **Robot Macro F1** | **0.740551** |
| n | 2219 |
| Seal | `CLAIM_SEAL.json` (one-shot) |
| Kill | Multi-bb soft **hurts** robot material (leaf→twig 51→129); hand cascade gate did not predict domain failure |

**Lesson:** Hand-selectable multi-bb blend is a **false friend** under camera shift. Pure paper correctly sealed a failure; no retune on robot.

## Non-claim diagnostics (not used for HP selection / not claim numbers)

| Probe | Robot Macro F1 | Note |
|---|---:|---|
| Contact stack only | 0.8346 | amb-lift + secondary |
| meta + b_leaf=-1.3 (clear v2) | **0.8508** | reproduces seal |
| multi-bb alone | 0.591 | no transfer |
| blend α>0 on multi-bb | ≤0.79 | monotonic harm |
| wood flip multi-bb on twig→trunk | ≤0.8532 | ~v1 ceiling |
| wood flip audio total240+w2v | ≤0.8544 | tiny lift |
| highsr bandlimit soft blend | ≤0.8525 | scan of cached softs |
| resnet segment proba fusion | ≤0.77 | worse |

**Frozen-feature pure-paper ceiling observed:** ~**0.85–0.854**. Historical material-bias oracle on 0.837 contact base ~**0.879** still **< 0.9**.

## Why >0.9 is blocked in this workspace

1. **No raw media** → cannot run P0.1 domain-aug FT / joint train from pixels/waveforms.  
2. **Linear/nonlinear probes on frozen caches** do not recover trunk↔twig / leaf↔twig under robot shift.  
3. **Contact-only oracle 0.886 < 0.9** → even perfect amb rescue is insufficient.  
4. Pure paper forbids robot UDA / test calibration that could close the gap non-honestly.

## What would unblock (still pure paper)

1. Restore `audio_visual_dataset_{default,robo_default}` images+audio.  
2. Hand-only domain-randomized material FT + worst-view gates (roadmap P0.1–P0.2).  
3. New claim dir; one-shot robot only after hand gates; success iff Macro F1 > 0.9.

## Code delivered (pure-paper machinery)

| Path | Role |
|---|---|
| `paper_claim/gt09_core.py` | Cache-based contact + material core |
| `paper_claim/RECIPE_gt09_material_multibb.json` | Frozen recipe |
| `run_paper_claim_gt09_hand_selection.py` | Hand-only selection |
| `run_paper_claim_gt09_final_test_once.py` | One-shot robot + seal |
| `outputs/paper_claim_gt09/*` | Lock, leaderboard, metrics, seal |
| `test_paper_claim_gt09_protocol.py` | Protocol + honesty tests |

## Honesty statement

No claim of Macro F1 > 0.9 is made. The pure-paper attempt that was opened once scored **0.7406**. Best pre-existing pure seals remain **0.8508 / 0.8532**.


## Attempt B–D — frozen-feature pure diagnostics (no new claim seal / no HP retune)

| Method | Robot Macro F1 | Hand gate note |
|---|---:|---|
| highsr×3 + CLAP + w2v + CLIP LogReg multiview material blend | ≤0.85 (α=0 best) | hand OOF material 0.918 |
| same with HGB blend | ≤0.847 | α>0 always hurts vs meta |
| HGB wood binary flip twig→trunk | ≤0.826 | hand nflip large; robot destroys twigs |
| HGB soft rules (twig↔leaf/trunk) | ≤0.808 | no pure lift |

**Conclusion reinforced:** without raw media FT, pure-paper ceiling remains ~0.85.


## Attempt E — highsr robot_mix amb-lift (diagnostic)

| Setting | Robot Macro F1 | trunk→amb |
|---|---:|---:|
| clear baseline | 0.8508 | 45 |
| highsr multiview amb + default th | 0.8239 | **101** (worse) |
| oracle th search on robot for this pack | 0.8239 | worse |

No pure-paper gain; do not open a second claim for this.

## Final status

- Pure-paper infrastructure: **delivered** and tested (17 pytest).
- Sealed pure attempt: **0.7406** (`outputs/paper_claim_gt09`).
- Best known pure seals (pre-existing): **0.8508 / 0.8532**.
- Target **>0.9**: **not achieved**; blocked without raw media for representation learning.
