# Paper Claim v2 — Multimodal Fusion

Sealed **audio–visual fusion** pipeline for 4-class contact material classification on a held-out robot domain.

| | |
|:--|:--|
| **Task** | 4-class contact material classification |
| **Classes** | `ambient` · `leaf` · `trunk` · `twig` |
| **Primary metric** | Macro F1 (4-class) on robot/test |
| **Test size** | *n* = 2 219 windows |
| **Paper claim (fusion v2)** | **Macro F1 = 0.850781** |
| **Protocol** | Hand-only HP selection · robot opened once · sealed claim |

This repository contains **only** the paper fusion v2 path: documentation, claim code, and sealed results.

---

## Sealed results (robot / test)

| Metric | Value |
|:--|--:|
| **Macro F1 (4-class)** | **0.850781** |
| Accuracy (4-class) | 0.891393 |
| Binary Macro F1 (ambient / contact) | 0.965652 |

| Per-class F1 | ambient | leaf | trunk | twig |
|:--|--:|--:|--:|--:|
| Fusion v2 | 0.9675 | 0.8577 | 0.8095 | 0.7684 |

| Artifact | Path |
|:--|:--|
| Seal | [`results/CLAIM_SEAL.json`](results/CLAIM_SEAL.json) |
| Full metrics | [`results/final_test_metrics.json`](results/final_test_metrics.json) |
| Targets | [`shared/sealed_targets.json`](shared/sealed_targets.json) |
| Recipe | `paper_claim_v2_simple_leaf_bias` · free HP: \(b_{\mathrm{leaf}} = -1.3\) (hand only) |

---

## Method overview

Fusion claim **v2** is a **four-step cascade** over frozen audio segment predictions and multimodal detectors:

```text
Inputs
  ├─ Audio segment stack (v2 base: hard / meta soft)
  ├─ Audio features (total240 / robot_mix) + wav2vec2
  └─ Image embeddings (CLIP)

Detectors  ──fit on HAND only──►  trunk / contact / binary scores

[1] Base ........ hier_trunk_meta_else on v2 segment artifact
[2] Amb-lift .... ambient → trunk if ts ≥ 0.5 and max(cs, bin) ≥ 0.1
[3] Secondary ... ambient → trunk if CLIP trunk ≥ 0.625 and bin ≥ 0.75
[4] Material .... leaf/twig redecode with logit bias b_leaf = −1.3
                  (trunk predictions protected)

→ 4-class labels on robot windows  ·  Macro F1 = 0.850781
```

**Claim code:** [`code/paper_claim/clear_pipeline.py`](code/paper_claim/clear_pipeline.py)  
**Design write-up:** [`docs/PAPER_CLAIM_V2_FUSION_GUIDE.md`](docs/PAPER_CLAIM_V2_FUSION_GUIDE.md)

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── reproduce.py                 # official sealed entry point
├── code/
│   ├── evaluate_fusion.py
│   └── paper_claim/
│       ├── clear_pipeline.py    # paper-facing clear architecture
│       ├── RECIPE_v2_simple.json
│       ├── ARCHITECTURE_CLEAR.md
│       └── architecture_clear.svg
├── data/
│   ├── manifests/               # hand + robot dataset CSVs (tracked)
│   └── features/                # precomputed features (local; not in git)
├── results/                     # CLAIM_SEAL + metrics + locks
├── docs/                        # fusion claim guide + diagram
├── shared/                      # metrics helpers + sealed_targets.json
└── tests/                       # sealed F1 regression tests
```

---

## Reproduce

Requires precomputed features under `data/features/` (local / portable pack; git tracks manifests + seals only).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python reproduce.py
# or:
python code/evaluate_fusion.py
pytest tests/ -q
```

Expected: exit code **0**, Macro F1 **0.8507810400075331** within **1e-9**.

Optional:

```bash
export TREE_BUNDLE_ROOT=/absolute/path/to/this/repo
python reproduce.py
```

---

## Protocol

| Rule | Status |
|:--|:--|
| Hyperparameters selected on **hand** data only | Yes |
| Robot / test used **once** for the sealed claim | Yes (`CLAIM_SEAL`) |
| No HP search on robot Macro F1 | Yes |
| No class-from-filename features | Yes |
| Grouping for CV / selection | `specimen_group` |
| Equality tolerance vs sealed targets | \(10^{-9}\) |

---

## Feature stack

```text
X_amb = hstack([ total240/robot_mix (240) , wav2vec2 (1536) , CLIP (768) ])  # dim 2544
```

Raw media (`.wav` / JPEG) is **not** included. Sealed F1 does not need it when features are present.
