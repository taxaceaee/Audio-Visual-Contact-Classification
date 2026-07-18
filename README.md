# Audio-Visual Contact Classification

**Multimodal classification of physical contact material on tree structures for agricultural robotics.**

This repository provides a reproducible **audio–visual fusion** pipeline that predicts contact class on a held-out **robot** domain, together with sealed evaluation artifacts for paper reporting.

| | |
|:--|:--|
| **Task** | 4-class contact material classification |
| **Classes** | `ambient` · `leaf` · `trunk` · `twig` |
| **Primary metric** | Macro F1 (4-class) on robot/test |
| **Test size** | *n* = 2 219 windows |
| **Paper claim (fusion v2)** | **Macro F1 = 0.850781** |
| **Protocol** | Hand-only hyperparameter selection · robot opened once · sealed claim |

---

## Highlights

- **Sealed paper path** — pure single-shot fusion cascade with frozen recipe and `CLAIM_SEAL` (no test-set hyperparameter search).
- **Clear architecture** — simplified claim code (`clear_pipeline`) matches the sealed Macro F1 bit-for-bit (tolerance \(10^{-9}\)).
- **Strong binary contact signal** — ambient vs non-ambient Macro F1 ≈ **0.966** under the same fusion claim.
- **Honest baselines** — audio-only and exploratory ablations are separated from the paper claim number.
- **Leakage controls** — specimen-group selection, no filename class features, no robot UDA as a claimed score.

---

## Sealed results (robot / test)

All numbers below are taken from sealed artifacts in this repository.  
**Evaluation split:** `robot_test_final` · **n = 2219**.

### Multimodal fusion — paper claim v2

| Metric | Value |
|:--|--:|
| **Macro F1 (4-class)** | **0.850781** |
| Accuracy (4-class) | 0.891393 |
| Macro precision | 0.882036 |
| Macro recall | 0.837340 |
| Weighted F1 | 0.890301 |
| Binary Macro F1 (ambient / contact) | 0.965652 |

| Per-class F1 | ambient | leaf | trunk | twig |
|:--|--:|--:|--:|--:|
| Fusion v2 | 0.9675 | 0.8577 | 0.8095 | 0.7684 |

| Artifact | Path |
|:--|:--|
| Seal | [`pipelines/audio_fusion_v2/fusion_v2/results/CLAIM_SEAL.json`](pipelines/audio_fusion_v2/fusion_v2/results/CLAIM_SEAL.json) |
| Full metrics | [`pipelines/audio_fusion_v2/fusion_v2/results/final_test_metrics.json`](pipelines/audio_fusion_v2/fusion_v2/results/final_test_metrics.json) |
| Targets | [`pipelines/audio_fusion_v2/shared/sealed_targets.json`](pipelines/audio_fusion_v2/shared/sealed_targets.json) |
| Recipe | `paper_claim_v2_simple_leaf_bias` · free HP: \(b_{\mathrm{leaf}} = -1.3\) (hand only) |

### Audio-only baseline (locked recipe)

| Metric | Value |
|:--|--:|
| **Macro F1 (4-class)** | **0.702672** |
| Accuracy (4-class) | 0.796755 |
| Binary Macro F1 | 0.929116 |
| Contact Macro F1 | 0.625050 |

| Artifact | Path |
|:--|:--|
| Metrics | [`pipelines/audio_fusion_v2/audio_only/results/metrics.json`](pipelines/audio_fusion_v2/audio_only/results/metrics.json) |
| Predictions | [`pipelines/audio_fusion_v2/audio_only/results/predictions.csv`](pipelines/audio_fusion_v2/audio_only/results/predictions.csv) |

> **Note.** A higher historical stack (paper claim v1 / BEST_STRICT) reaches Macro F1 **0.853226** with a more complex contact stack. The **paper-facing simplified path is v2 (0.850781)** — fewer free parameters, clearer architecture, identical clear/claim F1.

---

## Method overview

Fusion claim **v2** is **not** an end-to-end joint network. It is a **four-step cascade** over frozen audio segment predictions and multimodal detectors:

```text
Inputs
  ├─ Audio segment stack (v2 base: hard / meta soft)
  ├─ Audio features (total240, stress views) + wav2vec2
  └─ Image embeddings (CLIP)

Detectors  ──fit on HAND only──►  trunk / contact / binary scores

[1] Base ........ hier_trunk_meta_else on v2 segment artifact
[2] Amb-lift .... ambient → trunk if ts ≥ 0.5 and max(cs, bin) ≥ 0.1
[3] Secondary ... ambient → trunk if CLIP trunk ≥ 0.625 and bin ≥ 0.75
[4] Material .... leaf/twig redecode with logit bias b_leaf = −1.3
                  (trunk predictions protected)

→ 4-class labels on robot windows  ·  Macro F1 = 0.850781
```

**Implementation (paper figure):**  
[`pipelines/audio_fusion_v2/fusion_v2/code/paper_claim/clear_pipeline.py`](pipelines/audio_fusion_v2/fusion_v2/code/paper_claim/clear_pipeline.py)

**Full design write-up:**  
[`docs/paper/PAPER_CLAIM_V2_FUSION_GUIDE.md`](docs/paper/PAPER_CLAIM_V2_FUSION_GUIDE.md)

---

## Evaluation protocol

| Rule | Status |
|:--|:--|
| Hyperparameters selected on **hand** data only | Yes |
| Robot / test used **once** for the sealed claim | Yes (`CLAIM_SEAL`) |
| No HP search on robot Macro F1 | Yes |
| No class-from-filename features | Yes |
| No robot UDA as paper score | Yes |
| Grouping for CV / selection | `specimen_group` |
| Equality tolerance vs sealed targets | \(10^{-9}\) |

Exploratory runs under `experiments/` (including multi-peek multimodal searches) are **research history**. They must not replace the sealed claim numbers above.

---

## Repository structure

```text
.
├── README.md
├── pipelines/
│   └── audio_fusion_v2/          # ★ sealed audio-only + fusion v2 package
│       ├── audio_only/           # evaluate, train_chain (reference), sealed results
│       ├── fusion_v2/            # clear claim code + CLAIM_SEAL metrics
│       ├── e2e_code/             # claim runners, segment stack v2 upstream
│       ├── shared/               # metrics helpers + sealed_targets.json
│       ├── tests/
│       ├── reproduce_two.py      # dual sealed entry point
│       ├── requirements.txt
│       └── README.md
├── paper_claim/                  # importable clear claim package (workspace copy)
├── docs/
│   ├── paper/                    # fusion claim guide
│   ├── guides/                   # roadmaps, stress-view notes
│   └── attempts/                 # attempt logs & protocol catalogs
├── experiments/                  # ablations (not the paper claim path)
│   ├── audio/ | image/ | multimodal/ | segment/
│   ├── paper_claim/ | misc/
├── notebooks/
└── tools/                        # diagrams, reporting helpers
```

| Path | Role |
|:--|:--|
| `pipelines/audio_fusion_v2/` | **Authoritative paper package** (code + sealed results) |
| `paper_claim/` | Standalone clear pipeline + recipes |
| `docs/paper/` | Architecture and claim protocol documentation |
| `experiments/` | Historical experiments only |

Large feature tensors (`.npy`), model bundles, and distribution archives (`.zip` / `.tar.gz`) are **gitignored**. Ship them separately via a portable feature bundle when full re-inference is required.

---

## Getting started

### Requirements

- Python **3.10+** (3.10–3.13 tested with pinned wheels)
- CPU is sufficient for sealed evaluation

Pinned dependencies live in  
[`pipelines/audio_fusion_v2/requirements.txt`](pipelines/audio_fusion_v2/requirements.txt):

```text
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
scipy>=1.11,<1.18
pytest>=8.0
```

### Install

```bash
git clone https://github.com/taxaceaee/Audio-Visual-Contact-Classification.git
cd Audio-Visual-Contact-Classification

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r pipelines/audio_fusion_v2/requirements.txt
```

### Inspect sealed metrics (no features required)

```bash
# Fusion claim
python3 - <<'PY'
import json
from pathlib import Path
p = Path("pipelines/audio_fusion_v2/fusion_v2/results/final_test_metrics.json")
m = json.loads(p.read_text())["metrics"]
print({k: round(v, 6) for k, v in m.items()})
PY

# Audio-only baseline
python3 - <<'PY'
import json
from pathlib import Path
p = Path("pipelines/audio_fusion_v2/audio_only/results/metrics.json")
m = json.loads(p.read_text())["metrics"]
print({k: round(v, 6) for k, v in m.items()})
PY
```

### Reproduce sealed evaluation (requires feature bundle)

This git tree ships **code + seals**. Full fusion re-inference needs precomputed features (CLIP, total240, wav2vec2, v2 base), provided in a portable bundle, not in git.

```bash
cd pipelines/audio_fusion_v2
# place features under fusion_v2/data/features/ as documented in package README
python3 reproduce_two.py
# optional
python3 -m pytest tests/ -q
```

Expected: exit code **0**, both Macro F1 values matching sealed targets within \(10^{-9}\).

---

## Documentation

| Document | Description |
|:--|:--|
| [`docs/paper/PAPER_CLAIM_V2_FUSION_GUIDE.md`](docs/paper/PAPER_CLAIM_V2_FUSION_GUIDE.md) | Full fusion v2 claim guide (architecture, HP lock, seal) |
| [`pipelines/audio_fusion_v2/README.md`](pipelines/audio_fusion_v2/README.md) | Portable package usage |
| [`paper_claim/ARCHITECTURE_CLEAR.md`](paper_claim/ARCHITECTURE_CLEAR.md) | Clear-pipeline diagram notes |
| [`docs/attempts/`](docs/attempts/) | Multimodal / audio / image attempt logs |
| [`docs/guides/`](docs/guides/) | Roadmaps and stress-view explanations |
| [`experiments/README.md`](experiments/README.md) | How experiment folders relate to the claim |

---

## Citation

If you use this code or sealed results, please cite the associated paper (update when published):

```bibtex
@inproceedings{audio_visual_contact_classification,
  title     = {Audio-Visual Contact Classification for Tree Structures in Agriculture},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

Sealed claim identifiers for tables and reproducibility statements:

```text
recipe_id:   paper_claim_v2_simple_leaf_bias
protocol:    paper_claim_v2_simple_pure_single_shot
macro_f1:    0.8507810400075331
n_test:      2219
```

---

## License

Specify the project license here (e.g. MIT / Apache-2.0 / academic-only).  
Until a `LICENSE` file is added, all rights are reserved by the authors.

---

## Acknowledgments

Developed for agricultural robotics research on tree-structure contact sensing.  
Experimental ablations in `experiments/` supported the design of the sealed fusion cascade; only the sealed path is intended for paper tables.

---

## Contact

- Repository: [taxaceaee/Audio-Visual-Contact-Classification](https://github.com/taxaceaee/Audio-Visual-Contact-Classification)
- Active packaging branch: `feature/audio-fusion-v2-only`
