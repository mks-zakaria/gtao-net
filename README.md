# gTAO-Net

**Does multimodal fusion help classroom attention estimation? A controlled diagnostic of gated
ordinal fusion on DIPSER, DAiSEE and EngageNet**

gTAO-Net is an instrumented gated ordinal fusion network (face, face-mesh, head-pose and
wrist-physiology streams, Tchebichef-moment front-ends). It is used as a diagnostic, not as a
new state of the art: four design axes are crossed in a full factorial and tested under one
fixed, subject-disjoint protocol on [DIPSER](https://arxiv.org/abs/2502.20209) (in-person
classroom) and two webcam benchmarks, [DAiSEE](https://arxiv.org/abs/1609.01885) and
[EngageNet](https://dl.acm.org/doi/10.1145/3577190.3614164).

> **v2 results (nested early stopping) are the reference results.** In the v1 runs, early
> stopping selected the epoch on the same fold that was scored, so v1 QWKs are optimistic.
> v2 selects the epoch on an inner subject split inside each training fold. v1 files are kept
> under `results/` for comparison only.

---

## Key results (v2)

| Finding | Evidence |
|---|---|
| No fusion configuration beats the best single stream on DIPSER | face mesh + soft target QWK 0.256 ± 0.068; best fusion 0.124 ± 0.071 (495 fits: 33 configs × 5 subject folds × 3 seeds) |
| The learned gate is inert | ends one-hot in 108 of 180 folds; gate vs mean pooling Wilcoxon p = 0.68 over 12 pairs |
| The global input-to-head link helps on DIPSER… | better in 10 of 12 pairs, p = 0.002 |
| …but hurts on DAiSEE | worse in 0/8 (P1) and 1/8 (P2) pairs |
| The soft multi-rater target beats CORAL on DIPSER | 7 of 8 pairs, p = 0.02; on single-label data the soft loss equals CORAL exactly |
| The absent fusion gain transfers | DAiSEE fusion vs best stream −0.001 / +0.010 QWK; EngageNet best fusion 0.170 vs face mesh 0.366 |
| Permuted-label controls stay near zero | DIPSER 0.009, DAiSEE P2 0.024, EngageNet 0.035 |

Limitation: every EngageNet clip in the cached features is one frame repeated, so EngageNet
tests static geometry only.

---

## Repository layout

```
gtao-net/
├── README.md
├── LICENSE                      MIT
├── CITATION.cff
├── assets/architecture.png
├── analysis/                    scripts that regenerate every table of the paper
│   ├── stats.py                 paired tests -> stats.json
│   ├── tables.py                LaTeX tables
│   ├── perclass.py              per-class recall -> perclass.json
│   └── comparison.py            DAiSEE comparison with published methods
└── results/
    ├── v2/                      reference results (nested early stopping)
    │   ├── DIPSER_grid_v1_metrics.json   (file name kept from v1 for the scripts)
    │   ├── DIPSER_h4_masked_eval.json
    │   ├── DAiSEE_metrics.json           P1, P2, H4, P1-Test
    │   ├── EngageNet_metrics.json
    │   ├── raw_dipser/ raw_daisee/ raw_engagenet/   per-seed out-of-fold predictions
    │   └── *.log                          Kaggle run logs
    └── *.json                   v1 results (epoch selected on the scored fold)
```

Regenerate the statistics from the v2 results:

```bash
GTAO_RESULTS=results/v2 python analysis/stats.py
GTAO_RESULTS=results/v2 python analysis/perclass.py
```

---

## Kaggle kernels

| Dataset | v2 (reference) | v1 (legacy) |
|---|---|---|
| DIPSER | [gtao-net-dipser-grid-v2-nested-early-stopping](https://www.kaggle.com/code/zakariamakhas/gtao-net-dipser-grid-v2-nested-early-stopping) | [gtao-net-grid-v1-full-factorial](https://www.kaggle.com/code/zakariamakhas/gtao-net-grid-v1-full-factorial) |
| DAiSEE | [gtao-net-daisee-v2-nested-early-stopping](https://www.kaggle.com/code/zakariamakhas/gtao-net-daisee-v2-nested-early-stopping) | [gtao-net-daisee-3-stream-benchmark-v1](https://www.kaggle.com/code/zakariamakhas/gtao-net-daisee-3-stream-benchmark-v1) |
| EngageNet | [gtao-net-engagenet-v2-nested-early-stopping](https://www.kaggle.com/code/zakariamakhas/gtao-net-engagenet-v2-nested-early-stopping) | [gtao-net-engagenet-scoped-p2-benchmark-v2](https://www.kaggle.com/code/zakariamakhas/gtao-net-engagenet-scoped-p2-benchmark-v2) |

Each kernel mounts its dataset, runs the protocol and writes its metric file to the Output tab.

## Datasets

| Dataset | Where |
|---|---|
| DIPSER | from the authors (Marquez-Carpintero et al., 2025); processed slice: [`zakariamakhas/dipseer-paper1-slice`](https://www.kaggle.com/datasets/zakariamakhas/dipseer-paper1-slice) |
| DAiSEE | [`olgaparfenova/daisee`](https://www.kaggle.com/datasets/olgaparfenova/daisee), [`mahisharamesh/daisee`](https://www.kaggle.com/datasets/mahisharamesh/daisee) |
| EngageNet | [`laavanayadhawan/engagenet-personalization`](https://www.kaggle.com/datasets/laavanayadhawan/engagenet-personalization), [`laavanayadhawan/openface-engagenet`](https://www.kaggle.com/datasets/laavanayadhawan/openface-engagenet) |

---

## Cite this work

```bibtex
@article{makhkhas2026gtaonet,
  title   = {Does Multimodal Fusion Help Classroom Attention Estimation? A Controlled Diagnostic of Gated Ordinal Fusion on DIPSER, DAiSEE and EngageNet},
  author  = {Makhkhas, Zakaria and Rifai, Amal and Sadiq, Abdelalim},
  year    = {2026},
  note    = {Submitted to Intelligent Systems with Applications},
  url     = {https://github.com/mks-zakaria/gtao-net}
}
```

## License

Code and results: [MIT](./LICENSE). The datasets remain the property of their owners and are
used under their terms.
