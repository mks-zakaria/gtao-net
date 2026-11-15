# gTAO-Net

**Does multimodal fusion help classroom attention estimation? A controlled diagnostic of gated
ordinal fusion on DIPSER, DAiSEE and EngageNet**

gTAO-Net is an instrumented gated ordinal fusion network (face, face-mesh, head-pose and
wrist-physiology streams, Tchebichef-moment front-ends). It is used as a diagnostic, not as a
new state of the art: four design axes are crossed in a full factorial and tested under one
fixed, subject-disjoint protocol on [DIPSER](https://arxiv.org/abs/2502.20209) (in-person
classroom) and two webcam benchmarks, [DAiSEE](https://arxiv.org/abs/1609.01885) and
[EngageNet](https://dl.acm.org/doi/10.1145/3577190.3614164).

**Links**
- Results and verification: https://github.com/mks-zakaria/gtao-net
- Kaggle kernels (training code, v2 reference runs):
  [DIPSER](https://www.kaggle.com/code/zakariamakhas/gtao-net-dipser-grid-v2-nested-early-stopping) ·
  [DAiSEE](https://www.kaggle.com/code/zakariamakhas/gtao-net-daisee-v2-nested-early-stopping) ·
  [EngageNet](https://www.kaggle.com/code/zakariamakhas/gtao-net-engagenet-v2-nested-early-stopping)
- DIPSER processed slice: https://www.kaggle.com/datasets/zakariamakhas/dipseer-paper1-slice

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

Limitation: DIPSER folds are grouped by session-subject (`group/experiment/subject`). The slice
holds two sessions of the same 20 students, so for 5–20% of scored windows (depending on the
seed) the same student's other session is in training. `verify/person_disjoint_sensitivity.py`
rescores every run on person-disjoint windows only: no fusion beats the best stream (0/24),
the gate stays inert (p = 0.79) and the link still helps (10/12, p = 0.021); the soft-vs-CORAL
gain weakens to 6/8 pairs (p = 0.078).

---

## Repository layout

```
gtao-net/
├── README.md
├── LICENSE                      MIT
├── CITATION.cff
├── assets/architecture.png
├── requirements.txt             numpy, scipy (verification and analysis)
├── SHA256SUMS                   checksums of every result file
├── verify/                      independent verification (see below)
│   ├── verify_results.py        recomputes every reported number from the raw predictions
│   ├── person_disjoint_sensitivity.py   DIPSER rescored on person-disjoint windows
│   ├── claims.json              numbers quoted in the abstract
│   └── dipser_session_subjects.txt      the 40 session-subjects the DIPSER folds are dealt from
├── tables/                      the LaTeX tables exactly as printed in the paper
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

## Verify the results

```bash
pip install -r requirements.txt
python verify/verify_results.py                  # ~1 min, exit code 0 = all checks passed
python verify/person_disjoint_sensitivity.py     # optional DIPSER sensitivity analysis
sha256sum -c SHA256SUMS                          # file integrity (shasum -a 256 -c on macOS)
```

`verify_results.py` does not reuse the training or analysis code for its core checks. It

1. recomputes QWK, accuracy, MAE, n and n_subjects of all 288 out-of-fold prediction files
   (3 seeds × every configuration, DIPSER / DAiSEE P1, P2, P1-Test / EngageNet);
2. checks every per-seed QWK, mean and standard deviation in the four metrics files against them;
3. rebuilds every outer fold with the kernels' rule (sorted subjects, `numpy.random.default_rng(seed)`
   shuffle, round-robin into 5 folds), checks the fold sizes against the recorded `n_val`, and checks
   that no inner-validation subject used for epoch selection belongs to the scored fold;
4. checks the permuted-label controls (|QWK| ≤ 0.10) and the nested-split lines in the Kaggle logs;
5. reruns `analysis/*.py` in a temporary folder and requires `analysis/*.json` and every
   `tables/*.tex` to come out byte-identical;
6. compares the numbers quoted in the abstract (`verify/claims.json`) with the results.

The same checks run on every push (`.github/workflows/verify.yml`).

To regenerate the tables yourself: `GTAO_RESULTS=results/v2 python analysis/stats.py`, then
`tables.py`, `perclass.py` and `comparison.py` (they write into `tables/`).

The training code is the three public Kaggle kernels below; the versions that produced
`results/v2` are DIPSER version 1, DAiSEE version 3 (versions 2 + 3, resumed), EngageNet version 2.

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
