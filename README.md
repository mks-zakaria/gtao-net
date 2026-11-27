# gTAO-Net

**The Evaluation Protocol Outweighs Fusion in Classroom Attention Estimation: A Controlled
Study on DIPSER, DAiSEE and EngageNet**

> **Educational use.** Copyright (c) 2026 Zakaria Makhkhas, Amal Rifai and Abdelalim Sadiq.
> This work may be used for educational and research purposes with citation; see
> [NOTICE.md](./NOTICE.md). The code is under the MIT licence.

Current manuscript: [`paper/ieee_access/main.pdf`](./paper/ieee_access/main.pdf)
(rewritten 2026-11-27, in preparation for IEEE Access; not yet submitted). The earlier title
was *Does multimodal fusion help classroom attention estimation? A controlled diagnostic of
gated ordinal fusion on DIPSER, DAiSEE and EngageNet*.

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


## Findings on the full DIPSER release (57 students, 35,649 windows)

| Finding | Evidence |
|---|---|
| The fusion result is replicated with held-out students | 0 of 24 fusions above the best single stream (face mesh, QWK 0.116, 95% CI [0.077, 0.158]); difference -0.016 [-0.052, +0.020] |
| The sensor signal is mostly shared by the class | no stream beats a control trained on a classmate's label; label-only references reach QWK 0.372 (same lesson) and 0.378 (same moment) |
| Seven attempts to improve the sensor model fail | search, higher-order and local moments, face crop, time context, class context, three frozen face encoders (`paper/tables/fullcorpus_tests.tex`) |
| The split changes the score more than any model choice | same model: QWK 0.342 on random windows, 0.122 on held-out students, 0.085 on a held-out class; accuracy 65.4 / 57.2 / 53.9% against a 54.4% majority rate |
| Label persistence explains the sparse-probe gain | probe every 32 s: 74.2% with sensors against 72.3% for repeating the last label |

Reproduce the tables and figures of the manuscript from the stored results:

```bash
python analysis_v3/paper_fullcorpus.py    # paper/tables/fullcorpus_*.tex
python analysis_v3/published_qwk.py       # QWK of published DAiSEE systems from their confusion matrices
python analysis_v3/paper_figures.py       # paper/figures/fig_*.pdf (needs matplotlib)
```

The analysis plan with the criteria fixed before each full-release run is
[`docs/PLAN_accuracy.md`](./docs/PLAN_accuracy.md). The notebooks of the full-release
experiments are in `kaggle/` (source) with their outputs summarised in `results/v3b` ...
`results/v12_3enc`. On Kaggle, the three v2 notebooks listed below are public; the
full-release analysis notebooks are being made public, and the data-extraction notebooks
stay private because they handle the dataset access credentials and face-derived features.

## Full article workspace (added 2026-11-25)

Everything behind the current manuscript, beyond the v2 reference runs above:

| Folder | Contents |
|---|---|
| `paper/` | Manuscript LaTeX source and PDF (`main.tex`, `supplementary.tex`), tables, figures, drafts, and the IEEE Access version (`paper/ieee_access/`) |
| `kaggle/` | Source of every Kaggle kernel: v1 original, v2 nested, v3 standardised, v3b full corpus, v4 Bayesian search, v5/v6 feature re-extraction, v8 time context, v9 statistics, v10 class context, v11 sparse probe, v12 face encoders |
| `analysis_v3/` | Post-hoc analyses and the generator of the full-corpus tables (`paper_fullcorpus.py`) |
| `pc_pipeline/` | Local extraction pipeline for the full DIPSER release (download, Tchebichef features, face crops, pretrained face embeddings) |
| `results/v1`, `results/v3` … `results/v12_3enc`, `results/phase*` | Summaries, reports and logs of those runs |
| `tools/`, `env/`, `requirements-full.txt` | Helper scripts, Kaggle setup notes, dependencies for the full pipeline |

Raw data and feature caches are not included: get DIPSER, DAiSEE and EngageNet from their
publishers under their own licences. Credentials are never stored here; the extraction
kernels read them from environment variables or Kaggle Secrets.

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
│   ├── late_fusion_baseline.py  decision-level fusion (vote of the 4 single streams), §6.1
│   ├── local_cost_benchmark.py  time / memory / parameters on a CPU-only laptop (Table 5)
│   ├── local_cost_benchmark.json  measured on Apple M5, 4 threads, no GPU
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
python verify/late_fusion_baseline.py            # late-fusion baseline quoted in §6.1
python verify/local_cost_benchmark.py --kernel <gtao_grid.py from the DIPSER kernel>   # optional, needs torch
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
  title   = {The Evaluation Protocol Outweighs Fusion in Classroom Attention Estimation: A Controlled Study on DIPSER, DAiSEE and EngageNet},
  author  = {Makhkhas, Zakaria and Rifai, Amal and Sadiq, Abdelalim},
  year    = {2026},
  note    = {Manuscript in preparation},
  url     = {https://github.com/mks-zakaria/gtao-net}
}
```

## License

Code: [MIT](./LICENSE). The whole work may be used for educational and research purposes
with citation; the terms for each part are in [NOTICE.md](./NOTICE.md). The datasets remain
the property of their owners and are used under their terms (DIPSER is distributed by its
authors on request).
