# v9: every statistic, recomputed on Kaggle (`gtao-v9-statistics`)

Kernel `zakariamakhas/gtao-v9-statistics` (`kaggle/v9_analysis`) runs the committed analysis
scripts, with their original command lines, on Kaggle kernel outputs and the private corpus
dataset only. Outputs here; logs (stdout of each script) in `results/logs/`.

Comparison with the copies computed earlier on the Mac (2026-11-21):
- rich_vs_paper.json, phase7.json: byte-identical.
- v3/report.json, phase1b/whowhen.json, v3b/gate3.json: same values up to the last float
  digit (max difference 6e-16, CPU/numpy rounding); gate3.json key order also differs
  (built from a Python set).
- split_leakage.json (retrains the models): random split identical (65.4%, QWK 0.342);
  person-grouped 57.2% / QWK 0.122 on Kaggle vs 57.3% / 0.114 on the Mac (training
  non-determinism across CPUs). Conclusion unchanged. The Kaggle numbers are of record.
- label_prior (was printed only): experiment prior QWK 0.372, class-moment prior 0.378,
  variance shares person 0.13 / group-experiment 0.20 / session 0.49 (as reported).
- phase8.json (time context, statistics kernel v3): same values as the Mac copy.
- phase9.json (class context, statistics kernel v4): same values as the Mac copy.
