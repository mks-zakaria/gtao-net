#!/usr/bin/env bash
# CPU smoke test of the DIPSER grid (FAST mode, a few synthetic subjects).
# Needs a Python with env/requirements-local-smoke.txt and DIPSEER_DATA_ROOT pointing at a folder
# of *.zipbin subject bundles (e.g. a few files downloaded from zakariamakhas/dipseer-paper1-slice,
# or synthetic ones: python3 tools/make_synthetic_dipser.py <dir>).
set -euo pipefail
: "${DIPSEER_DATA_ROOT:?set DIPSEER_DATA_ROOT to a folder with *.zipbin files}"
PY=${PY:-python3}
rm -rf /tmp/gtao_grid_out
GRID_FAST=1 "$PY" kaggle/v2_nested/dipser_grid/gtao_grid.py | tail -5
echo "--- v2 with GTAO_NESTED=0 must equal v1 exactly:"
rm -rf /tmp/gtao_grid_out; GRID_FAST=1 GTAO_NESTED=0 "$PY" kaggle/v2_nested/dipser_grid/gtao_grid.py >/dev/null
cp /tmp/gtao_grid_out/grid_v1_metrics.json /tmp/_v2n0.json
rm -rf /tmp/gtao_grid_out; GRID_FAST=1 "$PY" kaggle/v1_original/dipser_grid/gtao_grid.py >/dev/null
python3 - <<'PYEOF'
import json
v2 = {r["config"]: r["qwk_per_seed"] for r in json.load(open("/tmp/_v2n0.json"))["summary"]["rows"]}
v1 = {r["config"]: r["qwk_per_seed"] for r in json.load(open("/tmp/gtao_grid_out/grid_v1_metrics.json"))["summary"]["rows"]}
# v2 adds the unimodal_<s>_cw_soft baselines; every config v1 has must match exactly
missing = [c for c in v1 if c not in v2]
same = not missing and all(v1[c] == v2[c] for c in v1)
print(f"{len(v1)} shared configs:", "IDENTICAL" if same else f"DIFFERENT (missing {missing}) — investigate before trusting v2")
print("extra in v2:", sorted(set(v2) - set(v1)))
PYEOF
