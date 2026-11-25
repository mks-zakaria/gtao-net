#!/usr/bin/env bash
# Pull a COMPLETE v2 kernel and file its metrics under results/v2 with the names the
# paper scripts expect. Run from the bundle root.
set -euo pipefail
which=${1:?usage: tools/pull_v2.sh dipser|daisee|engagenet}
dir=kaggle/v2_nested/$([ "$which" = dipser ] && echo dipser_grid || echo "$which")
slug=$(python3 -c "import json;print(json.load(open('$dir/kernel-metadata.json'))['id'])")
status=$(kaggle kernels status "$slug")
echo "$status"
echo "$status" | grep -qi complete || { echo "not COMPLETE yet — not pulling"; exit 1; }
tmp=$(mktemp -d); kaggle kernels output "$slug" -p "$tmp" >/dev/null
mkdir -p results/v2
case "$which" in
  dipser)    cp "$tmp/grid_v1_metrics.json"       results/v2/DIPSER_grid_v1_metrics.json
             cp "$tmp/h4_masked_eval.json"         results/v2/DIPSER_h4_masked_eval.json ;;
  daisee)    cp "$tmp/daisee_gt_metrics.json"      results/v2/DAiSEE_metrics.json ;;
  engagenet) cp "$tmp/engagenet_gt_metrics.json"   results/v2/EngageNet_metrics.json ;;
esac
cp "$tmp"/*.log results/v2/ 2>/dev/null || true
# keep per-config/per-seed predictions + gate traces (needed for confusion matrices, gate plots)
mkdir -p "results/v2/raw_$which"
dest="$PWD/results/v2/raw_$which"
(cd "$tmp" && find . -name 'oof.json' -exec cp --parents {} "$dest/" \;)
n=$(find "$dest" -name 'oof.json' | wc -l)
echo "kept $n oof.json file(s) in $dest"
[ "$n" -gt 0 ] || echo "WARNING: no oof.json found - per-class recall and gate plots will be unavailable"
echo "filed into results/v2/ (raw output kept in $tmp)"
