#!/usr/bin/env bash
# Push the three v2 (nested early-stopping) kernels to Kaggle. Run from the bundle root.
# Respect the 2-GPU cap: push two, wait for one to finish, then push the third.
set -euo pipefail
which=${1:-}
case "$which" in
  dipser|daisee|engagenet) ;;
  *) echo "usage: tools/push_v2.sh dipser|daisee|engagenet"; exit 1;;
esac
dir=kaggle/v2_nested/$([ "$which" = dipser ] && echo dipser_grid || echo "$which")
kaggle kernels push -p "$dir"
slug=$(python3 -c "import json;print(json.load(open('$dir/kernel-metadata.json'))['id'])")
echo "pushed $slug — check with: kaggle kernels status $slug"
