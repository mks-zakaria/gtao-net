# Kaggle setup on the new computer

The API key is deliberately **not** in this bundle: it is a password.

1. Copy `~/.kaggle/kaggle.json` from the old machine by hand (USB stick or password manager), or
   create a new one: kaggle.com → Settings → API → *Create New Token*.
2. `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. `pip install kaggle` then `kaggle kernels list --mine | head` to confirm it works.

## Account and data
- Kaggle user: `zakariamakhas`
- DIPSER features: the dataset `zakariamakhas/dipseer-paper1-slice` (public). The grid script reads
  its `*.zipbin` subject bundles from `/kaggle/input/dipseer-paper1-slice`.
- DAiSEE: `olgaparfenova/daisee` and `mahisharamesh/daisee` (public mirrors).
- EngageNet: `laavanayadhawan/engagenet-personalization` and `laavanayadhawan/openface-engagenet`.

## Accelerator and limits (learned the hard way, 2026-09)
- Every kernel: GPU on, TPU off. DAiSEE and EngageNet need **internet on** (they `pip install`
  mediapipe 0.10.14 and protobuf 4.25.3 at start). DIPSER needs internet off.
- Session cap is **5 h** in practice. Measured wall times for v1: DIPSER 1.0 h,
  DAiSEE 2.6 h, EngageNet (6-config scoped grid) 1.6 h.
- At most **2 GPU sessions** at a time. `kaggle kernels cancel` does not exist; cancel in the web UI.
- Pull outputs only after `kaggle kernels status <slug>` reports COMPLETE, into an **empty** folder.

## Published v1 kernels (keep them untouched; the paper cites them)
| Dataset | Public slug |
|---|---|
| DIPSER | `zakariamakhas/gtao-net-grid-v1-full-factorial` |
| DAiSEE | `zakariamakhas/gtao-net-daisee-3-stream-benchmark-v1` |
| EngageNet | `zakariamakhas/gtao-net-engagenet-scoped-p2-benchmark-v2` |
| Smoke | `zakariamakhas/dipseer-gtao-net-smoke-single-subject-multimodal` |

Note: `kaggle/v1_original/dipser_grid/kernel-metadata.json` says `"id": "gtao-grid"`, the local
working name. The public slug is the one in the table above. Do not push v1 again.
The v2 kernels in `kaggle/v2_nested/` use **new slugs**, so pushing them never overwrites v1.
