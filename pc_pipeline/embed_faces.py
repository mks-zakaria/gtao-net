"""Phase 10B, step 2 (PC, GPU): pretrained face embeddings from the crops of extract_face_crops.py.

    python embed_faces.py --crops C:\\dipser_crops --out C:\\dipser_crops\\face_embeddings.npz

Encoders (frozen, no training): resnet18 (ImageNet, torchvision, required), dinov2_vits14
(torch.hub, optional), hsemotion (EfficientNet-B0 trained on facial expressions, package
`hsemotion`, optional). An optional encoder that cannot be loaded is skipped and reported.
Per window: mean and standard deviation of the frame embeddings over the frames that show
the student's face (zeros if none) -> emb_<encoder> [n, 2 * dim]. Also merged: keys,
subjects, y, face_frac, votes [n, R] and rater_names (union over subjects, 0 = no vote).
Works on CPU too (slow). --fp32 disables fp16 autocast (on the GTX 1650, fp16 gave NaN for
resnet18 on 0.17% of the windows). --update FILE replaces or adds the encoders run here in an
existing output (keys must be identical) and keeps its other encoders.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder(name, device):
    if name == "resnet18":
        import torchvision
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
    elif name == "dinov2_vits14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif name == "hsemotion":
        from hsemotion.facial_emotions import HSEmotionRecognizer
        fer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew", device=str(device))
        model = fer.model
        model.classifier = torch.nn.Identity()
    else:
        raise ValueError(name)
    return model.eval().to(device)


@torch.no_grad()
def embed(model, images, device, batch, fp16=True):
    """images: uint8 [m, 112, 112, 3] -> float32 [m, dim]."""
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = (x - MEAN.to(device)) / STD.to(device)
        with torch.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda"):
            z = model(x)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crops", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoders", default="resnet18,dinov2_vits14,hsemotion")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--only", default=None, help="restrict to file names containing this text")
    parser.add_argument("--fp32", action="store_true", help="no fp16 autocast")
    parser.add_argument("--update", default=None, help="existing output to update in place of a new file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(Path(args.crops).glob("*__subject_*.npz"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    print(f"{len(files)} crop files, device {device}", flush=True)

    # labels and votes first (no GPU)
    keys, subjects, labels, face_frac, vote_parts = [], [], [], [], []
    for f in files:
        with np.load(f) as z:
            keys.append(z["keys"]), subjects.append(z["subjects"]), labels.append(z["y"])
            valid, has_face = z["valid"], z["has_face"]
            face_frac.append(has_face.sum(1) / np.maximum(1, valid.sum(1)))
            vote_parts.append((list(z["rater_names"].astype(str)), z["votes"]))
    names = sorted(set(n for part, _ in vote_parts for n in part))
    votes = []
    for part, v in vote_parts:
        full = np.zeros((len(v), len(names)), np.int8)
        for r, name in enumerate(part):
            full[:, names.index(name)] = v[:, r]
        votes.append(full)
    result = {"keys": np.concatenate(keys), "subjects": np.concatenate(subjects),
              "y": np.concatenate(labels), "face_frac": np.concatenate(face_frac).astype(np.float32),
              "votes": np.concatenate(votes), "rater_names": np.asarray(names)}
    print(f"{len(result['keys'])} windows, raters: {names}", flush=True)

    report = {"windows": int(len(result["keys"])), "device": str(device), "raters": names, "encoders": {}}
    if args.update:
        with np.load(args.update, allow_pickle=False) as old:
            if not np.array_equal(old["keys"], result["keys"]):
                raise SystemExit(f"--update: keys of {args.update} differ from the crops")
            result.update({k: old[k] for k in old.files if k.startswith("emb_")})
        report["encoders"] = json.loads(Path(args.update).with_suffix(".json").read_text())["encoders"]
    for name in [e for e in args.encoders.split(",") if e]:
        start = time.time()
        try:
            model = load_encoder(name, device)
        except Exception as error:
            print(f"encoder {name}: SKIPPED ({type(error).__name__}: {error})", flush=True)
            report["encoders"][name] = {"status": "skipped", "error": f"{type(error).__name__}: {error}"}
            if name == "resnet18":
                raise
            continue
        rows = []
        for k, f in enumerate(files):
            with np.load(f) as z:
                crops, has_face = z["crops"], z["has_face"]
            z_frames = embed(model, crops[has_face], device, args.batch, fp16=not args.fp32)
            dim = z_frames.shape[1] if len(z_frames) else rows[-1].shape[1] // 2 if rows else 1
            out = np.zeros((len(crops), 2 * dim), np.float32)
            position = np.cumsum(has_face.ravel()).reshape(has_face.shape) - 1
            for w in range(len(crops)):
                idx = position[w][has_face[w]]
                if len(idx):
                    out[w, :dim] = z_frames[idx].mean(0)
                    out[w, dim:] = z_frames[idx].std(0)
            rows.append(out)
            if (k + 1) % 25 == 0 or k + 1 == len(files):
                print(f"  {name}: {k + 1}/{len(files)} files ({time.time() - start:.0f}s)", flush=True)
        width = max(r.shape[1] for r in rows)
        rows = [r if r.shape[1] == width else np.zeros((len(r), width), np.float32) for r in rows]
        result[f"emb_{name}"] = np.concatenate(rows)
        nonfinite = int((~np.isfinite(result[f"emb_{name}"]).all(1)).sum())
        report["encoders"][name] = {"status": "ok", "dim": int(width), "seconds": round(time.time() - start, 1),
                                    "precision": "fp32" if args.fp32 or device.type != "cuda" else "fp16",
                                    "nonfinite_windows": nonfinite}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **result)
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
