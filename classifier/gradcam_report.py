import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import habana_frameworks.torch.core as htcore
import argparse
import re

FNAME_RE = re.compile(r"^pic\d+_([0-9.eE+\-]+)_([0-9.eE+\-]+)\.jpg$")


def parse_filenames(data_dir):
    records = []
    for f in sorted(os.listdir(data_dir)):
        m = FNAME_RE.match(f)
        if not m:
            continue
        v1, v2 = float(m.group(1)), float(m.group(2))
        records.append({"file": f, "v1": v1, "v2": v2})
    return records


def assign_quadrants(records):
    v1 = np.array([r["v1"] for r in records])
    v2 = np.array([r["v2"] for r in records])
    log_v2 = np.log10(np.clip(v2, 1e-12, None))
    v1_thresh = np.median(v1)
    v2_thresh = np.median(log_v2)
    for r, lv2 in zip(records, log_v2):
        lo_v1 = r["v1"] <= v1_thresh
        lo_v2 = lv2 <= v2_thresh
        r["label"] = (0 if lo_v1 else 2) + (0 if lo_v2 else 1)
    return records


class ImprovedCNN(nn.Module):
    # CHANGED: matches the architecture actually saved in model_final_best.pt
    # (the autoresearch loop's winning candidate), not the original small
    # BaselineCNN. Loading a checkpoint into a mismatched architecture fails
    # with a state_dict shape error, so this class must always match whatever
    # checkpoint --checkpoint points to.
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


def grad_cam_full(model, img_tensor, target_class, device):
    model.eval()
    grads = {}

    def save_grad(grad):
        grads["value"] = grad

    x = img_tensor.unsqueeze(0).to(device)
    x.requires_grad_(True)

    feats = model.features(x)
    feats.register_hook(save_grad)
    pooled = model.classifier[0](feats)
    flat = model.classifier[1](pooled)
    dropped = model.classifier[2](flat)
    logits = model.classifier[3](dropped)

    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    grad_vals = grads["value"][0]
    feat_vals = feats.detach()[0]
    weights = grad_vals.mean(dim=(1, 2))

    cam = torch.zeros(feat_vals.shape[1:], device=device)
    for c, w in enumerate(weights):
        cam += w * feat_vals[c]
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), logits.detach().cpu().numpy()[0]


def save_gradcam_overlay(orig_img_path, cam, out_path):
    img = Image.open(orig_img_path).convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
    ) / 255.0
    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    plt.imsave(out_path, overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/pics")
    ap.add_argument("--checkpoint", required=True,
                     help="path to a model.pt saved by quadrant_classifier.py")
    ap.add_argument("--out_dir", required=True,
                     help="e.g. /data/run_output/gradcam_before or .../gradcam_after")
    ap.add_argument("--n_gradcam", type=int, default=8)
    ap.add_argument("--fixed_sample_file", default="/data/run_output/gradcam_sample.txt",
                     help="Shared filename list so before/after runs use identical images.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("hpu")
    records = parse_filenames(args.data_dir)
    records = assign_quadrants(records)
    fname_to_record = {r["file"]: r for r in records}
    all_fnames = list(fname_to_record.keys())

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = ImprovedCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if os.path.exists(args.fixed_sample_file):
        with open(args.fixed_sample_file) as f:
            fixed_fnames = [line.strip() for line in f if line.strip()]
        sample_fnames = [fn for fn in fixed_fnames if fn in fname_to_record]
        print(f"Loaded {len(sample_fnames)} fixed Grad-CAM filenames.")
    else:
        sample_fnames = random.sample(all_fnames, min(args.n_gradcam, len(all_fnames)))
        with open(args.fixed_sample_file, "w") as f:
            f.write("\n".join(sample_fnames))
        print(f"No fixed sample file found. Picked {len(sample_fnames)} images and saved the list.")

    for fname in sample_fnames:
        label = fname_to_record[fname]["label"]
        img = Image.open(os.path.join(args.data_dir, fname)).convert("RGB")
        img_tensor = transform(img)
        cam, logits = grad_cam_full(model, img_tensor, target_class=label, device=device)
        pred = int(np.argmax(logits))
        out_path = os.path.join(args.out_dir, f"{fname}_true{label}_pred{pred}.png")
        save_gradcam_overlay(os.path.join(args.data_dir, fname), cam, out_path)

    print(f"Done. Grad-CAM overlays written to {args.out_dir}")


if __name__ == "__main__":
    main()
