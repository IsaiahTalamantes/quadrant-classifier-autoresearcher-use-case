import os
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import habana_frameworks.torch.core as htcore
import json

# ---------------------------------------------------------------#
# Fixed configuration
# ---------------------------------------------------------------#
# NOTE FOR THE AGENT: do not turn these into required CLI args.
# This script must run as `python3 quadrant_classifier.py` with no
# interactive input, so any new knobs you add must have defaults.
DATA_DIR = os.environ.get("QUADRANT_DATA_DIR", "/data/pics")
OUT_DIR = os.environ.get("QUADRANT_OUT_DIR", "/data/run_output")
CHECKPOINT_PATH = os.path.join(OUT_DIR, "model.pt")

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 30
SEED = int(os.environ.get("SEED", 0))

FNAME_RE = re.compile(r"^pic\d+_([0-9.eE+\-]+)_([0-9.eE+\-]+)\.jpg$")


def parse_filenames(data_dir):
    records = []
    for f in sorted(os.listdir(data_dir)):
        m = FNAME_RE.match(f)
        if not m:
            continue
        v1, v2 = float(m.group(1)), float(m.group(2))
        records.append({"file": f, "v1": v1, "v2": v2})
    if not records:
        raise RuntimeError(f"No files match this pattern from {data_dir}")
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

    print(f"v1 median threshold: {v1_thresh:.4g}")
    print(f"log10(v2) median threshold: {v2_thresh:.4g} (v2 ~= {10**v2_thresh:.4g})")
    counts = np.bincount([r["label"] for r in records], minlength=4)
    print(f"Class counts: {dict(enumerate(counts))}")
    return records


class QuadrantDataset(Dataset):
    def __init__(self, records, data_dir, transform):
        self.records = records
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(os.path.join(self.data_dir, r["file"])).convert("RGB")
        img = self.transform(img)
        return img, r["label"], r["file"]


class ImprovedCNN(nn.Module):
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
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


def train(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            htcore.mark_step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} loss={loss_sum/total:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return model, best_val_acc, best_state


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("hpu")
    print("Using device:", device)

    records = parse_filenames(DATA_DIR)
    records = assign_quadrants(records)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = QuadrantDataset(records, DATA_DIR, transform)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ImprovedCNN(num_classes=4).to(device)

    model, best_val_acc, best_state = train(
        model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE
    )

    print(f"Best val accuracy: {best_val_acc:.3f}")

    torch.save(best_state, CHECKPOINT_PATH)

    print(json.dumps({"status": "ok", "metrics": {"val_acc": best_val_acc}}))


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)
