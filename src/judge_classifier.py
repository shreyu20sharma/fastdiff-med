"""ResNet-18 for 32x32 lesions: used (a) as the "judge" that scores the class accuracy of
generated samples and (b) as the downstream classifier in the augmentation experiment.

    python -m src.judge_classifier --ckpt_dir <ckpt_dir>      # trains & saves judge.pt

The judge uses class-weighted cross-entropy so that it is reasonably accurate on rare
classes too (otherwise "class accuracy" of rare-class samples would mostly measure the judge).
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.data import NUM_CLASSES, SHORT_NAMES, load_tensors, random_flip
from src.utils import atomic_save, autocast, get_device, set_seed


def build_resnet18(num_classes=NUM_CLASSES):
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # small-image stem
    m.maxpool = nn.Identity()
    return m


# ----------------------------------------------------------------------------- metrics
def confusion(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = confusion(y_true, y_pred, num_classes)
    support = cm.sum(1)
    tp = np.diag(cm).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    pred_cnt = cm.sum(0)
    precision = np.divide(tp, pred_cnt, out=np.zeros_like(tp), where=pred_cnt > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return dict(accuracy=float(tp.sum() / max(cm.sum(), 1)),
                balanced_accuracy=float(recall[present].mean()),
                macro_f1=float(f1[present].mean()),
                recall=recall)


@torch.no_grad()
def predict(model, x, batch=256):
    model.eval()
    out = []
    for i in range(0, x.shape[0], batch):
        with autocast(x.device):
            out.append(model(x[i:i + batch]).argmax(1))
    return torch.cat(out).cpu().numpy()


# ----------------------------------------------------------------------------- training
def train_classifier(x_tr, y_tr, x_val, y_val, epochs=30, batch=128, lr=1e-3, wd=0.05,
                     class_weighted=False, seed=0, verbose=True):
    """Train ResNet-18 on GPU-resident tensors; keep the epoch with best val balanced accuracy."""
    device = x_tr.device
    set_seed(seed)
    model = build_resnet18().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, x_tr.shape[0] // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.15)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weight = None
    if class_weighted:
        cnt = torch.bincount(y_tr, minlength=NUM_CLASSES).float().clamp(min=1)
        weight = (cnt.sum() / cnt) ** 0.5
        weight = (weight / weight.mean()).to(device)

    best, best_state = -1.0, None
    n = x_tr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            xb, yb = random_flip(x_tr[idx]), y_tr[idx]
            with autocast(device, amp):
                loss = F.cross_entropy(model(xb), yb, weight=weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        m = metrics(y_val.cpu().numpy(), predict(model, x_val))
        if m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1:3d}/{epochs} val bal-acc {m['balanced_accuracy']:.3f} "
                  f"(best {best:.3f})", flush=True)
    model.load_state_dict(best_state)
    return model.eval()


def load_judge(path, device=None):
    device = device or get_device()
    m = build_resnet18()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    device = get_device()
    x_tr, y_tr = load_tensors("train", device)
    x_val, y_val = load_tensors("val", device)
    x_te, y_te = load_tensors("test", device)
    t0 = time.time()
    model = train_classifier(x_tr, y_tr, x_val, y_val, epochs=args.epochs, batch=args.batch, class_weighted=True,
                             seed=args.seed)
    m = metrics(y_te.cpu().numpy(), predict(model, x_te))
    print(f"judge test: acc {m['accuracy']:.3f}  balanced acc {m['balanced_accuracy']:.3f}  "
          f"macro-F1 {m['macro_f1']:.3f}  ({time.time() - t0:.0f}s)")
    atomic_save(dict(model=model.state_dict(), test_metrics={k: v for k, v in m.items() if k != "recall"}),
                os.path.join(args.ckpt_dir, "judge.pt"))
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "judge.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ("accuracy", "balanced_accuracy", "macro_f1"):
            w.writerow([k, f"{m[k]:.4f}"])
        for c, r in zip(SHORT_NAMES, m["recall"]):
            w.writerow([f"recall_{c}", f"{r:.4f}"])


if __name__ == "__main__":
    main()
