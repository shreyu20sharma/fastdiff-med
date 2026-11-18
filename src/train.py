"""Train one ablation run (A-D). Same U-Net and recipe for every run; only schedule/target differ.

    python -m src.train --run D --steps 12000 --ckpt_dir /content/drive/MyDrive/fastdiff-med/ckpt

Writes <ckpt_dir>/<run>/last.pt (full state, resumable, every --save_every steps),
<ckpt_dir>/<run>/ema.pt (EMA weights only, used for sampling) and loss.csv.
Re-running the same command resumes from last.pt (handy after a Colab disconnect).
"""
import argparse
import copy
import csv
import os
import time

import torch

from src.data import NUM_CLASSES, load_tensors, random_flip
from src.diffusion import GaussianDiffusion
from src.unet import UNet
from src.utils import RUNS, atomic_save, autocast, ema_update, get_device, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=list(RUNS), help="shortcut that sets --schedule/--pred")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--pred", choices=["eps", "v"], default="v")
    p.add_argument("--run_name", default=None)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--label_dropout", type=float, default=0.1)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    a = p.parse_args(argv)
    if a.run:
        a.schedule, a.pred = RUNS[a.run]
    a.run_name = a.run_name or a.run or f"{a.schedule}_{a.pred}"
    return a


def main(argv=None):
    args = parse_args(argv)
    device = get_device()
    set_seed(args.seed)
    amp = not args.no_amp and device.type == "cuda"
    run_dir = os.path.join(args.ckpt_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    x_train, y_train = load_tensors("train", device)
    n = x_train.shape[0]
    print(f"[{args.run_name}] schedule={args.schedule} pred={args.pred} | {n} train images | "
          f"device={device} amp={amp}")

    diffusion = GaussianDiffusion(args.schedule, args.pred)
    model = UNet(num_classes=NUM_CLASSES, base=args.base).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"[{args.run_name}] U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    config = dict(schedule=args.schedule, pred=args.pred, T=diffusion.T, base=args.base,
                  num_classes=NUM_CLASSES, run_name=args.run_name, distilled=False)
    step = 0
    last_path = os.path.join(run_dir, "last.pt")
    if os.path.exists(last_path) and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step = ck["step"]
        print(f"[{args.run_name}] resumed from step {step}")

    log_path = os.path.join(run_dir, "loss.csv")
    if step == 0 or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "it_per_s"])

    def save(path, full):
        obj = dict(config=dict(config, train_steps=step), model=ema.state_dict(), step=step)
        if full:
            obj.update(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                       scaler=scaler.state_dict())
        atomic_save(obj, path)

    model.train()
    t0, running, k = time.time(), 0.0, 0
    while step < args.steps:
        idx = torch.randint(0, n, (args.batch,), device=device)
        xb, yb = random_flip(x_train[idx]), y_train[idx]
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast(device, amp):
            loss = diffusion.training_loss(model, xb, yb, args.label_dropout)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))

        running += loss.item()
        k += 1
        if step % args.log_every == 0 or step == args.steps:
            dt = time.time() - t0
            ips = k / max(dt, 1e-9)
            eta = (args.steps - step) / max(ips, 1e-9) / 60
            print(f"[{args.run_name}] step {step:6d}/{args.steps} loss {running / k:.4f} "
                  f"{ips:.1f} it/s  ETA {eta:.1f} min", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{running / k:.6f}", f"{ips:.2f}"])
            t0, running, k = time.time(), 0.0, 0
        if step % args.save_every == 0 or step == args.steps:
            save(last_path, full=True)
            save(os.path.join(run_dir, "ema.pt"), full=False)

    print(f"[{args.run_name}] done -> {os.path.join(run_dir, 'ema.pt')}")


if __name__ == "__main__":
    main()
