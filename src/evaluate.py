"""Benchmark every configuration in the ablation: FID, judged class accuracy, network calls,
batched ms/image and single-request latency.

    python -m src.evaluate --ckpt_dir <ckpt_dir> [--n_fid 5000] [--n_fid_ddpm 1000]

Outputs: results/benchmark.csv, results/fid_vs_steps.png, results/latency.png,
results/loss_curves.png, results/grids/*.png, results/sample_grid_compare.png,
and docs/phase3_results.md (via src.report).

Label distribution: FID compares the generated distribution with the real test set, which is
~67% nevi. Generating class-balanced samples would inflate FID purely through the class-mix
mismatch, so by default labels follow the training class frequencies (--labels train).
Class accuracy is reported per class and averaged (balanced), so rare classes still count.
Use --labels balanced to follow the original plan literally.
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from src import report
from src.data import NUM_CLASSES, SHORT_NAMES, class_counts, load_numpy, tensor_to_uint8
from src.judge_classifier import load_judge, predict
from src.sampling import nfe_per_image, sample_images, sync, timed
from src.utils import RUNS, get_device, load_model

# ------------------------------------------------------------------ chart style (reference palette)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"D": "#2a78d6", "A": "#eb6834", "B": "#1baf7a", "C": "#eda100", "E": "#e87ba4"}
LABELS = {"A": "A  linear + ε (baseline)", "B": "B  cosine + ε", "C": "C  linear + v",
          "D": "D  cosine + v (Stage 1)", "E": "E  distilled student (Stage 2)"}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


class U8Dataset(Dataset):
    """uint8 N x H x W x 3 numpy -> items 3 x H x W uint8 tensors (torch-fidelity format)."""

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return torch.from_numpy(self.imgs[i]).permute(2, 0, 1).contiguous()


def compute_fid(gen_u8, real_u8, cuda):
    import torch_fidelity

    m = torch_fidelity.calculate_metrics(input1=U8Dataset(gen_u8), input2=U8Dataset(real_u8),
                                         cuda=cuda, fid=True, verbose=False)
    return float(m["frechet_inception_distance"])


def make_labels(n, mode, train_labels):
    if mode == "balanced":
        return torch.arange(n) % NUM_CLASSES
    p = class_counts(train_labels) / len(train_labels)
    cnt = np.floor(p * n).astype(int)
    cnt[np.argsort(-(p * n - cnt))[: n - cnt.sum()]] += 1  # largest remainders
    cnt = np.maximum(cnt, 1)
    lab = np.concatenate([np.full(c, k) for k, c in enumerate(cnt)])
    return torch.from_numpy(lab).long()


def find_configs(ckpt_dir, steps_list, guidance):
    cfgs = []
    for run in ["A", "B", "C", "D"]:
        path = os.path.join(ckpt_dir, run, "ema.pt")
        if not os.path.exists(path):
            print(f"  (skipping run {run}: {path} not found)")
            continue
        if run == "A":
            cfgs.append(dict(run=run, path=path, sampler="ddpm", steps=1000, guidance=guidance))
        for s in steps_list:
            cfgs.append(dict(run=run, path=path, sampler="ddim", steps=s, guidance=guidance))
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "distill", "student_*step.pt"))):
        s = int(os.path.basename(path).split("_")[1].replace("step.pt", ""))
        cfgs.append(dict(run="E", path=path, sampler="ddim", steps=s, guidance=1.0))
    for c in cfgs:
        c["name"] = f"{c['run']}-{c['sampler']}-{c['steps']}"
    return cfgs


def save_grid(x, path, nrow=8):
    import torchvision

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image((x + 1) / 2, path, nrow=nrow, padding=1)


def grid_labels(per_class=8):
    return torch.arange(NUM_CLASSES).repeat_interleave(per_class)


# ------------------------------------------------------------------ charts
def plot_fid_vs_steps(rows, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    for run in ["A", "B", "C", "D", "E"]:
        pts = sorted((r["steps"], r["fid"]) for r in rows if r["run"] == run and r["sampler"] == "ddim"
                     and r["fid"] == r["fid"])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=COLORS[run], lw=2, ms=6, mec=SURFACE, mew=1.5, label=LABELS[run])
        ax.annotate(run, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, fontweight="bold")
    base = [r for r in rows if r["sampler"] == "ddpm" and r["fid"] == r["fid"]]
    if base:
        ax.axhline(base[0]["fid"], color=INK2, lw=1.2, ls="--")
        ax.annotate(f"A  DDPM 1000 steps: FID {base[0]['fid']:.1f}", (1.0, base[0]["fid"]),
                    xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ticks = sorted({r["steps"] for r in rows if r["sampler"] == "ddim"})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("sampling steps (DDIM, log scale)")
    ax.set_ylabel("FID  (lower is better)")
    ax.set_title("Sample quality vs. number of steps", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_latency(rows, path):
    import matplotlib.pyplot as plt

    keep = [r for r in rows if r["name"] in ("A-ddpm-1000", "A-ddim-20", "D-ddim-20", "D-ddim-10")
            or r["run"] == "E"]
    keep = sorted(keep, key=lambda r: -r["single_latency_ms"])
    if not keep:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(keep) + 1.4), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    names = [f"{r['name']}  ({r['nfe']} calls)" for r in keep]
    vals = [r["single_latency_ms"] for r in keep]
    ax.barh(names, vals, color=COLORS["D"], height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:,.0f} ms" if v >= 10 else f"{v:.1f} ms", (v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("single-request latency, 1 image (ms, log scale)")
    ax.set_title("Time to generate one image", loc="left", fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_loss_curves(ckpt_dir, path):
    import matplotlib.pyplot as plt

    runs = [r for r in ["A", "B", "C", "D"] if os.path.exists(os.path.join(ckpt_dir, r, "loss.csv"))]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3 * len(runs), 2.8), dpi=150,
                             facecolor=SURFACE, squeeze=False)
    for ax, run in zip(axes[0], runs):
        style_axes(ax)
        data = np.genfromtxt(os.path.join(ckpt_dir, run, "loss.csv"), delimiter=",", names=True)
        data = np.atleast_1d(data)
        ax.plot(data["step"], data["loss"], color=COLORS[run], lw=2)
        ax.set_title(LABELS[run], loc="left", fontsize=9)
        ax.set_xlabel("step")
    axes[0][0].set_ylabel("training loss")
    fig.suptitle("Training loss (ε- and v-targets have different scales; compare shapes, not levels)",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_sample_compare(grids, path):
    import matplotlib.pyplot as plt

    if not grids:
        return
    per_class = 6
    fig, axes = plt.subplots(1, len(grids), figsize=(2.6 * len(grids), 3.4), dpi=150, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (title, x) in zip(axes, grids):
        x = x.view(NUM_CLASSES, -1, 3, 32, 32)[:, :per_class]
        img = tensor_to_uint8(x.reshape(-1, 3, 32, 32)).reshape(NUM_CLASSES, per_class, 32, 32, 3)
        img = img.transpose(0, 2, 1, 3, 4).reshape(NUM_CLASSES * 32, per_class * 32, 3)
        ax.imshow(img)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xticks([])
        ax.set_yticks(np.arange(NUM_CLASSES) * 32 + 16)
        ax.set_yticklabels(SHORT_NAMES if ax is axes[0] else [""] * NUM_CLASSES, fontsize=7, color=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--n_fid", type=int, default=5000)
    p.add_argument("--n_fid_ddpm", type=int, default=1000,
                   help="samples for the 1000-step DDPM baseline (it is ~50x slower)")
    p.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--labels", choices=["train", "balanced"], default="train")
    p.add_argument("--latency_repeats", type=int, default=3)
    p.add_argument("--only", nargs="*", default=None, help="evaluate only these config names")
    p.add_argument("--skip_fid", action="store_true", help="for offline smoke tests")
    p.add_argument("--docs_dir", default="docs")
    args = p.parse_args(argv)

    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    real_u8, _ = load_numpy("test")
    _, train_labels = load_numpy("train")
    judge_path = os.path.join(args.ckpt_dir, "judge.pt")
    judge = load_judge(judge_path, device) if os.path.exists(judge_path) else None
    if judge is None:
        print("  (no judge.pt - class accuracy will be NaN; run src.judge_classifier first)")

    configs = find_configs(args.ckpt_dir, args.steps, args.guidance)
    if args.only:
        configs = [c for c in configs if c["name"] in args.only]
    print(f"Evaluating {len(configs)} configurations on {device}")

    rows, cache, grids = [], {}, {}
    for c in configs:
        if c["path"] not in cache:
            cache.clear()
            cache[c["path"]] = load_model(c["path"], device)
        model, diffusion, _ = cache[c["path"]]
        n = args.n_fid_ddpm if c["sampler"] == "ddpm" else args.n_fid
        labels = make_labels(n, args.labels, train_labels)
        print(f"[{c['name']}] generating {n} samples ...", flush=True)

        # warm-up (cuDNN autotune, allocator) so timings are fair
        sample_images(model, diffusion, labels[:8], "ddim", 2, c["guidance"], seed=123)
        x, secs = timed(lambda: sample_images(model, diffusion, labels, c["sampler"], c["steps"],
                                              c["guidance"], batch=args.batch, seed=0), device)
        ms_per_image = 1000 * secs / n

        single = []
        for r in range(args.latency_repeats if c["sampler"] == "ddim" else 1):
            _, s1 = timed(lambda: sample_images(model, diffusion, labels[:1], c["sampler"], c["steps"],
                                                c["guidance"], batch=1, seed=r), device)
            single.append(1000 * s1)
        sync(device)

        gen_u8 = tensor_to_uint8(x)
        t0 = time.time()
        fid = float("nan") if args.skip_fid else compute_fid(gen_u8, real_u8, cuda=device.type == "cuda")
        class_acc, per_class = float("nan"), [float("nan")] * NUM_CLASSES
        if judge is not None:
            pred = predict(judge, x.to(device))
            lab = labels.numpy()
            per_class = [float((pred[lab == k] == k).mean()) if (lab == k).any() else float("nan")
                         for k in range(NUM_CLASSES)]
            class_acc = float(np.nanmean(per_class))
        row = dict(name=c["name"], run=c["run"], schedule=diffusion.schedule, pred=diffusion.pred,
                   sampler=c["sampler"], steps=c["steps"], guidance=c["guidance"],
                   nfe=nfe_per_image(c["sampler"], c["steps"], c["guidance"]), n_samples=n,
                   fid=fid, class_acc=class_acc, ms_per_image=ms_per_image,
                   single_latency_ms=float(np.median(single)))
        row.update({f"acc_{k}": v for k, v in zip(SHORT_NAMES, per_class)})
        rows.append(row)
        print(f"[{c['name']}] FID {fid:.2f}  class-acc {class_acc:.3f}  {ms_per_image:.2f} ms/img  "
              f"single {row['single_latency_ms']:.0f} ms  (metrics {time.time() - t0:.0f}s)", flush=True)

        g = sample_images(model, diffusion, grid_labels(), c["sampler"], c["steps"], c["guidance"],
                          batch=args.batch, seed=7)
        save_grid(g, os.path.join(args.results_dir, "grids", f"{c['name']}.png"))
        grids[c["name"]] = g

        with open(os.path.join(args.results_dir, "benchmark.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    plot_fid_vs_steps(rows, os.path.join(args.results_dir, "fid_vs_steps.png"))
    plot_latency(rows, os.path.join(args.results_dir, "latency.png"))
    plot_loss_curves(args.ckpt_dir, os.path.join(args.results_dir, "loss_curves.png"))
    student = sorted([n for n in grids if n.startswith("E-")], key=lambda s: int(s.split("-")[-1]))
    order = [("A-ddpm-1000", "Baseline · 1000 steps"), ("A-ddim-20", "Baseline · 20 steps"),
             ("D-ddim-20", "FastDiff · 20 steps")]
    if student:
        order.append((student[0], f"Student · {student[0].split('-')[-1]} steps"))
    plot_sample_compare([(t, grids[n]) for n, t in order if n in grids],
                        os.path.join(args.results_dir, "sample_grid_compare.png"))
    report.write(args.results_dir, args.docs_dir)


if __name__ == "__main__":
    main()
