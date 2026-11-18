"""Downstream utility: does FastDiff synthetic data help a real classifier on rare lesions?

    python -m src.augment_experiment --ckpt_dir <ckpt_dir> [--seeds 0 1 2] [--epochs 30]

For each data setting (100% and 20% of the real training set) we train ResNet-18 with 3 seeds on
  * real          : real images only
  * real+oversample: minority classes topped up by duplicating real images (the cheap baseline
                    any reviewer will ask about)
  * real+synthetic: minority classes topped up with 4-step student samples
Top-up rule: every class except the largest is filled to the size of the 2nd-largest class.
Metrics on the official test split: balanced accuracy, macro-F1, per-class recall (mean ± std).
Also times how long the synthetic set takes with the student vs. the 1000-step baseline
(the baseline is timed on --baseline_time_n images and extrapolated, marked as such).
"""
import argparse
import csv
import os

import numpy as np
import torch

from src import report
from src.data import NUM_CLASSES, SHORT_NAMES, class_counts, load_tensors, stratified_subset
from src.judge_classifier import metrics, predict, train_classifier
from src.sampling import sample_images, timed
from src.utils import get_device, load_model

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = ["real", "real+oversample", "real+synthetic"]
ARM_COLORS = {"real+synthetic": "#2a78d6", "real+oversample": "#eb6834", "real": "#1baf7a"}


def topup_deficits(labels):
    cnt = class_counts(labels)
    target = np.sort(cnt)[-2]
    deficit = np.maximum(target - cnt, 0)
    deficit[np.argmax(cnt)] = 0
    return cnt, int(target), deficit


def oversample(x, y, deficit, seed):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [x], [y]
    for k, d in enumerate(deficit):
        if d <= 0:
            continue
        idx = torch.nonzero(y == k).flatten().cpu()
        pick = idx[torch.randint(0, len(idx), (int(d),), generator=g)].to(x.device)
        xs.append(x[pick])
        ys.append(y[pick])
    return torch.cat(xs), torch.cat(ys)


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = sorted({r["setting"] for r in summary}, key=lambda s: -float(s.rstrip("%")))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150, facecolor=SURFACE)
    width = 0.26
    for ax, (metric, title) in zip(axes, [("balanced_accuracy", "Balanced accuracy"),
                                          ("macro_f1", "Macro-F1")]):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for j, arm in enumerate(ARMS):
            means, stds = [], []
            for st in settings:
                r = next((r for r in summary if r["setting"] == st and r["arm"] == arm), None)
                means.append(r[f"{metric}_mean"] if r else np.nan)
                stds.append(r[f"{metric}_std"] if r else 0)
            xs = np.arange(len(settings)) + (j - 1) * (width + 0.02)
            ax.bar(xs, means, width, yerr=stds, color=ARM_COLORS[arm], label=arm,
                   error_kw=dict(ecolor=INK2, lw=1, capsize=3))
            for x_, m in zip(xs, means):
                if m == m:
                    ax.annotate(f"{m:.3f}", (x_, m), xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=7, color=INK)
        ax.set_xticks(np.arange(len(settings)))
        ax.set_xticklabels([f"{s} real data" for s in settings])
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        lo = np.nanmin([r[f"{metric}_mean"] for r in summary])
        ax.set_ylim(max(0, lo - 0.1), None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=INK2, loc="upper right", ncol=3)
    fig.suptitle("ResNet-18 on DermaMNIST test set (mean ± std over seeds)", fontsize=9,
                 color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--student", default=None, help="default: <ckpt_dir>/distill/student_4step.pt")
    p.add_argument("--baseline", default=None, help="default: <ckpt_dir>/A/ema.pt")
    p.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--guidance", type=float, default=2.0, help="for the non-distilled baseline/fallback")
    p.add_argument("--baseline_time_n", type=int, default=250)
    p.add_argument("--docs_dir", default="docs")
    args = p.parse_args(argv)

    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    student_path = args.student or os.path.join(args.ckpt_dir, "distill", "student_4step.pt")
    baseline_path = args.baseline or os.path.join(args.ckpt_dir, "A", "ema.pt")

    if os.path.exists(student_path):
        gen_model, gen_diff, gcfg = load_model(student_path, device)
        gen_spec = dict(sampler="ddim", steps=gcfg.get("steps", 4), guidance=1.0,
                        name=f"Student-{gcfg.get('steps', 4)}")
    else:  # fallback keeps the experiment alive if distillation was skipped
        fallback = os.path.join(args.ckpt_dir, "D", "ema.pt")
        print(f"  student not found, falling back to FastDiff-20 ({fallback})")
        gen_model, gen_diff, _ = load_model(fallback, device)
        gen_spec = dict(sampler="ddim", steps=20, guidance=args.guidance, name="FastDiff-20")

    x_tr_all, y_tr_all = load_tensors("train", device)
    x_val, y_val = load_tensors("val", device)
    x_te, y_te = load_tensors("test", device)
    y_te_np = y_te.cpu().numpy()

    rows, time_rows = [], []
    for frac in args.fractions:
        setting = f"{int(round(frac * 100))}%"
        if frac < 1.0:
            idx = torch.from_numpy(stratified_subset(y_tr_all, frac, seed=0)).to(device)
            x_tr, y_tr = x_tr_all[idx], y_tr_all[idx]
        else:
            x_tr, y_tr = x_tr_all, y_tr_all
        cnt, target, deficit = topup_deficits(y_tr)
        print(f"\n=== setting {setting}: {len(y_tr)} real images, counts {dict(zip(SHORT_NAMES, cnt))}, "
              f"top-up target {target}, synthetic needed {int(deficit.sum())}")

        syn_labels = torch.cat([torch.full((int(d),), k) for k, d in enumerate(deficit) if d > 0]).long()
        x_syn, secs = timed(lambda: sample_images(gen_model, gen_diff, syn_labels, gen_spec["sampler"],
                                                  gen_spec["steps"], gen_spec["guidance"], seed=1000),
                            device)
        time_rows.append(dict(setting=setting, engine=gen_spec["name"], n_images=len(syn_labels),
                              measured_images=len(syn_labels), seconds=secs, extrapolated=False))
        print(f"  generated {len(syn_labels)} synthetic images with {gen_spec['name']} in {secs:.1f}s")
        if os.path.exists(baseline_path) and args.baseline_time_n > 0:
            bm, bd, _ = load_model(baseline_path, device)
            nb = min(args.baseline_time_n, len(syn_labels))
            _, bsecs = timed(lambda: sample_images(bm, bd, syn_labels[:nb], "ddpm", 1000, args.guidance,
                                                   seed=1000), device)
            est = bsecs / nb * len(syn_labels)
            time_rows.append(dict(setting=setting, engine="Baseline-1000 (DDPM)", n_images=len(syn_labels),
                                  measured_images=nb, seconds=est, extrapolated=nb < len(syn_labels)))
            print(f"  baseline DDPM-1000: {bsecs:.1f}s for {nb} images -> ~{est:.0f}s for the full set")
            del bm
        if setting == "100%":
            from src.evaluate import save_grid
            per = 8
            show = torch.cat([x_syn[syn_labels == k][:per] for k in range(NUM_CLASSES)
                              if (syn_labels == k).sum() >= per])
            save_grid(show, os.path.join(args.results_dir, "grids", "augment_synthetic.png"), nrow=per)

        x_syn, syn_labels = x_syn.to(device), syn_labels.to(device)
        for seed in args.seeds:
            data = {
                "real": (x_tr, y_tr),
                "real+oversample": oversample(x_tr, y_tr, deficit, seed),
                "real+synthetic": (torch.cat([x_tr, x_syn]), torch.cat([y_tr, syn_labels])),
            }
            for arm in ARMS:
                xa, ya = data[arm]
                print(f"  [{setting} | {arm} | seed {seed}] training on {len(ya)} images", flush=True)
                model = train_classifier(xa, ya, x_val, y_val, epochs=args.epochs, batch=args.batch,
                                         seed=seed, verbose=False)
                m = metrics(y_te_np, predict(model, x_te))
                row = dict(setting=setting, arm=arm, seed=seed, n_train=len(ya), accuracy=m["accuracy"],
                           balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"])
                row.update({f"recall_{c}": r for c, r in zip(SHORT_NAMES, m["recall"])})
                rows.append(row)
                print(f"    bal-acc {m['balanced_accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}", flush=True)
                del model
            _write(rows, time_rows, args.results_dir)

    summary = _write(rows, time_rows, args.results_dir)
    plot(summary, os.path.join(args.results_dir, "augmentation.png"))
    report.write(args.results_dir, args.docs_dir)


def _write(rows, time_rows, results_dir):
    def dump(path, rs):
        if not rs:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            for r in rs:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    dump(os.path.join(results_dir, "augmentation.csv"), rows)
    dump(os.path.join(results_dir, "gen_time.csv"), time_rows)
    summary = []
    keys = ["balanced_accuracy", "macro_f1", "accuracy"] + [f"recall_{c}" for c in SHORT_NAMES]
    for setting in dict.fromkeys(r["setting"] for r in rows):
        for arm in ARMS:
            rs = [r for r in rows if r["setting"] == setting and r["arm"] == arm]
            if not rs:
                continue
            s = dict(setting=setting, arm=arm, n_seeds=len(rs), n_train=rs[0]["n_train"])
            for k in keys:
                v = np.array([r[k] for r in rs])
                s[f"{k}_mean"], s[f"{k}_std"] = float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)
            summary.append(s)
    dump(os.path.join(results_dir, "augmentation_summary.csv"), summary)
    return summary


if __name__ == "__main__":
    main()
