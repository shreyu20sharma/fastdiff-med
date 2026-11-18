"""DermAugment: Gradio demo for FastDiff-Med.

Tab "Generate": pick a lesion class and an engine, watch a live timer, or compare all engines.
Tab "Impact":   the augmentation experiment (tables + charts from the results folder).

Used three ways:
  * notebook 04:  from app.app import build_demo, load_engines
                  build_demo(load_engines(CKPT), results_dir=RESULTS).launch(share=True)
  * locally:      python -m app.app --ckpt_dir checkpoints --results_dir results
  * HF Spaces:    copy this file to the Space root as app.py next to src/, checkpoints/, results/
                  (env FASTDIFF_CKPT_DIR / FASTDIFF_RESULTS_DIR override the defaults).
"""
import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):  # works from app/ in the repo and from a Space root
    if os.path.isdir(os.path.join(_p, "src")) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import CLASS_NAMES, SHORT_NAMES, tensor_to_uint8  # noqa: E402
from src.sampling import nfe_per_image, sampler_iter  # noqa: E402
from src.utils import autocast, get_device, load_model  # noqa: E402

BASELINE, FASTDIFF, STUDENT = "Baseline · DDPM-1000", "FastDiff · DDIM-20", "Student · 4 steps"
DISCLAIMER = ("⚠️ **Research and education only.** Images are synthetic 32×32 samples from a model "
              "trained on DermaMNIST. They are not real patients and must not be used for diagnosis "
              "or clinical decisions.")


# ----------------------------------------------------------------------------- engines
def load_engines(ckpt_dir, device=None, guidance=2.0):
    """Returns {engine name: dict(model, diffusion, sampler, steps, guidance)} for whatever exists."""
    device = device or get_device()
    specs = [
        (BASELINE, os.path.join(ckpt_dir, "A", "ema.pt"), "ddpm", 1000, guidance),
        (FASTDIFF, os.path.join(ckpt_dir, "D", "ema.pt"), "ddim", 20, guidance),
        (STUDENT, os.path.join(ckpt_dir, "distill", "student_4step.pt"), "ddim", 4, 1.0),
    ]
    engines = {}
    for name, path, sampler, steps, g in specs:
        if not os.path.exists(path):
            print(f"[app] {name}: {path} not found, engine disabled")
            continue
        model, diffusion, _ = load_model(path, device)
        engines[name] = dict(model=model, diffusion=diffusion, sampler=sampler, steps=steps, guidance=g)
        print(f"[app] loaded {name} from {path}")
    return engines


def to_grid(x, nrow, scale=4):
    """N x 3 x 32 x 32 in [-1, 1] -> one uint8 image, upscaled (nearest) so 32px lesions are visible."""
    imgs = tensor_to_uint8(x)
    n, h, w, _ = imgs.shape
    rows = int(np.ceil(n / nrow))
    pad = 2
    canvas = np.full((rows * (h + pad) + pad, nrow * (w + pad) + pad, 3), 255, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        canvas[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = im
    return canvas.repeat(scale, 0).repeat(scale, 1)


def generate_stream(engine, class_idx, n, seed=0, updates=20):
    """Yields (done, step, total_steps, elapsed_s, x or None). Runs the sampler step by step."""
    model, diffusion = engine["model"], engine["diffusion"]
    device = next(model.parameters()).device
    y = torch.full((n,), int(class_idx), dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    every = max(1, engine["steps"] // updates)
    t0 = time.perf_counter()
    x = None
    with torch.no_grad(), autocast(device):
        for i, total, x, _ in sampler_iter(model, diffusion, y, engine["sampler"], engine["steps"],
                                           engine["guidance"], g):
            if (i + 1) % every == 0 and i + 1 < total:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                yield False, i + 1, total, time.perf_counter() - t0, None
    if device.type == "cuda":
        torch.cuda.synchronize()
    yield True, engine["steps"], engine["steps"], time.perf_counter() - t0, x.float().clamp(-1, 1).cpu()


def _status(name, engine, done, step, total, secs, n):
    calls = nfe_per_image(engine["sampler"], engine["steps"], engine["guidance"])
    if not done:
        return f"⏳ **{name}**: step {step}/{total} · {secs:.1f} s elapsed"
    return (f"✅ **{name}**: {n} image{'s' if n > 1 else ''} in **{secs:.2f} s** · "
            f"{engine['steps']} steps · {calls} network calls per image")


# ----------------------------------------------------------------------------- impact tab helpers
def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def impact_markdown(results_dir):
    from src.report import headline

    bench = _read_csv(os.path.join(results_dir, "benchmark.csv"))
    summary = _read_csv(os.path.join(results_dir, "augmentation_summary.csv"))
    times = _read_csv(os.path.join(results_dir, "gen_time.csv"))
    if not (bench or summary):
        return (f"_No results found in `{results_dir}` yet. Run notebook 03 "
                "(src.evaluate and src.augment_experiment) first._")
    lines = ["### Headline", ""] + [f"* {h}" for h in headline(bench, summary, times)]
    if summary:
        lines += ["", "### ResNet-18 trained on real vs. real + synthetic data (DermaMNIST test set)", "",
                  "| Real data | Training set | Balanced accuracy | Macro-F1 |", "|---|---|---:|---:|"]
        for r in summary:
            lines.append(f"| {r['setting']} | {r['arm']} | {float(r['balanced_accuracy_mean']):.3f} ± "
                         f"{float(r['balanced_accuracy_std']):.3f} | {float(r['macro_f1_mean']):.3f} ± "
                         f"{float(r['macro_f1_std']):.3f} |")
        lines += ["", "Rare-class recall (mean over seeds):", "",
                  "| Real data | Training set | " + " | ".join(SHORT_NAMES) + " |",
                  "|---|---|" + "---:|" * len(SHORT_NAMES)]
        for r in summary:
            lines.append(f"| {r['setting']} | {r['arm']} | " +
                         " | ".join(f"{float(r[f'recall_{c}_mean']):.2f}" for c in SHORT_NAMES) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- UI
def build_demo(engines, results_dir="results"):
    import gradio as gr

    names = [n for n in (STUDENT, FASTDIFF, BASELINE) if n in engines]
    choices = [f"{s} · {c}" for s, c in zip(SHORT_NAMES, CLASS_NAMES)]
    device = next(next(iter(engines.values()))["model"].parameters()).device if engines else "cpu"

    def run_one(cls, n, name, seed):
        if name not in engines:
            yield None, f"Engine **{name}** is not available (checkpoint missing)."
            return
        k, n = choices.index(cls), int(n)
        for done, step, total, secs, x in generate_stream(engines[name], k, n, seed):
            status = _status(name, engines[name], done, step, total, secs, n)
            yield (to_grid(x, nrow=min(n, 8)) if done else gr.update()), status

    def run_all(cls, n, seed):
        k, n = choices.index(cls), int(n)
        imgs = [gr.update()] * 3
        stats = [f"_{nm}: waiting…_" if nm in engines else f"_{nm}: not available_"
                 for nm in (STUDENT, FASTDIFF, BASELINE)]
        for j, nm in enumerate((STUDENT, FASTDIFF, BASELINE)):
            if nm not in engines:
                continue
            for done, step, total, secs, x in generate_stream(engines[nm], k, n, seed):
                stats[j] = _status(nm, engines[nm], done, step, total, secs, n)
                if done:
                    imgs[j] = to_grid(x, nrow=min(n, 4))
                yield (*imgs, *stats)

    with gr.Blocks(title="DermAugment · FastDiff-Med") as demo:
        gr.Markdown("# DermAugment: FastDiff-Med\n"
                    "Class-conditional synthetic skin-lesion images for augmenting rare classes. "
                    "Same U-Net, three samplers: the 1000-step DDPM baseline, the re-engineered "
                    "20-step model (cosine schedule + v-prediction + trailing DDIM) and the 4-step "
                    f"distilled student with guidance folded in. Running on **{device}**.")
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    cls = gr.Dropdown(choices, value=choices[3], label="Lesion class")
                    n = gr.Slider(1, 32, value=8, step=1, label="Number of images")
                    engine = gr.Radio(names, value=names[0] if names else None, label="Engine")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    go = gr.Button("Generate", variant="primary")
                    cmp_btn = gr.Button("Compare all engines")
                    if BASELINE in engines and str(device) == "cpu":
                        gr.Markdown("_On CPU the 1000-step baseline takes minutes; that gap is the point._")
                with gr.Column(scale=2):
                    status = gr.Markdown()
                    out = gr.Image(label="Samples", type="numpy", interactive=False)
            gr.Markdown("### Compare all engines (same class, seed and number of images)")
            with gr.Row():
                cmp_imgs, cmp_stats = [], []
                for nm in (STUDENT, FASTDIFF, BASELINE):
                    with gr.Column():
                        cmp_stats.append(gr.Markdown(f"_{nm}_"))
                        cmp_imgs.append(gr.Image(label=nm, type="numpy", interactive=False))
            go.click(run_one, [cls, n, engine, seed], [out, status])
            cmp_btn.click(run_all, [cls, n, seed], cmp_imgs + cmp_stats)
        with gr.Tab("Impact"):
            gr.Markdown(impact_markdown(results_dir))
            charts = [("augmentation.png", "Augmentation: real vs real+oversample vs real+synthetic"),
                      ("fid_vs_steps.png", "Sample quality vs. number of steps"),
                      ("latency.png", "Time to generate one image"),
                      ("sample_grid_compare.png", "Samples per class from each engine")]
            for fname, label in charts:
                path = os.path.join(results_dir, fname)
                if os.path.exists(path):
                    gr.Image(path, label=label, interactive=False)
        gr.Markdown(DISCLAIMER)
    return demo


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=os.environ.get("FASTDIFF_CKPT_DIR", "checkpoints"))
    p.add_argument("--results_dir", default=os.environ.get("FASTDIFF_RESULTS_DIR", "results"))
    p.add_argument("--share", action="store_true")
    a = p.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    build_demo(load_engines(a.ckpt_dir), a.results_dir).queue().launch(share=a.share)


if __name__ == "__main__":
    main()
