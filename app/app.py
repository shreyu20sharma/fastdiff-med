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
STUDENT8 = "Student · 8 steps"
ENGINE_ORDER = (STUDENT, STUDENT8, FASTDIFF, BASELINE)  # fastest first
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
        (STUDENT8, os.path.join(ckpt_dir, "distill", "student_8step.pt"), "ddim", 8, 1.0),
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


_REAL = {}


def real_examples(class_idx, n, seed=0, split="test"):
    """Grid of n real DermaMNIST photos of one class (None if the dataset cannot be loaded)."""
    from src.data import load_numpy, uint8_to_tensor

    if split not in _REAL:
        try:
            _REAL[split] = load_numpy(split)
        except Exception as e:  # offline / dataset unavailable: the demo still works without it
            print(f"[app] real examples unavailable: {e}")
            _REAL[split] = None
    if _REAL[split] is None:
        return None
    imgs, labels = _REAL[split]
    idx = np.flatnonzero(labels == int(class_idx))
    pick = np.random.default_rng(int(seed)).choice(idx, size=min(int(n), len(idx)), replace=False)
    return pick, uint8_to_tensor(imgs[pick])


def real_grid(class_idx, n, seed=0, nrow=8):
    r = real_examples(class_idx, n, seed)
    return None if r is None else to_grid(r[1], nrow=min(int(n), nrow))


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

    names = [n for n in ENGINE_ORDER if n in engines]
    choices = [f"{s} · {c}" for s, c in zip(SHORT_NAMES, CLASS_NAMES)]
    device = next(next(iter(engines.values()))["model"].parameters()).device if engines else "cpu"
    real_note = ("**Real DermaMNIST test photos** of the same class, at the same 28×28 resolution and "
                 "the same 4× enlargement. The softness you see is the dataset's resolution, not the generator.")

    def show_real(cls, n, seed):
        return real_grid(choices.index(cls), n, seed)

    def run_one(cls, n, name, seed):
        real = show_real(cls, n, seed)
        if name not in engines:
            yield None, f"Engine **{name}** is not available (checkpoint missing).", real
            return
        k, n = choices.index(cls), int(n)
        for done, step, total, secs, x in generate_stream(engines[name], k, n, seed):
            status = _status(name, engines[name], done, step, total, secs, n)
            yield (to_grid(x, nrow=min(n, 8)) if done else gr.update()), status, real

    def run_all(cls, n, seed):
        k, n = choices.index(cls), int(n)
        real = real_grid(k, n, seed, nrow=4)
        imgs = [gr.update()] * len(ENGINE_ORDER)
        stats = [f"_{nm}: waiting…_" if nm in engines else f"_{nm}: not available_" for nm in ENGINE_ORDER]
        yield (real, *imgs, *stats)
        for j, nm in enumerate(ENGINE_ORDER):
            if nm not in engines:
                continue
            for done, step, total, secs, x in generate_stream(engines[nm], k, n, seed):
                stats[j] = _status(nm, engines[nm], done, step, total, secs, n)
                if done:
                    imgs[j] = to_grid(x, nrow=min(n, 4))
                yield (real, *imgs, *stats)

    with gr.Blocks(title="DermAugment · FastDiff-Med") as demo:
        gr.Markdown("# DermAugment: FastDiff-Med\n"
                    "Class-conditional synthetic skin-lesion images for augmenting rare classes. "
                    "Same U-Net, four engines: the 1000-step DDPM baseline, the re-engineered 20-step model "
                    "(cosine schedule + v-prediction + trailing DDIM) and the distilled 8-step and 4-step "
                    f"students with guidance folded in. Running on **{device}**.")
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    cls = gr.Dropdown(choices, value=choices[3], label="Lesion class")
                    n = gr.Slider(1, 32, value=8, step=1, label="Number of images")
                    engine = gr.Radio(names, value=names[0] if names else None, label="Engine",
                                      info="4 steps = fastest · 8 steps = sharper, still ~120× faster than the baseline")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    go = gr.Button("Generate", variant="primary")
                    cmp_btn = gr.Button("Compare all engines")
                    if BASELINE in engines and str(device) == "cpu":
                        gr.Markdown("_On CPU the 1000-step baseline takes minutes; that gap is the point. "
                                    "Use 1–2 images when comparing._")
                with gr.Column(scale=2):
                    status = gr.Markdown()
                    out = gr.Image(label="Generated samples", type="numpy", interactive=False)
                    gr.Markdown(real_note)
                    real_out = gr.Image(label="Real photos (same class)", type="numpy", interactive=False)
            gr.Markdown("### Compare all engines (same class, seed and number of images)")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Real photos** (reference)")
                    cmp_real = gr.Image(label="Real DermaMNIST", type="numpy", interactive=False)
                cmp_imgs, cmp_stats = [], []
                for nm in ENGINE_ORDER:
                    with gr.Column():
                        cmp_stats.append(gr.Markdown(f"_{nm}_"))
                        cmp_imgs.append(gr.Image(label=nm, type="numpy", interactive=False))
            go.click(run_one, [cls, n, engine, seed], [out, status, real_out])
            cmp_btn.click(run_all, [cls, n, seed], [cmp_real] + cmp_imgs + cmp_stats)
            cls.change(show_real, [cls, n, seed], real_out)
            demo.load(show_real, [cls, n, seed], real_out)
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
