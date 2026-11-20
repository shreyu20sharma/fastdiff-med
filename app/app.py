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

from src.data import SHORT_NAMES, tensor_to_uint8  # noqa: E402
from src.sampling import nfe_per_image, sampler_iter  # noqa: E402
from src.utils import autocast, get_device, load_model  # noqa: E402

BASELINE, FASTDIFF, STUDENT = "Baseline · DDPM-1000", "FastDiff · DDIM-20", "Student · 4 steps"
STUDENT8 = "Student · 8 steps"
ENGINE_ORDER = (STUDENT, STUDENT8, FASTDIFF, BASELINE)  # fastest first

# Plain-language labels shown in the UI (internal engine names stay technical)
FRIENDLY = {
    STUDENT: "⚡ Instant (4 steps)",
    STUDENT8: "✨ Sharper (8 steps)",
    FASTDIFF: "🔧 Improved model (20 steps)",
    BASELINE: "🐢 Original, slow (1000 steps)",
}
ENGINE_HELP = {
    STUDENT: "our fastest version",
    STUDENT8: "a little slower, a little sharper",
    FASTDIFF: "our improved recipe before shortcut training",
    BASELINE: "the standard method we started from",
}
# (plain name, what it is, training photos available)
LESIONS = [
    ("Actinic keratosis", "rough, scaly sun-damage patch that can turn into cancer", 228),
    ("Basal cell carcinoma", "the most common skin cancer; grows slowly", 359),
    ("Benign keratosis", "harmless 'age spot' type growth", 769),
    ("Dermatofibroma", "harmless small firm bump", 80),
    ("Melanoma", "the most dangerous type of skin cancer", 779),
    ("Mole (nevus)", "an ordinary, harmless mole", 4693),
    ("Vascular lesion", "red or purple spot made of blood vessels", 99),
]


def lesion_label(k):
    name, what, n = LESIONS[k]
    rare = "  · RARE" if n < 300 else ""
    return f"{name}: {what}{rare}"


INTRO = """# 🩺 DermAugment: AI-made example images of skin conditions

**The problem:** computer programs that help spot skin diseases learn from example photos. Some diseases
are **rare**, so there are very few photos of them, and the programs get bad at recognising them.

**What this tool does:** it uses AI to **create new, realistic example pictures** of a chosen skin condition,
so those programs get more practice on the rare ones. Our version does this in **seconds instead of
minutes**, about 240× faster than the standard method.

**How to use it:** ① pick a skin condition → ② pick a speed → ③ press **Create pictures**.
"""

DISCLAIMER = ("⚠️ **For research and learning only.** These are computer-made pictures, not real patients. "
              "They are tiny (28×28 pixels) and **must never be used to diagnose anyone**. "
              "If you are worried about a spot on your skin, see a doctor.")


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
    label = FRIENDLY.get(name, name)
    if not done:
        return f"⏳ **{label}**: drawing… step {step} of {total} · {secs:.1f} s so far"
    calls = nfe_per_image(engine["sampler"], engine["steps"], engine["guidance"])
    return (f"✅ **{label}**: made {n} picture{'s' if n > 1 else ''} in **{secs:.1f} seconds**"
            f"<br><small>{engine['steps']} steps · {calls} AI calls per picture</small>")


# ----------------------------------------------------------------------------- impact tab helpers
def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plain_summary(results_dir):
    """'What did we find?' in everyday words, with numbers read from the result CSVs."""
    bench = {r["name"]: r for r in _read_csv(os.path.join(results_dir, "benchmark.csv"))}
    summary = _read_csv(os.path.join(results_dir, "augmentation_summary.csv"))
    times = _read_csv(os.path.join(results_dir, "gen_time.csv"))
    if not (bench or summary):
        return "_No results yet. Run the experiments (notebook 03) first._"
    out = ["## What did we find?", ""]
    base, fast = bench.get("A-ddpm-1000"), bench.get("E-ddim-4")
    if base and fast:
        b, f = float(base["single_latency_ms"]) / 1000, float(fast["single_latency_ms"]) / 1000
        out.append(f"**1. Much faster.** One picture takes **{f:.2f} seconds** with our fast version, compared "
                   f"with **{b:.0f} seconds** with the original method: about **{b / f:.0f}× faster**.")
    t_fast = next((t for t in times if t["setting"] == "100%" and not t["engine"].startswith("Baseline")), None)
    t_slow = next((t for t in times if t["setting"] == "100%" and t["engine"].startswith("Baseline")), None)
    if t_fast and t_slow:
        out.append(f"Making a full set of **{int(t_fast['n_images']):,} extra pictures** took "
                   f"**{float(t_fast['seconds']):.0f} seconds** instead of about "
                   f"**{float(t_slow['seconds']) / 60:.0f} minutes**.")
    if base and fast:
        out += ["", f"**2. Pictures still look right.** An independent checker program recognised the intended "
                f"condition in **{float(fast['class_acc']):.0%}** of our fast pictures, vs "
                f"**{float(base['class_acc']):.0%}** for the original slow method. The fast pictures are slightly "
                "smoother and less detailed, so overall quality is a little lower."]
    real = next((r for r in summary if r["setting"] == "20%" and r["arm"] == "real"), None)
    syn = next((r for r in summary if r["setting"] == "20%" and r["arm"] == "real+synthetic"), None)
    real100 = next((r for r in summary if r["setting"] == "100%" and r["arm"] == "real"), None)
    syn100 = next((r for r in summary if r["setting"] == "100%" and r["arm"] == "real+synthetic"), None)
    if real and syn:
        gain = 100 * (float(syn["balanced_accuracy_mean"]) - float(real["balanced_accuracy_mean"]))
        df0, df1 = float(real["recall_df_mean"]), float(syn["recall_df_mean"])
        out += ["", "**3. The extra pictures help find rare conditions.** We trained a skin-condition "
                "recognition program with and without our extra pictures, then tested it on real photos it had "
                "never seen.",
                f"* **When real photos are scarce** (20% of the data): it got **{gain:.1f} points better** on "
                f"average across conditions. For the rarest one, dermatofibroma, it went from finding "
                f"**{df0:.0%}** of cases to **{df1:.0%}**."]
        if real100 and syn100:
            g100 = 100 * (float(syn100["balanced_accuracy_mean"]) - float(real100["balanced_accuracy_mean"]))
            out.append(f"* **With all the real photos:** a smaller gain ({g100:+.1f} points). The extra pictures "
                       "matter most when real data is limited.")
        out.append("* Simply **copying** the existing rare photos did **not** help. New pictures did.")
    return "\n".join(out)


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
    lesion_choices = [(lesion_label(k), k) for k in range(len(LESIONS))]
    engine_choices = [(f"{FRIENDLY[nm]}: {ENGINE_HELP[nm]}", nm) for nm in names]
    device = next(next(iter(engines.values()))["model"].parameters()).device if engines else "cpu"

    def show_real(k, n, seed):
        return real_grid(int(k), n, seed)

    def run_one(k, n, name, seed):
        real = show_real(k, n, seed)
        if name not in engines:
            yield None, "This version is not available right now.", real
            return
        k, n = int(k), int(n)
        for done, step, total, secs, x in generate_stream(engines[name], k, n, seed):
            status = _status(name, engines[name], done, step, total, secs, n)
            yield (to_grid(x, nrow=min(n, 8)) if done else gr.update()), status, real

    def run_all(k, n, seed):
        k, n = int(k), int(n)
        real = real_grid(k, n, seed, nrow=4)
        imgs = [gr.update()] * len(ENGINE_ORDER)
        stats = [f"_{FRIENDLY[nm]}: waiting its turn…_" if nm in engines else f"_{FRIENDLY[nm]}: not available_"
                 for nm in ENGINE_ORDER]
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
        gr.Markdown(INTRO)
        with gr.Tab("🎨 Create pictures"):
            with gr.Row():
                with gr.Column(scale=1):
                    cls = gr.Dropdown(lesion_choices, value=3, label="① Skin condition",
                                      info="RARE = very few real photos exist, so extra pictures help most")
                    engine = gr.Radio(engine_choices, value=names[0] if names else None, label="② Speed")
                    n = gr.Slider(1, 32, value=8, step=1, label="How many pictures")
                    seed = gr.Number(value=0, precision=0, label="Variation number",
                                     info="Change it to get different pictures. The same number always "
                                          "gives the same pictures.")
                    go = gr.Button("③ Create pictures", variant="primary")
                    if BASELINE in engines and str(device) == "cpu":
                        gr.Markdown("_Tip: the 🐢 original version is very slow on a normal computer "
                                    "(minutes). That slowness is exactly what this project fixes._")
                with gr.Column(scale=2):
                    status = gr.Markdown("Your pictures will appear here.")
                    out = gr.Image(label="AI-made pictures", type="numpy", interactive=False)
                    gr.Markdown("**Real photos of the same condition**, for comparison. Real photos in this "
                                "dataset are also tiny and soft, so the blur is normal and not a fault of the AI.")
                    real_out = gr.Image(label="Real photos", type="numpy", interactive=False)

            gr.Markdown("---\n### ⏱️ Speed race: all versions make the same pictures\n"
                        "Each version draws the same pictures one after another, so you can compare their "
                        "speed and quality with real photos. **Use 1–2 pictures**, because the 🐢 original "
                        "version is slow.")
            cmp_btn = gr.Button("Start the speed race")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**📷 Real photos**")
                    cmp_real = gr.Image(label="Real photos", type="numpy", interactive=False)
                cmp_imgs, cmp_stats = [], []
                for nm in ENGINE_ORDER:
                    with gr.Column():
                        cmp_stats.append(gr.Markdown(f"**{FRIENDLY[nm]}**"))
                        cmp_imgs.append(gr.Image(label=FRIENDLY[nm], type="numpy", interactive=False))
            go.click(run_one, [cls, n, engine, seed], [out, status, real_out])
            cmp_btn.click(run_all, [cls, n, seed], [cmp_real] + cmp_imgs + cmp_stats)
            cls.change(show_real, [cls, n, seed], real_out)
            demo.load(show_real, [cls, n, seed], real_out)

        with gr.Tab("📊 Results"):
            gr.Markdown(plain_summary(results_dir))
            charts = [("augmentation.png", "Does adding AI pictures help? (taller bars = better at "
                                           "recognising all conditions, including rare ones)"),
                      ("latency.png", "Time to make one picture (shorter = faster)"),
                      ("sample_grid_compare.png", "Example pictures from each version (one row per condition)"),
                      ("fid_vs_steps.png", "Picture quality vs. number of steps (lower = better)")]
            for fname, label in charts:
                path = os.path.join(results_dir, fname)
                if os.path.exists(path):
                    gr.Image(path, label=label, interactive=False)
            with gr.Accordion("🔬 Detailed numbers (for experts)", open=False):
                gr.Markdown(impact_markdown(results_dir))

        with gr.Tab("ℹ️ How it works"):
            gr.Markdown("""
### How does the AI make pictures?
It's a **diffusion model**. During training it practised on thousands of real skin photos: we added random
"static" (noise) to each photo, and the AI learned to remove it. To make a **new** picture, it starts from pure
static and removes the noise bit by bit until a realistic skin spot appears.

### Why was it slow, and what did we change?
The standard method removes the noise in **1000 tiny steps**, so 1000 runs of the AI per picture
(2000 with the extra "follow the chosen condition" check). We made it fast in two stages:

1. **A smarter recipe (1000 → 20 steps).** We changed *how* noise is added during training (a smoother
   schedule) and *what* the AI predicts (a more stable target). Then each step does more useful work, so
   20 steps are enough.
2. **Shortcut training (20 → 4 steps).** A "student" copy of the AI watched the "teacher" and learned to do
   two steps in one. We repeated this until only **4 steps** were needed. The student also learned the
   "follow the chosen condition" check, so each step needs just one run of the AI.

**Result:** 4 runs of the AI per picture instead of 2000, the same network and the same training data.

### What are the limits?
* Pictures are tiny (28×28 pixels), like the real dataset.
* The 4-step pictures are slightly less detailed than the slow original's.
* This is a research prototype, **not a medical tool**.
""")
            with gr.Accordion("🔬 Technical summary (for experts)", open=False):
                gr.Markdown("Class-conditional U-Net trained on DermaMNIST (32×32). Baseline: linear β schedule, "
                            "ε-prediction, 1000-step DDPM ancestral sampling with classifier-free guidance (w = 2). "
                            "Stage 1: cosine schedule + v-prediction + DDIM with trailing timesteps (20 steps). "
                            "Stage 2: progressive distillation 32 → 16 → 8 → 4 steps with guidance folded into the "
                            "student, truncated-SNR weighting. Metrics: FID (torch-fidelity), class accuracy via a "
                            "ResNet-18 judge, latency on a T4, and downstream ResNet-18 balanced accuracy / macro-F1 "
                            f"with synthetic rare-class augmentation. Running on **{device}**.")
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
