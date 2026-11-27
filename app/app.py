"""DermAugment: Gradio demo for FastDiff-Med.

Tabs: Home (the story), Create images (generate + download), Speed race (all engines side by side),
Watch it work (step through the denoising process).

Used three ways:
  * notebook 04:  from app.app import build_demo, launch_demo, load_engines
                  launch_demo(build_demo(load_engines(CKPT), results_dir=RESULTS), share=True)
  * locally:      python -m app.app --ckpt_dir checkpoints --results_dir results
  * HF Spaces:    copy this file to the Space root as app.py next to src/, checkpoints/, results/
                  (env FASTDIFF_CKPT_DIR / FASTDIFF_RESULTS_DIR override the defaults).
"""
import argparse
import csv
import os
import random
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


DISCLAIMER = ("⚠️ **For research and learning only.** These are computer-made images, not real patients. "
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
    """Yields (done, step, total_steps, elapsed_s, x). Runs the sampler step by step.

    While running, x is the current partly denoised batch (a live preview: static turning into an image);
    at the end it is the final sample. Time spent preparing previews is excluded from elapsed_s.
    """
    model, diffusion = engine["model"], engine["diffusion"]
    device = next(model.parameters()).device
    y = torch.full((n,), int(class_idx), dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    every = max(1, engine["steps"] // updates)
    t0 = time.perf_counter()
    excluded = 0.0
    x = None
    with torch.no_grad(), autocast(device):
        for i, total, x, _ in sampler_iter(model, diffusion, y, engine["sampler"], engine["steps"],
                                           engine["guidance"], g):
            if (i + 1) % every == 0 and i + 1 < total:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0 - excluded
                p0 = time.perf_counter()
                # everything from here until the generator resumes (copying the preview and the page drawing
                # it) is display work, not generation, so it is kept out of the timer
                yield False, i + 1, total, elapsed, x.float().clamp(-1, 1).cpu()
                excluded += time.perf_counter() - p0
    if device.type == "cuda":
        torch.cuda.synchronize()
    yield True, engine["steps"], engine["steps"], time.perf_counter() - t0 - excluded, x.float().clamp(-1, 1).cpu()


def _status(name, engine, done, step, total, secs, n):
    label = FRIENDLY.get(name, name)
    if not done:
        return f"⏳ **{label}**: drawing… step {step} of {total} · {secs:.1f} s so far"
    calls = nfe_per_image(engine["sampler"], engine["steps"], engine["guidance"])
    return (f"✅ **{label}**: made {n} image{'s' if n > 1 else ''} in **{secs:.1f} seconds**"
            f"<br><small>{engine['steps']} steps · {calls} AI calls per image</small>")


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


# ----------------------------------------------------------------------------- UI helpers
CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.hero h1 {margin-bottom: 0.2em;}
.stat-row {display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 4px;}
.stat {flex: 1 1 200px; border: 1px solid var(--border-color-primary); border-radius: 12px;
       padding: 14px 16px; background: var(--block-background-fill);}
.stat .num {font-size: 1.9em; font-weight: 700; color: var(--color-accent); line-height: 1.1;}
.stat .lbl {font-size: 0.92em; opacity: 0.85; margin-top: 4px;}
.card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 4px 14px;
       background: var(--block-background-fill); height: 100%;}
.note {font-size: 0.92em; opacity: 0.85;}
footer {visibility: hidden;}
"""


def _theme():
    import gradio as gr
    return gr.themes.Soft(primary_hue="blue", neutral_hue="slate")


def headline_numbers(results_dir):
    """(speed-up, calls before -> after, balanced-accuracy gain at 20% data) from the result CSVs."""
    bench = {r["name"]: r for r in _read_csv(os.path.join(results_dir, "benchmark.csv"))}
    summary = _read_csv(os.path.join(results_dir, "augmentation_summary.csv"))
    base, fast = bench.get("A-ddpm-1000"), bench.get("E-ddim-4")
    speed = (f"{float(base['single_latency_ms']) / float(fast['single_latency_ms']):.0f}× faster"
             if base and fast else "≈240× faster")
    calls = f"{base['nfe']} → {fast['nfe']}" if base and fast else "2000 → 4"
    real = next((r for r in summary if r["setting"] == "20%" and r["arm"] == "real"), None)
    syn = next((r for r in summary if r["setting"] == "20%" and r["arm"] == "real+synthetic"), None)
    gain = (f"+{100 * (float(syn['balanced_accuracy_mean']) - float(real['balanced_accuracy_mean'])):.1f} pts"
            if real and syn else "")
    return speed, calls, gain


def stat_cards(results_dir):
    speed, calls, gain = headline_numbers(results_dir)
    cards = [(speed, "to create one image, compared with the standard method"),
             (calls, "AI calls per image: the work we removed"),
             (gain, "better at recognising skin conditions when real photos are scarce")]
    return ('<div class="stat-row">' + "".join(
        f'<div class="stat"><div class="num">{n}</div><div class="lbl">{l}</div></div>' for n, l in cards if n)
        + "</div>")


def save_zip(x, class_idx, engine_name, seed):
    """Write the images at their native 28x28 size (padding removed) into a ZIP; returns its path."""
    import tempfile
    import zipfile

    from PIL import Image

    name = LESIONS[int(class_idx)][0].split(" (")[0].lower().replace(" ", "_")
    steps = engine_name.split("·")[-1].strip().split(" ")[0].replace("DDPM-", "").replace("DDIM-", "")
    path = os.path.join(tempfile.mkdtemp(prefix="dermaugment_"), f"synthetic_{name}_{steps}steps_seed{seed}.zip")
    with zipfile.ZipFile(path, "w") as z:
        for i, im in enumerate(tensor_to_uint8(x)):
            buf = __import__("io").BytesIO()
            Image.fromarray(im[2:-2, 2:-2]).save(buf, format="PNG")
            z.writestr(f"{name}_{i:03d}.png", buf.getvalue())
        z.writestr("README.txt", "Synthetic DermaMNIST-style images made with DermAugment (FastDiff-Med).\n"
                   f"Class: {LESIONS[int(class_idx)][0]}\nEngine: {engine_name}\nVariation number: {seed}\n"
                   "28x28 RGB PNG, like DermaMNIST. For research and education only; not for diagnosis.\n")
    return path


def denoise_frames(engine, class_idx, n, seed):
    """Run one sampling pass and keep, for every step, the current noisy image and the model's guess
    of the finished image (x0 prediction). Returns two lists of image grids."""
    model, diffusion = engine["model"], engine["diffusion"]
    device = next(model.parameters()).device
    y = torch.full((n,), int(class_idx), dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    x_T = torch.randn((n, 3, 32, 32), device=device, generator=g)
    current, guess = [to_grid(x_T.clamp(-1, 1).cpu(), nrow=n)], [None]
    with torch.no_grad(), autocast(device):
        for _, _, x, x0 in diffusion.ddim_sample_iter(model, y, engine["steps"], guidance=engine["guidance"],
                                                      x_T=x_T, generator=g):
            current.append(to_grid(x.float().clamp(-1, 1).cpu(), nrow=n))
            guess.append(to_grid(x0.float().clamp(-1, 1).cpu(), nrow=n))
    return current, guess


def launch_demo(demo, **kw):
    """launch() that also applies the theme and CSS (Gradio 6 takes them in launch, older versions in Blocks)."""
    import gradio as gr

    if int(gr.__version__.split(".")[0]) >= 6:
        kw.setdefault("theme", _theme())
        kw.setdefault("css", CSS)
    return demo.queue().launch(**kw)


# ----------------------------------------------------------------------------- UI
def build_demo(engines, results_dir="results"):
    import gradio as gr

    names = [n for n in ENGINE_ORDER if n in engines]
    lesion_choices = [(lesion_label(k), k) for k in range(len(LESIONS))]
    engine_choices = [(f"{FRIENDLY[nm]}: {ENGINE_HELP[nm]}", nm) for nm in names]
    device = next(next(iter(engines.values()))["model"].parameters()).device if engines else "cpu"
    explorer_engine = FASTDIFF if FASTDIFF in engines else (names[0] if names else None)

    def pick_seed(seed, surprise):
        return random.randint(1, 99_999) if surprise else int(seed or 0)

    def real_tiles(k, n, s):
        return real_grid(int(k), int(n), s)

    def run_one(k, n, name, seed, surprise):
        k, n, s = int(k), int(n), pick_seed(seed, surprise)
        real = real_tiles(k, n, s)
        if name not in engines:
            yield None, "This version is not available right now.", real, s, gr.update(visible=False)
            return
        yield (None, f"⏳ **{FRIENDLY[name]}**: starting from random static…", real, s,
               gr.update(value=None, visible=False))
        for done, step, total, secs, x in generate_stream(engines[name], k, n, s):
            status = _status(name, engines[name], done, step, total, secs, n)
            if done:
                yield to_grid(x, nrow=min(n, 8)), status, real, s, gr.update(value=save_zip(x, k, name, s), visible=True)
            else:
                yield to_grid(x, nrow=min(n, 8)), status, real, s, gr.update()

    def run_all(k, n, seed, surprise):
        k, n, s = int(k), int(n), pick_seed(seed, surprise)
        real = real_grid(k, n, s, nrow=4)
        imgs = [None] * len(ENGINE_ORDER)
        stats = [f"_{FRIENDLY[nm]}: waiting its turn…_" if nm in engines else f"_{FRIENDLY[nm]}: not available_"
                 for nm in ENGINE_ORDER]
        yield (real, *imgs, *stats, s)
        for j, nm in enumerate(ENGINE_ORDER):
            if nm not in engines:
                continue
            for done, step, total, secs, x in generate_stream(engines[nm], k, n, s):
                stats[j] = _status(nm, engines[nm], done, step, total, secs, n)
                imgs[j] = to_grid(x, nrow=min(n, 4))
                yield (real, *imgs, *stats, s)

    def explore(k, seed, surprise):
        s = pick_seed(seed, surprise)
        current, guess = denoise_frames(engines[explorer_engine], int(k), 4, s)
        last = len(current) - 1
        return (current, guess, gr.update(maximum=last, value=0, interactive=True),
                current[0], None, step_text(0, last), s)

    def step_text(i, last):
        if i == 0:
            return ("**Step 0: pure random static.** Every picture starts as noise. Move the slider to the right "
                    "to watch the AI remove it step by step.")
        if i == last:
            return f"**Step {i} of {last}: finished.** The noise is gone and four new skin-spot images remain."
        return (f"**Step {i} of {last}.** Left: the image so far, still partly noisy. Right: the AI's current "
                "guess of the finished image. Early guesses are blurry outlines; each step sharpens them.")

    def show_step(i, current, guess):
        if not current:
            return None, None, "Press **Run** first."
        i = int(i)
        return current[min(i, len(current) - 1)], guess[min(i, len(guess) - 1)], step_text(i, len(current) - 1)

    blocks_kw = {} if int(gr.__version__.split(".")[0]) >= 6 else {"theme": _theme(), "css": CSS}
    with gr.Blocks(title="DermAugment · FastDiff-Med", **blocks_kw) as demo:
        gr.Markdown("# 🩺 DermAugment\n**Realistic example images of skin conditions, made by AI in seconds**, "
                    "to help computer programs learn to recognise rare skin diseases.", elem_classes="hero")
        gr.HTML(stat_cards(results_dir))

        # ------------------------------------------------------------------ Home
        with gr.Tab("🏠 Home"):
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 1. The problem\nPrograms that help doctors spot skin diseases learn from "
                                "example photos. Some diseases are **rare**: in our dataset there are **4,693** "
                                "photos of ordinary moles but only **80** of dermatofibroma. With so few examples, "
                                "the programs rarely learn to recognise them.")
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 2. Our idea\nUse a **diffusion model**, an AI that draws new images by "
                                "removing noise step by step, to create extra example images of the rare "
                                "diseases. The catch: the standard method needs **1,000 steps** per image. "
                                "We changed the recipe and trained a shortcut so it needs only **4**.")
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 3. The result\nOne image now takes **0.07 seconds instead of 16**. A set of "
                                "2,360 extra images takes **8 seconds instead of about an hour**. Adding them "
                                "helped a recognition program find rare diseases, especially when real photos "
                                "were scarce.")
            gr.Markdown("### What you can do here\n"
                        "| Tab | What it does |\n|---|---|\n"
                        "| 🎨 **Create images** | Pick a skin condition and make new example images; download them |\n"
                        "| ⏱️ **Speed race** | All four versions make the same images, side by side with a timer |\n"
                        "| 🔍 **Watch it work** | Step through the process: random noise slowly becomes a skin spot |\n\n"
                        "Full results, the method and its limits are in the project report.")

        # ------------------------------------------------------------------ Create
        with gr.Tab("🎨 Create images"):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        cls = gr.Dropdown(lesion_choices, value=6, label="① Skin condition",
                                          info="RARE = very few real photos exist, so extra images help most")
                        engine = gr.Radio(engine_choices, value=names[0] if names else None, label="② Speed")
                        n = gr.Slider(1, 32, value=8, step=1, label="③ How many images")
                    go = gr.Button("Create images", variant="primary", size="lg")
                    with gr.Accordion("Options", open=False):
                        surprise = gr.Checkbox(value=True, label="🎲 New images every click",
                                               info="Untick to choose a variation number yourself")
                        seed = gr.Number(value=0, precision=0, label="Variation number",
                                         info="The number used last time is shown here. Untick 🎲 and keep it "
                                              "to get exactly the same images again.")
                    if BASELINE in engines and str(device) == "cpu":
                        gr.Markdown("💡 The 🐢 original version is very slow on a normal computer (minutes). "
                                    "That slowness is exactly what this project fixes.", elem_classes="note")
                with gr.Column(scale=7):
                    status = gr.Markdown("Your images will appear here. While the AI works you'll see random "
                                         "static slowly turning into skin spots.")
                    gr.Markdown("**AI-made images** (use the corner button to view full screen)")
                    out = gr.Image(type="numpy", interactive=False, show_label=False)
                    download = gr.File(label="⬇️ Download these images (ZIP, 28×28 PNG like the real dataset)",
                                       visible=False)
                    gr.Markdown("**Real photos of the same condition**, for comparison")
                    real_out = gr.Image(type="numpy", interactive=False, show_label=False)
                    gr.Markdown("Real photos in this dataset are also tiny (28×28 pixels) and soft, so the blur is "
                                "normal and not a fault of the AI.", elem_classes="note")
            go.click(run_one, [cls, n, engine, seed, surprise], [out, status, real_out, seed, download])
            cls.change(real_tiles, [cls, n, seed], real_out)
            demo.load(real_tiles, [cls, n, seed], real_out)

        # ------------------------------------------------------------------ Speed race
        with gr.Tab("⏱️ Speed race"):
            gr.Markdown("All four versions make **the same images from the same random static**, one after "
                        "another, so you can compare speed and quality fairly. Use **1–2 images**, because the "
                        "🐢 original version is slow.")
            with gr.Row():
                race_cls = gr.Dropdown(lesion_choices, value=6, label="Skin condition", scale=3)
                race_n = gr.Slider(1, 8, value=2, step=1, label="How many images", scale=2)
                race_btn = gr.Button("Start the race", variant="primary", scale=1)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**📷 Real photos**")
                    cmp_real = gr.Image(label="Real photos", type="numpy", interactive=False)
                cmp_imgs, cmp_stats = [], []
                for nm in ENGINE_ORDER:
                    with gr.Column():
                        cmp_stats.append(gr.Markdown(f"**{FRIENDLY[nm]}**"))
                        cmp_imgs.append(gr.Image(label=FRIENDLY[nm], type="numpy", interactive=False))
            gr.Markdown("💡 **Why do ⚡, ✨ and 🔧 make almost identical images?** The fast versions were trained to "
                        "copy the 🔧 improved model's result in fewer steps, so the same starting static leads to "
                        "the same image. That is the proof the shortcut works. The 🐢 original is a separate "
                        "model, so its images look different.", elem_classes="note")
            race_seed = gr.State(0)
            race_surprise = gr.State(True)
            race_btn.click(run_all, [race_cls, race_n, race_seed, race_surprise],
                           [cmp_real] + cmp_imgs + cmp_stats + [race_seed])

        # ------------------------------------------------------------------ Watch it work
        with gr.Tab("🔍 Watch it work"):
            gr.Markdown("A diffusion model starts from **pure random noise** and removes a little at each step. "
                        "Press **Run**, then drag the slider to move through the steps. On the right you can see "
                        "what the AI *thinks* the finished image will look like at that moment.")
            with gr.Row():
                ex_cls = gr.Dropdown(lesion_choices, value=4, label="Skin condition", scale=3)
                ex_btn = gr.Button("Run", variant="primary", scale=1)
            ex_text = gr.Markdown("Press **Run** to start.")
            ex_step = gr.Slider(0, 20, value=0, step=1, label="Step", interactive=False)
            with gr.Row():
                ex_now = gr.Image(label="The image so far", type="numpy", interactive=False)
                ex_guess = gr.Image(label="The AI's guess of the finished image", type="numpy", interactive=False)
            gr.Markdown(f"Uses the {FRIENDLY.get(explorer_engine, '')} version. Mathematically, the right-hand "
                        "image is the model's prediction of x₀ (the clean image) from the noisy image xₜ.",
                        elem_classes="note")
            ex_current, ex_guesses = gr.State([]), gr.State([])
            ex_seed, ex_surprise = gr.State(0), gr.State(True)
            if explorer_engine:
                ex_btn.click(explore, [ex_cls, ex_seed, ex_surprise],
                             [ex_current, ex_guesses, ex_step, ex_now, ex_guess, ex_text, ex_seed])
            ex_step.change(show_step, [ex_step, ex_current, ex_guesses], [ex_now, ex_guess, ex_text])

        gr.Markdown(DISCLAIMER, elem_classes="note")
    return demo


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=os.environ.get("FASTDIFF_CKPT_DIR", "checkpoints"))
    p.add_argument("--results_dir", default=os.environ.get("FASTDIFF_RESULTS_DIR", "results"))
    p.add_argument("--share", action="store_true")
    a = p.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    launch_demo(build_demo(load_engines(a.ckpt_dir), a.results_dir), share=a.share)


if __name__ == "__main__":
    main()
