# One-Day Build Plan — FastDiff-Med

Put `phase1_comparative_study.md`, `phase2_proposal.md` and this file in `docs/`, then work
through the blocks below in Claude Code. Training runs on **Google Colab (T4 GPU)**. Write the
code locally with Claude Code, push it to GitHub, and in Colab run
`!git clone <repo> && cd <repo> && pip install -r requirements.txt`.

## Timeline (about 10–12 hours)
| # | Block | Time | Output |
|---|---|---|---|
| 1 | Repo scaffold and code (Claude Code) | 1.5 h | `src/`, `app/`, `notebooks/colab_train.ipynb` |
| 2 | Smoke test on Colab (200 steps per model) | 0.5 h | catch bugs before long runs |
| 3 | Train ablation runs A–D, ~25 min each | 1.75 h | 4 checkpoints → Google Drive |
| 4 | Progressive distillation, 32→16→8→4 (~12 min per round) | 1 h | `student_4step.pt` |
| 5 | Benchmark: FID, class accuracy, latency | 1 h | `results/*.csv`, charts, grids |
| 6 | Downstream augmentation experiment | 1 h | utility table |
| 7 | Gradio demo plus a public HF Space or Colab share link | 1 h | `app/app.py`, screenshot/GIF |
| 8 | README, results write-up, push | 1 h | final GitHub repo |
| — | Buffer | 1–2 h | — |

**If you fall behind:** drop runs B and C (keep A vs. D), then stop distillation at 8 steps.
Never drop block 6. The real-world proof is what earns marks.

## Repo structure
```
fastdiff-med/
├── README.md                  # overview, results charts, how to run, demo GIF
├── requirements.txt           # torch, torchvision, medmnist, torch-fidelity, gradio, matplotlib, scipy
├── docs/                      # phase1, phase2, BUILD_PLAN, phase3_results (auto-generated)
├── src/
│   ├── data.py                # DermaMNIST loader, 32×32, [-1,1], class counts
│   ├── unet.py                # class-conditional U-Net (same for all runs)
│   ├── diffusion.py           # schedules, eps/v targets, DDPM & DDIM (trailing) samplers, CFG
│   ├── train.py               # --schedule {linear,cosine} --pred {eps,v}; EMA; ckpt to Drive
│   ├── distill.py             # progressive distillation with CFG folded in
│   ├── evaluate.py            # FID (torch-fidelity), class acc, latency → CSV + charts + report
│   ├── augment_experiment.py  # ResNet-18 real vs. real+synthetic → balanced acc, macro-F1
│   └── judge_classifier.py    # classifier used for "class accuracy" of samples
├── app/app.py                 # Gradio demo (2 tabs)
├── notebooks/colab_train.ipynb# runs the whole pipeline end to end on Colab
├── results/                   # CSVs, PNG charts, sample grids
└── checkpoints/               # (gitignored, or small final ckpts only)
```

## Claude Code prompts (paste one at a time)

**Prompt 1: scaffold and core**
> Read docs/phase2_proposal.md. Build the repo in docs/BUILD_PLAN.md. Implement src/data.py
> (medmnist DermaMNIST, size 28, pad to 32, RGB, normalise to [-1,1]), src/unet.py (class-
> conditional U-Net, channels 64/128/256, attention at 8×8, sinusoidal time embedding, null label
> for CFG), and src/diffusion.py (linear and cosine schedules; eps and v prediction; DDPM ancestral
> sampler; DDIM sampler with trailing timesteps, x0 clipping and CFG; plus a function that returns
> the DDIM x0 target for distillation). Add unit tests checking that v→x0/eps conversions invert
> exactly and that trailing timesteps start at T-1.

**Prompt 2: training and distillation**
> Implement src/train.py (AdamW 2e-4, batch 128, EMA 0.9995, label dropout 0.1, h/v-flip
> augmentation, --steps, checkpoints every 1000 steps to a --ckpt_dir) and src/distill.py
> following section 3 "Stage 2" of the proposal exactly (teacher 2 DDIM steps → solve x̃0 →
> student v-loss with weight max(SNR,1); CFG w=2 folded into teacher; rounds 32→16→8→4; log each
> round). Mixed precision on CUDA.

**Prompt 3: evaluation and augmentation**
> Implement src/evaluate.py: for each config in the ablation table, generate 5000 class-balanced
> samples, compute FID with torch-fidelity against the test set, class accuracy with
> judge_classifier, network calls, ms/image and single-request latency (CUDA-synchronised).
> Save results/benchmark.csv, plots fid_vs_steps.png, latency.png, sample grids, and generate
> docs/phase3_results.md with tables. Then implement src/augment_experiment.py: ResNet-18
> (3 seeds) on real-only vs real+synthetic (top up each minority class to the size of the 2nd
> largest class using the 4-step student), plus the same at 20% real data. Report balanced
> accuracy and macro-F1, mean ± std, and wall-clock time to generate the synthetic set with the
> baseline vs. the student.

**Prompt 4: notebook, demo and README**
> Create notebooks/colab_train.ipynb that mounts Drive, installs requirements, and runs smoke
> test → training A–D → distillation → evaluate → augment experiment. Build app/app.py in
> Gradio: tab 1 "Generate" (class dropdown, n images, engine: Baseline-1000 / FastDiff-20 /
> Student-4, live timer, side-by-side compare button); tab 2 "Impact" showing the augmentation
> results table and charts. Add launch(share=True) for Colab. Write a README with a project
> summary, key result numbers, embedded charts, ablation table, architecture diagram (Mermaid),
> how-to-run, and a medical disclaimer.

## Tips that earn marks
* Put the **headline numbers at the top of the README**, for example: "4 steps vs 1000 · ~500×
  fewer network calls · FID X vs Y · +Z pts macro-F1 on rare lesions".
* Include the **ablation table**. It shows *why* each change matters.
* Be honest about limitations and failed attempts in a "Limitations" section. Teachers reward
  this.
* Record a 30-second GIF of the demo comparing Baseline and FastDiff and put it in the README.
* Make many small, meaningful commits throughout the day. Graders look at commit history.
