# Phase 2 — Bottleneck Selection & Proposed Modification

**Project title:** *FastDiff-Med: 250× faster diffusion sampling for synthetic medical-image augmentation*

## 1. Selected architecture and limitation
* **Architecture:** Denoising Diffusion Probabilistic Model (DDPM) with a class-conditional U-Net.
* **Limitation:** **slow generation.** The baseline needs **T = 1000 sequential U-Net calls**
  per image, or 2000 with classifier-free guidance. Simply skipping steps with the baseline
  formulation sharply degrades quality. This makes diffusion impractical for interactive tools
  or for producing large synthetic datasets on modest hardware.

## 2. Why the baseline cannot just take fewer steps
The baseline uses (a) a **linear β schedule** and (b) **ε-prediction**. Both hurt few-step
sampling:

1. **Linear schedule wastes steps.** With β_t linear from 1e-4 to 0.02, ᾱ_t (the fraction
   of signal kept) drops close to zero by about t ≈ 600. Roughly the last 40% of timesteps are
   almost pure noise. A coarse 10-step sampler spends most of its steps there and gives too
   few steps to the low-noise range, where the image detail forms.
2. **ε-prediction is ill-conditioned at high noise.** The clean image estimate is
   x̂₀ = (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t. When ᾱ_t → 0, dividing by √ᾱ_t **amplifies any error in ε̂
   enormously**. With 1000 small steps this is corrected gradually. With a few large steps
   the error is not corrected.
3. **Leading timestep spacing never starts at pure noise.** The usual spacing (0, 100, …, 900)
   starts sampling at t = 900 instead of t = 999, which creates a train/test mismatch
   (Lin et al., 2024).
4. **Even a perfect ODE solver needs about 20 steps.** Sampling follows a curved trajectory,
   and no numerical solver can follow that curve accurately in 1–4 steps. Going further needs
   a *learned* shortcut.

## 3. Proposed modification: two stages
The network architecture, parameter count, data and training recipe are **kept identical**
to the baseline, so every gain comes from the formulation.

### Stage 1: re-engineer the diffusion process (1000 → 20 steps)

**3.1 Cosine noise schedule** (Nichol & Dhariwal, 2021)
  ᾱ_t = f(t)/f(0),  f(t) = cos²( (t/T + s)/(1 + s) · π/2 ),  s = 0.008
The log-SNR falls almost linearly in t, so information is destroyed at a steady rate. Every
step of a coarse sampler then covers an equal amount of denoising work.

**3.2 v-prediction** (Salimans & Ho, 2022)
The network predicts v_t = √ᾱ_t · ε − √(1−ᾱ_t) · x₀. The clean image and noise are recovered
*without division*:
  x̂₀ = √ᾱ_t · x_t − √(1−ᾱ_t) · v̂,   ε̂ = √(1−ᾱ_t) · x_t + √ᾱ_t · v̂
Since ‖∂x̂₀/∂v̂‖ = √(1−ᾱ_t) ≤ 1, errors are never amplified, even at pure noise. The v-loss
equals an x₀-loss weighted by (1 + SNR), which is balanced across noise levels.

**3.3 Deterministic DDIM sampler with trailing timesteps** (Song et al., 2021; Lin et al., 2024)
  x_{τ_{i+1}} = √ᾱ_{τ_{i+1}} · x̂₀ + √(1−ᾱ_{τ_{i+1}}) · ε̂     (η = 0)
The timestep subsequence always starts at t = T−1, and x̂₀ is clipped to [−1, 1].

### Stage 2: progressive distillation (20 → 4 steps) (Salimans & Ho, 2022)
A *student* network, initialised from the teacher, learns to match **two** teacher DDIM
steps with **one** of its own. For a student step from t to t'' via the midpoint t':
  1. The teacher takes two DDIM steps, x_t → x_{t'} → x_{t''}.
  2. Solve for the x̃₀ that a *single* DDIM step t → t'' would need to land exactly on
     x_{t''}:
     x̃₀ = ( x_{t''} − (σ_{t''}/σ_t)·x_t ) / ( α_{t''} − (σ_{t''}/σ_t)·α_t ),
     where α = √ᾱ and σ = √(1−ᾱ).
  3. Train the student, in v-space, to predict x̃₀, with loss weight max(SNR, 1)
     ("truncated SNR").
  4. The student becomes the new teacher. Repeat: 32 → 16 → 8 → 4 steps (or down to 2 if time
     allows).
Classifier-free guidance is folded into the teacher's targets, so the student needs **one**
forward pass per step instead of two.

### Expected effect
| Model | Steps | Network calls per image |
|---|---:|---:|
| Baseline DDPM (with guidance) | 1000 | 2000 |
| Stage 1 (with guidance) | 20 | 40 |
| Stage 2 (guidance distilled) | 4 | **4** (≈ 500× fewer) |

## 4. Experimental plan (Phase 3)
**Dataset:** **DermaMNIST** (MedMNIST v2). It has 10,015 dermatoscopic images of 7 skin-lesion
types, which we resize to 32×32 RGB. It is *heavily imbalanced*: nevi make up about 67% of
images, while dermatofibroma has only about 115. That imbalance is what motivates synthetic
augmentation.
*Fallback if time is short: BloodMNIST, which has 8 classes and cleaner images.*

**Ablation (same U-Net, same training budget)**
| Run | Schedule | Target | Purpose |
|---|---|---|---|
| A (baseline) | linear | ε | reference |
| B | cosine | ε | effect of the schedule alone |
| C | linear | v | effect of the target alone |
| D (Stage 1) | cosine | v | full Stage 1 |
| E (Stage 2) | D + distillation | — | 4-step student |

Each model is evaluated with DDIM at 5, 10, 20 and 50 steps, and the baseline also with
1000-step DDPM.

**Metrics**
1. **Quality:** FID (Inception features, via `torch-fidelity`, on 5k samples vs. the test set)
   and the class accuracy of generated samples, judged by a separately trained classifier.
2. **Speed:** number of network calls, ms per image (batched on a Colab T4 GPU) and latency
   for one request.
3. **Real-world utility (main result):** train a ResNet-18 classifier on (i) real data only,
   (ii) real data plus Stage 2 synthetic images that balance the rare classes. Report
   **balanced accuracy and macro-F1** on the official test split. Also report the time needed
   to generate the synthetic set with the baseline vs. the improved model.

**Deliverables:** an FID-vs-steps chart, a latency chart, an ablation table, sample grids, a
utility table and a training-loss curve.

## 5. Real-world use case (Phase 4)
**DermAugment**, a Gradio tool for clinical ML teams. Pick a rare lesion class and get synthetic
training images in under a second, with the baseline and FastDiff side by side and live timers.
A second tab shows the classifier result: how much the augmentation improved detection of
rare classes.
*Disclaimer: research/education only. Synthetic images are not for diagnosis.*

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Distillation fails or loses quality | Stage 1 alone is a complete result; report distillation as a stretch result either way |
| Augmentation gives no gain | Report honestly; also run a low-data setting (20% of real data), where gains are larger |
| Colab session limit | Save checkpoints to Google Drive every 1k steps; shorten runs B and C in the ablation |
