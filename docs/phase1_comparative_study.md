# Phase 1 — Comparative Study of Generative Architectures

This report compares the four architectures named in the assignment (Transformer, GAN, VAE and
Diffusion model) on four axes: training stability, output quality, generation speed, and
how easy the latent space is to interpret. It ends by choosing the architecture taken forward
to Phases 2–4.

---

## 1. How each architecture works

### 1.1 Autoregressive Transformer
A Transformer (Vaswani et al., 2017) is built from stacked **self-attention** layers. Each token
builds a query **Q**, key **K** and value **V**, and attention is computed as
`softmax(QKᵀ/√d)·V`. As a *generative* model (GPT style), it factorises the data likelihood
into a product of conditionals:

  p(x) = ∏ᵢ p(xᵢ | x₁ … xᵢ₋₁)

It is trained with teacher-forced cross-entropy (maximum likelihood) and generates **one token
at a time**. Images are handled by flattening pixels or VQ codes into a sequence (ImageGPT,
Parti, DALL·E 1).

### 1.2 Generative Adversarial Network (GAN)
A GAN (Goodfellow et al., 2014) sets two networks against each other. A **generator** G maps
noise z ~ N(0, I) to a sample, and a **discriminator** D tries to tell real samples from
generated ones. They play a minimax game:

  min_G max_D 𝔼ₓ[log D(x)] + 𝔼_z[log(1 − D(G(z)))]

At the optimum, G's distribution equals the data distribution. Sampling takes **one forward
pass**. StyleGAN, BigGAN and GigaGAN are well-known examples.

### 1.3 Variational Autoencoder (VAE)
A VAE (Kingma & Welling, 2014) has an **encoder** q_φ(z|x) that outputs a Gaussian over a
low-dimensional latent, and a **decoder** p_θ(x|z). It maximises the evidence lower bound
(ELBO):

  log p(x) ≥ 𝔼_q[log p_θ(x|z)] − KL(q_φ(z|x) ‖ N(0, I))

The *reparameterisation trick* (z = μ + σ·ε) makes this differentiable. Sampling is one
decoder pass on z ~ N(0, I).

### 1.4 Diffusion model (DDPM)
A diffusion model (Sohl-Dickstein 2015; Ho et al. 2020) sets up a fixed **forward process**
that gradually adds Gaussian noise over T steps:

  q(x_t | x₀) = N(√ᾱ_t · x₀, (1 − ᾱ_t) I)

A neural network (usually a U-Net) learns the **reverse process**, removing a little noise at
each step. In practice it is trained to predict the added noise ε with a plain MSE loss:

  L = 𝔼_{t, x₀, ε} ‖ε − ε_θ(√ᾱ_t x₀ + √(1−ᾱ_t) ε, t)‖²

Generation starts from pure noise x_T and applies the network **T times** (usually T = 1000).
Diffusion powers Stable Diffusion, DALL·E 2/3, Imagen and Sora-class video models.

---

## 2. Side-by-side comparison

| Criterion | Transformer (AR) | GAN | VAE | Diffusion |
|---|---|---|---|---|
| **Training stability** | High: plain cross-entropy, scales predictably | **Low**: adversarial game; mode collapse, oscillation, needs careful tuning (spectral norm, R1, TTUR) | High: single ELBO objective; posterior collapse can occur | **High**: simple regression loss, no adversary, rarely diverges |
| **Output quality** | High (needs a very large scale for images) | High sharpness, but **low diversity** (mode dropping) | **Low/blurry**: Gaussian likelihood averages modes | **State of the art** in both fidelity and diversity (best FID on ImageNet, COCO) |
| **Generation speed** | Slow: O(sequence length) sequential steps | **Fastest**: 1 forward pass | **Fast**: 1 decoder pass | **Slowest**: 100–1000 sequential network calls |
| **Latent-space interpretability** | Low: no global latent; control comes from prompts | Medium: StyleGAN's W space is disentangled and editable, but there is no encoder (needs inversion) | **High**: explicit, smooth, low-dimensional latent with an encoder | Medium: x_T is the same size as the data; DDIM inversion makes it deterministic and editable, but it is not compact |
| **Likelihood** | Exact | None | Lower bound | Lower bound (variational) |
| **Mode coverage** | Good | Poor | Good | **Excellent** |
| **Main failure mode** | Exposure bias, slow decoding | Mode collapse, instability | Blurriness | **Slow sampling**, high compute |

### Discussion
* **Stability vs. quality.** GANs give sharp images cheaply but are fragile to train. VAEs are
  stable but blurry. Diffusion models are the only family that is both stable to train *and*
  top of the table for quality and diversity. That is why they replaced GANs as the default
  for image, audio and video generation.
* **Speed is the price diffusion pays.** The quality comes from refining the sample over many
  small denoising steps, and each step is a full U-Net call. A 1000-step DDPM is roughly
  1000× slower at inference than a GAN with the same backbone. For interactive products
  (design tools, on-device apps, real-time augmentation) this is the main bottleneck.
* **Interpretability.** VAEs give the most usable latent. Diffusion latents are high-dimensional
  but, with a deterministic sampler (DDIM), each noise map maps to exactly one image. That
  allows interpolation and inversion.

---

## 3. Advantages and disadvantages of the chosen architecture: Diffusion

**Advantages**
1. Stable, non-adversarial training with a simple MSE objective.
2. State-of-the-art sample quality *and* diversity (no mode collapse).
3. Flexible conditioning (class, text, image) and classifier-free guidance, which trades
   diversity for fidelity at inference time.
4. Supports inpainting, editing and super-resolution without retraining.
5. Has a principled likelihood bound and a score-based / SDE interpretation.

**Disadvantages**
1. **Slow sampling**: hundreds to thousands of sequential network evaluations per image.
2. High training compute and memory.
3. The latent is not compact (same dimensionality as the data).
4. Design choices interact (noise schedule, prediction target, sampler); a poor combination
   breaks few-step sampling.
5. Likelihood estimates are less tight than autoregressive models.

## 4. How the architectures converge
The four families are increasingly combined rather than competing:
* **Latent Diffusion / Stable Diffusion** runs diffusion inside a **VAE's** latent space, so
  the diffusion model works on a compact latent instead of full-size pixels.
* **Diffusion Transformers (DiT, SD3, Sora)** replace the U-Net with a **Transformer** backbone.
* **Adversarial distillation (SDXL-Turbo)** uses a **GAN**-style discriminator to compress a
  diffusion sampler into 1–4 steps.
* **VQ-VAE + Transformer** (DALL·E 1, Parti) generates discrete VAE codes autoregressively.

Diffusion is the common thread in today's state-of-the-art systems, and its main weakness,
speed, is exactly what the other families are borrowed to fix.

## 5. Decision

We take the **Diffusion model** forward and target **disadvantage #1, slow sampling**. It is
the limitation with the biggest practical impact, it can be measured objectively (latency,
number of function evaluations, FID at a fixed step budget), and it can be fixed with principled
mathematical changes plus a learned shortcut (progressive distillation) rather than more
compute. See [Phase 2](phase2_proposal.md).

### References
Ho, Jain & Abbeel (2020) *Denoising Diffusion Probabilistic Models*. ·
Song, Meng & Ermon (2021) *Denoising Diffusion Implicit Models*. ·
Nichol & Dhariwal (2021) *Improved DDPM*. ·
Salimans & Ho (2022) *Progressive Distillation for Fast Sampling of Diffusion Models*. ·
Lin et al. (2024) *Common Diffusion Noise Schedules and Sample Steps are Flawed*. ·
Ho & Salimans (2022) *Classifier-Free Diffusion Guidance*. ·
Goodfellow et al. (2014) *GANs*. · Kingma & Welling (2014) *Auto-Encoding Variational Bayes*. ·
Vaswani et al. (2017) *Attention Is All You Need*. · Karras et al. (2019) *StyleGAN*.
