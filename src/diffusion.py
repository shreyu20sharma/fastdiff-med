"""Diffusion process: schedules, eps/v targets, DDPM & DDIM samplers, CFG, distillation target.

Conventions
-----------
* Timesteps t are integers in [0, T-1]; alpha_bar[t] is the signal fraction after t+1 noising
  steps. We also allow t = -1, meaning "clean data" (alpha_bar = 1), which is where the last
  DDIM step lands.
* alpha_t = sqrt(alpha_bar_t), sigma_t = sqrt(1 - alpha_bar_t), so x_t = alpha_t x0 + sigma_t eps.
* v-prediction: v = alpha_t eps - sigma_t x0, so x0 = alpha_t x_t - sigma_t v and
  eps = sigma_t x_t + alpha_t v (no division, see proposal section 3.2).
* Classifier-free guidance: out = uncond + w * (cond - uncond); w = 1 means no guidance.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def linear_alpha_bar(T, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    return np.cumprod(1.0 - betas)


def cosine_alpha_bar(T, s=0.008, max_beta=0.999):
    def f(t):
        return math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2

    betas = np.array([min(1 - f(i + 1) / f(i), max_beta) for i in range(T)], dtype=np.float64)
    return np.cumprod(1.0 - betas)


class GaussianDiffusion:
    def __init__(self, schedule="cosine", pred="v", T=1000):
        assert schedule in ("linear", "cosine") and pred in ("eps", "v")
        self.schedule, self.pred, self.T = schedule, pred, T
        ab = linear_alpha_bar(T) if schedule == "linear" else cosine_alpha_bar(T)
        self.alpha_bar = ab  # float64 numpy, index t in [0, T-1]
        # extended table: index t+1, so t = -1 maps to alpha_bar = 1
        ab_ext = np.concatenate([[1.0], ab])
        self._alpha = torch.tensor(np.sqrt(ab_ext), dtype=torch.float64)
        self._sigma = torch.tensor(np.sqrt(1.0 - ab_ext), dtype=torch.float64)
        betas = 1.0 - ab / np.concatenate([[1.0], ab[:-1]])
        ab_prev = np.concatenate([[1.0], ab[:-1]])
        self._post_c0 = torch.tensor(np.sqrt(ab_prev) * betas / (1 - ab))
        self._post_ct = torch.tensor(np.sqrt(1 - betas) * (1 - ab_prev) / (1 - ab))
        post_var = betas * (1 - ab_prev) / (1 - ab)
        self._post_logvar = torch.tensor(np.log(np.maximum(post_var, post_var[1] if T > 1 else 1e-20)))
        self._cache = {}

    # ------------------------------------------------------------------ coefficients
    def _table(self, name, device):
        key = (name, str(device))
        if key not in self._cache:
            self._cache[key] = getattr(self, name).to(device=device, dtype=torch.float32)
        return self._cache[key]

    def coef(self, t, x):
        """(alpha_t, sigma_t) shaped (B,1,1,1) for integer timesteps t (tensor, -1 allowed)."""
        t = self._as_t(t, x)
        a = self._table("_alpha", x.device)[t + 1]
        s = self._table("_sigma", x.device)[t + 1]
        return a.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1)

    @staticmethod
    def _as_t(t, x):
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), int(t), dtype=torch.long)
        t = t.to(x.device).long()
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return t

    def snr(self, t, x):
        a, s = self.coef(t, x)
        return a ** 2 / s ** 2

    # ------------------------------------------------------------------ forward process
    def q_sample(self, x0, t, eps):
        a, s = self.coef(t, x0)
        return a * x0 + s * eps

    def get_v(self, x0, eps, t):
        a, s = self.coef(t, x0)
        return a * eps - s * x0

    def v_to_x0(self, x_t, v, t):
        a, s = self.coef(t, x_t)
        return a * x_t - s * v

    def v_to_eps(self, x_t, v, t):
        a, s = self.coef(t, x_t)
        return s * x_t + a * v

    def eps_to_x0(self, x_t, eps, t):
        a, s = self.coef(t, x_t)
        return (x_t - s * eps) / a

    def output_to_x0_eps(self, out, x_t, t):
        if self.pred == "v":
            return self.v_to_x0(x_t, out, t), self.v_to_eps(x_t, out, t)
        return self.eps_to_x0(x_t, out, t), out

    # ------------------------------------------------------------------ training
    def training_loss(self, model, x0, y, label_dropout=0.1):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        target = eps if self.pred == "eps" else self.get_v(x0, eps, t)
        if label_dropout > 0:
            drop = torch.rand(b, device=x0.device) < label_dropout
            y = torch.where(drop, torch.full_like(y, model.null_label), y)
        out = model(x_t, t, y)
        return F.mse_loss(out.float(), target)

    # ------------------------------------------------------------------ model call + CFG
    def model_output(self, model, x_t, t, y, guidance=1.0):
        t = self._as_t(t, x_t)
        if guidance == 1.0:
            return model(x_t, t, y).float()
        null = torch.full_like(y, model.null_label)
        out = model(torch.cat([x_t, x_t]), torch.cat([t, t]), torch.cat([y, null])).float()
        cond, uncond = out.chunk(2)
        return uncond + guidance * (cond - uncond)

    def predict_x0_eps(self, model, x_t, t, y, guidance=1.0, clip=True):
        out = self.model_output(model, x_t, t, y, guidance)
        x0, eps = self.output_to_x0_eps(out, x_t, t)
        if clip:
            x0 = x0.clamp(-1, 1)
            a, s = self.coef(t, x_t)
            eps = (x_t - a * x0) / s  # keep (x0, eps) consistent with x_t after clipping
        return x0, eps

    # ------------------------------------------------------------------ DDIM
    def ddim_timesteps(self, n, spacing="trailing"):
        """Descending timesteps for an n-step sampler. 'trailing' always starts at T-1."""
        if spacing == "trailing":
            ts = np.round(self.T - np.arange(n) * self.T / n).astype(np.int64) - 1
        elif spacing == "leading":
            ts = (np.arange(n) * (self.T // n))[::-1].astype(np.int64)
        else:
            raise ValueError(spacing)
        return [int(t) for t in ts]

    def ddim_step(self, x0, eps, t_next):
        a, s = self.coef(t_next, x0)
        return a * x0 + s * eps

    @torch.no_grad()
    def ddim_sample_iter(self, model, y, steps, guidance=1.0, clip=True, spacing="trailing",
                         x_T=None, generator=None, img_shape=(3, 32, 32)):
        """Yields (i, n, x_next, x0_pred) after each step. The final x_next is the sample."""
        device = y.device
        x = x_T if x_T is not None else torch.randn((y.shape[0], *img_shape), device=device,
                                                       generator=generator)
        ts = self.ddim_timesteps(steps, spacing)
        nxt = ts[1:] + [-1]
        for i, (t, tn) in enumerate(zip(ts, nxt)):
            x0, eps = self.predict_x0_eps(model, x, t, y, guidance, clip)
            x = self.ddim_step(x0, eps, tn)
            yield i, steps, x, x0

    def ddim_sample(self, model, y, steps, **kw):
        x = None
        for _, _, x, _ in self.ddim_sample_iter(model, y, steps, **kw):
            pass
        return x

    # ------------------------------------------------------------------ DDPM (ancestral, all T steps)
    @torch.no_grad()
    def ddpm_sample_iter(self, model, y, guidance=1.0, clip=True, x_T=None, generator=None,
                         img_shape=(3, 32, 32)):
        device = y.device
        x = x_T if x_T is not None else torch.randn((y.shape[0], *img_shape), device=device,
                                                       generator=generator)
        c0 = self._table("_post_c0", device)
        ct = self._table("_post_ct", device)
        lv = self._table("_post_logvar", device)
        for i, t in enumerate(range(self.T - 1, -1, -1)):
            x0, _ = self.predict_x0_eps(model, x, t, y, guidance, clip)
            mean = c0[t] * x0 + ct[t] * x
            if t > 0:
                noise = torch.randn(x.shape, device=device, generator=generator)
                x = mean + torch.exp(0.5 * lv[t]) * noise
            else:
                x = mean
            yield i, self.T, x, x0

    def ddpm_sample(self, model, y, **kw):
        x = None
        for _, _, x, _ in self.ddpm_sample_iter(model, y, **kw):
            pass
        return x

    # ------------------------------------------------------------------ progressive distillation
    def x0_target_from_endpoints(self, x_t, x_next, t, t_next):
        """x~0 such that ONE DDIM step t -> t_next from x_t lands exactly on x_next.

        x~0 = (x_next - (s''/s) x_t) / (a'' - (s''/s) a)   (proposal section 3, Stage 2)
        """
        a, s = self.coef(t, x_t)
        a2, s2 = self.coef(t_next, x_t)
        r = s2 / s
        return (x_next - r * x_t) / (a2 - r * a)

    @torch.no_grad()
    def teacher_two_step_target(self, teacher, x_t, t, t_mid, t_next, y, guidance=1.0, clip=True):
        x0, eps = self.predict_x0_eps(teacher, x_t, t, y, guidance, clip)
        x_mid = self.ddim_step(x0, eps, t_mid)
        x0b, epsb = self.predict_x0_eps(teacher, x_mid, t_mid, y, guidance, clip)
        x_next = self.ddim_step(x0b, epsb, t_next)
        return self.x0_target_from_endpoints(x_t, x_next, t, t_next)

    def distill_loss(self, student, x_t, t, y, x0_target):
        """Student predicts v; loss is on x0 with truncated-SNR weight max(SNR, 1)."""
        out = student(x_t, self._as_t(t, x_t), y).float()
        x0_pred, _ = self.output_to_x0_eps(out, x_t, t)
        w = self.snr(t, x_t).clamp(min=1.0)
        return (w * (x0_pred - x0_target) ** 2).mean()
