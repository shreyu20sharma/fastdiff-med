"""Batched sampling + timing helpers shared by evaluate.py, augment_experiment.py and the app."""
import time

import torch

from src.utils import autocast


def nfe_per_image(sampler, steps, guidance):
    n = steps if sampler == "ddim" else 1000
    return n * (2 if guidance != 1.0 else 1)


def sampler_iter(model, diffusion, y, sampler, steps, guidance, generator=None):
    if sampler == "ddpm":
        return diffusion.ddpm_sample_iter(model, y, guidance=guidance, generator=generator)
    return diffusion.ddim_sample_iter(model, y, steps, guidance=guidance, generator=generator)


@torch.no_grad()
def sample_images(model, diffusion, labels, sampler="ddim", steps=20, guidance=2.0, batch=500,
                  seed=0, progress=False):
    """labels: 1-D LongTensor/list. Returns float tensor in [-1, 1] on CPU (N x 3 x 32 x 32)."""
    device = next(model.parameters()).device
    labels = torch.as_tensor(labels, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, len(labels), batch):
        y = labels[i:i + batch].to(device)
        with autocast(device):
            x = None
            for _, _, x, _ in sampler_iter(model, diffusion, y, sampler, steps, guidance, g):
                pass
        out.append(x.float().clamp(-1, 1).cpu())
        if progress:
            print(f"    {min(i + batch, len(labels))}/{len(labels)}", flush=True)
    return torch.cat(out)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    t0 = time.perf_counter()
    r = fn()
    sync(device)
    return r, time.perf_counter() - t0
