"""Small shared helpers: device, seeding, EMA, checkpoint I/O, run registry."""
import copy
import os
import random

import numpy as np
import torch

# Ablation registry: run letter -> (schedule, prediction target)
RUNS = {
    "A": ("linear", "eps"),   # baseline
    "B": ("cosine", "eps"),   # schedule only
    "C": ("linear", "v"),     # target only
    "D": ("cosine", "v"),     # full Stage 1
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ema_update(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def atomic_save(obj, path):
    """Write to a temp file then rename, so a Colab disconnect never leaves a corrupt ckpt."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def autocast(device, enabled=True):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


def load_model(path, device=None):
    """Load an EMA/student checkpoint -> (model, diffusion, config). Model is in eval mode."""
    from src.diffusion import GaussianDiffusion
    from src.unet import UNet

    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = UNet(num_classes=cfg["num_classes"], base=cfg.get("base", 64))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    diffusion = GaussianDiffusion(cfg["schedule"], cfg["pred"], T=cfg.get("T", 1000))
    return model, diffusion, cfg


def clone_frozen(model):
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
