import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.diffusion import GaussianDiffusion  # noqa: E402
from src.unet import UNet  # noqa: E402

SCHEDULES = ["linear", "cosine"]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_v_conversions_invert_exactly(schedule):
    d = GaussianDiffusion(schedule, "v")
    torch.manual_seed(0)
    x0 = torch.rand(16, 3, 32, 32) * 2 - 1
    eps = torch.randn_like(x0)
    t = torch.randint(0, d.T, (16,))
    x_t = d.q_sample(x0, t, eps)
    v = d.get_v(x0, eps, t)
    assert torch.allclose(d.v_to_x0(x_t, v, t), x0, atol=1e-5)
    assert torch.allclose(d.v_to_eps(x_t, v, t), eps, atol=1e-5)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_eps_conversion_inverts(schedule):
    d = GaussianDiffusion(schedule, "eps")
    x0 = torch.rand(8, 3, 32, 32) * 2 - 1
    eps = torch.randn_like(x0)
    t = torch.randint(0, 500, (8,))  # moderate noise; eps->x0 is ill-conditioned near T
    x_t = d.q_sample(x0, t, eps)
    assert torch.allclose(d.eps_to_x0(x_t, eps, t), x0, atol=1e-4)


@pytest.mark.parametrize("n", [1, 4, 5, 10, 16, 20, 32, 50, 64])
def test_trailing_timesteps_start_at_T_minus_1(n):
    d = GaussianDiffusion("cosine", "v")
    ts = d.ddim_timesteps(n)
    assert ts[0] == d.T - 1
    assert len(ts) == n and len(set(ts)) == n
    assert all(a > b for a, b in zip(ts, ts[1:])) and ts[-1] >= 0


def test_leading_timesteps_do_not_start_at_T_minus_1():
    d = GaussianDiffusion("linear", "eps")
    assert d.ddim_timesteps(10, "leading")[0] == 900


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_student_grid_is_every_other_teacher_step(n):
    d = GaussianDiffusion("cosine", "v")
    assert d.ddim_timesteps(2 * n)[::2] == d.ddim_timesteps(n)


def test_schedules_monotone_and_cosine_keeps_signal_longer():
    lin, cos = GaussianDiffusion("linear", "v"), GaussianDiffusion("cosine", "v")
    for d in (lin, cos):
        assert np.all(np.diff(d.alpha_bar) < 0)
    # linear destroys almost all signal by t~600; cosine does not
    assert lin.alpha_bar[600] < 0.05 and cos.alpha_bar[600] > 0.3


def test_distill_x0_target_reproduces_teacher_endpoint():
    """One DDIM step using x~0 must land exactly where two teacher steps landed."""
    d = GaussianDiffusion("cosine", "v")
    torch.manual_seed(0)
    x_t = torch.randn(4, 3, 8, 8, dtype=torch.float32)
    x_next = torch.randn_like(x_t)
    for t, tn in [(999, 749), (499, 249), (249, -1)]:
        tt = torch.full((4,), t)
        tnn = torch.full((4,), tn)
        x0_tilde = d.x0_target_from_endpoints(x_t, x_next, tt, tnn)
        a, s = d.coef(tt, x_t)
        eps_tilde = (x_t - a * x0_tilde) / s
        assert torch.allclose(d.ddim_step(x0_tilde, eps_tilde, tnn), x_next, atol=1e-4)


def test_unet_shapes_and_null_label():
    m = UNet(num_classes=7, base=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0, 999])
    y = torch.tensor([3, m.null_label])
    assert m(x, t, y).shape == x.shape


def test_samplers_run_end_to_end():
    torch.manual_seed(0)
    m = UNet(num_classes=7, base=32).eval()
    y = torch.tensor([0, 6])
    for pred in ("eps", "v"):
        d = GaussianDiffusion("cosine", pred)
        out = d.ddim_sample(m, y, 4, guidance=2.0)
        assert out.shape == (2, 3, 32, 32) and out.abs().max() <= 1.0 + 1e-5
    d = GaussianDiffusion("linear", "eps", T=10)  # short DDPM chain for speed
    assert d.ddpm_sample(m, y, guidance=2.0).shape == (2, 3, 32, 32)


def test_training_and_distill_losses_backprop():
    m = UNet(num_classes=7, base=32)
    d = GaussianDiffusion("cosine", "v")
    x0 = torch.rand(4, 3, 32, 32) * 2 - 1
    y = torch.randint(0, 7, (4,))
    d.training_loss(m, x0, y).backward()
    teacher = UNet(num_classes=7, base=32).eval()
    t = torch.full((4,), 999)
    x_t = d.q_sample(x0, t, torch.randn_like(x0))
    tgt = d.teacher_two_step_target(teacher, x_t, t, torch.full((4,), 749),
                                    torch.full((4,), 499), y, guidance=2.0)
    loss = d.distill_loss(m, x_t, t, y, tgt)
    assert torch.isfinite(loss)
    loss.backward()
