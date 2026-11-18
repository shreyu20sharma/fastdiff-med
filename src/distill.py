"""Progressive distillation (proposal section 3, Stage 2) with classifier-free guidance folded in.

    python -m src.distill --teacher <ckpt_dir>/D/ema.pt --ckpt_dir <ckpt_dir>

Round k: the student (initialised from the teacher) learns to match TWO teacher DDIM steps with
ONE of its own. For a student step t -> t'' with teacher midpoint t':
  1. teacher: x_t -> x_t' -> x_t''  (deterministic DDIM, CFG w applied in round 1 only)
  2. x~0 = (x_t'' - (s''/s) x_t) / (a'' - (s''/s) a)
  3. student loss: max(SNR_t, 1) * || x0_hat(student v-output) - x~0 ||^2
  4. student becomes the teacher; halve the steps.
With --start_steps 32 --end_steps 4: teacher@32 -> student@16 -> 8 -> 4.
Writes <ckpt_dir>/distill/student_{N}step.pt and distill/log.csv.
"""
import argparse
import copy
import csv
import os
import time

import torch

from src.data import load_tensors, random_flip
from src.utils import atomic_save, autocast, clone_frozen, get_device, load_model, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, help="EMA checkpoint of run D (cosine + v)")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--start_steps", type=int, default=32)
    p.add_argument("--end_steps", type=int, default=4)
    p.add_argument("--iters_per_round", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--guidance", type=float, default=2.0, help="CFG scale folded into round 1")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--fresh", action="store_true", help="retrain rounds even if students exist")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = get_device()
    set_seed(args.seed)
    amp = not args.no_amp and device.type == "cuda"
    out_dir = os.path.join(args.ckpt_dir, "distill")
    os.makedirs(out_dir, exist_ok=True)

    teacher, diffusion, cfg = load_model(args.teacher, device)
    if cfg["pred"] != "v":
        print("WARNING: distillation is designed for a v-prediction teacher (run D).")
    x_train, y_train = load_tensors("train", device)
    n = x_train.shape[0]
    teacher = clone_frozen(teacher)
    teacher_guidance = args.guidance

    log_path = os.path.join(out_dir, "log.csv")
    if args.fresh or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "student_steps", "iter", "loss"])

    n_teacher = args.start_steps
    rnd = 0
    while n_teacher > args.end_steps:
        rnd += 1
        n_student = n_teacher // 2
        ts_teacher = diffusion.ddim_timesteps(n_teacher)
        ts_student = diffusion.ddim_timesteps(n_student)
        assert ts_teacher[::2] == ts_student, "student grid must be every other teacher step"
        t_cur = torch.tensor(ts_student, device=device)
        t_mid = torch.tensor(ts_teacher[1::2], device=device)
        t_nxt = torch.tensor(ts_student[1:] + [-1], device=device)

        path = os.path.join(out_dir, f"student_{n_student}step.pt")
        if os.path.exists(path) and not args.fresh:  # resume: round already finished
            print(f"[round {rnd}] found {path}, skipping", flush=True)
            teacher = clone_frozen(load_model(path, device)[0])
            teacher_guidance = 1.0
            n_teacher = n_student
            continue

        student = copy.deepcopy(teacher).train()
        for p in student.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        print(f"[round {rnd}] teacher {n_teacher} steps (CFG w={teacher_guidance}) -> "
              f"student {n_student} steps", flush=True)

        t0, running, k = time.time(), 0.0, 0
        for it in range(1, args.iters_per_round + 1):
            lr = args.lr * (1 - (it - 1) / args.iters_per_round)  # linear decay to 0 per round
            for g in opt.param_groups:
                g["lr"] = lr
            idx = torch.randint(0, n, (args.batch,), device=device)
            x0, y = random_flip(x_train[idx]), y_train[idx]
            i = torch.randint(0, n_student, (args.batch,), device=device)
            t, tm, tn = t_cur[i], t_mid[i], t_nxt[i]
            x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))
            with autocast(device, amp):
                x0_tgt = diffusion.teacher_two_step_target(teacher, x_t, t, tm, tn, y,
                                                           guidance=teacher_guidance)
            x0_tgt = x0_tgt.float()
            with autocast(device, amp):
                loss = diffusion.distill_loss(student, x_t, t, y, x0_tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            k += 1
            if it % args.log_every == 0 or it == args.iters_per_round:
                ips = k / max(time.time() - t0, 1e-9)
                print(f"[round {rnd}] it {it:5d}/{args.iters_per_round} loss {running / k:.5f} "
                      f"{ips:.1f} it/s", flush=True)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([rnd, n_student, it, f"{running / k:.6f}"])
                t0, running, k = time.time(), 0.0, 0

        student.eval()
        scfg = dict(cfg, distilled=True, steps=n_student, guidance_folded=args.guidance,
                    run_name=f"E{n_student}")
        atomic_save(dict(config=scfg, model=student.state_dict()), path)
        print(f"[round {rnd}] saved {path}")
        teacher = clone_frozen(student)
        teacher_guidance = 1.0  # guidance already folded in after round 1
        n_teacher = n_student


if __name__ == "__main__":
    main()
