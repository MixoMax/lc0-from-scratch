"""Offline teacher-to-student distillation from a pinned snapshot and its replay."""

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from checkpoints import Remote, Store, file_hash
from hardware import amp, configure_device
from network import ChessNet, legal_logits
from replay import Replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--teacher-dir",
        type=Path,
        required=True,
        help="a trainer workdir or an empty download directory",
    )
    p.add_argument(
        "--server",
        default=os.getenv("CHECKPOINT_URL"),
        help="used only when the teacher-dir is empty",
    )
    p.add_argument("--output", type=Path, default=Path("student.pt"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--steps", type=int, default=20000, help="absolute number of student updates")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--save-every", type=int, default=250)
    args = p.parse_args()
    if min(args.steps, args.batch, args.save_every) <= 0:
        p.error("steps, batch and save-every must be positive")
    if args.output.resolve() == (args.teacher_dir / "latest.pt").resolve():
        p.error("student output must be a separate file")
    device, dtype = configure_device(args.device, args.dtype)
    torch.set_num_threads(2)
    remote = None
    if args.server:
        remote = Remote(args.server, os.environ.get("CHECKPOINT_TOKEN"))
    store = Store(args.teacher_dir, remote)
    try:
        source = store.load(pinned_local=True)
        if source is None or not source["rows"]:
            raise ValueError("a trained teacher and completed replay are required")
        identity = file_hash(store.latest)
        cfg = source["config"]
        teacher = ChessNet(cfg["width"], cfg["blocks"]).to(
            device, memory_format=torch.channels_last
        )
        teacher.load_state_dict(source["model"])
        teacher.eval().requires_grad_(False)
        buffer = Replay(min(source["rows"], cfg["capacity"]))
        buffer.restore(store.refs, store.shards, source["rows"])
    finally:
        store.close()
    del source
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    student = ChessNet(width=88, blocks=3, key_size=16).to(
        device, memory_format=torch.channels_last
    )
    opt = torch.optim.AdamW(student.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    step = 0
    if args.output.exists():
        state = torch.load(args.output, map_location="cpu", weights_only=True)
        if state["teacher"] != identity:
            raise ValueError("existing student uses a different teacher snapshot")
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["numpy_rng"]
        step = state["step"]
        del state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False

    def stop(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save():
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        torch.save(
            {
                "format": 1,
                "teacher": identity,
                "step": step,
                "architecture": {"width": 88, "blocks": 3, "key_size": 16},
                "model": student.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "numpy_rng": rng.bit_generator.state,
            },
            tmp,
        )
        os.replace(tmp, args.output)

    print(
        f"teacher={identity[:12]} positions={len(buffer)} student_parameters={sum(x.numel() for x in student.parameters())} "
        f"start_step={step} device={device} dtype={dtype}",
        flush=True,
    )
    t, last = time.monotonic(), step
    while step < args.steps and not interrupted:
        x, actions, _, _ = buffer.sample(rng, args.batch)
        x = torch.as_tensor(x, device=device)
        ids = torch.as_tensor(actions, dtype=torch.long, device=device)
        with torch.no_grad(), amp(device, dtype):
            p, v = teacher(x)
            policy = legal_logits(p, ids).softmax(1)
            value = v.float().softmax(1)
        opt.zero_grad(set_to_none=True)
        with amp(device, dtype):
            p, v = student(x)
            loss = (
                -(policy * legal_logits(p, ids).log_softmax(1)).sum(1).mean()
                - (value * v.float().log_softmax(1)).sum(1).mean()
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled()
        )
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < scale:
            continue
        step += 1
        if step % args.save_every == 0:
            save()
            print(
                f"student_step={step} loss={loss.item():.4f} rows_s={(step - last) * args.batch / (time.monotonic() - t):.1f}",
                flush=True,
            )
            t, last = time.monotonic(), step
    save()
    print(f"saved {args.output} step={step}", flush=True)


if __name__ == "__main__":
    main()
