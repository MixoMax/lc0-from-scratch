"""Single-GPU AlphaZero-style chess training. Run python train.py --help."""

import argparse
import json
import math
import multiprocessing as mp
import os
import signal
import threading
import time
from pathlib import Path
from queue import Empty


def arguments():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--server",
        default=os.getenv("CHECKPOINT_URL"),
        help="HTTPS base URL of the persistent store",
    )
    p.add_argument("--local-only", action="store_true", help="explicitly use only the workdir")
    p.add_argument("--workdir", type=Path, default=Path("run"))
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument(
        "--seconds", type=float, help="wall-time budget for this invocation including startup"
    )
    p.add_argument(
        "--max-steps", type=int, help="absolute optimizer step at which to save and exit"
    )
    p.add_argument("--checkpoint-minutes", type=float, default=15)
    p.add_argument("--log-seconds", type=float, default=30)
    p.add_argument(
        "--gather-ms", type=float, default=1.5, help="maximum wait to coalesce ready actor batches"
    )
    p.add_argument(
        "--tune-seconds",
        type=float,
        default=5,
        help="GPU end-to-end trial duration per actor layout; zero skips",
    )
    p.add_argument(
        "--actors", type=int, help="CPU search processes (default uses CPU quota and RAM)"
    )
    p.add_argument("--games-per-actor", type=int, help="concurrent games in each search process")
    p.add_argument(
        "--inference-batch", type=int, help="cap on GPU inference batch (default benchmarks GPU)"
    )
    p.add_argument("--microbatch", type=int, help="training microbatch (default benchmarks GPU)")
    for name, desc in RECIPE.items():
        p.add_argument(
            "--" + name.replace("_", "-"),
            type=type(desc),
            help=f"new-run default {desc}; restored on resume",
        )
    args = p.parse_args()
    if not args.local_only and not args.server:
        p.error("set CHECKPOINT_URL or pass --server or --local-only")
    for k in (
        "seconds",
        "checkpoint_minutes",
        "log_seconds",
        "actors",
        "games_per_actor",
        "inference_batch",
        "microbatch",
    ):
        if getattr(args, k) is not None and getattr(args, k) <= 0:
            p.error(f"{k} must be positive")
    if args.gather_ms < 0 or args.tune_seconds < 0:
        p.error("gather-ms and tune-seconds must be nonnegative")
    return args


RECIPE = {
    "width": 96,
    "blocks": 10,
    "batch": 512,
    "visits": 192,
    "plies": 192,
    "capacity": 1000000,
    "warm_rows": 4096,
    "reuse": 4,
    "seed": 1234,
    "lr": 0.05,
    "decay_steps": 400000,
}


def recipe(args, state):
    cfg = state["config"].copy() if state else RECIPE.copy()
    for k in RECIPE:
        value = getattr(args, k)
        if value is not None:
            if state and cfg[k] != value:
                raise ValueError(
                    f"--{k} changes a saved learning setting ({cfg[k]}); use a separate store for a new run"
                )
            cfg[k] = value
    if any(v <= 0 for k, v in cfg.items() if k != "seed") or cfg["seed"] < 0:
        raise ValueError("recipe values must be positive and seed nonnegative")
    if (
        cfg["warm_rows"] > cfg["capacity"]
        or cfg["batch"] & (cfg["batch"] - 1)
        or cfg["visits"] > 32000
    ):
        raise ValueError(
            "warm-rows must fit replay, batch must be a power of two, visits must be <=32000"
        )
    return cfg


class Actors:
    def __init__(self, n, games, cfg, seeds):
        from encoding import MAX_LEGAL, STATE_BYTES
        from search import actor, shared_views

        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue()
        self.procs, self.events, self.views = [], [], []
        self.raw = []
        try:
            for i in range(n):
                raw = (
                    ctx.RawArray("B", games * STATE_BYTES),
                    ctx.RawArray("H", games * MAX_LEGAL),
                    ctx.RawArray("f", games * (MAX_LEGAL + 1)),
                )
                event = ctx.Event()
                proc = ctx.Process(
                    target=actor,
                    args=(
                        i,
                        raw,
                        event,
                        self.queue,
                        games,
                        cfg["visits"],
                        cfg["plies"],
                        int(seeds[i]),
                    ),
                    daemon=True,
                    name=f"search-{i}",
                )
                proc.start()
                self.raw.append(raw)
                self.procs.append(proc)
                self.events.append(event)
                self.views.append(shared_views(raw, games))
        except BaseException:
            self.close()
            raise

    def check(self):
        for proc in self.procs:
            if proc.exitcode is not None:
                raise RuntimeError(f"{proc.name} exited with code {proc.exitcode}")

    def close(self):
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        self.queue.close()


class Inference:
    def __init__(self, net, device, dtype, cap, views, events):
        import torch

        from encoding import MAX_LEGAL, STATE_BYTES

        self.net, self.device, self.dtype = net, device, dtype
        self.cap, self.views, self.events = cap, views, events
        pin = device.type == "cuda"
        self.states = torch.empty((cap, STATE_BYTES), dtype=torch.uint8, pin_memory=pin)
        self.ids = torch.empty((cap, MAX_LEGAL), dtype=torch.int32, pin_memory=pin)
        self.out = torch.empty((cap, MAX_LEGAL + 1), dtype=torch.float32, pin_memory=pin)
        self.s, self.a, self.y = self.states.numpy(), self.ids.numpy(), self.out.numpy()

    def serve(self, messages, games):
        import torch

        from hardware import amp
        from network import legal_logits

        width = max(m[1] for m in messages)
        total = len(messages) * games
        batches = 0
        # A single actor may also be larger than the GPU cap.
        for base in range(0, total, self.cap):
            count = min(self.cap, total - base)
            spans = []
            at = 0
            while at < count:
                slot, start = divmod(base + at, games)
                idx = messages[slot][0]
                n = min(count - at, games - start)
                src = slice(start, start + n)
                dst = slice(at, at + n)
                xs, ids, _ = self.views[idx]
                self.s[dst] = xs[src]
                self.a[dst, :width] = ids[src, :width]
                spans.append((idx, src, dst))
                at += n
            with torch.inference_mode(), amp(self.device, self.dtype):
                x = self.states[:count].to(self.device, non_blocking=True)
                ids = self.ids[:count, :width].to(self.device, dtype=torch.long, non_blocking=True)
                p, v = self.net(x)
                probs = legal_logits(p, ids).softmax(1)
                wdl = v.float().softmax(1)
                result = torch.cat((probs, (wdl[:, 0] - wdl[:, 2])[:, None]), 1)
                self.out[:count, : width + 1].copy_(result)
            for idx, src, dst in spans:
                target = self.views[idx][2]
                target[src, :width] = self.y[dst, :width]
                target[src, -1] = self.y[dst, width]
                if src.stop == games:
                    self.events[idx].set()
            batches += 1
        return batches


def collect(queue, actors, games, cap, gather_ms, timeout):
    messages = []
    try:
        messages.append(queue.get(timeout=timeout))
        until = time.monotonic() + gather_ms / 1000
        while len(messages) < actors and len(messages) * games < cap:
            messages.append(queue.get(timeout=max(0, until - time.monotonic())))
    except Empty:
        pass
    return messages


def tune_search(args, cfg, published, device, dtype, n, games, cap, stop, deadline):
    if device.type != "cuda" or not args.tune_seconds or (args.actors and args.games_per_actor):
        return n, games
    # Respect explicit settings and the host's previously established total game budget.
    import numpy as np

    layouts = [(n, games)]
    if not args.games_per_actor:
        layouts.append((n, max(1, games // 2)))
    if not args.actors:
        fewer = max(1, n // 2)
        layouts.append((fewer, games if args.games_per_actor else min(256, n * games // fewer)))
    best, best_rate = layouts[0], 0
    for workers, slots in dict.fromkeys(layouts):
        if stop.is_set() or time.monotonic() >= deadline:
            break
        seeds = np.random.default_rng(cfg["seed"]).integers(2**63, size=workers, dtype=np.int64)
        pool = Actors(workers, slots, cfg, seeds)
        try:
            engine = Inference(published, device, dtype, cap, pool.views, pool.events)
            seen = set()
            warm_start = None
            searches = 0
            while not stop.is_set() and time.monotonic() < deadline:
                pool.check()
                msgs = collect(pool.queue, workers, slots, cap, args.gather_ms, 0.05)
                if msgs:
                    engine.serve(msgs, slots)
                    seen.update(m[0] for m in msgs)
                now = time.monotonic()
                if len(seen) == workers and warm_start is None:
                    warm_start = now
                if warm_start is None or now - warm_start < 1:
                    continue
                searches += sum(m[2] for m in msgs)
                interval = now - warm_start - 1
                if interval >= args.tune_seconds:
                    rate = searches / interval
                    print(
                        f"selfplay_probe actors={workers} games_per_actor={slots} real_searches_s={rate:.1f}",
                        flush=True,
                    )
                    if rate > best_rate:
                        best, best_rate = (workers, slots), rate
                    break
        finally:
            pool.close()
    print(
        f"selected actors={best[0]} games_per_actor={best[1]} measured_searches_s={best_rate:.1f}",
        flush=True,
    )
    return best


def update(model, optimizer, scaler, replay, rng, cfg, micro, device, dtype, step):
    import torch

    from hardware import amp
    from network import legal_logits

    batch = cfg["batch"]
    fraction = min(1.0, max(0, step - 100) / cfg["decay_steps"])
    lr = cfg["lr"] * min(1.0, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))
    optimizer.param_groups[0]["lr"] = lr
    states, actions, counts, outcomes = replay.sample(rng, batch)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    packed, legal, visits, result = (
        torch.as_tensor(a, device=device, dtype=dt)
        for a, dt in (
            (states, torch.uint8),
            (actions, torch.long),
            (counts, torch.float32),
            (outcomes, torch.float32),
        )
    )
    for start in range(0, batch, micro):
        take = slice(start, start + micro)
        x, ids, n, z = packed[take], legal[take], visits[take], result[take]
        with amp(device, dtype):
            p, v = model(x)
            pi = n / n.sum(1, keepdim=True)
            pol = -(pi * legal_logits(p, ids).log_softmax(1)).sum(1).mean()
            val = -(z * v.float().log_softmax(1)).sum(1).mean()
            loss = (pol + val) * (micro / batch)
        scaler.scale(loss).backward()
        loss_sum += loss.detach()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled()
    )
    old = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return loss_sum.item(), scaler.get_scale() >= old


def main():
    started = time.monotonic()
    args = arguments()
    # Spawned actors only import NumPy and chess; each uses one CPU thread.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import copy

    import numpy as np
    import torch

    from checkpoints import Remote, Store
    from hardware import Utilization, configure_device, layout
    from network import ChessNet
    from replay import Replay

    torch.set_num_threads(1 if args.device == "cuda" else 2)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    device, dtype = configure_device(args.device, args.dtype)
    remote = None if args.local_only else Remote(args.server, os.getenv("CHECKPOINT_TOKEN"))
    store = Store(args.workdir, remote)
    actors = None
    utilization = Utilization(device)
    try:
        state = store.load()
        cfg = recipe(args, state)
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        net = ChessNet(cfg["width"], cfg["blocks"]).to(device, memory_format=torch.channels_last)
        if state:
            net.load_state_dict(state["model"])
        n, games, cap, micro = layout(args, cfg, net, device, dtype)
        opt = torch.optim.SGD(
            net.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=1e-4
        )
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        step = credit = completed = capped = 0
        replay = Replay(cfg["capacity"])
        if state:
            opt.load_state_dict(state["optimizer"])
            if scaler.is_enabled() and state["scaler"]:
                scaler.load_state_dict(state["scaler"])
            step, credit, completed, capped = (
                state[k] for k in ("step", "credit", "games", "capped")
            )
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], device)
            replay.restore(store.refs, store.shards, state["rows"])
        published = copy.deepcopy(net).eval().requires_grad_(False)
        del state
        print(
            f"start step={step} games={completed} replay={len(replay)} parameters={sum(p.numel() for p in net.parameters())} "
            f"recipe={json.dumps(cfg, sort_keys=True)}",
            flush=True,
        )
        deadline = started + args.seconds if args.seconds else float("inf")
        n, games = tune_search(args, cfg, published, device, dtype, n, games, cap, stop, deadline)
        actors = Actors(n, games, cfg, rng.integers(2**63, size=n, dtype=np.int64))
        inference = Inference(published, device, dtype, cap, actors.views, actors.events)

        def save():
            snap = {
                "format": 1,
                "config": cfg,
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "rows": replay.total,
                "games": completed,
                "capped": capped,
                "credit": credit,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            return store.submit(snap, replay)

        last_log = last_save = time.monotonic()
        searched = leaves = batches = updates = 0
        infer_s = learn_s = idle_s = 0.0
        last_rows, last_games = replay.total, completed
        loss = None
        while not stop.is_set():
            now = time.monotonic()
            if (args.seconds and now - started >= args.seconds) or (
                args.max_steps is not None and step >= args.max_steps
            ):
                break
            actors.check()
            store.poll()
            if now - last_save >= args.checkpoint_minutes * 60 and store.job is None:
                save()
                last_save = now
            can_train = len(replay) >= cfg["warm_rows"] and credit >= cfg["batch"]
            t = time.monotonic()
            messages = collect(
                actors.queue, n, games, cap, args.gather_ms, 0 if can_train else 0.05
            )
            idle_s += time.monotonic() - t
            if messages:
                t = time.monotonic()
                batches += inference.serve(messages, games)
                infer_s += time.monotonic() - t
                leaves += len(messages) * games
                for _, _, simulations, finished in messages:
                    searched += simulations
                    for x, ids, visits, z, was_capped in finished:
                        replay.append(x, ids, visits, z)
                        credit += cfg["reuse"] * len(x)
                        completed += 1
                        capped += int(was_capped)
            if len(replay) >= cfg["warm_rows"] and credit >= cfg["batch"]:
                t = time.monotonic()
                loss, advanced = update(
                    net, opt, scaler, replay, rng, cfg, micro, device, dtype, step
                )
                if advanced:
                    step += 1
                    credit -= cfg["batch"]
                    updates += 1
                    if step % 8 == 0:
                        published.load_state_dict(net.state_dict())
                if not math.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                learn_s += time.monotonic() - t
            elapsed = time.monotonic() - last_log
            if elapsed >= args.log_seconds:
                stats = {
                    "step": step,
                    "games": completed,
                    "capped_games": capped,
                    "replay": len(replay),
                    "credit": credit,
                    "searches_s": round(searched / elapsed, 1),
                    "leaves_s": round(leaves / elapsed, 1),
                    "mean_infer_batch": round(leaves / max(1, batches), 1),
                    "games_h": round((completed - last_games) * 3600 / elapsed, 1),
                    "new_rows_s": round((replay.total - last_rows) / elapsed, 1),
                    "train_rows_s": round(updates * cfg["batch"] / elapsed, 1),
                    "infer_wall_pct": round(100 * infer_s / elapsed, 1),
                    "train_wall_pct": round(100 * learn_s / elapsed, 1),
                    "gpu_busy_pct": utilization.mean(),
                    "actor_wait_pct": round(100 * idle_s / elapsed, 1),
                    "loss": loss,
                }
                print(json.dumps(stats), flush=True)
                searched = leaves = batches = updates = 0
                infer_s = learn_s = idle_s = 0.0
                last_rows, last_games, last_log = replay.total, completed, time.monotonic()
        actors.close()
        actors = None
        store.poll(wait=True)
        save()
        store.poll(wait=True)
        print(f"finished at step={step} games={completed} rows={replay.total}", flush=True)
    finally:
        if actors is not None:
            actors.close()
        utilization.close()
        store.close()


if __name__ == "__main__":
    main()
