"""Hardware probes are disposable. A resumed model keeps all learned state."""

import copy
import os
import statistics
import time
from pathlib import Path

import psutil
import torch

from encoding import N_ACTIONS, STATE_BYTES
from network import legal_logits

GIB = 1024**3


def number(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def host_limits():
    cpu = len(os.sched_getaffinity(0))
    ram = psutil.virtual_memory().available
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpu = min(cpu, max(1, int(quota) // int(period)))
    except OSError:
        quota = number("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = number("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and quota > 0 and period:
            cpu = min(cpu, max(1, quota // period))
    for limit, usage in (
        ("memory.max", "memory.current"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"),
    ):
        cap = number("/sys/fs/cgroup/" + limit)
        used = number("/sys/fs/cgroup/" + usage)
        if cap is not None and used is not None:
            ram = min(ram, max(0, cap - used))
    return cpu, ram


def configure_device(name, dtype):
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install a compatible PyTorch GPU image, or use --device cpu for a smoke test."
            )
        major, _ = torch.cuda.get_device_capability(device)
        if dtype == "auto":
            dtype = (
                "bf16"
                if major >= 8 and torch.cuda.is_bf16_supported()
                else ("fp16" if major >= 7 else "fp32")
            )
        if dtype == "bf16" and (major < 8 or not torch.cuda.is_bf16_supported()):
            raise ValueError("native BF16 is unavailable on this GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Variable inference and legal-policy widths make per-shape exhaustive search expensive.
        torch.backends.cudnn.benchmark = False
        try:
            (torch.ones((8, 8), device=device) @ torch.ones((8, 8), device=device)).sum().item()
        except RuntimeError as exc:
            raise RuntimeError(
                "PyTorch could not execute on this GPU; check the wheel architecture and driver"
            ) from exc
    else:
        if dtype not in ("auto", "fp32"):
            raise ValueError("CPU smoke mode uses FP32")
        dtype = "fp32"
    return device, {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def amp(device, dtype):
    return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)


def probe(net, device, dtype, training, limit):
    """Measure candidate sizes using real layers, packed inputs, and both heads."""
    best_n, best_rate = None, 0
    model = copy.deepcopy(net).train(training)
    sizes = [2**i for i in range(limit.bit_length()) if 2**i <= limit]
    sizes = [n for n in sizes if n >= min(16, limit)]

    def run(model, x, ids):
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), amp(device, dtype):
            p, v = model(x)
            z = legal_logits(p, ids)
            if training:
                loss = -z.log_softmax(1)[:, 0].mean() - v.float().log_softmax(1)[:, 0].mean()
            else:
                torch.cat((z.softmax(1), v.float().softmax(1)), 1).cpu()
        if training:
            loss.backward()

    for n in sizes:
        try:
            x = torch.randint(0, 256, (n, STATE_BYTES), device=device, dtype=torch.uint8)
            ids = torch.randint(N_ACTIONS, (n, 48), device=device)
            run(model, x, ids)
            torch.cuda.synchronize()
            times = []
            for _ in range(3):
                t = time.perf_counter()
                run(model, x, ids)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            rate = n / statistics.median(times)
            if rate > best_rate * 1.025:
                best_n, best_rate = n, rate
        except torch.cuda.OutOfMemoryError:
            break
    del model
    torch.cuda.empty_cache()
    if best_n is None:
        raise RuntimeError(
            "insufficient GPU memory for calibration; reduce the model or set a smaller explicit batch"
        )
    print(
        f"probe {'train' if training else 'inference'} batch={best_n} synthetic_positions/s={best_rate:.0f}",
        flush=True,
    )
    return best_n


def layout(args, cfg, net, device, dtype):
    cpu, ram = host_limits()
    gpu = device.type == "cuda"
    free = torch.cuda.mem_get_info(device)[0] if gpu else 0
    effective = cfg["batch"]
    micro = args.microbatch or (
        probe(net, device, dtype, True, effective) if gpu else min(8, effective)
    )
    if micro > effective or effective % micro:
        raise ValueError("--microbatch must divide the checkpoint's effective batch")
    max_infer = 1024 if free >= 32 * GIB else 512
    infer = args.inference_batch or (probe(net, device, dtype, False, max_infer) if gpu else 16)
    # Reserve room for the learner, ring buffer and a frozen checkpoint snapshot.
    available = ram - 2.5 * cfg["capacity"] * 1400 - (400 << 20)
    if available < 64 << 20:
        raise RuntimeError(
            "insufficient free host RAM for this replay capacity; reduce --capacity on a new run"
        )
    actors = args.actors or min(
        48 if gpu else 2, max(1, cpu - 2), max(1, int(available // (200 << 20)))
    )
    games = args.games_per_actor or min(
        128 if gpu else 8, max(4, (3 * infer + actors - 1) // actors)
    )
    # Estimate per-game tree + history from the search budget and cap.
    game_bytes = (cfg["visits"] + 1) * 6500 + cfg["plies"] * 650
    if args.games_per_actor is None:
        games = max(1, min(games, int(available * 0.6 / actors / game_bytes)))
    gpu_name = torch.cuda.get_device_name(device) if gpu else "CPU"
    print(
        f"hardware={gpu_name} cuda={torch.version.cuda} dtype={dtype} cpu_quota={cpu} "
        f"ram_free_GiB={ram / GIB:.1f} vram_free_GiB={free / GIB:.1f} actors={actors} games/actor={games} "
        f"parallel_games={actors * games} inference_cap={infer} microbatch={micro} effective_batch={effective}",
        flush=True,
    )
    return actors, games, infer, micro


class Utilization:
    """Average NVML device-busy samples over each reporting interval."""

    def __init__(self, device):
        import threading

        self.values = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        if device.type == "cuda":
            self.thread = threading.Thread(target=self._run, args=(device,), daemon=True)
            self.thread.start()

    def _run(self, device):
        while not self.stop.is_set():
            try:
                pct = torch.cuda.utilization(device)
            except (RuntimeError, ImportError, OSError, AttributeError) as exc:
                print(
                    f"GPU utilization telemetry unavailable ({type(exc).__name__}); throughput counters remain active",
                    flush=True,
                )
                return
            with self.lock:
                self.values.append(pct)
            self.stop.wait(1)

    def mean(self):
        with self.lock:
            result = round(sum(self.values) / len(self.values), 1) if self.values else None
            self.values.clear()
        return result

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
