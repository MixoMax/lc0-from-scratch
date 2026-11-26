# Chess Zero

Single-trainer, single-GPU chess self-play on interruptible Linux machines. Run the checkpoint service on the always-on host and launch `train.py` on whichever GPU you rent. All custom search, replay, and training code is Python. PyTorch runs the neural network and python-chess handles standard chess rules.

## Persistent host

Use Python 3.11 or 3.12. Extract the archive on both machines and enter its `chess-zero` directory. On the persistent host, create a virtual environment, set a token once, and retain it across service restarts.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-server.txt
umask 077
# Run this line once to create the shared secret.
python -c 'import secrets; print(secrets.token_urlsafe(32))' > checkpoint.token
export CHECKPOINT_TOKEN="$(cat checkpoint.token)"
export CHECKPOINT_DIR=/mnt/checkpoints/chess-zero
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

Point your HTTPS reverse proxy or Cloudflare Tunnel at `http://127.0.0.1:8000`. Set the proxy request-body allowance to at least 16 MiB and bypass caching for this host. Use a writable directory on the HDD for `CHECKPOINT_DIR`. A service manager can launch the same command with the same two environment variables after reboot. One data directory represents one training run. The server retains committed versions and their content-addressed replay objects.

## GPU host and one-hour pilot

Use an Ubuntu image with a working NVIDIA driver and CUDA-enabled PyTorch. Keep the image's GPU build visible if you create a virtual environment. The commands below use the image's Python, with PyTorch 2.8 or newer. Installation examples for hosts without PyTorch appear below.

```bash
python3 -m pip install -r requirements.txt
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
export CHECKPOINT_URL=https://chkpt.linush.org
read -rsp 'Checkpoint token: ' CHECKPOINT_TOKEN && echo
export CHECKPOINT_TOKEN
set -o pipefail
python3 -u train.py --seconds 3300 2>&1 | tee pilot.log
```

Use the subdomain you configured. Enter the same token as on the persistent host. The 55-minute pilot leaves five minutes of a one-hour rental for final transfer and inspection, subject to setup time and link speed. The wall-time option is honored between work units. A successful final upload is acknowledged in the log before process exit.

On the RTX PRO 6000 host, extract the same package, install the trainer dependencies, export the same URL and token, then run:

```bash
python -u train.py
```

The trainer fetches the latest teacher checkpoint and missing replay shards. An authenticated empty-store response starts from random weights. A wrong URL, missing token, corrupt download, or server error surfaces as an error. Hardware settings are calibrated anew. A recovered run keeps its architecture, effective batch, search budget, learning-rate schedule and data-reuse settings.

For a host without a working PyTorch installation, these are example PyTorch 2.8 wheels for Linux x86-64. Use a driver compatible with the selected wheel. The Blackwell card needs a build that includes its architecture. The P40 needs Pascal support and uses FP32 in this project.

```bash
# A modern rental image, including Blackwell
python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# A separate environment for the home P40
python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

[PyTorch wheel list](https://pytorch.org/get-started/previous-versions/) · [PyTorch architecture support](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128) · [Blackwell build guidance](https://discuss.pytorch.org/t/pytorch-support-for-sm-120/222119) · [NVIDIA legacy GPU list](https://developer.nvidia.com/cuda/gpus/legacy)

## Reading the pilot

Startup probes time forward and backward passes at several batch sizes. A second sweep measures **actual MCTS simulations per second** with different actor counts and parallel games. It uses temporary games and separate random seeds. The winning actor layout runs self-play. CPU quota, available RAM, free VRAM, and arithmetic support influence the choices. Search processes send packed positions through shared memory. One GPU owner coalesces requests across processes, evaluates legal moves in batches and trains while the actors are working. Four training presentations are funded per new completed-game position.

Every 30 seconds the trainer prints one JSON line. For the pilot, keep that log and the hardware line.

| Field | How to use it |
|---|---|
| `searches_s`, `leaves_s` | End-to-end simulations and neural evaluations per wall second. Compare hosts at the same recipe. |
| `mean_infer_batch`, `actor_wait_pct`, `gpu_busy_pct` | Small batches, sustained queue waits and low device busy time together suggest that the CPU side is starving the GPU. GPU busy comes from NVML; a missing sample is reported as null. |
| `new_rows_s`, `games_h`, `train_rows_s`, `step` | Confirm that games finish and produce optimizer updates. New positions arrive in bursts, especially before the first games finish. |
| `capped_games`, `games` | Compare their increments to see how often games reach the ply cap. Capped games use a searched value estimate for the final position. |
| `infer_wall_pct`, `train_wall_pct` | Main-process wall time serving inference or updating the network, including transfers. Actor work overlaps these intervals. |
| `checkpoint` line | A completed snapshot reports optimizer step, rows, payload size and transfer time. |

If the CPU side limits throughput on a rental, compare a host with more dedicated CPU cores, or repeat the pilot with `--actors`, `--games-per-actor` and `--gather-ms`. Compare searches/s **and** new rows/s. Raw GPU busy percentage alone cannot measure useful training progress. Explicit `--microbatch` and `--inference-batch` override the GPU probes. Setting both actor overrides skips the actor sweep. `--tune-seconds 0` also skips it. Arithmetic can be overridden with `--dtype`. A microbatch must divide the effective batch. Gradient accumulation preserves the examples per update; BatchNorm statistics and floating-point results may differ across hosts.

## Learning configuration

The default teacher has 8 residual blocks of width 96 and 1,391,531 parameters. Inputs follow the paper's 52-plane side-to-move representation, with scaled scalar features. The policy uses query/key scores over 1,880 origin-destination and promotion actions. Both training and search mask illegal moves. The value predicts win/draw/loss. Search uses PUCT, root noise and subtree reuse. The side-to-move value changes sign on each backup. Training uses legal root visits, completed-game outcomes, uniform replay, Nesterov SGD and a warmup/cosine schedule. The network used for self-play refreshes every eight optimizer updates.

| New-run option | Default |
|---|---:|
| `--width`, `--blocks` | 96, 8 |
| `--visits` | 64 total root visits per move, including reusable visits |
| `--plies` | 192 searched plies after a random legal opening of up to eight plies |
| `--capacity`, `--warm-rows` | 500,000 replay positions, 1,024 positions before learning |
| `--batch`, `--reuse` | 256 positions per update, four presentations per new position |
| `--lr`, `--decay-steps` | 0.04 peak after 100 warmup steps, 30,000 decay steps |

Standard chess only. Self-play takes a draw at the third occurrence of the current position or after 100 halfmoves, with mate taking precedence. At the ply cap, the searched value supplies a soft outcome target. The model learns chess solely through self-play. This fixed-size baseline adapts the paper to a single GPU and Python search. The paper's larger search budgets, progressive growth, replay prioritization, auxiliary heads and TensorRT pipeline are outside this recipe. Playing strength and throughput require measurement on your run.

## Checkpoints and interruption

The default 15-minute timer freezes weights, optimizer state, mixed-precision scaler, training counters, random-generator states and new replay rows at a consistent boundary. A worker thread compresses and transfers that snapshot. The trainer keeps playing and learning during network transfers. The server verifies SHA-256 and atomically publishes the latest pointer only after all referenced objects have arrived. Each model part fits in 16 MiB. Replay uses small compressed shards; previously committed rows are referenced again without retransmission. A bare GET to `/latest` with the bearer token returns the actual PyTorch checkpoint file. The trainer automatically fetches the associated replay objects on restore.

The default teacher weights occupy about **5.3 MiB** in FP32. Its SGD momentum needs about another 5.3 MiB once training starts. Replay can readily take a snapshot over 100 MB in aggregate. On a 100 Mbps link, 100 MB takes at least eight seconds before overhead. The upload limit applies to each request. [Cloudflare upload limits](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/)

SIGTERM or Ctrl-C requests a final save. A sudden eviction resumes the last committed snapshot. Unfinished games, transient search trees, and advances after that snapshot are lost. Actor scheduling and new games can diverge after a restart. A network outage delays remote commits while the worker retries. If shutdown exhausts the retry budget, the trainer reports failure; the previous committed snapshot remains downloadable. The local workdir keeps the latest checkpoint plus its replay window. Budget several GiB of working RAM for the default replay and actor processes.

## Home distillation

Use a dedicated teacher directory on the P40's persistent disk. On first launch this downloads the latest teacher and its replay window. Later launches reuse that frozen teacher and resume the student from `--output`. The student has **470,315 parameters**. It learns the teacher's legal-move distribution and WDL probabilities with soft cross-entropy. This uses up to 500,000 recent positions at the default capacity. The paper's separate student experiment used a larger frozen dataset. Student weights and optimizer state are saved locally, so on a rented GPU use persistent storage for the output and pinned teacher.

```bash
python distill.py --teacher-dir frozen-teacher --output student.pt --steps 250 --save-every 50
python distill.py --teacher-dir frozen-teacher --output student.pt --steps 20000
```

Use the first timing in `rows_s` to decide whether the home card is fast enough. At batch 128, 20,000 updates make 2.56 million training presentations. Divide that count by measured rows/s to estimate seconds. These step counts are pilot and starting budgets; playing strength requires separate evaluation.

## Local checks

```bash
python test_core.py
python train.py --device cpu --local-only --workdir smoke --width 16 --blocks 1 --batch 8 --microbatch 4 --visits 3 --plies 8 --warm-rows 8 --capacity 128 --actors 2 --games-per-actor 3 --max-steps 5
python train.py --device cpu --local-only --workdir smoke --actors 1 --games-per-actor 2 --max-steps 8
```

These small settings exercise the full learning loop and resume path. The GPU pilot should use the default recipe and a separate remote store. Run `python train.py --help` for all overrides.

To rehearse true eviction over HTTP on one host, install both requirements files and run `python test_resume.py`. It starts a loopback uvicorn server, kills a trainer with SIGKILL, and verifies learning from a fresh workdir with a different number of actors and a smaller microbatch.
