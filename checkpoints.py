"""The main thread freezes tensors and rows. A single background thread persists them."""

import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpx

PART = 16 * 1024**2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Remote:
    def __init__(self, url, token):
        parts = urlsplit(url)
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("checkpoint URL must use HTTPS (HTTP is supported on loopback)")
        if not token:
            raise ValueError("set CHECKPOINT_TOKEN")
        self.draining = threading.Event()
        self.client = httpx.Client(
            base_url=url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + token},
            timeout=httpx.Timeout(90, connect=10),
            trust_env=parts.hostname not in ("localhost", "127.0.0.1", "::1"),
        )

    def retry(self, func, upload=False):
        attempt = 0
        while True:
            try:
                return func()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code not in (408, 429)
                    and exc.response.status_code < 500
                ):
                    raise
                attempt += 1
                if attempt >= 5 and (not upload or self.draining.is_set()):
                    raise
                delay = min(30, 2 ** min(attempt - 1, 5))
                if upload:
                    print(
                        f"checkpoint transfer delayed ({type(exc).__name__}); retry in {delay}s",
                        flush=True,
                    )
                if attempt < 5:
                    time.sleep(delay)
                else:
                    self.draining.wait(delay)

    def download(self, route, dest, expected=None, allow_empty=False):
        def fetch():
            with self.client.stream("GET", route) as r:
                if allow_empty and r.status_code == 404 and r.headers.get("x-chess-store") == "1":
                    return None
                r.raise_for_status()
                want = expected or r.headers["x-checkpoint-sha256"]
                h = hashlib.sha256()
                tmp = dest.with_suffix(".download")
                try:
                    with tmp.open("wb") as f:
                        for chunk in r.iter_bytes(1024**2):
                            f.write(chunk)
                            h.update(chunk)
                    if want != h.hexdigest():
                        raise OSError(f"checksum mismatch downloading {route}")
                    os.replace(tmp, dest)
                finally:
                    tmp.unlink(missing_ok=True)
                return want

        return self.retry(fetch)

    def put(self, sha, data):
        self.retry(
            lambda: self.client.post("objects/" + sha, content=data).raise_for_status(), upload=True
        )

    def commit(self, parent, sha, parts, replay):
        body = {"parent": parent, "sha": sha, "parts": parts, "replay": [r[2] for r in replay]}
        self.retry(lambda: self.client.post("commit", json=body).raise_for_status(), upload=True)


def freeze_tensors(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: freeze_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(freeze_tensors(v) for v in obj)
    return obj


class Store:
    def __init__(self, directory, remote=None):
        self.dir = Path(directory)
        self.shards = self.dir / "replay"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.latest = self.dir / "latest.pt"
        self.remote = remote
        self.parent = None
        self.refs = []
        self.saved = 0
        self.job = None
        self.pool = ThreadPoolExecutor(max_workers=1)

    def load(self, pinned_local=False):
        import torch

        if self.remote and not (pinned_local and self.latest.exists()):
            self.parent = self.remote.download("latest", self.latest, allow_empty=True)
            if self.parent is None:
                if self.latest.exists():
                    raise RuntimeError(
                        "remote store is empty but workdir contains a checkpoint; use a fresh workdir"
                    )
                return None
        elif self.latest.exists():
            self.parent = file_hash(self.latest)
        else:
            return None
        data = torch.load(self.latest, map_location="cpu", weights_only=True)
        if data["format"] != 1:
            raise ValueError("unsupported checkpoint format")
        self.refs, self.saved = data["replay"], data["rows"]
        for lo, hi, sha in self.refs:
            if not re.fullmatch(r"[0-9a-f]{64}", sha) or not (0 <= lo < hi <= self.saved):
                raise ValueError("invalid replay reference")
            p = self.shards / sha
            if p.exists() and file_hash(p) == sha:
                continue
            if not self.remote:
                raise OSError(f"missing/corrupt local shard {sha}")
            self.remote.download("objects/" + sha, p, expected=sha)
        return data

    def poll(self, wait=False):
        if wait and self.remote:
            self.remote.draining.set()
        if self.job is None:
            return True
        if not wait and not self.job.done():
            return False
        self.parent, self.refs, self.saved = self.job.result()
        self.job = None
        return True

    def submit(self, state, replay):
        if not self.poll():
            return False
        chunks = replay.freeze(self.saved)
        refs = [r for r in self.refs if r[1] > replay.total - len(replay)]
        self.job = self.pool.submit(self._write, freeze_tensors(state), refs, chunks, self.parent)
        return True

    def _write(self, state, refs, chunks, parent):
        import torch

        from replay import encode_shard

        started = time.monotonic()
        uploaded = 0
        for lo, hi, arrays in chunks:
            data = encode_shard(arrays)
            sha = sha256(data)
            path = self.shards / sha
            if not path.exists():
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            if self.remote:
                self.remote.put(sha, data)
            refs.append((lo, hi, sha))
            uploaded += len(data)
        state["replay"] = refs
        tmp = self.dir / "pending.pt"
        torch.save(state, tmp)
        sha = file_hash(tmp)
        if self.remote:
            parts = []
            with tmp.open("rb") as f:
                while block := f.read(PART):
                    part = sha256(block)
                    self.remote.put(part, block)
                    parts.append(part)
                    uploaded += len(block)
            self.remote.commit(parent, sha, parts, refs)
        os.replace(tmp, self.latest)
        keep = {r[2] for r in refs}
        for p in self.shards.iterdir():
            if len(p.name) == 64 and p.name not in keep:
                p.unlink()
        print(
            f"checkpoint step={state['step']} rows={state['rows']} sha={sha[:12]} "
            f"payload_MiB={uploaded / 2**20:.1f} elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
        return sha, refs, state["rows"]

    def close(self):
        if self.remote:
            self.remote.draining.set()
        self.pool.shutdown(wait=True)
        if self.remote:
            self.remote.client.close()
