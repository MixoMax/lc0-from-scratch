"""Local HTTP integration test. Simulate eviction and a new machine with an empty disk."""

import hashlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx
import torch


class ResumeTest(unittest.TestCase):
    def test_eviction(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            token = "loopback-test-token"
            env = dict(os.environ, CHECKPOINT_TOKEN=token, CHECKPOINT_DIR=str(root / "store"))
            with (root / "run.log").open("w") as log:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "server:app",
                        "--port",
                        str(port),
                        "--no-access-log",
                    ],
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=log,
                )
                try:
                    with httpx.Client(
                        base_url=url, headers={"Authorization": "Bearer " + token}, trust_env=False
                    ) as client:
                        for _ in range(100):
                            try:
                                if client.get("/latest").status_code == 404:
                                    break
                            except httpx.TransportError:
                                pass
                            time.sleep(0.1)
                        else:
                            self.fail("HTTP store did not become ready")

                        def trainer(name, options):
                            return subprocess.Popen(
                                [
                                    sys.executable,
                                    "train.py",
                                    "--device",
                                    "cpu",
                                    "--server",
                                    url,
                                    "--workdir",
                                    str(root / name),
                                    "--checkpoint-minutes",
                                    ".04",
                                    *options,
                                ],
                                cwd=source,
                                env=env,
                                stdout=log,
                                stderr=log,
                                start_new_session=True,
                            )

                        first = trainer(
                            "spot1",
                            [
                                "--width",
                                "16",
                                "--blocks",
                                "1",
                                "--batch",
                                "8",
                                "--microbatch",
                                "8",
                                "--capacity",
                                "64",
                                "--warm-rows",
                                "8",
                                "--visits",
                                "3",
                                "--plies",
                                "8",
                                "--actors",
                                "2",
                                "--games-per-actor",
                                "3",
                            ],
                        )
                        try:
                            for _ in range(600):
                                self.assertIsNone(first.poll(), (root / "run.log").read_text())
                                path = root / "spot1/latest.pt"
                                if (
                                    path.exists()
                                    and torch.load(path, weights_only=True)["rows"] > 64
                                ):
                                    break
                                time.sleep(0.1)
                            else:
                                self.fail("first trainer never committed a populated replay")
                        finally:
                            if first.poll() is None:
                                os.killpg(first.pid, signal.SIGKILL)
                            first.wait()
                        checkpoint = client.get("/latest")
                        checkpoint.raise_for_status()
                        sha = checkpoint.headers["x-checkpoint-sha256"]
                        self.assertEqual(hashlib.sha256(checkpoint.content).hexdigest(), sha)
                        path = root / "committed.pt"
                        path.write_bytes(checkpoint.content)
                        old = torch.load(path, weights_only=True)
                        target = old["step"] + 8
                        second = trainer(
                            "spot2",
                            [
                                "--max-steps",
                                str(target),
                                "--microbatch",
                                "4",
                                "--actors",
                                "1",
                                "--games-per-actor",
                                "3",
                                "--inference-batch",
                                "2",
                            ],
                        )
                        try:
                            status = second.wait(timeout=60)
                            self.assertEqual(status, 0, (root / "run.log").read_text())
                        finally:
                            if second.poll() is None:
                                os.killpg(second.pid, signal.SIGKILL)
                                second.wait()
                        new = torch.load(root / "spot2/latest.pt", weights_only=True)
                        self.assertEqual(new["step"], target)
                        self.assertGreaterEqual(new["rows"], old["rows"])
                        self.assertEqual(new["config"], old["config"])
                        self.assertTrue(new["optimizer"]["state"])
                        self.assertFalse(
                            torch.equal(
                                new["model"]["stem.0.weight"], old["model"]["stem.0.weight"]
                            )
                        )
                        self.assertIn(
                            f"start step={old['step']} games={old['games']} replay=64",
                            (root / "run.log").read_text(),
                        )
                finally:
                    server.terminate()
                    server.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
