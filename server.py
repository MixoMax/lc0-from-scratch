"""Single-writer checkpoint store. Run one uvicorn worker on the persistent disk."""

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("CHECKPOINT_DIR", "checkpoint-store"))
TOKEN = os.environ.get("CHECKPOINT_TOKEN")
if not TOKEN:
    raise RuntimeError("Set CHECKPOINT_TOKEN before starting uvicorn")
OBJECTS = ROOT / "objects"
VERSIONS = ROOT / "versions"
for directory in (OBJECTS, VERSIONS):
    directory.mkdir(parents=True, exist_ok=True)
LATEST = ROOT / "latest.json"
LOCK = threading.Lock()
MAX_PART = 16 * 1024**2
HASH = re.compile(r"[0-9a-f]{64}")


def auth(authorization: str = Header(default="")):
    if not secrets.compare_digest(authorization.encode(), ("Bearer " + TOKEN).encode()):
        raise HTTPException(401, "invalid bearer token")


app = FastAPI(dependencies=[Depends(auth)], docs_url=None, redoc_url=None, openapi_url=None)


def digest(value):
    if not HASH.fullmatch(value):
        raise HTTPException(422, "invalid SHA-256")
    return value


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@app.post("/objects/{sha}")
async def put_object(sha: str, request: Request):
    dest = OBJECTS / digest(sha)
    h = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=OBJECTS, delete=False) as f:
        tmp = Path(f.name)
        try:
            async for data in request.stream():
                size += len(data)
                if size > MAX_PART:
                    raise HTTPException(413, "use parts of at most 16 MiB")
                h.update(data)
                f.write(data)
            if h.hexdigest() != sha or size == 0:
                raise HTTPException(422, "bad object digest or empty body")
            f.flush()
            os.fsync(f.fileno())
            os.replace(tmp, dest)
            sync_dir(OBJECTS)
        finally:
            tmp.unlink(missing_ok=True)
    return {"bytes": size}


@app.get("/objects/{sha}")
def get_object(sha: str):
    path = OBJECTS / digest(sha)
    if not path.is_file():
        raise HTTPException(404, "missing object")
    return FileResponse(
        path, media_type="application/octet-stream", headers={"Cache-Control": "private, no-store"}
    )


class Commit(BaseModel):
    parent: str | None
    sha: str
    parts: list[str] = Field(min_length=1, max_length=4096)
    replay: list[str] = Field(max_length=100000)


@app.post("/commit")
def commit(body: Commit):
    sha = digest(body.sha)
    parts = [OBJECTS / digest(s) for s in body.parts]
    replay = [OBJECTS / digest(s) for s in body.replay]
    with LOCK:
        current = json.loads(LATEST.read_text())["sha"] if LATEST.exists() else None
        if current == sha:
            return {"sha": sha}
        if current != body.parent:
            raise HTTPException(409, "latest changed; another writer has committed")
        if any(not p.is_file() for p in parts + replay):
            raise HTTPException(422, "missing checkpoint part or replay shard")
        with tempfile.NamedTemporaryFile(dir=VERSIONS, delete=False) as f:
            tmp = Path(f.name)
            try:
                h = hashlib.sha256()
                for part in parts:
                    with part.open("rb") as source:
                        while block := source.read(1024**2):
                            h.update(block)
                            f.write(block)
                if h.hexdigest() != sha:
                    raise HTTPException(422, "assembled checkpoint digest mismatch")
                f.flush()
                os.fsync(f.fileno())
                os.replace(tmp, VERSIONS / sha)
                sync_dir(VERSIONS)
            finally:
                tmp.unlink(missing_ok=True)
        # Immutable version behind the pointer keeps concurrent GET requests consistent.
        with tempfile.NamedTemporaryFile(dir=ROOT, delete=False, mode="w") as f:
            json.dump({"sha": sha}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(f.name, LATEST)
        sync_dir(ROOT)
    return {"sha": sha}


@app.get("/latest")
def latest():
    headers = {"Cache-Control": "private, no-store", "X-Chess-Store": "1"}
    if not LATEST.exists():
        return JSONResponse({"detail": "empty checkpoint store"}, status_code=404, headers=headers)
    sha = json.loads(LATEST.read_text())["sha"]
    headers["X-Checkpoint-SHA256"] = sha
    return FileResponse(
        VERSIONS / sha, filename="latest.pt", media_type="application/octet-stream", headers=headers
    )
