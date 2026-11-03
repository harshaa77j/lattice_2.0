import logging
import os
import re
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.coordinator import Coordinator, Evaluator, JobConflict
from shared.models import list_models
from shared.protocol import (
    ADMIN_KEY_HEADER, NODE_ID_MAX_LEN, NODE_ID_PATTERN, NODE_TOKEN_HEADER,
    NOT_REGISTERED, RESEARCHER_KEY_HEADER,
)
from shared.serialization import bytes_to_state_dict

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

import torch
torch.set_num_threads(int(os.getenv("LATTICE_TORCH_THREADS", "1")))
logging.getLogger("lattice").info("PyTorch threads: %d", torch.get_num_threads())

VERSION = "1.0.0"
MAX_UPDATE_BYTES = 20 * 1024 * 1024

def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to server/.env (or the host's environment).")
    return value

def _number(name: str, default: float) -> float:
    return float(os.getenv(name, default))

NODE_TOKEN = _secret("LATTICE_TOKEN")
RESEARCHER_KEY = _secret("LATTICE_RESEARCHER_KEY")
ADMIN_KEY = _secret("LATTICE_ADMIN_KEY")

SETTINGS = {
    "heartbeat_timeout": _number("LATTICE_HEARTBEAT_TIMEOUT", 20),
    "round_deadline": _number("LATTICE_ROUND_DEADLINE", 180),
    "min_nodes": int(_number("LATTICE_MIN_NODES", 2)),
}

coord = Coordinator(
    evaluate_fn=Evaluator(subset_size=int(_number("LATTICE_EVAL_SUBSET", 2000))),
    heartbeat_timeout=SETTINGS["heartbeat_timeout"],
    round_deadline=SETTINGS["round_deadline"],
    min_nodes=SETTINGS["min_nodes"],
    checkpoint_dir=os.getenv("LATTICE_CHECKPOINT_DIR", "checkpoints"),
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = threading.Event()
    loop = threading.Thread(target=coord.run_forever, args=(stop,), name="round-loop", daemon=True)
    loop.start()
    yield
    stop.set()

app = FastAPI(title="Lattice", version=VERSION, lifespan=lifespan)

def _key_check(header_name: str, secret: str, label: str):
    def check(value: Optional[str] = Header(default=None, alias=header_name)):
        if value is None or not secrets.compare_digest(value.encode(), secret.encode()):
            raise HTTPException(status_code=401, detail=f"Invalid or missing {label}")
    return check

require_node_token = _key_check(NODE_TOKEN_HEADER, NODE_TOKEN, "node token")
require_researcher_key = _key_check(RESEARCHER_KEY_HEADER, RESEARCHER_KEY, "researcher key")
require_admin_key = _key_check(ADMIN_KEY_HEADER, ADMIN_KEY, "admin key")

def _unknown_node():
    raise HTTPException(status_code=404, detail=NOT_REGISTERED)

class NodeRegistration(BaseModel):
    node_id: str = Field(min_length=1, max_length=NODE_ID_MAX_LEN, pattern=NODE_ID_PATTERN)
    cpu_count: int = Field(gt=0)
    device: Literal["cuda", "cpu", "mps"]
    available: bool = True

class Availability(BaseModel):
    available: bool

class JobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    dataset: Literal["cifar10"] = "cifar10"
    model: str = Field(min_length=1, max_length=64)
    target_accuracy: float = Field(gt=0, lt=1)
    max_rounds: int = Field(ge=1, le=200)
    local_epochs: int = Field(ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}

@app.get("/models")
def models():
    """The model library, for the website's model picker. No secrets in it."""
    return list_models()

@app.post("/register", dependencies=[Depends(require_node_token)])
def register(node: NodeRegistration):
    coord.register(node.node_id, node.cpu_count, node.device, node.available)
    return {"ok": True}

@app.post("/heartbeat/{node_id}", dependencies=[Depends(require_node_token)])
def heartbeat(node_id: str):
    if not coord.heartbeat(node_id):
        _unknown_node()
    return {"ok": True}

@app.post("/nodes/{node_id}/availability", dependencies=[Depends(require_node_token)])
def availability(node_id: str, body: Availability):
    if not coord.set_available(node_id, body.available):
        _unknown_node()
    return {"ok": True, "available": body.available}

@app.get("/task/{node_id}", dependencies=[Depends(require_node_token)])
def task(node_id: str):
    result = coord.get_task(node_id)
    if result is None:
        _unknown_node()
    return result

@app.get("/weights", dependencies=[Depends(require_node_token)])
def weights():
    current = coord.get_weights()
    if current is None:
        raise HTTPException(status_code=409, detail="no round is open")
    data, job_id, round_number = current
    return Response(content=data, media_type="application/octet-stream",
                    headers={"X-Lattice-Job": job_id, "X-Lattice-Round": str(round_number)})

@app.post("/update/{node_id}", dependencies=[Depends(require_node_token)])
async def update(
    node_id: str,
    request: Request,
    round: int = Query(ge=1),
    n_samples: Optional[int] = Query(default=None, gt=0),
):
    if not coord.is_registered(node_id):
        _unknown_node()
    body = await request.body()
    if len(body) > MAX_UPDATE_BYTES:
        raise HTTPException(status_code=413, detail="update too large")
    try:
        state_dict = bytes_to_state_dict(body)
    except Exception:
        raise HTTPException(status_code=400, detail="not a valid weights file")
    if not isinstance(state_dict, dict):
        raise HTTPException(status_code=400, detail="not a valid weights file")
    problem = coord.check_update_shape(state_dict)
    if problem:
        raise HTTPException(status_code=400, detail=problem)

    result = coord.submit_update(node_id, round, state_dict, claimed_samples=n_samples)
    messages = {
        "not_open": f"round {round} is not open",
        "not_selected": "this node was not selected for the round",
        "duplicate": "this node already reported for the round",
    }
    if result != "accepted":
        raise HTTPException(status_code=409, detail=messages[result])
    return {"accepted": True}

@app.post("/jobs", dependencies=[Depends(require_researcher_key)])
def create_job(body: JobCreate):
    try:
        job = coord.create_job(body.name, body.dataset, body.model, body.target_accuracy,
                               body.max_rounds, body.local_epochs)
    except ValueError as e:              # unknown or unavailable model
        raise HTTPException(status_code=422, detail=str(e))
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}

@app.get("/jobs", dependencies=[Depends(require_researcher_key)])
def jobs():
    return coord.list_jobs()

@app.get("/jobs/{job_id}", dependencies=[Depends(require_researcher_key)])
def job(job_id: str):
    detail = coord.job_detail(job_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="no such job")
    return detail

@app.get("/jobs/{job_id}/model", dependencies=[Depends(require_researcher_key)])
def job_model(job_id: str):
    result = coord.job_model(job_id)
    if result is None:
        raise HTTPException(status_code=409, detail="the model is available once the job has finished")
    data, name = result
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".pt"
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/admin/status", dependencies=[Depends(require_admin_key)])
def admin_status():
    return coord.admin_status(SETTINGS, VERSION)

@app.get("/admin/nodes", dependencies=[Depends(require_admin_key)])
def admin_nodes():
    return coord.admin_nodes()

@app.get("/admin/events", dependencies=[Depends(require_admin_key)])
def admin_events():
    return coord.admin_events()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")