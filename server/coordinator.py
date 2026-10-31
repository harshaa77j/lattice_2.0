import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch

from scheduler.select import choose_nodes
from server.aggregator import fedavg
from shared.data import slice_data
from shared.models import get_spec
from shared.serialization import state_dict_to_bytes

log = logging.getLogger("lattice.coordinator")

CIFAR10_TRAIN_SIZE = 50_000
ACTIVE = ("waiting_for_nodes", "training")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobConflict(Exception):
    """A job is already running (v1 runs one job at a time)."""

@dataclass
class Node:
    node_id: str
    cpu_count: int
    device: str
    registered_at: float                
    last_seen: float
    available: bool = True
    rounds_assigned: int = 0
    rounds_completed: int = 0
    outcomes: deque = field(default_factory=lambda: deque(maxlen=10))  
    avg_round_s: Optional[float] = None

    @property
    def reliability(self) -> float:
        return self.rounds_completed / self.rounds_assigned if self.rounds_assigned else 1.0

@dataclass
class Job:
    job_id: str
    name: str
    dataset: str
    model: str
    target_accuracy: float
    max_rounds: int
    local_epochs: int
    created_at: str
    status: str = "waiting_for_nodes"
    current_round: int = 0              
    best_accuracy: Optional[float] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    history: list = field(default_factory=list)
    weights: bytes = b""               

@dataclass
class Round:
    job_id: str
    number: int
    assignments: dict                   
    num_slices: int
    opened_at: float
    deadline: float
    updates: dict = field(default_factory=dict)   

class Coordinator:
    def __init__(
        self,
        evaluate_fn: Callable[[str, dict, bool], float],
        heartbeat_timeout: float = 20.0,
        round_deadline: float = 180.0,
        min_nodes: int = 2,
        checkpoint_dir: Optional[str] = "checkpoints",
        clock: Callable[[], float] = time.monotonic,
    ):
        self.evaluate_fn = evaluate_fn
        self.heartbeat_timeout = heartbeat_timeout
        self.round_deadline = round_deadline
        self.min_nodes = min_nodes
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.clock = clock

        self.lock = threading.Lock()
        self.nodes: dict[str, Node] = {}
        self.jobs: dict[str, Job] = {}
        self.round: Optional[Round] = None
        self.events: deque = deque(maxlen=200)
        self.started_at = clock()
        self.last_scheduler = "heuristic"
        self._shape_cache: dict[str, dict] = {}

    # ---------------------------------------------------------------- helpers
    def _event(self, kind: str, text: str):
        self.events.appendleft({"time": datetime.now().strftime("%H:%M:%S"), "kind": kind, "text": text})
        log.info(text)

    def _alive(self, node: Node) -> bool:
        return self.clock() - node.last_seen <= self.heartbeat_timeout

    def _active_job(self) -> Optional[Job]:
        for job in self.jobs.values():          # dicts keep creation order
            if job.status in ACTIVE:
                return job
        return None

    def _status_of(self, node: Node) -> str:
        if not self._alive(node):
            return "dropped"
        r = self.round
        if r and node.node_id in r.assignments and node.node_id not in r.updates:
            return "training"
        return "available" if node.available else "standby"

    def _features(self, node: Node) -> dict:
        """What the scheduler sees about a node (DESIGN.md D10)."""
        now = self.clock()
        return {
            "node_id": node.node_id,
            "cpu_count": node.cpu_count,
            "device": node.device,
            "reliability": round(node.reliability, 3),
            "session_age_s": round(now - node.registered_at, 1),
            "seconds_since_seen": round(now - node.last_seen, 1),
            "recent_drops": sum(1 for ok in node.outcomes if not ok),
            "avg_round_s": round(node.avg_round_s, 1) if node.avg_round_s else None,
        }

    def _reference_shapes(self, model_name: str) -> dict:
        if model_name not in self._shape_cache:
            sd = get_spec(model_name).make_model().state_dict()
            self._shape_cache[model_name] = {k: (tuple(v.shape), v.dtype) for k, v in sd.items()}
        return self._shape_cache[model_name]

    # ----------------------------------------------------------------- nodes
    def register(self, node_id: str, cpu_count: int, device: str, available: bool = True):
        with self.lock:
            now = self.clock()
            node = self.nodes.get(node_id)
            if node is None:
                self.nodes[node_id] = Node(node_id, cpu_count, device, registered_at=now,
                                           last_seen=now, available=available)
                self._event("info", f"{node_id} registered: {device}, {cpu_count} cores")
            else:
                # Re-registration keeps the node's history (counters, reliability).
                node.cpu_count, node.device, node.last_seen = cpu_count, device, now
                self._event("info", f"{node_id} re-registered")

    def is_registered(self, node_id: str) -> bool:
        with self.lock:
            return node_id in self.nodes

    def heartbeat(self, node_id: str) -> bool:
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                return False
            node.last_seen = self.clock()
            return True

    def set_available(self, node_id: str, available: bool) -> bool:
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                return False
            if node.available != available:
                node.available = available
                self._event("info", f"{node_id} is {'available' if available else 'on standby'}")
            return True

    # ------------------------------------------------------------------ jobs
    def create_job(self, name: str, dataset: str, model: str, target_accuracy: float,
                   max_rounds: int, local_epochs: int) -> Job:
        spec = get_spec(model)                       # raises ValueError for unknown models
        torch.manual_seed(0)
        initial = state_dict_to_bytes(spec.make_model().state_dict())
        with self.lock:
            if self._active_job():
                raise JobConflict("a job is already running; wait for it to finish")
            job = Job(job_id=uuid.uuid4().hex[:8], name=name, dataset=dataset, model=model,
                      target_accuracy=target_accuracy, max_rounds=max_rounds,
                      local_epochs=local_epochs, created_at=utc_now_iso(), weights=initial)
            self.jobs[job.job_id] = job
            self._event("info", f"job {name} submitted ({model}, target {target_accuracy:.0%})")
            return job

    # ---------------------------------------------------------- node protocol
    def get_task(self, node_id: str) -> Optional[dict]:
        """None if the node is unknown; otherwise a task or {"status": "wait"}."""
        with self.lock:
            if node_id not in self.nodes:
                return None
            r = self.round
            if r is None or node_id not in r.assignments or node_id in r.updates:
                return {"status": "wait"}
            job = self.jobs[r.job_id]
            return {
                "status": "train",
                "job_id": job.job_id,
                "model": job.model,
                "round": r.number,
                "slice_id": r.assignments[node_id],
                "num_slices": r.num_slices,
                "local_epochs": job.local_epochs,
            }

    def get_weights(self) -> Optional[tuple]:
        """(bytes, job_id, round number) for the open round, or None."""
        with self.lock:
            r = self.round
            if r is None:
                return None
            return self.jobs[r.job_id].weights, r.job_id, r.number

    def check_update_shape(self, state_dict: dict) -> Optional[str]:
        """Problem description if the weights don't fit the open round's model, else None."""
        with self.lock:
            r = self.round
            model_name = self.jobs[r.job_id].model if r else None
        if model_name is None:
            return None                  # submit_update will reject it as "not open"
        ref = self._reference_shapes(model_name)
        if set(state_dict) != set(ref):
            return "weights don't match the job's model (different layers)"
        for key, (shape, _) in ref.items():
            if tuple(state_dict[key].shape) != shape:
                return f"'{key}' has shape {tuple(state_dict[key].shape)}, expected {shape}"
        return None

    def submit_update(self, node_id: str, round_number: int, state_dict: dict,
                      claimed_samples: Optional[int] = None) -> str:
        """Returns "accepted", "not_open", "not_selected" or "duplicate"."""
        with self.lock:
            r = self.round
            if r is None or round_number != r.number:
                return "not_open"
            if node_id not in r.assignments:
                return "not_selected"
            if node_id in r.updates:
                return "duplicate"
            # The server trusts its own count, never the client's claim.
            n = len(slice_data(CIFAR10_TRAIN_SIZE, r.assignments[node_id], r.num_slices))
            if claimed_samples is not None and claimed_samples != n:
                log.warning("%s claimed %s samples, slice has %s; using %s",
                            node_id, claimed_samples, n, n)
            r.updates[node_id] = (state_dict, n)
            node = self.nodes.get(node_id)
            if node:
                took = self.clock() - r.opened_at
                node.avg_round_s = took if node.avg_round_s is None else 0.7 * node.avg_round_s + 0.3 * took
            return "accepted"

    # ------------------------------------------------------------- the loop
    def tick(self):
        """One step of the state machine. Called repeatedly by the loop thread."""
        with self.lock:
            job = self._active_job()
            if job is None:
                return
            if self.round is not None:
                closing = self._maybe_close_round(job)
                if closing is None:
                    return
            else:
                closing = None
                candidates = [n for n in self.nodes.values() if self._alive(n) and n.available]
                if len(candidates) < self.min_nodes:
                    job.status = "waiting_for_nodes"
                    return
                features = [self._features(n) for n in candidates]
                job_info = {"name": job.name, "model": job.model, "target_accuracy": job.target_accuracy,
                            "max_rounds": job.max_rounds, "local_epochs": job.local_epochs,
                            "rounds_done": job.current_round}

        if closing is not None:
            self._finish_round(job, closing)
            return

        chosen, used = choose_nodes(features, job_info)
        with self.lock:
            if self.round is not None or job.status not in ACTIVE:
                return
            alive_now = {n.node_id for n in self.nodes.values() if self._alive(n) and n.available}
            chosen = [nid for nid in chosen if nid in alive_now]
            if len(chosen) < self.min_nodes:
                return
            if used != self.last_scheduler:
                self._event("info", f"scheduler: {used}")
                self.last_scheduler = used
            now = self.clock()
            number = job.current_round + 1
            self.round = Round(job_id=job.job_id, number=number,
                               assignments={nid: i for i, nid in enumerate(chosen)},
                               num_slices=len(chosen), opened_at=now, deadline=now + self.round_deadline)
            for nid in chosen:
                self.nodes[nid].rounds_assigned += 1
            job.status = "training"
            job.started_at = job.started_at or utc_now_iso()
            self._event("info", f"round {number} started with {len(chosen)} volunteers")

    def _maybe_close_round(self, job: Job) -> Optional[dict]:
        """Called with the lock held. Closes the round if it's finished; returns its data."""
        r = self.round
        pending = [nid for nid in r.assignments if nid not in r.updates]
        pending_alive = [nid for nid in pending if nid in self.nodes and self._alive(self.nodes[nid])]
        if pending_alive and self.clock() < r.deadline:
            return None
        self.round = None
        for nid in r.assignments:
            node = self.nodes.get(nid)
            if node is None:
                continue
            ok = nid in r.updates
            node.outcomes.append(ok)
            if ok:
                node.rounds_completed += 1
        for nid in pending:
            reason = "missed the deadline" if nid in pending_alive else "dropped: no heartbeat"
            self._event("dropped", f"{nid} {reason} in round {r.number}")
        return {"round": r, "updates": list(r.updates.values()), "missing": pending,
                "seconds": self.clock() - r.opened_at}

    def _finish_round(self, job: Job, closing: dict):
        """Aggregate and evaluate outside the lock, then record the result."""
        r, updates = closing["round"], closing["updates"]
        if not updates:
            with self.lock:
                self._event("dropped", f"round {r.number} got no updates; retrying it")
            return

        new_state = fedavg(updates)
        accuracy = self.evaluate_fn(job.model, new_state, False)
        reached = accuracy >= job.target_accuracy
        if reached:
            accuracy = self.evaluate_fn(job.model, new_state, True)   # confirm on the full test set
            reached = accuracy >= job.target_accuracy
        new_bytes = state_dict_to_bytes(new_state)
        self._save_checkpoint(job.job_id, r.number, new_bytes)

        with self.lock:
            job.weights = new_bytes
            job.current_round = r.number
            job.best_accuracy = max(accuracy, job.best_accuracy or 0.0)
            job.history.append({"round": r.number, "accuracy": round(accuracy, 4),
                                "reported": len(updates), "dropped": len(closing["missing"]),
                                "seconds": round(closing["seconds"], 1)})
            self._event("round", f"round {r.number} closed: {len(updates)} of {len(r.assignments)} "
                                 f"reported, {accuracy:.1%}")
            if reached:
                job.status, job.finished_at = "done", utc_now_iso()
                self._event("round", f"job {job.name} reached {accuracy:.1%}: done")
            elif job.current_round >= job.max_rounds:
                job.status, job.finished_at = "stopped", utc_now_iso()
                self._event("info", f"job {job.name} stopped at the round limit ({accuracy:.1%})")

    def _save_checkpoint(self, job_id: str, number: int, data: bytes):
        if not self.checkpoint_dir:
            return
        try:
            folder = self.checkpoint_dir / job_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"round_{number:03d}.pt").write_bytes(data)
            (folder / "latest.pt").write_bytes(data)
        except OSError as e:
            log.warning("could not save checkpoint: %s", e)

    def run_forever(self, stop: threading.Event, interval: float = 0.5):
        while not stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("round loop error")
            stop.wait(interval)

    # ------------------------------------------------------------- read views
    def _volunteer_counts(self, job: Job) -> dict:
        r = self.round
        if r is None or r.job_id != job.job_id:
            return {"training": 0, "reported": 0, "dropped": 0}
        pending = [nid for nid in r.assignments if nid not in r.updates]
        dropped = [nid for nid in pending if nid not in self.nodes or not self._alive(self.nodes[nid])]
        return {"training": len(pending) - len(dropped), "reported": len(r.updates), "dropped": len(dropped)}

    def _job_view(self, job: Job, detail: bool) -> dict:
        shown_round = self.round.number if self.round and self.round.job_id == job.job_id else job.current_round
        view = {"job_id": job.job_id, "name": job.name, "dataset": job.dataset, "model": job.model,
                "status": job.status, "current_round": shown_round, "max_rounds": job.max_rounds,
                "target_accuracy": job.target_accuracy, "local_epochs": job.local_epochs,
                "best_accuracy": job.best_accuracy, "started_at": job.started_at,
                "finished_at": job.finished_at}
        if detail:
            view["volunteers"] = self._volunteer_counts(job)
            view["history"] = list(job.history)
        return view

    def list_jobs(self) -> list:
        with self.lock:
            return [self._job_view(j, False) for j in reversed(list(self.jobs.values()))]

    def job_detail(self, job_id: str) -> Optional[dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            return self._job_view(job, True) if job else None

    def job_model(self, job_id: str) -> Optional[tuple]:
        """(bytes, name) once the job has finished, else None."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status not in ("done", "stopped"):
                return None
            return job.weights, job.name

    def admin_status(self, settings: dict, version: str) -> dict:
        with self.lock:
            nodes = list(self.nodes.values())
            statuses = [self._status_of(n) for n in nodes]
            job = self._active_job()
            current = None
            if job:
                last = job.history[-1]["accuracy"] if job.history else None
                current = {"name": job.name, "current_round": self._job_view(job, False)["current_round"],
                           "max_rounds": job.max_rounds, "accuracy": last}
            return {
                "uptime_seconds": round(self.clock() - self.started_at),
                "version": version,
                "settings": {**settings, "scheduler": self.last_scheduler},
                "counts": {
                    "online": sum(s != "dropped" for s in statuses),
                    "total": len(nodes),
                    "standby": statuses.count("standby"),
                    "dropped": statuses.count("dropped"),
                    "training": statuses.count("training"),
                    "cpus": sum(n.cpu_count for n, s in zip(nodes, statuses) if s != "dropped"),
                },
                "current_job": current,
            }

    def admin_nodes(self) -> list:
        with self.lock:
            now = self.clock()
            return [{"node_id": n.node_id, "device": n.device, "cpu_count": n.cpu_count,
                     "status": self._status_of(n), "seconds_since_seen": round(now - n.last_seen, 1),
                     "reliability": round(n.reliability, 3)} for n in self.nodes.values()]

    def admin_events(self, limit: int = 50) -> list:
        with self.lock:
            return list(self.events)[:limit]


class Evaluator:
    """Server-side evaluation: a fixed test subset each round, the full set to confirm."""

    def __init__(self, subset_size: int = 2000, root: str = "./data"):
        self.subset_size = subset_size
        self.root = root
        self._test = None
        self._lock = threading.Lock()

    def _dataset(self):
        with self._lock:
            if self._test is None:
                from shared.data import get_cifar10
                self._test = get_cifar10(train=False, root=self.root)
            return self._test

    def __call__(self, model_name: str, state_dict: dict, full: bool) -> float:
        from torch.utils.data import DataLoader, Subset
        from shared.protocol import DATA_SEED
        from shared.training import evaluate

        model = get_spec(model_name).make_model()
        model.load_state_dict(state_dict)
        data = self._dataset()
        if not full and self.subset_size and self.subset_size < len(data):
            g = torch.Generator().manual_seed(DATA_SEED + 1)
            data = Subset(data, torch.randperm(len(data), generator=g)[: self.subset_size].tolist())
        return evaluate(model, DataLoader(data, batch_size=256, shuffle=False, num_workers=0), "cpu")