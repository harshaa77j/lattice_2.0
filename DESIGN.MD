# Lattice — Design Decisions

**Last updated:** 26-10-2026

## Goal and constraints
Have quite a few volunteer laptops; trained models should handle node scheduling and dropouts well while being faster than training on a single CPU.

- **Demo goal:**
5 volunteer laptops train a mid-sized model submitted from a sixth (researcher) laptop, while one laptop drops out midway. Show that a researcher on a weak laptop reaches 60% test accuracy sooner via Lattice than by training alone. (Machine count for the demo still open, see Open questions.)

- **Time available:**
Demo on 06-11-2026.

- **Team:**
Me: platform (server, volunteer client, FedAvg, deployment). Person B: RL scheduler. Claude: frontend and the volunteer app.

- **Networking:**
Star layout over plain HTTP: volunteers only ever connect out to the server, never to each other. The server runs on a public host (Northflank) over HTTPS, so no VPN is needed. Tailscale and the Raspberry Pi remain the development fallback.

---

## D1. Communication: Ray or plain HTTP?
- **Decision:**
Plain HTTP
- **Why:**
Ray needs every machine to reach every other machine, which fails across different home networks. With HTTP only the server has to be reachable, since volunteers start every connection. It also gives full control over what happens when a node drops, and Ray requires identical environments on every machine.
- **What I'm giving up:**
Ray sends data around the connected nodes by itself.
I now write the round logic, timing, and weight transfer myself.
- **Revisit if:**
Networking bugs take more than three days to fix, or we need to scale far beyond a handful of volunteers.

## D2. Detecting dropped nodes
(estimated values below, to be measured in the timing run)
- **Heartbeat interval:**
5 seconds
- **Timeout (when a node counts as dropped):**
Heartbeat × 4 = 20 seconds
- **Round deadline (for stragglers):**
3 × measured round time on the slowest volunteer laptop: [to measure]
- **What happens to updates that arrive late:**
Discard
- **Can a dropped node rejoin? How:**
Yes, for the rounds that are left. It re-registers and downloads the current global model; if the job is finished, it stays idle.
- **Why these numbers:**
A 4× margin over the heartbeat interval tolerates a few missed beats from a slow network without false alarms. The deadline comes from the timing run: [to fill in].
- **Revisit if:**
Healthy nodes get marked dropped (timeout too short), or rounds wait too long for dead nodes (too long).

## D3. What a round looks like

**Message sequence** (who sends what to whom, in order):
SETUP
1. Node → Server: register (node id, hardware info: CPU count, device).
   Server adds the node to its registry.
2. Node → Server: heartbeat every 5 s, from a separate thread so it
   keeps going during training. Server updates "last seen" for that node.
3. Researcher (6th laptop) → Server: submit job (model, dataset,
   number of rounds, local epochs, target accuracy).

EACH ROUND r
4. Server: waits until enough nodes are alive, picks nodes, assigns
   each one a slice number for THIS round (slice i of N), opens
   round r with a deadline.
5. Node → Server: "any task for me?" (polls every few seconds).
   Server → Node: "wait", or a task {round r, slice i, N, local epochs}.
6. Node → Server: download the current global weights.
7. Node trains on slice i for E local epochs (heartbeats continue).
8. Node → Server: upload weights + round number + sample count.
   Server accepts only if round r is still open; otherwise rejects as stale.
9. Server closes round r when:
   all selected nodes have reported, OR every node still missing
   has timed out, OR the deadline passes.
10. Server: weighted average of the received updates → new global
    model; evaluate on the test set; save a checkpoint; log who
    reported and who dropped.
11. Repeat from step 4 until the stop condition is met.

END
12. Server marks the job done (tasks now return "done");
    the researcher downloads the final model.

DROPOUT CASE
A node stops heartbeating mid-round → after 20 s the server marks it
dropped → step 9 closes the round without it → its slice isn't trained
that round → next round, step 4 reassigns slices among alive nodes.

- **Who starts each exchange (push or pull), and why:**
The nodes, since incoming connections to volunteer laptops aren't possible across networks (NAT).
- **How each node knows which data slice is its own:**
The server assigns each node a slice number for that particular round (slice i of N). Every machine shuffles the dataset indices with the same seed (`DATA_SEED`), and slice i takes every N-th index starting at position i, so all machines agree on the split.
- **What happens to a dropped node's slice that round:**
Skipped that round; next round's split covers all the data among the alive nodes.
- **How weights are combined:**
Weighted average of the received weights, weighted by each node's sample count (FedAvg).
- **When training stops:**
When test accuracy reaches 60% (at least for the demo), or after a maximum of N rounds (N to set after the timing run).

## D4. Model and dataset

- **Dataset:**
CIFAR-10 (50,000 training and 10,000 test images, 10 classes)
- **Model:**
CNN: 3 blocks of Conv2d (3×3, padding 1) → ReLU → MaxPool, with channels 3 → 32 → 64 → 128, then flatten (2048) → Linear 256 → ReLU → Linear 10.
- **Model size / time per round:**
620,362 parameters ≈ 2.5 MB, uploaded by each node every round.
Time per round on CPU: 30–60 s (estimated, to measure in the timing run).
- **Why this fits the demo:**
One CPU laptop alone needs an estimated 7.5–25 minutes to reach the target (to measure in the baseline run), while compute per round is much larger than the ~2.5 MB of weights transferred.
- **What I'm giving up:**
A small CNN on CIFAR-10 typically tops out around 65–75% test accuracy, and CPU rounds are slower than MNIST. That realistic ceiling sets the target at 60%.

## D5. Where the server runs

- **Decision:**
Northflank (Docker, built from GitHub, redeployed on every push to `main`) for the demo. The Raspberry Pi (Ubuntu, Python 3.10) and the Mac remain development servers and the fallback.
- **Options considered:**
Pi, one of the laptops, a cloud VM, Northflank
- **Why:**
A public HTTPS address means volunteers and the researcher's laptop connect without Tailscale, and the same address serves the website. Every push redeploys automatically.
- **What I'm giving up:**
The free plan has 0.1 shared CPU and 256 MB memory: fine for the registry today, but likely too little once the server loads PyTorch for averaging and evaluation. Deployment is paused until a card is added for verification.
- **Revisit if:**
Evaluation slows rounds or runs out of memory (see Open questions). The server address is read from config, so moving it is a one-line change on each node.

## D6. Web framework

- **Decision:**
FastAPI (served with uvicorn)
- **Options considered:**
FastAPI, Flask
- **Why:**
Automatic /docs page for testing endpoints without a client; request bodies validated through Pydantic models, so malformed requests get clear errors.
- **What I'm giving up:**
Flask's simplicity.

## D7. What a job submission contains

- **Decision:**
Configuration only (model name, dataset, rounds, local epochs, target accuracy). No code.
- **Options considered:**
Config only; researcher uploads model code
- **Why:**
In a volunteer system, accepting uploaded code means every volunteer laptop runs whatever anyone submits. That is a serious security hole.
- **What I'm giving up:**
Researchers can only train models already defined in the node client.
- **Revisit if:**
Researchers need custom models. Would need sandboxing, code review, or signed models first.

## D8. API contract (v1)

Three kinds of callers, each with its own key:

**Nodes** (header `X-Lattice-Token`, the shared node token)
- `POST /register`: {node_id, cpu_count, device}
- `POST /heartbeat/{node_id}`
- `POST /nodes/{node_id}/availability`: {available: true/false}. The app's Volunteer now / Pause buttons.
- `GET /task/{node_id}`: "wait" or {round, slice_id, num_slices, local_epochs}
- `GET /weights`: current global weights (bytes)
- `POST /update/{node_id}?round=&n_samples=`: trained weights (bytes); rejected if stale

**Researchers** (header `X-Researcher-Key`)
- `POST /jobs`: {name, dataset, model, target_accuracy, max_rounds, local_epochs} → {job_id, status}
- `GET /jobs`: list of jobs: {job_id, name, dataset, model, status, current_round, max_rounds, best_accuracy, started_at, finished_at}
- `GET /jobs/{job_id}`: the above plus volunteers {training, reported, dropped} and history [{round, accuracy, reported, dropped, seconds}]. Counts only, never node names.
- `GET /jobs/{job_id}/model`: final model file once status is "done"
- status is one of: waiting_for_nodes, training, done, stopped

**Admin** (header `X-Admin-Key`)
- `GET /admin/status`: {uptime_seconds, version, settings: {heartbeat_timeout, round_deadline, min_nodes, scheduler}, counts: {online, total, standby, dropped, training, cpus}, current_job: {name, current_round, max_rounds, accuracy}}
- `GET /admin/nodes`: [{node_id, device, cpu_count, status, seconds_since_seen, reliability}]; status is one of training, available, standby, dropped
- `GET /admin/events`: [{time, kind, text}]; kind is one of round, dropped, info

**Other**
- `GET /health`: open health check, no key.
- `GET /status`: temporary node list behind the node token; replaced by the admin endpoints.
- `/` serves the website (built React files), mounted after every API route.

**Rules**
- One job at a time in v1; a queue is future work.
- Each key is a separate secret in the server's environment, so a leaked node token can't open admin data or submit jobs.
- Accuracy is a fraction everywhere (0.56, not 56); the website converts it to a percentage.

## D9. Product and frontend decisions

- **No public node data.** Node lists and server status come only from the admin endpoints, behind `ADMIN_KEY`. Researchers see counts, never node names.
- **"Volunteer now" and "Pause" live in the local app,** as requests from the client to the server with its node token, so no browser-to-laptop pairing is needed.
- **The volunteer app is a small Python window (tkinter),** opened by a double-click installer with the server address and token filled in. Closing it stops Lattice.
- **Website:** React (Vite), built inside the Docker image and served by FastAPI from the same address. Pages use `#` addresses (e.g. `/#/admin`) so they never clash with API paths.

---

## Open questions
- Tailscale freezes on macOS during sign-in, so the Mac can't join the tailnet. Not debugging it (time); with a public server it's no longer needed. Limitation for the paper: not every volunteer machine could join the VPN.
- Demo machine count: need 5 volunteers + the researcher laptop; 4–5 machines expected. Decide the demo layout once they're available.
- Northflank free tier (256 MB, 0.1 CPU) is likely too small once the server averages and evaluates with PyTorch: paid plan, a laptop as training server, or evaluating on a test subset.
- The volunteer "Your history" page needs volunteer accounts (OAuth); in v1 the local app shows a laptop's own numbers.
- Agree with Person B: the `select_nodes` contract, the node features the server records, and the `SCHEDULER` switch (becomes D10).

## Change log
- 15-10-2026: D1–D4 written.
- 16-10-2026: D5 (Pi as dev server), D6 (FastAPI), D7 (config-only jobs). Mac reaches the server over local wifi instead of Tailscale.
- 24-10-2026: Demo moved to 06-11-2026.
- 25-10-2026: D8 API contract; shared node token added; frontend design locked.
- 26-10-2026: Server moves to Northflank (D5); D9 product decisions; health check moved to `/health`; Person B joins and owns the RL scheduler.