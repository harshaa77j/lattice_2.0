# Lattice — Design Decisions

**Last updated:15-10-2026** 

## Goal and constraints
Have quite a few volunteer laptops, trained models should handle node scheduling and drop outs well while being faster than single cpu training.

- **Demo goal:** 
Short Term Goal: Have 5 volunteer laptops, with a mid sized model to train. Upload from a sixth laptop, and train the model on the 5 laptops while one latop drops out midway. Show that a researcher on a weak laptop reaches faster 60% test accuracy in lesser time via Lattice than training alone.

- **Time available:** 
2 weeks for demo goal

- **Networking:** 
Tailscale with a star layout. Volunteers (Windows) reach the server over Tailscale; the Mac reaches it over local wifi (see Open questions).

---

## D1. Communication: Ray or plain HTTP?
- **Decision:** 
Plain HTTP
- **Why:** 
More flexible and high control. Ray requires every node to have the same environment. While the same environment can be simulated using docker, it is simpler to personally design on plain HTTP.
- **What I'm giving up:** 
Ray sends data around the connected nodes by itself.
I now write the round logic, timing, and weight transfer myself.
- **Revisit if:** 
Full control becomes separate personalization, too buggy.

## D2. Detecting dropped nodes
(estimated values given below, to be measured)
- **Heartbeat interval:** 
5 seconds 
- **Timeout (when a node counts as dropped):** 
Heartbeat*4
- **Round deadline (for stragglers):** 
3 x measured round time on the slowest volunteer laptop
- **What happens to updates that arrive late:** 
Discard
- **Can a dropped node rejoin? How:**
Yes, but only for the rounds that are left, if rounds are finished, leave it. 
- **Why these numbers:** 
Seems reasonably short but also enough to verify
- **Revisit if:** 
Becomes to slow/timout is too short 

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
Since HTTP is used, the nodes will initiate, since incomind connections arent allowed.
- **How each node knows which data slice is its own:** 
Server assigns each node a slice number for that particular round (slice i of N)
- **What happens to a dropped node's slice that round:** 
Redistributed: Skipped that round, next round's split covers all the data among the alive nodes.
- **How weights are combined:** 
Weighted average of the received weights from all nodes
- **When training stops:** 
Stop when test accuracy reaches 60%( atleast for demo) or after a maximum of N(to experiment) rounds.

## D4. Model and dataset

- **Dataset:** 
CIFAR-10
- **Model:** 
a modest CNN
- **Approx. model size / time per round:** 
1 MB – 5 MB/
GPU:2 – 5 seconds	CPU:30 – 60 seconds (estimeated)
- **Why this fits the demo:** 
One CPU laptop alone needs about N minutes to reach the target, while compute per round is much bigger than the ~1–5 MB of weights being transferred
N values(estimated):
Modern GPU (e.g., RTX 3060 / 4060 / Colab T4) ->30 seconds – 2 minutes
Standard CPU (e.g., Core i5 / Ryzen 5) ->7.5 – 25 minutes

- **What I'm giving up:** 
A small CNN on CIFAR-10 typically tops out around 65–75% test accuracy, and CPU rounds are slower than MNIST. That realistic ceiling should also set my target accuracy, so I have to pick something like 60% accuracy.
---

## D5. Where the server runs

- **Decision:** 
Raspberry Pi (Ubuntu, Python 3.10) as the development server. Demo-day host decided later.
- **Options considered:** 
Pi, one of the laptops, a cloud VM
- **Why:** 
Always on, low power, Tailscale works on Linux, frees a laptop to be a volunteer. Mac (on same wifi) reaches it without Tailscale.
- **What I'm giving up:** 
Slower CPU, so evaluating the global model each round may add time. Timings measured with the Pi as server may not match the demo setup.
- **Revisit if:** 
Evaluation on the Pi noticeably slows rounds, or the Pi can't be brought to the demo venue. Server address is read from config, so moving it is a one-line change.

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

-  **Open Questions:** 
1. Tailscale sign-in freezes on macOS Sonoma 14.0; working around it with Windows machines