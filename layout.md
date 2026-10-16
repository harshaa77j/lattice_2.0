# Lattice — Repo Layout

Code is grouped by **who runs it**. Anything both the server and nodes need
lives in `shared/`.

```
lattice/
├── DESIGN.md            design decisions (already pushed)
├── LAYOUT.md            this file
├── requirements.txt     Python packages, same on every machine
├── .gitignore
│
├── shared/              used by BOTH server and nodes
│   ├── model.py         the CNN definition (server and nodes must match exactly)
│   └── data.py          dataset loading + deterministic slicing (slice i of N)
│
├── server/              runs on the Pi (for now)
│   ├── app.py           HTTP endpoints: register, heartbeat, task, weights, update, status
│   ├── rounds.py        round loop: pick nodes, open/close rounds, deadlines
│   └── aggregator.py    fedavg()
│
├── node/                runs on every volunteer laptop
│   ├── client.py        register, heartbeat thread, poll for tasks, upload
│   └── trainer.py       local training on one slice
│
├── researcher/          runs on the researcher's laptop (Mac)
│   └── submit.py        submit a job (config only, no code), fetch final model
│
├── scheduler/           Person B
│   └── select.py        select_nodes(): heuristic first, RL policy later
│
├── experiments/         one-off scripts, not part of the running system
│   ├── time_epoch.py    timing one epoch per machine (fills in DESIGN.md D4)
│   └── baseline.py      single-laptop training to 60% (the comparison)
│
├── rl/                  Week 3: environment, PPO training, policy loading
│
└── tests/               small checks you can run without the network
    └── test_fedavg.py   e.g. averaging identical models returns the same model
```

## When each part gets filled in

| Week | Folders |
| 1 | `server/app.py` (register, heartbeat, status), `node/client.py` (register + heartbeat thread) |
| 2 | `shared/`, `node/trainer.py`, `server/rounds.py`, `server/aggregator.py`, `tests/`, `experiments/` |
| 3 | `rl/`, `scheduler/` hookup with Person B |
| 4 | `researcher/`, polish |
