# Reinforcement Learning-Based Scheduler (PPO)

## Lattice 2.0

## 1. Overview

This module implements a Reinforcement Learning (RL)-based node selection scheduler using Proximal Policy Optimization (PPO) for the Lattice 2.0 federated learning system.

The scheduler selects volunteer nodes for federated learning rounds using node characteristics such as reliability, device type, CPU capacity, recent dropouts, and training performance.

The objective is to improve training efficiency and reduce volunteer-node failures compared with heuristic scheduling.

## 2. Objectives

- Develop a PPO-based node selection policy.
- Use volunteer-node and job characteristics as input features.
- Improve job completion rates.
- Reduce volunteer dropout impact.
- Reduce total training time.
- Maintain compatibility with heuristic scheduling.
- Evaluate RL performance against baseline strategies.

## 3. Project Structure

```text
rl/
├── __init__.py
├── baselines.py
├── chaos.py
├── decision_log.py
├── evaluate.py
├── features.py
├── plot_curve.py
├── policy.py
├── sim_env.py
├── train_ppo.py
├── SPEC.md
└── models/
    ├── ppo_seed0.zip
    ├── ppo_seed1.zip
    ├── ppo_seed2.zip
    ├── ppo_seed3.zip
    └── ppo_seed4.zip
```

## 4. Technologies Used

- Python 3.11
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Pandas
- Matplotlib
- Pytest

## 5. Main Components

| File | Description |
|---|---|
| `features.py` | Converts node and job information into RL observations. |
| `policy.py` | Loads the trained PPO model and selects nodes. |
| `train_ppo.py` | Trains the PPO scheduling agent. |
| `evaluate.py` | Evaluates and compares scheduling strategies. |
| `baselines.py` | Implements baseline scheduling strategies. |
| `sim_env.py` | Provides the federated learning simulation environment. |
| `chaos.py` | Provides simulated dropout and slowdown functionality. |
| `decision_log.py` | Records scheduling decisions and round-level traces. |
| `plot_curve.py` | Generates training and evaluation plots. |

## 6. RL Environment

### State Space

The agent receives information about available volunteer nodes and the current federated learning job.

Node features include:

- CPU count
- Device type
- Reliability
- Session age
- Time since last heartbeat
- Recent dropouts
- Average round execution time

Job features include:

- Job name
- Model type
- Target accuracy
- Maximum rounds
- Local epochs
- Completed rounds

The observation representation supports up to 10 volunteer nodes.

### Action Space

The PPO agent uses a multi-binary action space.

- `1` — Select the volunteer node.
- `0` — Do not select the volunteer node.

### Reward

The agent is trained in a simulation environment to optimize federated learning scheduling performance.

The reward definition is implemented in the simulation environment.

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/harshaa77j/lattice_2.0.git
cd lattice_2.0
```

Switch to the RL branch:

```bash
git checkout rl
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements-rl.txt
```

## 8. Training

The PPO training implementation is available in:

```text
rl/train_ppo.py
```

View available training options:

```powershell
python -m rl.train_ppo --help
```

Trained PPO models are stored in:

```text
rl/models/
```

## 9. Evaluation

Run the evaluation:

```powershell
python -m rl.evaluate --seeds 0 1 2 3 4 --episodes 200 --out rl_results_live_baseline.csv
```

The evaluation compares:

1. PPO scheduler
2. Heuristic scheduler
3. All-nodes baseline

The experiment uses five training seeds and 200 evaluation episodes per seed for each scheduler.

## 10. Simulation Results

The following results were obtained from simulation experiments.

| Metric | PPO | Heuristic |
|---|---:|---:|
| Completion rate | 97.10% | 96.50% |
| Dropouts per 100 jobs | 357.50 | 593.10 |
| Average time (seconds) | 283.63 | 319.24 |
| Average rounds | 5.27 | 5.42 |

### Key Observations

- Completion rate improved by 0.60 percentage points.
- Average execution time decreased by approximately 11.15%.
- Dropouts per 100 jobs decreased by approximately 39.72%.
- Average rounds decreased from 5.42 to 5.27.

**Note:** These are simulation results, not real-laptop measurements. The reported standard deviations are calculated across five training-seed means.

## 11. Chaos Testing

The chaos helper is available in:

```text
rl/chaos.py
```

It provides functionality for simulating:

- Volunteer-node dropouts.
- Slower volunteer nodes.
- Temporary volunteer unavailability.
- Volunteer rejoining delays.

Environment variables:

```powershell
$env:LATTICE_CHAOS_DROP="0.3"
$env:LATTICE_CHAOS_SLOW="2.0"
$env:LATTICE_CHAOS_REJOIN_S="20"
```

Full volunteer-client integration, including heartbeat interruption, is a separate integration task.

## 12. Decision Logging

The decision logging implementation is available in:

```text
rl/decision_log.py
```

It supports recording:

- Round number.
- Job information.
- Scheduler type.
- Candidate volunteer nodes.
- Node selection scores.
- Selected and excluded nodes.
- Selection order.
- Round execution time.
- Reported and dropped nodes.
- Model accuracy.

Records can be appended in JSONL format for subsequent analysis.

## 13. Testing

Run the test suite:

```powershell
python -m pytest -q
```

Latest recorded test result:

```text
76 passed
```

Python syntax validation:

```powershell
python -m py_compile rl/decision_log.py rl/chaos.py rl/policy.py
```

## 14. Implementation Status

### Completed

- PPO scheduler implementation.
- RL feature extraction.
- Trained PPO models across five seeds.
- Policy loading and node selection.
- Corrected heuristic baseline.
- Simulation-based evaluation.
- Chaos testing helper.
- Decision logging helper.
- Decision logging unit tests.
- Evaluation results and training curve.

### Pending

- Server/coordinator decision logging integration.
- Dashboard visualization of node selection decisions.
- Volunteer-client chaos integration.
- Heartbeat interruption and safe rejoining.
- Real-laptop A/B testing.
- Final experimental analysis.

## 15. Future Work

1. Complete integration with the live federated learning coordinator.
2. Integrate decision logs with the dashboard.
3. Test node failures and recovery under real network conditions.
4. Conduct repeated experiments using actual volunteer laptops.
5. Compare PPO and heuristic scheduling using real execution traces.
6. Analyze scalability as volunteer counts increase.

## 16. Conclusion

The RL scheduler introduces a PPO-based approach to volunteer-node selection in the Lattice 2.0 federated learning system.

Initial simulation experiments show improvements in completion rate, execution time, and dropout metrics compared with the heuristic baseline.

The next stage is live-system integration and real-laptop experimentation to validate these improvements under practical distributed training conditions.

---

**Project:** Lattice 2.0  
**Module:** Reinforcement Learning Scheduler  
**Algorithm:** Proximal Policy Optimization (PPO)  
**Development Branch:** `rl`
