# PPO scheduler specification (live-feature version)

Everything here is defined by the code in `rl/`. Items marked **[RUN]** must be filled
from your own run of `train_ppo.py` / `evaluate.py`; do not copy numbers from anywhere else.

## 1. State (OBS_DIM = 114, float32)
10 node slots x 11 features + 4 job features. Slot order = order given by the server;
unused slots are zero (present = 0). The simulator shuffles slot order every round.

| Per-node feature | Source key | Normalisation |
|---|---|---|
| present | (slot filled) | 0 / 1 |
| cpu_count | cpu_count | min(x, 32) / 32 |
| device one-hot (cpu, mps, cuda) | device | 3 binary features |
| reliability | reliability | already 0-1 |
| session age | session_age_s | min(x, 3600) / 3600 |
| last heartbeat | seconds_since_seen | min(x, 20) / 20 |
| recent drops | recent_drops | min(x, 10) / 10 |
| avg round time | avg_round_s | min(x, 300) / 300; 0 if None |
| avg round time known | (avg_round_s is not None) | 0 / 1 |

Job features: rounds_done / max_rounds; min(local_epochs, 5) / 5; target_accuracy;
min(max_rounds, 50) / 50.

## 2. Action
`MultiBinary(10)`: one bit per slot. Bits beyond the number of offered nodes are ignored.
Selected nodes = slots with bit 1, returned highest P(bit=1) first (slice 0 first).
An empty selection falls back to all offered nodes. If more than 10 nodes are alive,
the 10 most reliable are shown to the policy and the rest are not selected.

## 3. Reward (per round)
r = clip(1 - T_round / 90, -1, 1) - 0.3 * D - 0.1 * S + 0.5 * 1[target reached] - 1.0 * 1[max_rounds hit without target]
- T_round: round duration in seconds; D: dropouts this round;
- S: offered nodes with reliability >= 0.9 that were not selected.
Weights 0.3 / 0.1 / 0.5 / 1.0 are inherited from `scheduler/reward.py`.
**Differences from reward.py:** it scores a whole job; here it is applied per round. The speed term
uses a fixed 90 s reference instead of estimated_hrs. The accuracy-gap bonus is replaced by a
completion bonus. Agree this with Person A before the paper describes it.

## 4. Network
SB3 `MlpPolicy`, separate actor and critic MLPs: 114 -> 128 -> 128 (Tanh) -> 10 Bernoulli logits (actor),
114 -> 128 -> 128 (Tanh) -> 1 value (critic).

## 5. PPO hyperparameters
lr 3e-4, n_steps 2048, batch_size 64, n_epochs 10, gamma 0.99, gae_lambda 0.95, clip_range 0.2,
ent_coef 0.01, vf_coef 0.5, max_grad_norm 0.5, total_timesteps 300,000 (one step = one round).

## 6. Library
Stable-Baselines3 + PyTorch (CPU), gymnasium. **[RUN]** record `pip show stable-baselines3 torch gymnasium`.

## 7. Simulator (rl/sim_env.py) - assumptions, NOT calibrated on real traces
- Pool: N ~ UniformInt[3, 10] nodes per job. Job: 6-12 rounds, target accuracy ~ U(0.55, 0.70), 1 local epoch.
- Device: cpu 0.6, mps 0.2, cuda 0.2. cpu_count uniform over {2,4,6,8,12,16}.
- Solo epoch time: 240 s x sqrt(4/cpu_count) x {cpu 1.0, mps 0.5, cuda 0.3} x LogNormal(0, 0.25).
- Per-round node time: (solo time / k selected) x LogNormal(0, 0.1) + 5 s overhead.
- Drop probability per round (hidden): 20% of nodes are flaky ~ Beta(4, 6), the rest ~ Beta(1.2, 8).
- Dropout timing: uniform in 10-90% of the node's run; the server detects it 20 s later; round deadline 180 s.
- After dropping, a node sits out the next round with probability 0.5 (rejoin).
- History before the job: assigned ~ UniformInt[0, 30]; completions ~ Binomial(assigned, 1 - p); session age ~ Exp(mean 1800 s);
  avg_round_s is None for nodes with no history. seconds_since_seen ~ U(0, 5) and carries no signal.
- Accuracy model: acc += (0.80 - acc) x 0.30 x (reporting nodes / selected nodes), starting at 0.10.
- Dropout hazard is a constant per-node Bernoulli per round (no time-of-day term).

## 8. Baseline
`rl/baselines.py::heuristic_top5_reliability` is a **placeholder**: sort by reliability, take 5.
**[RUN]** replace it with the exact rule from `scheduler/select.py` and re-run. Also reported: all-nodes.

## 9. Evaluation protocol
5 seeds (0-4); one PPO model trained per seed; each evaluated on 200 episodes whose initial pools are
identical across schedulers (round randomness is independent). Reported: mean +/- sample std (ddof=1) across
the 5 per-seed means of: completion %, dropouts per 100 jobs, total job time (s), rounds.

## 10. Training curve
`python -m rl.plot_curve` -> training_curve.png (rolling mean episode reward vs timesteps, one line per seed).

## Results  **[RUN]**
Paste the table printed by `python -m rl.evaluate`.
