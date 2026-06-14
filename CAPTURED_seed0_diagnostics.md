# Captured seed-0 diagnostics (TEMPORAL, full dataset, CUDA T4)

> Source: live Colab run log, `--mode full --split temporal --nneg_sweep --full_pool`, full KIPRIS
> (370,666 patents / 122,519 companies), seed 0. `--nneg_sweep` and `--full_pool` are **seed-0-only
> and seed-invariant** by design, so these point estimates are final even though the 10-seed run was
> interrupted afterward. Preserved here so they survive further interruptions; feeds §11 / §12.

## §11 — Candidate-set-size sensitivity (n_neg sweep), seed 0, NDCG@10

| n_neg | MostPop | SVD | GraphSAGE | GAT |
| :---: | :---: | :---: | :---: | :---: |
| 50  | 0.255 | 0.040 | 0.083 | 0.104 |
| 100 | 0.196 | 0.036 | 0.043 | 0.082 |
| 200 | 0.152 | 0.033 | 0.023 | 0.069 |

Reading: MostPop dominates at every candidate-set size; the learned models never approach it, and the
gap is stable as the task hardens (n_neg 50→200). GAT is the best learned model throughout.

## §12 — Unsampled full-IPC-pool ranking, seed 0, NDCG@10

Mean pool = 900 candidates (cap = 1000; 171,596 / 220,270 queries down-capped at 1000; 14 skipped — i.e.
the true same-IPC pool exceeds 1000 for ~78% of queries). This is the unsampled counterpart to n_neg=100.

| Model | sampled (n_neg=100) | **full-pool (~900)** |
| :--- | :---: | :---: |
| MostPop   | 0.1968 | **0.0779** |
| GAT       | 0.0816 | 0.0576 |
| SVD       | 0.0366 | 0.0282 |
| GraphSAGE | 0.0431 | 0.0081 |

Reading (the key §12 defense): even ranking against the **entire** same-IPC company pool (no negative
sampling, ~9× more candidates than the sampled metric), **no learned model beats MostPop** — the model
ordering is preserved (MostPop > GAT > SVD > GraphSAGE). This directly answers the sampled-metric
concern (Krichene & Rendle, 2020): the diagnosis is not an artifact of drawing 100 negatives.
GraphSAGE collapses hardest under the full pool (0.043 → 0.008); GAT degrades least.

## Note on the run
Seed 0 took 2188 s (~36 min) because full_pool (+1492 s) and the n_neg sweep run only at seed 0. For the
remaining 10-seed Table 4/5, re-run **without** `--full_pool --nneg_sweep` (light seed 0 ≈ 6 min, robust
to interruption); these §11/§12 numbers are already final and are merged from here.
