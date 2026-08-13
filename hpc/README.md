# Bandit fragment-selection grid on CURC Alpine

Sweeps the UCB1 fragment-selection arm in `scripts/exps/lead/run.py` against
random selection, across ~1,014 configurations in staged submissions.

## One-time setup

```bash
# On a login node
cd /projects/$USER
git clone <this fork> genmol && cd genmol

# Build the env on a compute node (NOT the login node)
acompile
bash hpc/setup_curc_env.sh
```

Then stage the V1 checkpoint (NGC `nvidia/clara/genmol_v1`) at
`/projects/$USER/genmol/model.ckpt`.

`setup_curc_env.sh` handles four things `env/setup.sh` does not: the
transformers pin conflict between `pyproject.toml` (4.52.4) and
`env/requirements.txt` (4.56.2), the `safe-mol` import patch, the `qvina02`
execute bit, and warming the HuggingFace/TDC caches so ~30 concurrent workers
are not all doing first-touch downloads.

## Running a stage

```bash
DRY_RUN=1 bash hpc/queue_grid.sh pilot     # inspect the sbatch lines first
bash hpc/queue_grid.sh pilot
```

Stages run in order; each screen parameterizes the next, so pass the winner
forward as an anchor:

| Stage | What it answers | Runs |
|---|---|---|
| `pilot` | Does it run, how long per run, aa100 vs ah200 | 4 |
| `1a` | Arm count (`pop_cap`) x exploration (`ucb_c`) | 300 |
| `1c` | Random baselines across `pop_cap` | 50 |
| `1b` | QED shaping (`lam_rq`) x memory (`q_alpha`) at the 1a winner | 150 |
| `1d` | `lam_rq=200` degenerate anchor | 10 |
| `2` | Q-initialisation, horizon, and an epsilon-greedy arm | 50 |
| `3` | Confirmation: 3 strategies x 5 targets x 3 start molecules x 10 seeds | 450 |

```bash
# after picking the 1a winner
cat > anchors/best_1a.json <<'JSON'
{"pop_cap": 50, "ucb_c": 1.0}
JSON
ANCHOR=anchors/best_1a.json bash hpc/queue_grid.sh 1b
```

Useful overrides: `ACCOUNT=`, `SKIP_AH200=1`, `AA100_WORKERS=`, `POOL_SIZE=`,
`WALLTIME=`, `FORCE=1`, `GENMOL_SCRATCH=`.

## Why the job is shaped this way

**One worker holds many runs.** `Sampler.mask_modification` processes a single
molecule per call, so one run leaves an A100 essentially idle while docking does
the real work. Each worker therefore runs `POOL_SIZE` configurations
concurrently, round-robining `CUDA_VISIBLE_DEVICES` across the job's GPUs.

**Sizing invariant:** `POOL_SIZE * GENMOL_NUM_SUB_PROC ~= cpus-per-task`. Each
docking worker is effectively one core, because `exhaustiveness=1` means qvina
runs a single Monte Carlo task and its `--cpu` flag is inert. Tune with
`seff <jobid>` after the pilot.

**Both GPU partitions.** The binding constraint is the per-user GRES quota, not
node availability:

| QOS | a100_80gb | a100-40gb | h200 |
|---|---|---|---|
| `gpu-normal` (24 h) | 3 | 6 | 4 |
| `gpu-long` (7 d) | 1 | 3 | 2 |

Quotas are per GRES type, so running `aa100` and `ah200` together gives ~10 GPUs
rather than 6. `a100-40gb` is preferred over `a100_80gb` because the model is
~110M parameters at batch size 1 and the quota is twice as large. `ami100`
(needs a ROCm torch build), `al40` (CU Anschutz only) and `artxpro6000`
(Blackwell sm_120, needs torch >= 2.7 + cu128 against the pinned `torch==2.6.0`)
are unusable without a second environment.

## Resuming

`run.py` writes `status.json` last and only on a clean finish, and skips any
configuration that already has one. Re-running the same `queue_grid.sh` command
after a walltime kill finishes only what is left. Use `FORCE=1` to override.

## Collecting results

```bash
python scripts/exps/lead/collect.py \
  --results-dir /scratch/alpine/$USER/genmol/results \
  -o runs.parquet --finals-output finals.parquet
```

`collect.py` joins each run's `config.json` to its `iterlog.csv`, so no
hyperparameter is ever parsed back out of a filename. It reports runs that did
not finish, rows with `rv_max > 15` (Vina failures, which return the 99.9
sentinel), and any iterlog whose `iter` column is non-monotonic — the signature
of two processes writing one file, which is what corrupted
`results_v1_unshaped/`.

The older `anal_sweep.py` / `analyze_sweep.py` / `analyze_lam.py` parse
hyperparameters out of filenames with a regex whose `\w` class matches neither
`-` nor `.`, so they raise `AttributeError` on the tags `run.py` emits for
bandit runs. Use `collect.py` instead.

## Checks before trusting a stage

```bash
python scripts/exps/lead/test_grid_safety.py          # runs without rdkit/openbabel
squeue -u $USER --format="%A %j %P %q %a %t %r"       # %a must be the project account
sacct -u $USER --format=JobID,JobName%20,State,ExitCode,Elapsed -X
seff <jobid>
```
