#!/bin/bash
#SBATCH --job-name=genmol_grid
#SBATCH --output=logs/genmol_grid_%j_%x.out
#SBATCH --error=logs/genmol_grid_%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=120G
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100-40gb:3
# c3gpu-g5-u1: confirmed-bad node, excluded by default across this group's
# scripts. CLI flags win over this directive if it is ever fixed.
#SBATCH --exclude=c3gpu-g5-u1

# Ensure log directory exists before SLURM opens the output file
mkdir -p logs

# GenMol bandit-vs-random grid worker.
#
# Takes a shard of a JSONL manifest (every line where index % TOTAL_WORKERS ==
# WORKER_ID) and runs POOL_SIZE configurations concurrently inside this one
# allocation.
#
# Why a pool instead of one job per run: Sampler.mask_modification processes a
# single molecule per call, so one run leaves an A100 essentially idle while
# docking does the real work. Packing runs also sidesteps the per-user GRES
# quota (gpu-normal allows only 6x a100-40gb / 4x h200), which would otherwise
# throttle the whole grid to a handful of concurrent runs.
#
# Sizing invariant: POOL_SIZE * GENMOL_NUM_SUB_PROC ~= cpus-per-task. Each
# docking worker is effectively one core, because exhaustiveness=1 means qvina
# runs a single Monte Carlo task and the --cpu flag is inert.
#
# Environment variables:
#   MANIFEST        JSONL manifest path                 (required)
#   WORKER_ID       this worker's shard index           (default 0)
#   TOTAL_WORKERS   number of workers over the manifest (default 1)
#   POOL_SIZE       concurrent runs in this job         (default: cpus/8)
#   OUT_DIR         results root                        (default scratch)
#   GENMOL_MODEL_PATH  absolute path to model.ckpt      (default $PROJ_DIR/model.ckpt)
#   FORCE           1 to re-run completed configurations
#
# Submit (usually via hpc/queue_grid.sh):
#   sbatch --account=ucb738_asc1 \
#     --export=ALL,MANIFEST=manifests/stage_1a.jsonl,WORKER_ID=0,TOTAL_WORKERS=3 \
#     hpc/slurm_genmol_grid.sh
#
# Author: Patrick Cooper

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/hpc/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/hpc/common.sh"
else
    source "$(cd "$(dirname "$0")" && pwd)/common.sh"
fi

: "${MANIFEST:?ERROR: MANIFEST must be set (path to a JSONL manifest)}"

WORKER_ID="${WORKER_ID:-0}"
TOTAL_WORKERS="${TOTAL_WORKERS:-1}"
NCPU="${SLURM_CPUS_PER_TASK:-8}"
FORCE="${FORCE:-0}"

genmol_print_job_header "GenMol Grid Worker ${WORKER_ID}/${TOTAL_WORKERS}"
genmol_activate_env
genmol_setup_paths

export GENMOL_MODEL_PATH="${GENMOL_MODEL_PATH:-${PROJ_DIR}/model.ckpt}"
genmol_preflight

OUT_DIR="${OUT_DIR:-${GENMOL_SCRATCH}/results}"
RUN_LOG_DIR="${GENMOL_SCRATCH}/logs/job${SLURM_JOB_ID:-local}_w${WORKER_ID}"
mkdir -p "${OUT_DIR}" "${RUN_LOG_DIR}"

# Docking scratch on node-local disk. DockingVina now mkdtemps under $TMPDIR
# rather than scanning docking/tmp/tmpN on the shared filesystem, which was a
# check-then-create race that crashed every concurrent task past the first.
export TMPDIR="${SLURM_TMPDIR:-/tmp/genmol_${SLURM_JOB_ID:-$$}}"
mkdir -p "${TMPDIR}"

if [ ! -f "${MANIFEST}" ]; then
    echo "FATAL: manifest not found: ${MANIFEST}" >&2
    exit 1
fi

# Default the pool so that pool * 8 docking workers ~= the CPU allocation.
POOL_SIZE="${POOL_SIZE:-$(( NCPU / 8 ))}"
[ "${POOL_SIZE}" -lt 1 ] && POOL_SIZE=1
SUB_PROC="${GENMOL_NUM_SUB_PROC:-$(( NCPU / POOL_SIZE ))}"
[ "${SUB_PROC}" -lt 1 ] && SUB_PROC=1
export GENMOL_NUM_SUB_PROC="${SUB_PROC}"

IFS=',' read -r -a GPUS <<< "$(genmol_gpu_list)"
NGPU="${#GPUS[@]}"

TOTAL_LINES=$(grep -c . "${MANIFEST}")

echo "Manifest  : ${MANIFEST} (${TOTAL_LINES} runs)"
echo "Shard     : indices where i %% ${TOTAL_WORKERS} == ${WORKER_ID}"
echo "Out dir   : ${OUT_DIR}"
echo "Run logs  : ${RUN_LOG_DIR}"
echo "TMPDIR    : ${TMPDIR}"
echo "Pool      : ${POOL_SIZE} concurrent runs x ${SUB_PROC} docking procs (${NCPU} cpus)"
echo "GPUs      : ${GPUS[*]} (${NGPU})"
echo ""

# Turn one manifest line into run.py flags. Every key in the manifest is a
# long-form flag on run.py, so this is a mechanical translation.
json_to_args() {
    python - "$1" <<'PY'
import json, shlex, sys
cfg = json.loads(sys.argv[1])
out = []
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            out.append(f'--{key}')
    else:
        out += [f'--{key}', str(value)]
print(' '.join(shlex.quote(tok) for tok in out))
PY
}

n_started=0
n_done=0
n_failed=0
running=0
declare -a FAILED_RUNS=()

launch() {
    local idx="$1" json="$2"
    local gpu="${GPUS[$(( idx % NGPU ))]}"
    local args
    args=$(json_to_args "${json}")
    local log="${RUN_LOG_DIR}/run_${idx}.log"

    (
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${gpu}" \
        python -u scripts/exps/lead/run.py ${args} \
            --model_path "${GENMOL_MODEL_PATH}" \
            --out_dir "${OUT_DIR}" \
            --num_sub_proc "${SUB_PROC}" \
            $( [ "${FORCE}" = "1" ] && echo --force ) \
            > "${log}" 2>&1
    ) &
}

idx=-1
while IFS= read -r line; do
    [ -z "${line}" ] && continue
    idx=$(( idx + 1 ))
    if [ $(( idx % TOTAL_WORKERS )) -ne "${WORKER_ID}" ]; then
        continue
    fi

    launch "${idx}" "${line}"
    n_started=$(( n_started + 1 ))
    running=$(( running + 1 ))

    if [ "${running}" -ge "${POOL_SIZE}" ]; then
        # `wait -n` needs bash >= 4.3; Alpine ships bash 5.
        if wait -n; then n_done=$(( n_done + 1 ));
        else n_failed=$(( n_failed + 1 )); fi
        running=$(( running - 1 ))
        echo "[$(date +%H:%M:%S)] started=${n_started} finished=$(( n_done + n_failed )) failed=${n_failed}"
    fi
done < "${MANIFEST}"

while [ "${running}" -gt 0 ]; do
    if wait -n; then n_done=$(( n_done + 1 ));
    else n_failed=$(( n_failed + 1 )); fi
    running=$(( running - 1 ))
done

echo ""
echo "======================================================================="
echo "Worker ${WORKER_ID}/${TOTAL_WORKERS} complete"
echo "Runs started : ${n_started}"
echo "Succeeded    : ${n_done}"
echo "Failed       : ${n_failed}"
echo "End          : $(date)"
if [ "${n_failed}" -gt 0 ]; then
    echo ""
    echo "Failed runs left no status.json and will be retried on resubmission."
    echo "Inspect: ${RUN_LOG_DIR}"
fi
echo ""
echo "Next: python scripts/exps/lead/collect.py --results-dir ${OUT_DIR} -o runs.parquet"
echo "======================================================================="

# A worker whose runs all failed must not look like a success to the driver.
[ "${n_failed}" -eq "${n_started}" ] && [ "${n_started}" -gt 0 ] && exit 1
exit 0
