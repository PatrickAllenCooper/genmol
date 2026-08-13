#!/bin/bash
# Submit one stage of the GenMol bandit grid across aa100 and ah200.
#
# The binding constraint on Alpine is the per-user GRES quota, not node
# availability:
#
#   QOS          a100_80gb  a100-40gb  h200
#   gpu-normal       3          6        4
#   gpu-long         1          3        2
#
# So: use a100-40gb rather than a100_80gb (the model is ~110M params at batch
# size 1, and the quota is twice as large), and run BOTH partitions, since the
# quotas are per GRES type. That is ~10 GPUs rather than 6.
#
# ami100 (AMD, needs a ROCm torch build), al40 (CU Anschutz only) and
# artxpro6000 (Blackwell sm_120, needs torch >= 2.7 + cu128 against the pinned
# torch==2.6.0) are all unusable without a second environment.
#
# Usage (from the repo root on a CURC login node):
#   DRY_RUN=1 bash hpc/queue_grid.sh pilot            # print what would run
#   bash hpc/queue_grid.sh pilot
#   bash hpc/queue_grid.sh 1a
#   ANCHOR=anchors/best_1a.json bash hpc/queue_grid.sh 1b
#   ACCOUNT=ucb736_asc1 bash hpc/queue_grid.sh 1a     # bill a different account
#   SKIP_AH200=1 bash hpc/queue_grid.sh 1a            # aa100 only
#
# ACCOUNT defaults to ucb738_asc1. Do NOT let this fall back to the SLURM
# default: on this project DefaultAccount is `ucb-general`, the Trailhead tier
# (~2,000 SU/month), and running it over allocation depresses LevelFS and
# leaves jobs in PD (Priority). Verify after submitting with the %a column of
#   squeue -u $USER --format="%A %j %P %q %a %t %r"
#
# Before running for real:
#   suuser $USER 30                                   # SU usage (curc-quota is disk only)
#   sacctmgr -p show associations user=$USER format=account,partition,qos
#
# Author: Patrick Cooper

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

STAGE="${1:-}"
if [ -z "${STAGE}" ]; then
    echo "usage: bash hpc/queue_grid.sh <pilot|1a|1b|1c|1d|2|3>" >&2
    exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-ucb738_asc1}"
EXCLUDE_NODES="${EXCLUDE_NODES:-c3gpu-g5-u1}"
ANCHOR="${ANCHOR:-}"
FORCE="${FORCE:-0}"

# Overridable so DRY_RUN=1 can be exercised off-cluster; the worker script and
# hpc/common.sh read the same variable.
SCRATCH="${GENMOL_SCRATCH:-/scratch/alpine/${USER}/genmol}"
LOGS="${SCRATCH}/logs"
OUT_DIR="${OUT_DIR:-${SCRATCH}/results}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRATCH}/manifests}"
MANIFEST="${MANIFEST_DIR}/stage_${STAGE}.jsonl"

# aa100: 3 GPUs / 64 cores per node, quota 6x a100-40gb -> 2 whole-node jobs.
# ah200: 4 GPUs / 128 cores per node, quota 4x h200      -> 1 whole-node job.
AA100_WORKERS="${AA100_WORKERS:-2}"
AH200_WORKERS="${AH200_WORKERS:-1}"
SKIP_AA100="${SKIP_AA100:-0}"
SKIP_AH200="${SKIP_AH200:-0}"
WALLTIME="${WALLTIME:-24:00:00}"

mkdir -p "${LOGS}" "${OUT_DIR}" "${MANIFEST_DIR}" logs

echo "======================================================================="
echo "GenMol grid -- stage ${STAGE}"
echo "Date     : $(date)"
echo "Dry run  : ${DRY_RUN}"
echo "Account  : ${ACCOUNT}"
echo "Exclude  : ${EXCLUDE_NODES:-none}"
echo "Out dir  : ${OUT_DIR}"
echo "======================================================================="
echo ""

# --- Build the manifest --------------------------------------------------
echo "--- Manifest ---"
GEN_ARGS=(--stage "${STAGE}" -o "${MANIFEST}")
if [ -n "${ANCHOR}" ]; then
    if [ ! -f "${ANCHOR}" ]; then
        echo "FATAL: anchor file not found: ${ANCHOR}" >&2
        exit 1
    fi
    GEN_ARGS+=(--anchor "${ANCHOR}")
fi
if ! python scripts/exps/lead/make_manifest.py "${GEN_ARGS[@]}"; then
    echo "FATAL: manifest generation failed (stages 1b/1d/2/3 need ANCHOR=...)" >&2
    exit 1
fi

N_RUNS=$(grep -c . "${MANIFEST}")
echo "  ${N_RUNS} runs -> ${MANIFEST}"

# Report how many are already done, so the printed plan reflects real work.
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 2 -name status.json 2>/dev/null | wc -l)
    echo "  ${N_DONE} completed run(s) already in ${OUT_DIR} (skipped unless FORCE=1)"
fi
echo ""

TOTAL_WORKERS=0
[ "${SKIP_AA100}" = "1" ] || TOTAL_WORKERS=$(( TOTAL_WORKERS + AA100_WORKERS ))
[ "${SKIP_AH200}" = "1" ] || TOTAL_WORKERS=$(( TOTAL_WORKERS + AH200_WORKERS ))
if [ "${TOTAL_WORKERS}" -eq 0 ]; then
    echo "FATAL: both partitions skipped, nothing to submit." >&2
    exit 1
fi

LAST_JOBID=""
JOBIDS=()

sb() {
    local label="$1"; shift
    local exclude_args=()
    if [ -n "${EXCLUDE_NODES}" ]; then
        exclude_args=(--exclude="${EXCLUDE_NODES}")
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [dry-run] ${label}"
        echo "            sbatch --account=${ACCOUNT} ${exclude_args[*]} $*"
        LAST_JOBID="DRYRUN"
        return 0
    fi
    local out
    if out=$(sbatch --parsable --account="${ACCOUNT}" "${exclude_args[@]}" "$@" 2>&1); then
        LAST_JOBID="${out##*$'\n'}"
        echo "  SUBMITTED  ${label}  ->  job ${LAST_JOBID}"
        JOBIDS+=("${LAST_JOBID}")
    else
        LAST_JOBID=""
        echo "  FAILED     ${label}"
        echo "${out}" | sed 's/^/             /'
    fi
}

submit_worker() {
    local label="$1" wid="$2"; shift 2
    sb "${label}" \
        --time="${WALLTIME}" \
        --job-name="genmol_${STAGE}_w${wid}" \
        --output="${LOGS}/grid_${STAGE}_w${wid}_%j.out" \
        --error="${LOGS}/grid_${STAGE}_w${wid}_%j.err" \
        --export=ALL,MANIFEST="${MANIFEST}",WORKER_ID="${wid}",TOTAL_WORKERS="${TOTAL_WORKERS}",OUT_DIR="${OUT_DIR}",FORCE="${FORCE}" \
        "$@" \
        hpc/slurm_genmol_grid.sh
}

wid=0

if [ "${SKIP_AA100}" != "1" ]; then
    echo "--- aa100 (${AA100_WORKERS} workers, 3x a100-40gb each) ---"
    for (( i = 0; i < AA100_WORKERS; i++ )); do
        submit_worker "aa100 worker ${wid}" "${wid}" \
            --partition=aa100 --qos=gpu-normal --gres=gpu:a100-40gb:3 \
            --cpus-per-task=60 --mem=120G
        wid=$(( wid + 1 ))
    done
    echo ""
fi

if [ "${SKIP_AH200}" != "1" ]; then
    echo "--- ah200 (${AH200_WORKERS} workers, 4x h200 each) ---"
    for (( i = 0; i < AH200_WORKERS; i++ )); do
        # CLI flags beat the #SBATCH headers, so one worker script covers both
        # partitions with no edits.
        submit_worker "ah200 worker ${wid}" "${wid}" \
            --partition=ah200 --qos=gpu-normal --gres=gpu:h200:4 \
            --cpus-per-task=120 --mem=240G
        wid=$(( wid + 1 ))
    done
    echo ""
fi

echo "======================================================================="
echo "Stage ${STAGE}: ${N_RUNS} runs across ${TOTAL_WORKERS} workers"
echo ""
echo "Watch    : squeue -u \$USER --format=\"%A %j %P %q %a %t %r\""
echo "           (the %a column must read ${ACCOUNT}, not ucb-general)"
echo "History  : sacct -u \$USER --format=JobID,JobName%20,State,ExitCode,Elapsed -X"
echo "Efficiency: seff <jobid>   # use this to retune POOL_SIZE"
echo ""
echo "Collect  : python scripts/exps/lead/collect.py \\"
echo "             --results-dir ${OUT_DIR} -o ${SCRATCH}/runs_${STAGE}.parquet"
echo ""
echo "Resubmitting this exact command is safe: completed runs are skipped via"
echo "status.json, so it only finishes what a walltime kill left behind."
echo "======================================================================="
