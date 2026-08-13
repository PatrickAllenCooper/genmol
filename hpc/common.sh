#!/usr/bin/env bash
# hpc/common.sh -- shared boilerplate for the GenMol grid SLURM scripts.
#
# Adapted from the DeFAb/blanc hpc/common.sh conventions. Source this near the
# top of each slurm_*.sh. Under sbatch, $0 points at the spooled copy in
# /var/spool/slurmd, so scripts must try $SLURM_SUBMIT_DIR (the repo root at
# submit time) first:
#
#   if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/hpc/common.sh" ]; then
#       source "${SLURM_SUBMIT_DIR}/hpc/common.sh"
#   else
#       source "$(dirname "$0")/common.sh"
#   fi
#
# Provides:
#   genmol_print_job_header  - standard SLURM job metadata banner
#   genmol_activate_env      - load anaconda and activate the genmol env
#   genmol_setup_paths       - PROJ_DIR, PYTHONPATH, caches on scratch
#   genmol_preflight         - fail loudly if the docking toolchain is broken
#   genmol_gpu_list          - comma-separated GPU indices visible to this job
#
# Author: Patrick Cooper

set -euo pipefail

genmol_print_job_header() {
    local title="${1:-GenMol Job}"
    echo "======================================================================="
    echo "$title"
    echo "======================================================================="
    echo "Job ID    : ${SLURM_JOB_ID:-<local>}"
    echo "Node      : ${SLURM_NODELIST:-<local>}"
    echo "Partition : ${SLURM_JOB_PARTITION:-<local>}"
    echo "GPUs      : ${SLURM_GPUS_ON_NODE:-0}"
    echo "CPUs      : ${SLURM_CPUS_PER_TASK:-?}"
    echo "Start     : $(date)"
    echo ""
}

genmol_activate_env() {
    local env_name="${GENMOL_CONDA_ENV:-genmol}"
    if ! command -v conda &>/dev/null; then
        module load anaconda 2>/dev/null \
            || module load Anaconda3 2>/dev/null \
            || true
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${env_name}"
    echo "Conda env : ${env_name} ($(python --version 2>&1))"
}

genmol_setup_paths() {
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        PROJ_DIR="${SLURM_SUBMIT_DIR}"
    else
        PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
    export PROJ_DIR

    # run.py does `from scripts.exps.lead.docking.docking import DockingVina`,
    # which only resolves from the repo root.
    cd "${PROJ_DIR}"
    export PYTHONPATH="${PROJ_DIR}:${PROJ_DIR}/src:${PYTHONPATH:-}"

    # Home is 2 GB on Alpine -- never cache anything there. Use $USER, never a
    # hard-coded username.
    export GENMOL_SCRATCH="${GENMOL_SCRATCH:-/scratch/alpine/${USER}/genmol}"
    export HF_HOME="${HF_HOME:-${GENMOL_SCRATCH}/hf_cache}"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    mkdir -p "${HF_HOME}" 2>/dev/null || true

    echo "Proj dir  : ${PROJ_DIR}"
    echo "Scratch   : ${GENMOL_SCRATCH}"
    echo "HF_HOME   : ${HF_HOME}"
    echo ""
}

# A broken openbabel or a non-executable qvina02 does NOT fail the run: every
# dock raises, is caught, and is scored with the 99.9 sentinel, so run.py exits
# 0 after producing ~1000 `gen_3d unexpected error` lines and rv mean=0.00.
# Across a 1000-run grid that silently yields a thousand empty successes.
genmol_preflight() {
    local ok=0
    if ! command -v obabel &>/dev/null; then
        echo "FATAL: obabel not on PATH (openbabel-wheel not installed?)" >&2
        ok=1
    fi
    if [ ! -x "${PROJ_DIR}/scripts/exps/lead/docking/qvina02" ]; then
        echo "FATAL: scripts/exps/lead/docking/qvina02 is not executable." >&2
        echo "       Git tracks it as mode 100755, so this means the tree was" >&2
        echo "       copied rather than cloned. Run: chmod +x that path." >&2
        ok=1
    fi
    if [ ! -f "${GENMOL_MODEL_PATH:-${PROJ_DIR}/model.ckpt}" ]; then
        echo "FATAL: checkpoint not found at ${GENMOL_MODEL_PATH:-${PROJ_DIR}/model.ckpt}" >&2
        echo "       Download nvidia/clara/genmol_v1 from NGC and stage it." >&2
        ok=1
    fi
    if ! python -c "import rdkit, torch" 2>/dev/null; then
        echo "FATAL: rdkit/torch not importable in the active env." >&2
        ok=1
    fi
    # model.py wraps forward() in torch.amp.autocast('cuda', ...), which merely
    # warns and no-ops when CUDA is absent. A worker that lost its GPU would
    # therefore run silently on CPU, several times slower, and blow the
    # walltime rather than failing. Check explicitly.
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "FATAL: torch.cuda.is_available() is False inside this allocation." >&2
        echo "       Check --gres actually granted a GPU (nvidia-smi), and that" >&2
        echo "       the torch build matches the node's driver." >&2
        ok=1
    fi
    [ "$ok" -eq 0 ] && echo "Preflight : OK" && echo ""
    return "$ok"
}

genmol_gpu_list() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "${CUDA_VISIBLE_DEVICES}"
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -
    else
        echo "0"
    fi
}
