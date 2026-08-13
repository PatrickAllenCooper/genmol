#!/bin/bash
# Build the genmol conda environment on CURC Alpine.
#
# IMPORTANT: run this on a compute node, not a login node.
#   cd /projects/$USER/genmol
#   acompile
#   bash hpc/setup_curc_env.sh
#
# Four things bite here that env/setup.sh does not handle:
#
#   1. `pip install -e .` downgrades transformers. pyproject.toml pins
#      transformers==4.52.4 while env/requirements.txt pins 4.56.2, and
#      env/setup.sh installs requirements FIRST and the editable package
#      SECOND -- so the editable install wins and drags transformers back.
#      Installing with --no-deps keeps the requirements.txt resolution.
#   2. safe-mol==0.1.14 cannot be imported against modern transformers until
#      env/fix_safe_imports.sh has patched its __init__.
#   3. qvina02 must be executable. Git tracks it as mode 100755, so a clone on
#      Alpine is fine, but a tree copied from Windows loses the bit.
#   4. The tokenizer and the TDC SA-score table download lazily at first use.
#      ~30 concurrent grid workers doing first-touch downloads is both slow and
#      a good way to get rate-limited, so warm the caches once here.
#
# Author: Patrick Cooper

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "${SCRIPT_DIR}")"
CONDA_ENV_NAME="${GENMOL_CONDA_ENV:-genmol}"
SCRATCH="/scratch/alpine/${USER}/genmol"

cd "${PROJ_DIR}"

echo "======================================================================="
echo "GenMol CURC environment setup"
echo "======================================================================="
echo "Repo     : ${PROJ_DIR}"
echo "Env      : ${CONDA_ENV_NAME}"
echo "Scratch  : ${SCRATCH}"
echo ""

HOSTNAME_NOW="$(hostname)"
if [[ ${HOSTNAME_NOW} == login* ]]; then
    echo "ERROR: you are on a login node (${HOSTNAME_NOW})."
    echo "Run 'acompile' first, then re-run this script."
    exit 1
fi

echo "--- Modules ---"
module purge 2>/dev/null || true
module load anaconda 2>/dev/null || module load Anaconda3 2>/dev/null || true
module load cuda/12.1 2>/dev/null || echo "  (cuda/12.1 module unavailable; relying on the torch wheel)"
eval "$(conda shell.bash hook)"
echo "  conda: $(which conda)"
echo ""

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Env '${CONDA_ENV_NAME}' already exists. Remove it first to rebuild:"
    echo "  conda env remove -n ${CONDA_ENV_NAME} -y"
    echo "Continuing with the existing environment."
else
    echo "--- Creating env (python 3.10; fix_safe_imports.sh hardcodes that path) ---"
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
fi

conda activate "${CONDA_ENV_NAME}"
echo "  python: $(python --version 2>&1)"
echo ""

echo "--- Dependencies ---"
pip install --upgrade pip setuptools wheel
pip install -r env/requirements.txt
# --no-deps: pyproject.toml's transformers==4.52.4 pin would otherwise override
# requirements.txt's 4.56.2 and break safe-mol differently.
pip install -e . --no-deps
# Deliberately NOT installing scikit-learn==1.2.2. env/setup.sh pins it for the
# PMO hit-generation oracles (gsk3b, jnk3), which the lead-optimization grid
# never touches, and the downgrade puts it below bionemo-moco's
# scikit-learn>=1.6.0 requirement -- and bionemo-moco supplies the MDLM the
# sampler runs on. Leave the resolved version alone.
echo ""

echo "--- Patching safe-mol ---"
bash env/fix_safe_imports.sh
echo ""

echo "--- Docking toolchain ---"
chmod +x scripts/exps/lead/docking/qvina02 || true
if [ -x scripts/exps/lead/docking/qvina02 ]; then
    echo "  qvina02: executable"
else
    echo "  ERROR: qvina02 is still not executable" >&2
    exit 1
fi
if command -v obabel &>/dev/null; then
    echo "  obabel : $(obabel -V 2>&1 | head -1)"
else
    echo "  ERROR: obabel not on PATH after installing openbabel-wheel" >&2
    exit 1
fi
echo ""

echo "--- Warming caches on scratch ---"
export HF_HOME="${SCRATCH}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${SCRATCH}"/{results,logs,manifests}
python - <<'PY'
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
try:
    from safe.tokenizer import SAFETokenizer
    SAFETokenizer.from_pretrained('datamol-io/safe-gpt')
    print('  tokenizer cached')
except Exception as exc:
    print(f'  WARNING: tokenizer prefetch failed: {exc}')
try:
    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer  # noqa: F401
    print('  SA scorer importable')
except Exception as exc:
    print(f'  WARNING: SA scorer import failed: {exc}')
PY
echo ""

echo "--- Verifying ---"
# cuda=False is EXPECTED here: acompile hands out CPU nodes. The worker's
# preflight checks torch.cuda.is_available() inside the real GPU allocation.
python -c "import torch; print(f'  torch {torch.__version__}, cuda={torch.cuda.is_available()} (False is expected on an acompile node)')"
python -c "import rdkit; print(f'  rdkit {rdkit.__version__}')"
python -c "import transformers; print(f'  transformers {transformers.__version__}')"
python -c "from openbabel import pybel; print('  openbabel importable')"
echo ""

echo "======================================================================="
echo "Setup complete."
echo ""
if [ -f "${PROJ_DIR}/model.ckpt" ]; then
    echo "Checkpoint: ${PROJ_DIR}/model.ckpt ($(stat -c %s "${PROJ_DIR}/model.ckpt") bytes)"
else
    echo "STILL NEEDED -- the V1 checkpoint. The NGC CLI is not installed on"
    echo "Alpine, so use NVIDIA's HuggingFace mirror (public, ungated, one file,"
    echo "1396949417 bytes). curl needs no conda env and, unlike hf_hub_download,"
    echo "does not leave a second 1.3 GB copy in the cache:"
    echo ""
    echo "  curl -L -o ${PROJ_DIR}/model.ckpt \\"
    echo "    https://huggingface.co/nvidia/NV-GenMol-89M-v1/resolve/main/model.ckpt"
fi
echo ""
echo "Note: this script activated '${CONDA_ENV_NAME}' in ITS OWN shell only."
echo "Your prompt is back to (base). Before running anything by hand:"
echo "  module load anaconda && eval \"\$(conda shell.bash hook)\" && conda activate ${CONDA_ENV_NAME}"
echo ""
echo "Then, from a login node:"
echo "  DRY_RUN=1 bash hpc/queue_grid.sh pilot"
echo "  bash hpc/queue_grid.sh pilot"
echo "======================================================================="
