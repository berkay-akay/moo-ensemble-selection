#!/usr/bin/zsh
#SBATCH --job-name=moo_es
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=c23ms
#SBATCH --chdir=/rwthfs/rz/cluster/hpcwork/ay047206/moo-ensemble-selection/assembled_ensembles
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ay047206/logs/slurm-%x-%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ay047206/logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=berkay.akay@rwth-aachen.de
# #SBATCH -A rwthXXXX   # <-- uncomment & set if required by your project

set -euo pipefail

# Create log directory if missing
mkdir -p "/rwthfs/rz/cluster/hpcwork/ay047206/logs"

# Tee stdout/stderr to job-specific files as well
LOGDIR="/rwthfs/rz/cluster/hpcwork/ay047206/logs"
LOG_OUT="${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
LOG_ERR="${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

echo "========== SLURM START =========="
echo "JobID=${SLURM_JOB_ID}  Name=${SLURM_JOB_NAME}  User=${SLURM_JOB_USER}  Acct=${SLURM_JOB_ACCOUNT:-N/A}"
echo "Node(s)=${SLURM_JOB_NODELIST}  CPUs=${SLURM_CPUS_PER_TASK}  Mem=${SLURM_MEM_PER_NODE:-NA}"
echo "Initial pwd=$(pwd)  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-N/A}  TMPDIR=${TMPDIR:-N/A}"
date

# Threading limits to avoid oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

# --- Conda activation ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at $HOME/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Activate the prebuilt environment (no installs here)
conda activate py395-ensemble

# Show interpreter info for sanity
python --version
python -c "import sys, sklearn, ConfigSpace; print('Python', sys.version); print('sklearn', sklearn.__version__); print('ConfigSpace', ConfigSpace.__version__)"

# Ensure we are in the repo
REPO_DIR="/rwthfs/rz/cluster/hpcwork/ay047206/moo-ensemble-selection/assembled_ensembles"
echo "cd to repo: ${REPO_DIR}"
cd "${REPO_DIR}" || { echo "ERROR: Repo dir not found: ${REPO_DIR}"; exit 2; }
echo "pwd after cd: $(pwd)"

# Sanity check: entrypoint exists
ENTRY="${REPO_DIR}/run_evaluate_ensemble_on_metatask.py"
if [ ! -f "${ENTRY}" ]; then
  echo "ERROR: Entrypoint not found: ${ENTRY}"
  ls -la
  exit 2
fi

# Make repo importable (if your code does relative imports)
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Helpful failure trap + command echo
trap 'rc=$?; echo "FAILED with exit code ${rc} at $(date)"; exit $rc' ERR
set -x

# Parameters (kept explicit to match your interactive run)
TASK_ID=-1
DATASET_TECHNIQUE=SiloTopN
METHOD="MOOEnsembleSelection|n_generations:1|population_size:2"
METRIC=balanced_accuracy
METATASK=minimal_example_12
STATE_NAME=ensemble_evaluations_qdo
SAVE_VAL_PRED=no
USE_CONF=no
SEED=-1
EVAL_MODE=QDO
CONF_TAG=moo_conf_0

# Run using srun
echo "========== RUN =========="
srun python -u "${ENTRY}" \
  ${TASK_ID} \
  ${DATASET_TECHNIQUE} \
  "${METHOD}" \
  ${METRIC} \
  ${METATASK} \
  ${STATE_NAME} \
  ${SAVE_VAL_PRED} \
  ${USE_CONF} \
  ${SEED} \
  ${EVAL_MODE} \
  ${CONF_TAG}

set +x
echo "========== DONE =========="
date
