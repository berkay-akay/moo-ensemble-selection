#!/usr/bin/zsh
#SBATCH --job-name=moo_es
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=c23ms
#SBATCH --chdir=/rwthfs/rz/cluster/hpcwork/ay047206/moo-ensemble-selection/assembled_ensembles
#SBATCH --array=0-9
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ay047206/logs/slurm-%x-%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ay047206/logs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=berkay.akay@rwth-aachen.de

set -euo pipefail

# Create log directory if missing
mkdir -p "/rwthfs/rz/cluster/hpcwork/ay047206/logs"

# Tee stdout/stderr to job-specific files (include array task id to avoid collisions)
LOGDIR="/rwthfs/rz/cluster/hpcwork/ay047206/logs"
LOG_OUT="${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-NA}.out"
LOG_ERR="${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-NA}.err"
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

echo "========== SLURM START =========="
echo "JobID=${SLURM_JOB_ID}  ArrayTaskID=${SLURM_ARRAY_TASK_ID:-NA}  Name=${SLURM_JOB_NAME}  User=${SLURM_JOB_USER}  Acct=${SLURM_JOB_ACCOUNT:-N/A}"
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

# Activate the prebuilt environment
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

# Make repo importable
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

trap 'rc=$?; echo "FAILED with exit code ${rc} at $(date)"; exit $rc' ERR
set -x

# Parameters
TASK_ID=59
PRUNER=SiloTopN
METHOD="MOOEnsembleSelection|n_generations:10|population_size:50"
METRIC=balanced_accuracy
METATASK=minimal_example_13
STATE_NAME=ensemble_evaluations_qdo
ISOLATE=no
LOAD_METHOD=eage
FOLD_IDX=${SLURM_ARRAY_TASK_ID}
CONFIG_SPACE=QDO
CONF_TAG=moo_conf_1


# Run using srun - pass 12 args expected by run_evaluate_ensemble_on_metatask.py
# Arg order:
# 1 openml_task_id | 2 pruner | 3 ensemble_method_name | 4 metric_name | 5 benchmark_name
# 6 evaluation_name | 7 isolate_execution("yes"/"no") | 8 load_method | 9 folds_to_run_on
# 10 config_space_name | 11 ens_save_name

echo "========== RUN (fold ${FOLD_IDX}) =========="
srun python -u "${ENTRY}" \
  ${TASK_ID} \
  ${PRUNER} \
  "${METHOD}" \
  ${METRIC} \
  ${METATASK} \
  ${STATE_NAME} \
  ${ISOLATE} \
  ${LOAD_METHOD} \
  ${FOLD_IDX} \
  ${CONFIG_SPACE} \
  ${CONF_TAG}

set +x
echo "========== DONE (fold ${FOLD_IDX}) =========="
date
