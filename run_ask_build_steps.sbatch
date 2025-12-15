#!/usr/bin/zsh
#SBATCH --job-name=ask_steps_2_3_folds_0_9
#SBATCH --array=0-9
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=c23ms
#SBATCH --chdir=/rwthfs/rz/cluster/hpcwork/ay047206/moo-ensemble-selection
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ay047206/logs/slurm-%x-%A_%a.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ay047206/logs/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=berkay.akay@rwth-aachen.de
# #SBATCH -A rwthXXXX   # <-- uncomment & set if required by your project

set -euo pipefail

# Create log directory if missing
mkdir -p "/rwthfs/rz/cluster/hpcwork/ay047206/logs"

# Tee stdout/stderr to job- and array-specific files as well
LOGDIR="/rwthfs/rz/cluster/hpcwork/ay047206/logs"
LOG_OUT="${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.out"
LOG_ERR="${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.err"
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

echo "========== SLURM START (STEPS 1–3 ARRAY) =========="
echo "JobID=${SLURM_JOB_ID}  ArrayJobID=${SLURM_ARRAY_JOB_ID:-NA}  TaskID=${SLURM_ARRAY_TASK_ID:-NA}  Name=${SLURM_JOB_NAME}"
echo "Node(s)=${SLURM_JOB_NODELIST}  CPUs=${SLURM_CPUS_PER_TASK}  Mem=${SLURM_MEM_PER_NODE:-NA}"
echo "Initial pwd=$(pwd)  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-N/A}  TMPDIR=${TMPDIR:-N/A}"
date

# Threading limits to avoid oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

# --- Conda activation (same as other sbatch) ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at $HOME/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Activate the prebuilt environment
conda activate py395

# Show interpreter info for sanity
python --version
python -c "import sys; print('Python', sys.version)"

# Ensure repo is importable (add project root)
export PYTHONPATH="/rwthfs/rz/cluster/hpcwork/ay047206/moo-ensemble-selection:${PYTHONPATH:-}"

FOLD_ID="${SLURM_ARRAY_TASK_ID}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

OPENML_TASK_ID=59
FOLDS_CSV="0,1,2,3,4,5,6,7,8,9"
METRIC="balanced_accuracy"
BASE_FOLDER="minimal_example_13"

STATE_DIR="/rwthfs/rz/cluster/hpcwork/ay047206/moo-ensemble-selection/benchmark/state/${BASE_FOLDER}/task_${OPENML_TASK_ID}"
OUTPUT_DIR="/rwthfs/rz/cluster/hpcwork/ay047206/moo-ensemble-selection/benchmark/output/${BASE_FOLDER}/task_${OPENML_TASK_ID}"

mkdir -p "${STATE_DIR}" || true

trap 'rc=$?; echo "FAILED with exit code ${rc} at $(date)"; exit $rc' ERR
set -x

# Step 1: run_ask_on_metatask.py for fold
srun python -u assembled_ask/build_steps/run_ask_on_metatask.py \
  ${OPENML_TASK_ID} \
  1 \
  1 \
  ${FOLD_ID} \
  ${METRIC} \
  ${BASE_FOLDER}

# Step 2: run_collect_predictor_data.py for fold
srun python -u assembled_ask/build_steps/run_collect_predictor_data.py \
  ${OPENML_TASK_ID} \
  ${FOLD_ID} \
  ${METRIC} \
  ${BASE_FOLDER} \
  norefit \
  save_space_false

set +x

# Coordinator: Only Task 0 proceeds to Step 3 after verifying all folds finished Step 2
if [ "${TASK_ID}" = "0" ]; then
  echo "Task 0 waiting for predictor data collection to finish for all folds..."
  set -x
  # Wait up to 6 hours (720 * 30s) for all 10 .done files
  MAX_TRIES=720
  SLEEP_SECS=30
  i=0
  while [ $i -lt $MAX_TRIES ]; do
    missing=0
    for f in 0 1 2 3 4 5 6 7 8 9; do
      if [ ! -f "${STATE_DIR}/run_collect_predictor_data_${f}.done" ]; then
        missing=$((missing+1))
      fi
    done
    if [ $missing -eq 0 ]; then
      echo "All run_collect_predictor_data_<fold>.done files found. Proceeding to step 3."
      break
    fi
    i=$((i+1))
    echo "${missing} folds still missing predictor data. Sleeping ${SLEEP_SECS}s (attempt ${i}/${MAX_TRIES})..."
    sleep ${SLEEP_SECS}
  done

  if [ $missing -ne 0 ]; then
    echo "ERROR: Timeout waiting for all predictor data .done files. Missing: ${missing}" >&2
    exit 3
  fi

  # Step 3a: TopN
  srun python -u assembled_ask/build_steps/run_full_metatask_builder.py \
    ${OPENML_TASK_ID} \
    1 \
    1 \
    ${METRIC} \
    csv \
    ${BASE_FOLDER} \
    TopN

  # Step 3b: SiloTopN
  srun python -u assembled_ask/build_steps/run_full_metatask_builder.py \
    ${OPENML_TASK_ID} \
    1 \
    1 \
    ${METRIC} \
    csv \
    ${BASE_FOLDER} \
    SiloTopN
  set +x
else
  echo "Task ${TASK_ID}: Completed step 2 for fold ${FOLD_ID}. Step 3 will be handled by Task 0."
fi

echo "========== DONE (TASK ${TASK_ID} / FOLD ${FOLD_ID}) =========="
date
