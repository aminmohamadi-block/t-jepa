#!/bin/bash
#SBATCH --job-name=probe_grid_search
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=probe_grid_%A_%a.out
#SBATCH --error=probe_grid_%A_%a.err
#SBATCH --array=1-PLACEHOLDER_ARRAY_SIZE

# Grid search script for linear probe hyperparameter tuning
# This script runs as a SLURM job array to test different hyperparameter combinations

# Environment setup
source ../bin/activate-hermit
source .venv/bin/activate

# Parse command line arguments passed from launch script
CHECKPOINT_PATH=""
DATASET_NAME="higgs"
DATA_PATH="./datasets"
USE_MLFLOW=""
MLFLOW_EXPERIMENT="probe_grid_search"
MAX_EPOCHS=100
BATCH_SIZE=512
PATIENCE=20
RESULTS_DIR="probe_grid_results"

# Read parameters from environment (set by launch script)
if [[ -n "$GRID_CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="$GRID_CHECKPOINT_PATH"
fi
if [[ -n "$GRID_DATASET_NAME" ]]; then
    DATASET_NAME="$GRID_DATASET_NAME"
fi
if [[ -n "$GRID_DATA_PATH" ]]; then
    DATA_PATH="$GRID_DATA_PATH"
fi
if [[ -n "$GRID_USE_MLFLOW" ]]; then
    USE_MLFLOW="--use_mlflow"
fi
if [[ -n "$GRID_MLFLOW_EXPERIMENT" ]]; then
    MLFLOW_EXPERIMENT="$GRID_MLFLOW_EXPERIMENT"
fi
if [[ -n "$GRID_MAX_EPOCHS" ]]; then
    MAX_EPOCHS="$GRID_MAX_EPOCHS"
fi
if [[ -n "$GRID_BATCH_SIZE" ]]; then
    BATCH_SIZE="$GRID_BATCH_SIZE"
fi
if [[ -n "$GRID_PATIENCE" ]]; then
    PATIENCE="$GRID_PATIENCE"
fi
if [[ -n "$GRID_RESULTS_DIR" ]]; then
    RESULTS_DIR="$GRID_RESULTS_DIR"
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Define parameter grid - read from file created by launch script
GRID_FILE="${RESULTS_DIR}/grid_params.txt"
if [[ ! -f "$GRID_FILE" ]]; then
    echo "Error: Grid parameters file not found: $GRID_FILE"
    exit 1
fi

# Get parameters for this array task
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$GRID_FILE")
if [[ -z "$PARAMS" ]]; then
    echo "Error: No parameters found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse parameters (format: lr,wd)
IFS=',' read -r LEARNING_RATE WEIGHT_DECAY <<< "$PARAMS"

echo "Starting grid search job:"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Learning Rate: $LEARNING_RATE"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_NAME"
echo "Results Dir: $RESULTS_DIR"

# Create unique run name
RUN_NAME="${DATASET_NAME}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_job${SLURM_ARRAY_TASK_ID}"

# Run the linear probe test
python src/benchmark/linear_probe_eval/test_linear_probe.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_name "$DATASET_NAME" \
    --data_path "$DATA_PATH" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    $USE_MLFLOW \
    --mlflow_experiment "$MLFLOW_EXPERIMENT" \
    --run_name "$RUN_NAME" \
    --device cuda

# Save job completion signal
echo "Job completed successfully at $(date)" > "${RESULTS_DIR}/job_${SLURM_ARRAY_TASK_ID}_complete.txt"

echo "Grid search job completed: $RUN_NAME"