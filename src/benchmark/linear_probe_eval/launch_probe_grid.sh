#!/bin/bash

# Launch SLURM grid search for linear probe hyperparameter tuning
# This script sets up the parameter grid and submits job arrays

function show_help() {
    echo "Launch SLURM grid search for linear probe hyperparameter tuning"
    echo ""
    echo "Usage: ./launch_probe_grid.sh --checkpoint <path> [options]"
    echo ""
    echo "Required:"
    echo "  --checkpoint PATH        Path to T-JEPA checkpoint file"
    echo ""
    echo "Grid Search Options:"
    echo "  --lr_values STR          Learning rates (comma-separated, default: '0.001,0.01,0.1')"
    echo "  --wd_values STR          Weight decays (comma-separated, default: '0.0001,0.001,0.01')"
    echo ""
    echo "Training Options:"
    echo "  --dataset_name STR       Dataset name (default: higgs)"
    echo "  --data_path STR          Path to dataset (default: ./datasets)"
    echo "  --max_epochs INT         Maximum epochs (default: 100)"
    echo "  --batch_size INT         Batch size (default: 512)"
    echo "  --patience INT           Early stopping patience (default: 20)"
    echo ""
    echo "SLURM Options:"
    echo "  --partition STR          SLURM partition (default: v100)"
    echo "  --gpus INT               GPUs per job (default: 1)"
    echo "  --cpus_per_task INT      CPUs per job (default: 8)"
    echo "  --mem STR                Memory per job (default: 64G)"
    echo "  --time STR               Time limit per job (default: 2:00:00)"
    echo ""
    echo "MLflow Options:"
    echo "  --use_mlflow             Enable MLflow logging"
    echo "  --mlflow_experiment STR  MLflow experiment name (default: probe_grid_search)"
    echo ""
    echo "Output Options:"
    echo "  --results_dir STR        Results directory (default: probe_grid_results)"
    echo ""
    echo "Examples:"
    echo "  # Basic grid search"
    echo "  ./launch_probe_grid.sh --checkpoint ./checkpoints/higgs/model.pth"
    echo ""
    echo "  # Custom parameter ranges with MLflow"
    echo "  ./launch_probe_grid.sh --checkpoint ./checkpoints/higgs/model.pth \\"
    echo "                         --lr_values '0.001,0.01,0.1,1.0' \\"
    echo "                         --wd_values '0.0,0.001,0.01' \\"
    echo "                         --use_mlflow"
    echo ""
    echo "  # Large cluster setup"
    echo "  ./launch_probe_grid.sh --checkpoint ./checkpoints/higgs/model.pth \\"
    echo "                         --partition a100 \\"
    echo "                         --gpus 1 \\"
    echo "                         --mem 128G \\"
    echo "                         --time 4:00:00"
    echo ""
}

# Default values
CHECKPOINT_PATH=""
LR_VALUES="0.001,0.01,0.1"
WD_VALUES="0.0001,0.001,0.01"
DATASET_NAME="higgs"
DATA_PATH="./datasets"
MAX_EPOCHS=20
BATCH_SIZE=512
PATIENCE=100
PARTITION="a100"
GPUS=1
CPUS_PER_TASK=8
MEM="64G"
TIME="2:00:00"
USE_MLFLOW=false
MLFLOW_EXPERIMENT="t-jepa-higgs-probe"
RESULTS_DIR="t_jepa_higgs_probe_grid_results_20epoch"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --lr_values)
            LR_VALUES="$2"
            shift 2
            ;;
        --wd_values)
            WD_VALUES="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --cpus_per_task)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --use_mlflow)
            USE_MLFLOW=true
            shift
            ;;
        --mlflow_experiment)
            MLFLOW_EXPERIMENT="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: --checkpoint is required"
    show_help
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Generate parameter grid
echo "Generating parameter grid..."
GRID_FILE="${RESULTS_DIR}/grid_params.txt"
> "$GRID_FILE"  # Clear file

# Convert comma-separated values to arrays
IFS=',' read -ra LR_ARRAY <<< "$LR_VALUES"
IFS=',' read -ra WD_ARRAY <<< "$WD_VALUES"

# Generate all combinations
for lr in "${LR_ARRAY[@]}"; do
    for wd in "${WD_ARRAY[@]}"; do
        echo "${lr},${wd}" >> "$GRID_FILE"
    done
done

# Count total jobs
TOTAL_JOBS=$(wc -l < "$GRID_FILE")

echo "Parameter grid generated:"
echo "Learning rates: ${LR_VALUES}"
echo "Weight decays: ${WD_VALUES}"
echo "Total combinations: $TOTAL_JOBS"
echo "Grid file: $GRID_FILE"
echo ""

# Create modified submit script with correct array size
SUBMIT_SCRIPT="${RESULTS_DIR}/submit_probe_grid_${TOTAL_JOBS}.sh"
sed "s/PLACEHOLDER_ARRAY_SIZE/$TOTAL_JOBS/" src/benchmark/linear_probe_eval/submit_probe_grid.sh > "$SUBMIT_SCRIPT"

# Make it executable
chmod +x "$SUBMIT_SCRIPT"

# Prepare environment variables for job
export GRID_CHECKPOINT_PATH="$CHECKPOINT_PATH"
export GRID_DATASET_NAME="$DATASET_NAME"
export GRID_DATA_PATH="$DATA_PATH"
export GRID_MAX_EPOCHS="$MAX_EPOCHS"
export GRID_BATCH_SIZE="$BATCH_SIZE"
export GRID_PATIENCE="$PATIENCE"
export GRID_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT"
export GRID_RESULTS_DIR="$RESULTS_DIR"

if [[ "$USE_MLFLOW" == true ]]; then
    export GRID_USE_MLFLOW="true"
fi

# Submit job array
echo "Submitting SLURM job array..."
JOB_ID=$(sbatch \
    --partition="$PARTITION" \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="probe_grid_${DATASET_NAME}" \
    --output="${RESULTS_DIR}/probe_grid_%A_%a.out" \
    --error="${RESULTS_DIR}/probe_grid_%A_%a.err" \
    "$SUBMIT_SCRIPT" | grep -o '[0-9]\+')

if [[ $? -eq 0 && -n "$JOB_ID" ]]; then
    echo "✓ Grid search submitted successfully!"
    echo "Job Array ID: $JOB_ID"
    echo "Total jobs: $TOTAL_JOBS"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -j $JOB_ID                    # Check job status"
    echo "  ls ${RESULTS_DIR}/job_*_complete.txt # Count completed jobs"
    echo "  tail -f ${RESULTS_DIR}/probe_grid_${JOB_ID}_*.out # Follow specific job"
    echo ""
    echo "Results will be saved to:"
    echo "  ${RESULTS_DIR}/"
    echo ""
    if [[ "$USE_MLFLOW" == true ]]; then
        echo "MLflow experiment: $MLFLOW_EXPERIMENT"
    fi
    
    # Save job info
    cat > "${RESULTS_DIR}/job_info.txt" << EOF
Grid Search Job Information
==========================
Job Array ID: $JOB_ID
Submitted: $(date)
Checkpoint: $CHECKPOINT_PATH
Dataset: $DATASET_NAME
Learning rates: $LR_VALUES
Weight decays: $WD_VALUES
Total jobs: $TOTAL_JOBS

SLURM Configuration:
- Partition: $PARTITION
- GPUs: $GPUS
- CPUs: $CPUS_PER_TASK
- Memory: $MEM
- Time limit: $TIME

Training Configuration:
- Max epochs: $MAX_EPOCHS
- Batch size: $BATCH_SIZE
- Patience: $PATIENCE
- MLflow: $USE_MLFLOW
- Results dir: $RESULTS_DIR
EOF
    
else
    echo "✗ Failed to submit grid search jobs"
    exit 1
fi