#!/bin/bash
#SBATCH --job-name=optuna_controller
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --output=optuna_controller_%j.out
#SBATCH --error=optuna_controller_%j.err

# This script runs the Optuna hyperparameter tuning controller
# It needs to run as a SLURM job because it will run for many hours/days
# monitoring and submitting batches of training jobs

# Environment Setup
source ../bin/activate-hermit
source .venv/bin/activate

# Default values
CONFIG=""
PROJECT_NAME=""
N_TRIALS=200
BATCH_SIZE=10
PARTITION="a100"
GPUS=1
CPUS_PER_TASK=8
MEM="100G"
TIME="6:00:00"
ENV_SETUP=""
SAMPLER="TPE"
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --n_trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
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
        --env_setup)
            ENV_SETUP="$2"
            shift 2
            ;;
        --sampler)
            SAMPLER="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG" || -z "$PROJECT_NAME" ]]; then
    echo "Error: --config and --project_name are required"
    echo "Usage: sbatch submit_tuning.sh --config <path> --project_name <name> [other options]"
    exit 1
fi


echo "Starting Optuna hyperparameter tuning controller"
echo "Config: $CONFIG"
echo "Project: $PROJECT_NAME"
echo "Trials: $N_TRIALS"
echo "Batch size: $BATCH_SIZE"
echo "Training job config: $PARTITION, $GPUS GPUs, $CPUS_PER_TASK CPUs, $MEM, $TIME"
echo "Start time: $(date)"

# Run the Optuna tuning
python src/benchmark/optuna_tune/tune_optuna.py \
    --config "$CONFIG" \
    --project_name "$PROJECT_NAME" \
    --n_trials "$N_TRIALS" \
    --batch_size "$BATCH_SIZE" \
    --partition "$PARTITION" \
    --gpus "$GPUS" \
    --cpus_per_task "$CPUS_PER_TASK" \
    --mem "$MEM" \
    --time "$TIME" \
    --env_setup "$ENV_SETUP" \
    --sampler "$SAMPLER" \
    $RESUME

echo "Optuna tuning completed at: $(date)"
echo "Results saved to: optuna_results_${PROJECT_NAME}/"