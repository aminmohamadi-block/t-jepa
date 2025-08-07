#!/bin/bash

# Wrapper script to easily submit Optuna hyperparameter tuning jobs
# This submits the controller as a SLURM job that will run for days

function show_help() {
    echo "Launch Optuna hyperparameter tuning on SLURM cluster"
    echo ""
    echo "Usage: ./launch_tuning.sh --config <yaml> --project_name <name> [options]"
    echo ""
    echo "Required:"
    echo "  --config          Path to YAML hyperparameter config"
    echo "  --project_name    Project name for result organization"
    echo ""
    echo "Tuning Options:"
    echo "  --n_trials        Number of trials to run (default: 100)"
    echo "  --batch_size      Parallel jobs per batch (default: 10)"
    echo "  --sampler         Optuna sampler: TPE|CmaEs|Random (default: TPE)"
    echo "  --resume          Resume existing study"
    echo "  --max_retries     Maximum retries for preempted jobs (default: 3)"
    echo ""
    echo "Training Job Resources:"
    echo "  --partition       SLURM partition for training jobs (default: a100)"
    echo "  --gpus            GPUs per training job (default: 1)"
    echo "  --cpus_per_task   CPUs per training job (default: 8)"
    echo "  --mem             Memory per training job (default: 100G)"
    echo "  --time            Time limit per training job (default: 6:00:00)"
    echo "  --env_setup       Additional environment commands for training"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  ./launch_tuning.sh --config scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml --project_name higgs_v1"
    echo ""
    echo "  # Large study with bigger batches"
    echo "  ./launch_tuning.sh --config scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml --project_name higgs_large \\"
    echo "                     --n_trials 200 --batch_size 20 --partition a100 --gpus 4"
    echo ""
    echo "  # Resume existing study"
    echo "  ./launch_tuning.sh --config scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml --project_name higgs_v1 --resume"
    echo ""
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Check for help flag
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        show_help
        exit 0
    fi
done

# Submit the controller job with all arguments passed through
echo "Submitting Optuna hyperparameter tuning controller job..."
echo "Arguments: $*"

job_id=$(sbatch src/benchmark/optuna_tune/submit_tuning.sh "$@" | grep -o '[0-9]\+')

if [ $? -eq 0 ]; then
    echo "✓ Controller job submitted successfully!"
    echo "Job ID: $job_id"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -j $job_id                              # Check job status"  
    echo "  tail -f optuna_controller_${job_id}.out        # Follow controller output (initially)"
    echo "  tail -f optuna_results_*/trials_results.csv   # Follow trial results"
    echo ""
    echo "Note: Controller logs will be moved to optuna_results_*/ when the job completes"
    echo ""
    echo "The controller will:"
    echo "  1. Submit batches of training jobs"
    echo "  2. Wait for each batch to complete"
    echo "  3. Collect results and submit next batch"
    echo "  4. Generate final report when done"
else
    echo "✗ Failed to submit controller job"
    exit 1
fi