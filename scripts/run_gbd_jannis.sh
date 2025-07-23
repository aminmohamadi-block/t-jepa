#!/bin/bash
#SBATCH --job-name=gbd_xgboost_jannis
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Assuming that this script is run from the root of the project

source ../bin/activate-hermit
source .venv/bin/activate

# Default configuration file
CONFIG_FILE="./src/benchmark/model_configs/gbd_configs/xgboost_jannis.json"

# Parse command line arguments
NUM_RUNS=5
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE="$2"
            shift
            shift
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift
            shift
            ;;
        --embedded)
            CONFIG_FILE="./src/benchmark/model_configs/gbd_configs/xgboost_jannis.json"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running XGBoost on jannis dataset..."
echo "Config file: $CONFIG_FILE"
echo "Number of runs: $NUM_RUNS"

# Run the experiment
python run_gbd.py --config_file="$CONFIG_FILE" --num_runs="$NUM_RUNS"