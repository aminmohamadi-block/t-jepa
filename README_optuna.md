# SLURM-based Optuna Hyperparameter Tuning for T-JEPA

This system provides SLURM cluster-based hyperparameter optimization using Optuna's Bayesian optimization.

## Overview

- **Input**: YAML config file (wandb format) + project name
- **Output**: CSV results + Optuna visualizations + best configuration
- **Method**: Submits individual SLURM jobs for each trial
- **Storage**: CSV files + SQLite for Optuna state

## Usage

⚠️ **Important**: The tuning controller runs for hours/days and must be submitted as a SLURM job (not run on login node).

### Basic Usage

```bash
# Submit controller as SLURM job
./launch_tuning.sh \
    --config scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml \
    --project_name higgs_tuning_v1 \
    --n_trials 50 \
    --batch_size 10
```

### Advanced Usage

```bash
./launch_tuning.sh \
    --config scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml \
    --project_name higgs_tuning_v1 \
    --n_trials 200 \
    --batch_size 20 \
    --partition a100 \
    --gpus 4 \
    --cpus_per_task 32 \
    --mem 256G \
    --time 12:00:00 \
    --env_setup "module load cuda/11.8" \
    --sampler TPE
```

### Resume Existing Study

```bash
./launch_tuning.sh \
    --config scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml \
    --project_name higgs_tuning_v1 \
    --resume
```

### Alternative: Direct SLURM Submission

```bash
# For more control over controller job resources
sbatch submit_tuning.sh \
    --config scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml \
    --project_name higgs_tuning_v1 \
    --n_trials 100 \
    --batch_size 10
```

## Output Structure

Results are saved to `optuna_results_{project_name}/`:

```
optuna_results_higgs_tuning_v1/
├── optuna_study.db                 # Optuna state database
├── trials_results.csv              # Real-time trial results
├── study_summary.csv               # Final comprehensive results
├── best_results.json               # Best configuration summary
├── optimization_history.html       # Optuna visualization
├── param_importances.html          # Parameter importance plot
├── parallel_coordinate.html        # Parameter relationship plot
├── trial_0.sh, trial_1.sh, ...    # Generated SLURM scripts
├── trial_0.out, trial_1.out, ...  # SLURM job outputs
├── trial_0.err, trial_1.err, ...  # SLURM job errors
└── completed_trials.log            # Completion tracking
```

## Key Features

### 1. Parallel Batch Execution
- Submits multiple SLURM jobs simultaneously (configurable batch size)
- **Much faster than serial execution**: 10x speedup with batch_size=10
- Waits for entire batch to complete before submitting next batch
- Efficient resource utilization

### 2. SLURM Integration
- Automatically generates SLURM scripts for each trial
- Configurable resource allocation (GPUs, CPUs, memory, time)
- Job monitoring and completion detection via `squeue`
- Proper environment setup with modular commands

### 3. Reproducible Training
- Each trial runs `./scripts/launch_tjepa.sh` with specific parameters
- Full run.py compatibility for later reproduction
- MLflow integration with trial tagging

### 4. Robust Monitoring
- Batch-wise job completion tracking
- Handles job failures gracefully
- Timeout protection (6 hours default per job)
- Multiple score extraction methods

### 5. Comprehensive Reporting
- Real-time CSV logging during trials
- Final comprehensive CSV with all results
- Interactive HTML visualizations
- JSON summary of best configuration

## Configuration Format

Uses the existing wandb YAML format. Example:

```yaml
parameters:
  model_dim_hidden:
    values: [64, 128, 256, 512]
  exp_lr:
    min: 0.00001
    max: 0.001
  batch_size:
    value: 512  # Fixed parameter
```

Supported parameter types:
- `values: [...]` → Categorical choice
- `min/max` → Continuous range (float/int)
- `value: X` → Fixed parameter

## Arguments Reference

### Required
- `--config`: Path to YAML hyperparameter config
- `--project_name`: Project name for result organization

### Optional
- `--n_trials`: Number of trials to run (default: 100)
- `--batch_size`: Parallel jobs per batch (default: 10)
- `--partition`: SLURM partition (default: v100)
- `--gpus`: GPUs per job (default: 1)
- `--cpus_per_task`: CPUs per task (default: 8)
- `--mem`: Memory per job (default: 100G)
- `--time`: Time limit per job (default: 6:00:00)
- `--env_setup`: Additional environment setup commands
- `--sampler`: Optuna sampler [TPE, CmaEs, Random] (default: TPE)
- `--resume`: Resume existing study

## Performance Benefits

### Parallel vs Serial Execution

**Serial (old approach):**
- Submit job → Wait 6 hours → Submit next job → Wait 6 hours → ...
- 100 trials = 600 hours = 25 days

**Parallel Batches (new approach):**
- Submit 10 jobs → Wait 6 hours for all → Submit next 10 → Wait 6 hours → ...
- 100 trials in batches of 10 = 10 × 6 hours = 60 hours = 2.5 days
- **10x faster!**

### Optimal Batch Sizing
- **Small clusters**: `batch_size=5-10`
- **Large clusters**: `batch_size=20-50`
- **Consider**: Queue limits, resource availability, job duration variability

## Monitoring Progress

### Controller Job Status
```bash
# Check if controller job is running
squeue -u $USER | grep optuna_controller

# Follow controller output (replace JOBID with actual job ID)
tail -f optuna_controller_JOBID.out

# Check controller resource usage
sstat -j JOBID --format=AveCPU,AveRSS,MaxRSS
```

### Trial Progress
```bash
# Follow trial results in real-time
tail -f optuna_results_PROJECT_NAME/trials_results.csv

# Count completed trials
wc -l optuna_results_PROJECT_NAME/trials_results.csv

# Check active training jobs
squeue -u $USER | grep "PROJECT_NAME_trial"
```

### Study Analysis
```bash
# View best results so far
cat optuna_results_PROJECT_NAME/best_results.json

# Open visualizations (copy to local machine)
scp cluster:path/optuna_results_PROJECT_NAME/*.html .
# Then open in browser locally
```

## Example Workflow

1. **Prepare config**: Use existing `scripts/tjepa_tuning/config_tjepa_linear_higgs.yaml`
2. **Start tuning**: `./launch_tuning.sh --config ... --project_name my_study --batch_size 10`
3. **Monitor progress**: Check controller output and trial CSV
4. **Resume if needed**: Add `--resume` flag to continue interrupted studies
5. **Analyze results**: Open HTML visualizations and check best_results.json
6. **Reproduce best**: Use best parameters with `./scripts/launch_tjepa.sh`

## Technical Details

### Two-Level Job Architecture

**Controller Job** (CPU-only, long-running):
- Partition: `cpu` (or login-accessible partition)
- Resources: 4 CPUs, 16GB RAM, 72 hours max
- Purpose: Manages Optuna study, submits/monitors training jobs
- Runs continuously until all trials complete

**Training Jobs** (GPU-based, parallel):
- Partition: `v100`/`a100` (GPU partitions)
- Resources: Configurable GPUs, CPUs, memory
- Purpose: Individual T-JEPA training trials
- Submitted in batches, monitored by controller

### Job Flow
1. **Controller starts**: Loads/creates Optuna study
2. **Batch submission**: Submits N training jobs simultaneously
3. **Monitoring**: Polls `squeue` every 60 seconds
4. **Collection**: Gathers results when batch completes
5. **Next batch**: Repeats until all trials done

### Score Extraction
The system extracts validation scores from:
1. `OPTUNA_SCORE: X.XXX` in job output
2. `Best validation score: X.XXX` in job output
3. Dedicated score files: `trial_N_score.txt`

### Job Management
- Each trial gets unique job name: `{study_name}_trial_{N}`
- Jobs run in project directory with full environment
- Completion detected via `squeue` monitoring
- Failed jobs return score of `-inf`

### Bayesian Optimization
- Uses Optuna's TPE sampler by default
- MedianPruner for early stopping unpromising trials
- Supports all Optuna samplers and pruners
- Persistent state in SQLite database

### Robustness Features
- **Controller recovery**: Resume interrupted studies with `--resume`
- **Job failure handling**: Continue with remaining trials if some fail
- **Timeout protection**: Both controller (72h) and training jobs (6h) have limits
- **SSH disconnection safe**: Controller runs as SLURM job, not affected by SSH

This system provides a production-ready hyperparameter tuning solution for SLURM clusters while maintaining full compatibility with the existing T-JEPA codebase.