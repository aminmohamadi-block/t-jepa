#!/usr/bin/env python3
"""
SLURM-based Optuna hyperparameter tuning for T-JEPA
Submits SLURM jobs for each trial and monitors completion
"""

import argparse
import os
import sys
import yaml
import optuna
import subprocess
import time
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import re


def parse_yaml_config(config_path: str) -> Dict[str, Any]:
    """Parse YAML config file and extract parameter definitions"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config['parameters']


def suggest_parameter(trial: optuna.Trial, param_name: str, param_config: Dict[str, Any]) -> Any:
    """Convert YAML parameter config to Optuna suggestion"""
    
    if 'values' in param_config:
        # Categorical parameter
        return trial.suggest_categorical(param_name, param_config['values'])
    
    elif 'min' in param_config and 'max' in param_config:
        # Continuous parameter
        min_val = param_config['min']
        max_val = param_config['max']
        
        # Check if it's a float or int range
        if isinstance(min_val, float) or isinstance(max_val, float):
            # Use log scale for learning rates
            if 'lr' in param_name.lower():
                return trial.suggest_float(param_name, min_val, max_val, log=True)
            else:
                return trial.suggest_float(param_name, min_val, max_val)
        else:
            return trial.suggest_int(param_name, min_val, max_val)
    
    elif 'value' in param_config:
        # Fixed parameter - return as is
        return param_config['value']
    
    else:
        raise ValueError(f"Unknown parameter config format for {param_name}: {param_config}")


def create_slurm_script(params: Dict[str, Any], study_name: str, trial_number: int, 
                       base_slurm_config: Dict[str, str], output_dir: str, project_name: str) -> str:
    """Create SLURM script for this trial"""
    
    script_path = f"{output_dir}/trial_{trial_number}.sh"
    
    # Convert parameters to command line arguments (space-separated format)
    cmd_args = []
    for param_name, param_value in params.items():
        cmd_args.extend([f"--{param_name}", str(param_value)])
    
    # Add study metadata (only supported arguments)
    cmd_args.extend([
        "--tag", f"optuna_trial_{trial_number}",
        "--project_name", project_name
    ])
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={study_name}_trial_{trial_number}
#SBATCH --partition={base_slurm_config.get('partition', 'a100')}
#SBATCH --gpus={base_slurm_config.get('gpus', '1')}
#SBATCH --cpus-per-task={base_slurm_config.get('cpus_per_task', '8')}
#SBATCH --mem={base_slurm_config.get('mem', '100G')}
#SBATCH --time={base_slurm_config.get('time', '6:00:00')}
#SBATCH --output={output_dir}/trial_{trial_number}.out
#SBATCH --error={output_dir}/trial_{trial_number}.err

# Environment setup
source ../bin/activate-hermit
source .venv/bin/activate
{base_slurm_config.get('env_setup', '')}

# Change to project directory
cd {os.getcwd()}

# Run T-JEPA training with Optuna parameters
./scripts/launch_tjepa.sh {' '.join(cmd_args)}

# Signal completion and write final score
echo "SLURM_JOB_COMPLETION: trial_{trial_number}" >> {output_dir}/completed_trials.log
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path


def submit_slurm_job(script_path: str) -> Optional[str]:
    """Submit SLURM job and return job ID"""
    try:
        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract job ID from sbatch output (e.g., "Submitted batch job 12345")
            job_id = re.search(r'Submitted batch job (\d+)', result.stdout)
            return job_id.group(1) if job_id else None
        else:
            print(f"Failed to submit job: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None




def read_trial_result(output_dir: str, trial_number: int) -> float:
    """Read the final validation score from trial output"""
    
    # Check multiple possible output files
    output_files = [
        f"{output_dir}/trial_{trial_number}.out",
        f"{output_dir}/trial_{trial_number}_score.txt"
    ]
    
    for output_file in output_files:
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    content = f.read()
                
                # Look for score markers
                lines = content.strip().split('\n')
                for line in reversed(lines):
                    if 'OPTUNA_SCORE:' in line:
                        score = float(line.split(':')[1].strip())
                        return score
                    elif 'Best validation score:' in line:
                        score = float(line.split(':')[1].strip())
                        return score
                        
            except Exception as e:
                print(f"Error reading {output_file}: {e}")
    
    # If no score found, return worst possible
    print(f"No score found for trial {trial_number}")
    return float('-inf')


class TrialJob:
    """Container for trial job information"""
    def __init__(self, trial_number: int, job_id: str, params: Dict[str, Any], submit_time: float):
        self.trial_number = trial_number
        self.job_id = job_id
        self.params = params
        self.submit_time = submit_time
        self.score = None
        self.completed = False


def submit_trial_job(params: Dict[str, Any], study_name: str, trial_number: int,
                    base_slurm_config: Dict[str, str], output_dir: str, project_name: str) -> Optional[TrialJob]:
    """Submit a single trial job and return TrialJob object"""
    
    # Create SLURM script
    script_path = create_slurm_script(params, study_name, trial_number, 
                                    base_slurm_config, output_dir, project_name)
    
    # Submit job
    job_id = submit_slurm_job(script_path)
    if not job_id:
        print(f"Failed to submit trial {trial_number}")
        return None
    
    print(f"Submitted trial {trial_number} as job {job_id}")
    
    return TrialJob(trial_number, job_id, params, time.time())


def check_job_status(job_id: str) -> bool:
    """Check if a SLURM job is still running. Returns True if completed/failed."""
    try:
        result = subprocess.run(['squeue', '-j', job_id, '-h'], capture_output=True, text=True)
        # If job not in queue or squeue fails, job is done
        return result.returncode != 0 or not result.stdout.strip()
    except Exception as e:
        print(f"Error checking job {job_id}: {e}")
        return True  # Assume completed on error


def wait_for_batch_completion(active_jobs: List[TrialJob], output_dir: str, 
                             timeout: int = 43200, check_interval: int = 60) -> List[TrialJob]:
    """Wait for all jobs in batch to complete and collect results"""
    start_time = time.time()
    completed_jobs = []
    
    print(f"Waiting for {len(active_jobs)} jobs to complete...")
    
    while active_jobs and (time.time() - start_time < timeout):
        # Check status of each active job
        still_active = []
        
        for job in active_jobs:
            if check_job_status(job.job_id):
                # Job completed - read result
                job.score = read_trial_result(output_dir, job.trial_number)
                job.completed = True
                completed_jobs.append(job)
                print(f"Trial {job.trial_number} completed with score: {job.score}")
            else:
                # Job still running
                still_active.append(job)
        
        active_jobs = still_active
        
        if active_jobs:
            print(f"{len(active_jobs)} jobs still running...")
            time.sleep(check_interval)
    
    # Handle timeouts
    for job in active_jobs:
        job.score = float('-inf')
        job.completed = True
        completed_jobs.append(job)
        print(f"Trial {job.trial_number} (job {job.job_id}) timed out")
    
    return completed_jobs


def run_batch_trials(trial_configs: List[Dict], study_name: str, 
                    base_slurm_config: Dict[str, str], output_dir: str) -> List[TrialJob]:
    """Submit and execute a batch of trials in parallel"""
    
    active_jobs = []
    
    # Submit all jobs in batch
    for trial_number, params in trial_configs:
        job = submit_trial_job(params, study_name, trial_number, 
                              base_slurm_config, output_dir)
        if job:
            active_jobs.append(job)
    
    # Wait for all to complete
    completed_jobs = wait_for_batch_completion(active_jobs, output_dir)
    
    return completed_jobs


class BatchOptunaTuner:
    """Batch-based Optuna tuner for parallel SLURM job execution"""
    
    def __init__(self, config_path: str, study_name: str, base_slurm_config: Dict[str, str], 
                 output_dir: str, project_name: str, batch_size: int = 10):
        self.param_configs = parse_yaml_config(config_path)
        self.study_name = study_name
        self.base_slurm_config = base_slurm_config
        self.output_dir = output_dir
        self.project_name = project_name
        self.batch_size = batch_size
        self.pending_trials = []  # Trials ready to submit (unused in new approach)
        self.trial_results = {}   # trial_number -> score mapping
        self.active_jobs = {}     # trial_number -> TrialJob mapping
        self.study = None         # Will be set later
    
    def suggest_parameters(self, trial):
        """Generate parameters for a single trial"""
        params = {}
        for param_name, param_config in self.param_configs.items():
            params[param_name] = suggest_parameter(trial, param_name, param_config)
        return params
    
    def objective(self, trial):
        """Optuna objective function - submits trial and returns placeholder"""
        
        # Generate parameters for this trial
        params = self.suggest_parameters(trial)
        
        # Submit this trial immediately (non-blocking)
        job = submit_trial_job(params, self.study_name, trial.number,
                              self.base_slurm_config, self.output_dir, self.project_name)
        
        if job:
            print(f"✓ Submitted trial {trial.number} as job {job.job_id}")
            # Store job for tracking
            self.active_jobs[trial.number] = job
            # Return a placeholder score - we'll update this asynchronously
            return 0.0  # Placeholder
        else:
            print(f"✗ Failed to submit trial {trial.number}")
            return float('-inf')
    
    def wait_for_trial_completion(self, trial_number: int, timeout: int = 21600) -> float:
        """Wait for a specific trial to complete and return its score"""
        job = self.active_jobs.get(trial_number)
        if not job:
            print(f"No active job found for trial {trial_number}")
            return float('-inf')
        
        start_time = time.time()
        check_interval = 60  # Check every minute
        
        print(f"Waiting for trial {trial_number} (job {job.job_id}) to complete...")
        
        while time.time() - start_time < timeout:
            if check_job_status(job.job_id):
                # Job completed - read result
                score = read_trial_result(self.output_dir, trial_number)
                job.score = score
                job.completed = True
                
                # Save to CSV
                save_trial_to_csv(trial_number, job.params, score, self.output_dir)
                
                # Clean up tracking
                del self.active_jobs[trial_number]
                
                print(f"Trial {trial_number} completed with score: {score}")
                return score
            else:
                # Job still running
                print(f"Trial {trial_number} still running... (elapsed: {time.time() - start_time:.0f}s)")
                time.sleep(check_interval)
        
        # Timeout
        print(f"Trial {trial_number} (job {job.job_id}) timed out after {timeout}s")
        if trial_number in self.active_jobs:
            del self.active_jobs[trial_number]
        return float('-inf')
    
    def start_background_monitor(self):
        """Start background thread to monitor job completion"""
        import threading
        def monitor_jobs():
            while True:
                completed_jobs = []
                for trial_number, job in list(self.active_jobs.items()):
                    if check_job_status(job.job_id):  # Job completed
                        score = read_trial_result(self.output_dir, trial_number)
                        job.score = score
                        job.completed = True
                        
                        # Save results
                        save_trial_to_csv(trial_number, job.params, score, self.output_dir)
                        print(f"✓ Trial {trial_number} completed with score: {score}")
                        
                        completed_jobs.append(trial_number)
                
                # Clean up completed jobs
                for trial_number in completed_jobs:
                    del self.active_jobs[trial_number]
                
                # Exit if no more jobs
                if not self.active_jobs:
                    print("All jobs completed - monitor thread exiting")
                    break
                    
                time.sleep(60)  # Check every minute
        
        monitor_thread = threading.Thread(target=monitor_jobs, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def wait_for_all_jobs(self, timeout: int = 7200):
        """Wait for all active jobs to complete (2 hour timeout)"""
        if not self.active_jobs:
            return
        
        print(f"Waiting for {len(self.active_jobs)} remaining jobs to complete...")
        start_time = time.time()
        
        while self.active_jobs and (time.time() - start_time < timeout):
            completed_jobs = []
            for trial_number, job in list(self.active_jobs.items()):
                if check_job_status(job.job_id):
                    score = read_trial_result(self.output_dir, trial_number)
                    save_trial_to_csv(trial_number, job.params, score, self.output_dir)
                    print(f"✓ Final cleanup: Trial {trial_number} completed with score: {score}")
                    completed_jobs.append(trial_number)
            
            for trial_number in completed_jobs:
                del self.active_jobs[trial_number]
            
            if self.active_jobs:
                time.sleep(30)  # Check every 30 seconds for cleanup
        
        if self.active_jobs:
            print(f"⚠️ Timeout: {len(self.active_jobs)} jobs still running after {timeout}s")
    
    def process_batch(self):
        """Submit and process a batch of trials"""
        if not self.pending_trials:
            return
        
        print(f"Processing batch of {len(self.pending_trials)} trials...")
        
        # Submit batch
        completed_jobs = run_batch_trials(
            self.pending_trials, 
            self.study_name, 
            self.base_slurm_config, 
            self.output_dir
        )
        
        # Store results
        for job in completed_jobs:
            self.trial_results[job.trial_number] = job.score
            
            # Save to CSV
            save_trial_to_csv(job.trial_number, job.params, job.score, self.output_dir)
            
            print(f"Trial {job.trial_number}: Score = {job.score}")
        
        # Clear pending trials
        self.pending_trials = []
    
    def get_trial_score(self, trial_number: int) -> float:
        """Get score for a specific trial, processing batch if needed"""
        
        # If score already available, return it
        if trial_number in self.trial_results:
            return self.trial_results[trial_number]
        
        # Process any remaining batch to get the score
        self.process_batch()
        
        # Return score (should be available now)
        return self.trial_results.get(trial_number, float('-inf'))
    
    def finalize(self):
        """Process any remaining trials in batch"""
        if self.pending_trials:
            self.process_batch()


def save_trial_to_csv(trial_number: int, params: Dict[str, Any], score: float, output_dir: str):
    """Save trial results to CSV file"""
    csv_path = f"{output_dir}/trials_results.csv"
    
    # Prepare row data
    row_data = {
        'trial_number': trial_number,
        'score': score,
        **params
    }
    
    # Create DataFrame
    df = pd.DataFrame([row_data])
    
    # Append to CSV (create if doesn't exist)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)


def generate_report(study: optuna.Study, output_dir: str):
    """Generate comprehensive report with CSV and visualizations"""
    
    # CSV summary
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'trial_number': trial.number,
            'score': trial.value if trial.value is not None else float('-inf'),
            'state': trial.state.name,
            **trial.params
        }
        trials_data.append(trial_data)
    
    df = pd.DataFrame(trials_data)
    df.to_csv(f"{output_dir}/study_summary.csv", index=False)
    
    # Best results summary
    best_summary = {
        'best_trial': study.best_trial.number,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
    
    with open(f"{output_dir}/best_results.json", 'w') as f:
        json.dump(best_summary, f, indent=2)
    
    # Generate Optuna visualizations
    try:
        import optuna.visualization as vis
        import plotly.graph_objects as go
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(f"{output_dir}/optimization_history.html")
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(f"{output_dir}/param_importances.html")
        
        # Parameter relationships
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(f"{output_dir}/parallel_coordinate.html")
        
        print(f"Visualizations saved to {output_dir}/")
        
    except ImportError:
        print("Plotly not available - skipping visualizations")


def run_parallel_optimization(study: optuna.Study, batch_tuner, n_trials: int):
    """Run optimization with controlled batch sizes"""
    
    batch_size = batch_tuner.batch_size
    trials_completed = 0
    
    print(f"Starting batch optimization: {n_trials} trials in batches of {batch_size}")
    
    while trials_completed < n_trials:
        # Calculate batch size for this round
        remaining_trials = n_trials - trials_completed
        current_batch_size = min(batch_size, remaining_trials)
        
        print(f"\n=== Batch {trials_completed//batch_size + 1}: Submitting {current_batch_size} trials ===")
        
        # Submit batch of jobs
        submitted_jobs = {}
        for i in range(current_batch_size):
            # Create trial and get parameters
            trial = study.ask()
            params = batch_tuner.suggest_parameters(trial)
            
            # Submit job
            job = submit_trial_job(params, batch_tuner.study_name, trial.number,
                                  batch_tuner.base_slurm_config, batch_tuner.output_dir, 
                                  batch_tuner.project_name)
            
            if job:
                print(f"✓ Submitted trial {trial.number} as job {job.job_id}")
                submitted_jobs[trial.number] = {'trial': trial, 'job': job, 'params': params}
                batch_tuner.active_jobs[trial.number] = job
            else:
                print(f"✗ Failed to submit trial {trial.number}")
                # Tell Optuna this trial failed
                study.tell(trial, float('-inf'))
        
        # Wait for this batch to complete
        print(f"Waiting for batch of {len(submitted_jobs)} jobs to complete...")
        completed_count = 0
        
        while submitted_jobs:
            time.sleep(60)  # Check every minute
            
            completed_trials = []
            for trial_number, job_info in list(submitted_jobs.items()):
                job = job_info['job']
                
                if check_job_status(job.job_id):  # Job completed
                    # Read result
                    score = read_trial_result(batch_tuner.output_dir, trial_number)
                    
                    # Update Optuna study with actual result
                    study.tell(job_info['trial'], score)
                    
                    # Save to CSV
                    save_trial_to_csv(trial_number, job_info['params'], score, batch_tuner.output_dir)
                    
                    completed_count += 1
                    trials_completed += 1
                    print(f"✓ Trial {trial_number} completed: score = {score} ({trials_completed}/{n_trials} total)")
                    
                    completed_trials.append(trial_number)
            
            # Remove completed trials
            for trial_number in completed_trials:
                del submitted_jobs[trial_number]
                if trial_number in batch_tuner.active_jobs:
                    del batch_tuner.active_jobs[trial_number]
            
            # Progress update
            if submitted_jobs:
                print(f"Batch progress: {completed_count}/{current_batch_size} completed, {len(submitted_jobs)} still running...")
        
        print(f"✓ Batch completed! Total progress: {trials_completed}/{n_trials}")
    
    print(f"✅ All {n_trials} trials completed!")


def main():
    parser = argparse.ArgumentParser(description='SLURM-based Optuna hyperparameter tuning for T-JEPA')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--project_name', required=True, help='Project name for organization')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of parallel jobs per batch')
    parser.add_argument('--partition', default='v100', help='SLURM partition')
    parser.add_argument('--gpus', default='1', help='Number of GPUs per job')
    parser.add_argument('--cpus_per_task', default='8', help='CPUs per task')
    parser.add_argument('--mem', default='100G', help='Memory per job')
    parser.add_argument('--time', default='6:00:00', help='Time limit per job')
    parser.add_argument('--env_setup', default='', help='Additional environment setup commands')
    parser.add_argument('--sampler', default='TPE', choices=['TPE', 'CmaEs', 'Random'], 
                       help='Optuna sampler algorithm')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = f"optuna_results_{args.project_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # SLURM configuration
    base_slurm_config = {
        'partition': args.partition,
        'gpus': args.gpus,
        'cpus_per_task': args.cpus_per_task,
        'mem': args.mem,
        'time': args.time,
        'env_setup': args.env_setup
    }
    
    # Use SQLite storage
    storage_path = f"sqlite:///{output_dir}/optuna_study.db"
    
    # Set up sampler
    if args.sampler == 'TPE':
        sampler = optuna.samplers.TPESampler()
    elif args.sampler == 'CmaEs':
        sampler = optuna.samplers.CmaEsSampler()
    else:
        sampler = optuna.samplers.RandomSampler()
    
    # Create or load study
    study_name = f"{args.project_name}_study"
    if args.resume and os.path.exists(f"{output_dir}/optuna_study.db"):
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_path
        )
        print(f"Resumed study '{study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',  # Maximize validation accuracy
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner()
        )
        print(f"Created new study '{study_name}'")
    
    # Create batch tuner
    batch_tuner = BatchOptunaTuner(
        args.config, 
        study_name, 
        base_slurm_config, 
        output_dir, 
        args.project_name,
        args.batch_size
    )
    
    # Run optimization with parallel execution
    print(f"Starting parallel optimization with {args.n_trials} trials...")
    print(f"Results will be saved to {output_dir}/")
    
    try:
        # Custom parallel optimization loop
        run_parallel_optimization(study, batch_tuner, args.n_trials)
    finally:
        # Wait for any remaining jobs
        batch_tuner.wait_for_all_jobs()
    
    # Generate final report
    generate_report(study, output_dir)
    
    # Print summary
    print("\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nResults saved to {output_dir}/")
    print(f"- study_summary.csv: All trials data")
    print(f"- best_results.json: Best configuration")
    print(f"- *.html: Optuna visualizations")


if __name__ == "__main__":
    main()