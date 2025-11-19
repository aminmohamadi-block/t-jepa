#!/bin/bash
#SBATCH --job-name=p3_pq_amp
#SBATCH --output=logs/phase3_pq_amp_%j.out
#SBATCH --error=logs/phase3_pq_amp_%j.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

echo "========================================"
echo "Phase 3 PARQUET AMP ENABLED batch_size=4096 (h100)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

cd /home/aminmohamadi_squareup_com/projects/t-jepa-code-optimization
source ../bin/activate-hermit

timeout 300 python run.py \
    --use_parquet_dataset=True \
    --parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train \
    --parquet_data_files=chunk_10.parquet \
    --parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train/stats_bootstrap.pq \
    --parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv \
    --parquet_num_features=128 \
    --parquet_categorize_nan=True \
    --parquet_scaling_method=IQR \
    --parquet_infill_value=zero \
    --parquet_transform=asinh \
    --project_name phase3_pq_amp \
    --exp_train_total_epochs 1 \
    --batch_size 4096 \
    --exp_patience 999 \
    --exp_cache_cadence 999 \
    --probe_cadence 0 \
    --model_dim_hidden 64 \
    --model_num_layers 4 \
    --model_num_heads 4 \
    --pred_num_layers 2 \
    --use_vectorized_masking=True \
    --model_amp=True \
    --profiling_level=DETAILED

exit_code=$?
echo ""
echo "Exit code: $exit_code (0 or 124 = success)"
echo "Completed at: $(date)"
exit $exit_code
