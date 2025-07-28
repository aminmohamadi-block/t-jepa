#!/bin/bash
#SBATCH --job-name=test_tjepa_jannis_default_1a100_resume
#SBATCH --partition=a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00 

# Assuming that this script is run from the root of the project

source ../bin/activate-hermit
source .venv/bin/activate

# Parse command line arguments for tag
TAG_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG_ARG="--tag $2"
            shift
            shift
            ;;
        *)
            # Unknown option - ignore and continue
            shift
            ;;
    esac
done

# ./scripts/launch_tjepa.sh --data_path ./datasets --data_set jannis --load_from_checkpoint=True --load_path=./checkpoints/jannis/jannis__model_nlyrs_16_nheads_2_hdim_64__pred_ovrlap_F_npreds_4__nlyrs_16_activ_relunenc_1_inter_ctx_0.13628349247854785_0.36819012681604135_inter_trgt_0.1556321308543076_0.6222278244105446__lr_0.0003658682841082736_start_0.0_final_0.0_20250717_184031/epoch_120.pth
./scripts/launch_tjepa.sh --data_path ./datasets --data_set jannis $TAG_ARG
