#!/bin/bash

# Easy wrapper for testing linear probe optimization
# Usage: ./run_probe_test.sh <checkpoint_path> [options]

if [ $# -eq 0 ]; then
    echo "Usage: ./run_probe_test.sh <checkpoint_path> [options]"
    echo ""
    echo "Examples:"
    echo "  # Basic test with default hyperparameters"
    echo "  ./run_probe_test.sh ./checkpoints/higgs/model.pth"
    echo ""
    echo "  # Test with MLflow logging"
    echo "  ./run_probe_test.sh ./checkpoints/higgs/model.pth --use_mlflow --run_name higgs_probe_test_v1"
    echo ""
    echo "  # Test with custom hyperparameters"  
    echo "  ./run_probe_test.sh ./checkpoints/higgs/model.pth --learning_rate 0.1 --max_epochs 200"
    echo ""
    echo "Available options:"
    echo "  --learning_rate FLOAT     Learning rate (default: 0.01)"
    echo "  --weight_decay FLOAT      Weight decay (default: 0.001)" 
    echo "  --max_epochs INT          Max epochs (default: 100)"
    echo "  --batch_size INT          Batch size (default: 128)"
    echo "  --patience INT            Early stopping patience (default: 20)"
    echo "  --use_mlflow             Enable MLflow logging"
    echo "  --run_name STR           MLflow run name"
    echo "  --dataset_name STR       Dataset name (default: higgs)"
    echo "  --data_path STR          Data path (default: ./datasets)"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift

echo "Testing linear probe optimization..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Additional args: $*"
echo ""

python src/benchmark/linear_probe_eval/test_linear_probe.py --checkpoint_path "$CHECKPOINT_PATH" "$@"