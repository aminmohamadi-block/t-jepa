#!/bin/bash
#SBATCH --job-name=check_pytorch
#SBATCH --output=logs/check_pytorch_%j.out
#SBATCH --error=logs/check_pytorch_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=8G

echo "=========================================="
echo "PyTorch Environment Check"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Load hermit environment if available
if [ -f ../bin/activate-hermit ]; then
    source ../bin/activate-hermit
fi

python3 << 'PYEOF'
import torch
import sys

print("=" * 60)
print("PYTORCH VERSION INFO")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print("")

print("=" * 60)
print("MIXED PRECISION (AMP) SUPPORT")
print("=" * 60)
from torch.cuda.amp import GradScaler, autocast
print("✅ torch.cuda.amp.GradScaler available")
print("✅ torch.cuda.amp.autocast available")
print("")

print("=" * 60)
print("FLASH ATTENTION (SDPA) SUPPORT")
print("=" * 60)
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    print("✅ F.scaled_dot_product_attention available")
    print(f"   PyTorch >= 2.0 detected")

    # Check if we're using the fast path
    import torch.nn as nn
    if hasattr(nn, 'MultiheadAttention'):
        mha = nn.MultiheadAttention(64, 4, batch_first=True, device='cuda')
        print(f"   nn.MultiheadAttention uses backend: {mha}")
else:
    print("❌ F.scaled_dot_product_attention NOT available")
    print("   PyTorch < 2.0 - consider upgrading")
print("")

print("=" * 60)
print("TORCH.COMPILE SUPPORT")
print("=" * 60)
if hasattr(torch, 'compile'):
    print("✅ torch.compile available")
    print(f"   PyTorch >= 2.0 detected")
    print(f"   Can use torch.compile for additional speedups")

    # Test basic compilation
    try:
        @torch.compile
        def test_fn(x):
            return x * 2
        test_tensor = torch.tensor([1.0], device='cuda')
        result = test_fn(test_tensor)
        print(f"   ✅ torch.compile test: SUCCESS")
    except Exception as e:
        print(f"   ⚠️  torch.compile test FAILED: {e}")
else:
    print("❌ torch.compile NOT available")
    print("   PyTorch < 2.0 - consider upgrading for torch.compile support")
print("")

print("=" * 60)
print("GPU CAPABILITIES")
print("=" * 60)
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_props.name}")
    print(f"Compute capability: {device_props.major}.{device_props.minor}")
    print(f"Total memory: {device_props.total_memory / 1024**3:.2f} GB")
    print(f"Multi-processor count: {device_props.multi_processor_count}")

    # Check tensor core support
    if device_props.major >= 7:  # Volta and newer
        print(f"✅ Tensor Cores available")
        print(f"   FP16 operations will be significantly faster")
    else:
        print(f"⚠️  No Tensor Cores (compute < 7.0)")
print("")

print("=" * 60)
print("RECOMMENDATIONS FOR PHASE 3")
print("=" * 60)
recommendations = []

# AMP recommendation
if torch.cuda.is_available() and device_props.major >= 7:
    recommendations.append("✅ Enable AMP (--model_amp=True) - Expected 1.5-2x speedup")
else:
    recommendations.append("⚠️  AMP may have limited benefit (no Tensor Cores)")

# Flash Attention recommendation
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    recommendations.append("✅ Flash Attention already available in MultiheadAttention")
else:
    recommendations.append("📦 Upgrade PyTorch to 2.0+ for Flash Attention")

# torch.compile recommendation
if hasattr(torch, 'compile'):
    recommendations.append("✅ Consider torch.compile for additional 10-30% speedup")
else:
    recommendations.append("📦 Upgrade PyTorch to 2.0+ for torch.compile")

for rec in recommendations:
    print(rec)

print("")
print("=" * 60)
print("Check complete!")
print("=" * 60)
PYEOF

echo ""
echo "Script completed at $(date)"
