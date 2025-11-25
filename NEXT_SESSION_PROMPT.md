# Next Session Prompt: Phase 6 Optimizations

Use this prompt to continue the optimization work in the next Claude Code session.

---

## Prompt to Use

```
I'm continuing optimization work on the T-JEPA codebase. Please read the context files first, then implement the remaining optimizations.

## Context Files to Read (in order)

1. **CLAUDE.md** - Project overview and architecture understanding
2. **DEEP_BOTTLENECK_ANALYSIS.md** - Comprehensive profiling analysis with all findings
3. **src/predictors.py** - Contains MLP predictor that needs vectorization (lines 120-147)
4. **src/encoder.py** - Contains categorical encoding that may be dead code (lines 311-323)

## Current State

- Branch: `code-optimization/runtime-perf`
- Latest commit: `1573ef6 Add Phase 5 optimizations: EMA _foreach, GPU grad stats, expand optimization`
- Previous phases implemented:
  - Phase 1-3: Forward pass optimizations, MLflow fix, mixed precision
  - Phase 4: torch.compile investigation (limited benefit due to dynamic masking)
  - Phase 5: EMA _foreach (21.4x), GPU grad stats (4.4x), expand optimization (5.0x)

## Remaining Tasks

### Task 1: MLP Predictor Vectorization (HIGH PRIORITY)
**Location**: `src/predictors.py:120-147`

**Current Problem**:
```python
def forward_mlp(self, x, mask_pred):
    out = []
    for mask in mask_pred:  # 4 iterations
        out_mask = []
        for col_idx in range(self.num_features):  # 256 iterations!
            out_batch = self.forward_predictor_k(x, col_idx)
            out_mask.append(out_batch)
        # ... processing
```

This executes 256 × 4 = 1024 sequential MLP forward passes. This is a major bottleneck when using the MLP predictor type (`--pred_type mlp`).

**Goal**: Vectorize to batch all feature predictions in parallel instead of sequential loops.

**Constraints**:
- Each feature has its own MLP (self.predictor_k is a ModuleList of MLPs)
- Must maintain same output semantics
- Consider using grouped convolutions or batched linear layers
- May need to restructure the MLP architecture slightly

### Task 2: Categorical Encoding Fix (MEDIUM PRIORITY)
**Location**: `src/encoder.py:311-323`

**Current Problem**:
```python
if x_cat is not None:
    x_cat = x_cat.detach().cpu().numpy()  # CPU transfer!
    categories = [list(range(card[1])) for card in self.cardinalities]
    ohe = OneHotEncoder(sparse_output=False, categories=categories).fit(x_cat)
    x_cat = torch.tensor(ohe.transform(x_cat), device=x_num.device)  # Back to GPU!
```

**Issues**:
1. CPU-GPU round trip on every forward pass
2. OneHotEncoder created and fitted on every forward pass
3. This code path appears to be dead when categorical features use embeddings (which is the default)

**Goal**:
- Verify if this code path is ever executed (check if x_cat is ever not None when cardinalities exist)
- If dead code: Remove it or add a warning
- If used: Pre-fit encoder in __init__, use torch-native operations instead of sklearn

## How to Proceed

1. Read the context files listed above
2. Understand the current MLP predictor architecture in `src/predictors.py`
3. Design a vectorized approach for Task 1
4. Implement and test Task 1
5. Investigate and fix Task 2
6. Run verification benchmarks using `scripts/verify_optimizations.py` pattern
7. Submit SLURM jobs to verify correctness with actual training
8. Commit changes with detailed message

## Verification

After implementation, run training to verify correctness:
```bash
sbatch scripts/compare_phase5_optimizations.sh
```

Or create a new test script for Phase 6 specifically.

## Important Notes

- The MLP predictor is used when `--pred_type mlp` is set (default is `transformer`)
- The categorical encoding issue only affects datasets with categorical features
- Maintain backward compatibility - don't break existing functionality
- Use profiling to verify speedups: `--profiling_level DETAILED`

Ultrathink and get this done. Take your time to deeply analyze the MLP predictor architecture before implementing the vectorization. Feel free to submit SLURM jobs and use sleep commands to wait for results.
```

---

## Quick Reference

### Key Files
| File | Purpose |
|------|---------|
| `src/predictors.py` | MLP and Transformer predictor implementations |
| `src/encoder.py` | Encoder with categorical encoding |
| `DEEP_BOTTLENECK_ANALYSIS.md` | Full profiling analysis |
| `scripts/micro_benchmark.py` | Benchmark template |
| `scripts/verify_optimizations.py` | Verification template |

### Git Commands
```bash
# Check current state
git log --oneline -5
git status

# After implementation
git add <files>
git commit -m "Add Phase 6: MLP vectorization, categorical encoding fix"
```

### SLURM Job Template
```bash
#!/bin/bash
#SBATCH --job-name=phase6_test
#SBATCH --output=logs/phase6_%j.out
#SBATCH --error=logs/phase6_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=01:00:00

source ../bin/activate-hermit
python run.py --data_set jannis --pred_type mlp --exp_train_total_epochs 2 --profiling_level DETAILED
```
