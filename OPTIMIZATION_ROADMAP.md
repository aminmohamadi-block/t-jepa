# T-JEPA Performance Optimization Roadmap

## Current Status: Phase 1 & 2 Complete ✅

**Branch**: `code-optimization/runtime-perf`
**Baseline**: 516.11ms/iteration (h100, parquet, batch_size=4096)
**After Phase 1+2**: 511.44ms/iteration
**Combined Speedup**: 1.009x (+0.91%)

---

## Completed Optimizations

### ✅ Mask Collation Optimization
**Impact**: 16.9x speedup (293ms → 17ms)
**Technique**: Vectorized batch operations with NumPy

### ✅ Phase 1: Forward Pass Code Optimizations
**Impact**: +0.91% overall (h100, parquet, batch_size=4096)
**Files**: src/encoder.py, src/predictors.py
**Details**: See `PHASE1_COMPLETE_SUMMARY.md`

**Implementations**:
1. ✅ Cached CLS/REG token templates (register_buffer + .expand())
2. ✅ Cached bias zero padding (register_buffer)
3. ✅ Cached feature index embedding zeros (register_buffer)
4. ✅ Eliminated redundant positional embedding in predictor

### ✅ Phase 2: apply_masks_from_idx Optimization
**Impact**: +0.47% (h100, parquet, batch_size=4096)
**File**: src/utils/train_utils.py
**Details**: See `PHASE2_RESULTS.md`

**Implementation**:
- Replaced gather-based indexing with advanced indexing
- Eliminated temporary tensor creation
- Benefits scale with batch size (use batch_size >= 4096)

### ✅ MLflow Hang Fix
**Impact**: Jobs complete in ~30-40s instead of hanging for 30 minutes
**File**: src/train.py:673-677
**Issue**: Profiling MLflow logging was outside run_context, creating orphaned run
**Fix**: Moved profiling save/log inside `with run_context:` block

---

## Remaining Phases

### Phase 3: Mixed Precision
**Status**: ⏳ NOT STARTED (Highest Priority)
**Expected Impact**: 1.5-2x overall speedup
**Risk**: Low (PyTorch built-in)
**See**: `PHASE3_PLAN.md`

**Quick Summary**:
- Enable `--model_amp=True`
- Verify numerical stability
- Largest remaining optimization opportunity

---

## Bottleneck Summary (Current State)

From profiling after Phase 1:

| Component | Time (ms) | % | Status | Optimization Opportunity |
|-----------|-----------|---|--------|-------------------------|
| **backward_pass** | 223 | 46.4% | ⏳ | Phase 3a: AMP (1.5-2x) |
| **predictor** | 112 | 23.2% | ⏳ | Phase 3a: AMP (1.3-1.5x) |
| **forward_pass** | 106 | 22.2% | ✅ | Phase 1 complete, Phase 3a: AMP |
| **context_encoder** | 29 | 6.0% | ✅ | Phase 1 complete |
| **mask_collation** | 17 | 3.6% | ✅ | Vectorization complete |
| **Other** | 11 | 2.3% | - | Minor |

**Key Insight**: Biggest remaining opportunity is Mixed Precision (AMP) - affects all components

---

## Projected Final Performance

### After All Phases:
```
Current (Phase 1):       480ms/iteration
After Phase 2:          ~475ms/iteration  (apply_masks optimization)
After Phase 3 (AMP):    ~298-317ms/iteration  (1.5-1.6x speedup)

TOTAL IMPROVEMENT: 1.6x speedup
```

### Training Time Impact (100 epochs):
```
Baseline (original):     17.5 hours  (with slow mask collation)
After mask vectorization: 11.1 hours  (16.9x mask speedup)
After Phase 1:           ~11.0 hours  (small ops optimized)
After Phase 2:           ~10.8 hours  (apply_masks optimized)
After Phase 3 (AMP):     ~6.7-7.1 hours  (mixed precision)

TOTAL TIME SAVED: 10.4-10.8 hours (59-62% faster than original)
```

---

## Implementation Checklist

### Phase 1: ✅ DONE
- [x] Deep code analysis
- [x] Identify redundant allocations
- [x] Implement 4 optimizations
- [x] Test and verify
- [x] Document results

### Phase 2: ⏳ TODO
- [ ] Read `PHASE2_PLAN.md`
- [ ] Implement apply_masks_from_idx optimization
- [ ] Create and run unit tests
- [ ] Benchmark performance
- [ ] Run smoke test on SLURM
- [ ] Profile full epoch
- [ ] Document results in `PHASE2_RESULTS.md`

### Phase 3: ⏳ TODO
- [ ] Read `PHASE3_PLAN.md`
- [ ] Check PyTorch version and Flash Attention availability
- [ ] Enable `--model_amp=True`
- [ ] Run convergence test (5-10 epochs)
- [ ] Verify numerical stability (no NaN/Inf)
- [ ] Profile with AMP enabled
- [ ] Compare performance vs FP32
- [ ] Document results in `PHASE3_RESULTS.md`

---

## Quick Start Guide

### To Continue Optimization Work:

```bash
# Check current status
git status
cat PHASE1_COMPLETE_SUMMARY.md  # Review what's done

# Start Phase 2
cat PHASE2_PLAN.md  # Read the plan
# Then implement following steps in the plan

# OR Start Phase 3 (higher impact)
cat PHASE3_PLAN.md  # Read the plan
# Enable AMP and test
```

### To Review Current Changes:

```bash
# See code modifications
git diff src/encoder.py
git diff src/predictors.py

# See all documentation
ls -1 PHASE*.md FORWARD*.md IMPLEMENTATION*.md
```

### To Test Current State:

```bash
# Quick smoke test
sbatch scripts/test_phase1_fixes.sh

# Full profiling
sbatch scripts/profile_phase1_optimized.sh
```

---

## Key Files Reference

### Documentation:
- **OPTIMIZATION_ROADMAP.md** (this file) - Master plan
- **PHASE1_COMPLETE_SUMMARY.md** - Phase 1 detailed results
- **PHASE2_PLAN.md** - Phase 2 implementation guide
- **PHASE3_PLAN.md** - Phase 3 implementation guide
- **FORWARD_PASS_DEEP_ANALYSIS.md** - Complete code analysis
- **BACKWARD_PASS_ANALYSIS.md** - Backward pass understanding

### Code Changes:
- **src/encoder.py** - Phase 1 optimizations
- **src/predictors.py** - Phase 1 optimizations
- **src/utils/train_utils.py** - Phase 2 target (not yet modified)

### Test Scripts:
- **scripts/test_phase1_fixes.sh** - Phase 1 smoke test
- **scripts/profile_phase1_optimized.sh** - Phase 1 profiling
- **scripts/test_phase2_fixes.sh** - (To be created for Phase 2)
- **scripts/profile_amp.sh** - (To be created for Phase 3)

### Profiling Baseline:
- **profiling_parquet_dataset...195640.json** - Original baseline (batch_size=4096)
- **profiling_jannis...165856.json** - Phase 1 results (batch_size=2048)

---

## Success Criteria

### Phase 2 Success:
- [ ] apply_masks_from_idx: 2x+ speedup measured
- [ ] Unit tests pass
- [ ] Smoke test completes
- [ ] Full epoch profiling shows improvement

### Phase 3 Success:
- [ ] 1.4x+ overall iteration speedup
- [ ] Training converges normally (loss within 5% of FP32)
- [ ] No NaN/Inf values
- [ ] 50+ epoch stability test passes

### Overall Success:
- [ ] 1.5-1.6x total speedup achieved
- [ ] All tests passing
- [ ] Code production-ready
- [ ] Training time: 10.7h → ~6.7-7.1h

---

## Recommendations

### Immediate Priority (Highest Impact):
1. **Phase 3 (AMP)** - Just enable flag, 1.5-2x speedup
2. Phase 2 (apply_masks) - Code optimization, smaller but safe

### Rationale:
- AMP has much higher impact (1.5-2x vs 0.5-1%)
- AMP is lower risk (PyTorch built-in vs custom code)
- AMP is faster to implement (config change vs code refactor)

### Suggested Order:
1. Phase 3 (AMP) - Get biggest win first
2. Phase 2 (apply_masks) - Clean up remaining code inefficiency
3. Re-profile to measure combined effect

---

## Long-term Vision

### Code Optimizations (Phases 1-2):
- Target: Eliminate inefficiencies in existing code
- Approach: Profiling-driven, targeted fixes
- Impact: Incremental (~1-2% total)
- Risk: Low

### System Optimizations (Phase 3):
- Target: Use modern PyTorch features (AMP, Flash Attention)
- Approach: Enable built-in optimizations
- Impact: Major (~1.5-2x)
- Risk: Low

### Future (Beyond Phase 3):
- Architectural changes (reduce layers, predictions)
- Custom CUDA kernels (high effort)
- Model distillation
- Distributed training optimization

---

## Notes for Future Developers

### Best Practices Learned:
1. **Profile first, optimize second**: Always measure before optimizing
2. **Start with easy wins**: AMP > code optimizations for effort/impact ratio
3. **Test thoroughly**: Correctness >> Performance
4. **Document everything**: Future you will thank you
5. **Verify assumptions**: Check if features are actually being used

### Common Pitfalls:
1. **Profiling overhead**: Fine-grained profiling can make jobs 10x slower
2. **Dataset differences**: Hard to compare across datasets
3. **Memory constraints**: Profiling + large batch = OOM
4. **GPU vs CPU**:  Not all operations benefit from GPU

### Tools That Worked:
- PyTorch profiler for operation-level timing
- SLURM for long-running profiling jobs
- Git for safe experimentation
- Thorough code reading before implementing

### Tools That Didn't:
- Fine-grained sub-operation profiling (too much overhead)
- GPU-based mask generation (86x slower than CPU)
- Cross-dataset performance comparison

---

## Contact and Continuity

All work is on branch: `code-optimization/runtime-perf`
All documentation is in project root (PHASE*.md files)
All test scripts are in `scripts/` directory

To resume:
1. Read this file (OPTIMIZATION_ROADMAP.md)
2. Check PHASE1_COMPLETE_SUMMARY.md for what's done
3. Read PHASE2_PLAN.md or PHASE3_PLAN.md for next steps
4. Follow implementation checklists in those files
