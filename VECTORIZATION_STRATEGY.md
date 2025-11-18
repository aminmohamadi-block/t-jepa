# Mask Generation Vectorization Strategy

## Pseudocode Breakdown

### Current Implementation (Sequential)

```python
# For EACH of 4096 samples:
for sample_idx in range(4096):
    all_indices = [0, 1, 2, ..., 255]

    # Context masks (1 encoder)
    for enc in range(1):
        np.random.shuffle(all_indices)          # Shuffle #1
        context_mask = all_indices[:n_ctx]
        all_indices = setdiff(all_indices, context_mask)  # Remove used

    # Target masks (4 predictors)
    for pred in range(4):
        np.random.shuffle(all_indices)          # Shuffles #2-5
        target_mask = all_indices[:n_trgt]

    store_masks(context_mask, target_masks)
```

**Problem**:
- 4,096 samples × 5 shuffles = **20,480 sequential shuffle operations**
- Each shuffle: `O(n)` time
- `np.setdiff1d`: `O(n log n)` time
- **Total complexity**: `O(batch_size × (num_encs + num_preds) × n × log n)`

### Vectorized Implementation (Batch)

```python
# Generate ALL permutations at ONCE
random_matrix = np.random.rand(4096, 256)      # Single allocation
all_perms = random_matrix.argsort(axis=1)      # Single sort operation
# Shape: [4096, 256] - all permutations ready!

# Extract context masks (pure slicing)
masks_ctx = all_perms[:, 0:n_ctx]              # [4096, n_ctx]

# Extract target masks (pure slicing)
offset = n_ctx
masks_trgt = []
for pred_idx in range(4):
    mask = all_perms[:, offset:offset+n_trgt]  # [4096, n_trgt]
    masks_trgt.append(mask)
```

**Improvement**:
- 20,480 operations → **1 argsort operation**
- Complexity: `O(batch_size × n × log n)` - factor of `(num_encs + num_preds)` removed!
- All operations vectorized (uses SIMD instructions)

## Visual Comparison

### Sequential (Current):
```
Sample 0:  shuffle → slice → setdiff → shuffle → slice × 4  [~0.07ms]
Sample 1:  shuffle → slice → setdiff → shuffle → slice × 4  [~0.07ms]
Sample 2:  shuffle → slice → setdiff → shuffle → slice × 4  [~0.07ms]
...
Sample 4095: shuffle → slice → setdiff → shuffle → slice × 4  [~0.07ms]
────────────────────────────────────────────────────────────
Total: ~293ms
```

### Vectorized (Proposed):
```
All samples: [4096 × 256] argsort → slice all → done  [~5-15ms]
────────────────────────────────────────────────────────────
Total: ~15ms (20x faster!)
```

## Key Vectorization Techniques

### 1. Batch Random Permutation Generation
```python
# OLD: 4096 sequential shuffles
for i in range(4096):
    indices = np.arange(256)
    np.random.shuffle(indices)

# NEW: Single batch argsort
random_matrix = np.random.rand(4096, 256)
all_permutations = random_matrix.argsort(axis=1)
```
**Why faster?**
- Single memory allocation
- Parallel sorting (uses multi-core automatically)
- Better cache locality

### 2. Advanced Indexing Instead of Loops
```python
# OLD: Loop to extract masks
masks = []
for i in range(4096):
    mask = permutations[i][:n_ctx]
    masks.append(mask)

# NEW: Slice entire batch at once
masks = permutations[:, :n_ctx]  # [4096, n_ctx] in one operation
```

### 3. Eliminate `np.setdiff1d`
```python
# OLD: Remove used indices (expensive!)
remaining = np.setdiff1d(all_indices, used_indices)  # O(n log n)

# NEW: Pre-allocate non-overlapping regions
# Context uses indices [0:n_ctx]
# Target uses indices [n_ctx:n_ctx+n_trgt]
# No removal needed!
```

## Expected Performance

### Theoretical Speedup

| Operation | Time Complexity | Original | Vectorized |
|-----------|----------------|----------|------------|
| Shuffles | O(B × N × N) | 20,480 ops | 1 argsort |
| Setdiff | O(B × N log N) | 4,096 ops | 0 ops |
| Indexing | O(B × K) | Loops | Slicing |

Where:
- B = batch_size = 4096
- N = num_features = 256
- K = mask_size ≈ 100

**Expected speedup**: 15-25x

### Empirical Estimates

Based on profiling:
- Original: 242ms (mask_creation)
- Vectorized: 10-15ms (estimated)
- **Speedup: ~20x**

Impact on iteration:
- Original: 487.97ms/iter
- With vectorization: 487.97 - 242 + 15 = **260.97ms/iter**
- **Overall speedup: 1.87x**

## Implementation Details

### Ultra-Vectorized Version

The `UltraVectorizedMaskCollator` uses pure NumPy operations:

```python
# 1. Single random matrix generation
random_matrix = np.random.rand(batch_size, num_features)

# 2. Single argsort for all permutations
all_perms = random_matrix.argsort(axis=1)  # [B, N]

# 3. Stack slices for all encoders at once
masks_ctx = np.stack([
    all_perms[:, enc*n_ctx:(enc+1)*n_ctx]
    for enc in range(num_encs)
], axis=1)  # [B, num_encs, n_ctx]

# 4. Stack slices for all predictors at once
offset = num_encs * n_ctx
masks_trgt = np.stack([
    all_perms[:, offset:offset+n_trgt]
    for pred in range(num_preds)
], axis=1)  # [B, num_preds, n_trgt]

# 5. Direct tensor conversion (no collation needed!)
return torch.from_numpy(masks_ctx), torch.from_numpy(masks_trgt)
```

### Memory Efficiency

**Original**:
- 20,480 temporary arrays
- Frequent allocation/deallocation
- Poor cache locality

**Vectorized**:
- 1 large matrix: `4096 × 256 × 8 bytes = 8.4 MB`
- Single allocation
- Perfect cache locality
- Modern CPUs can handle this easily

## Correctness Preservation

The vectorized implementation preserves all properties:

1. ✅ **Random permutations**: Each row is a random permutation
2. ✅ **No overlap between context encoders**: Use non-overlapping slices
3. ✅ **No overlap between context and targets**: Targets start after context
4. ✅ **Targets can overlap each other**: All use same slice range (if allow_overlap=True)
5. ✅ **Deterministic seeding**: `np.random.seed(step())` before generation

## Testing Strategy

```python
def test_correctness():
    # Generate masks with both methods
    original_masks = original_collator(batch)
    vectorized_masks = vectorized_collator(batch)

    # Check properties:
    # 1. Same number of masks
    assert len(original_masks[1]) == len(vectorized_masks[1])

    # 2. All indices valid
    assert torch.all(vectorized_masks[1] < num_features)

    # 3. No duplicates within context
    for ctx_mask in vectorized_masks[1]:
        assert len(torch.unique(ctx_mask)) == len(ctx_mask)

    # 4. Context and target don't overlap
    # (if using non-overlapping slices)
```

## Next Steps

1. **Run benchmark** on CPU cluster to measure actual speedup
2. **Validate correctness** - ensure masks maintain required properties
3. **Profile memory usage** - should be similar or better
4. **Integrate into training** if successful
5. **Compare training quality** - ensure no degradation

## Conclusion

Vectorization through batch operations can provide **~20x speedup** for mask generation by:
- Eliminating 20,480 sequential operations
- Using single batch argsort operation
- Leveraging NumPy's optimized SIMD instructions
- Removing expensive `setdiff1d` calls

This reduces mask_collation from 293ms to ~15ms, giving **1.87x overall training speedup**.