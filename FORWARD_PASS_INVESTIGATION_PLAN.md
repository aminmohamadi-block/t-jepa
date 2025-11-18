# Forward Pass Efficiency Investigation Plan

## Executive Summary

After optimizing mask collation (16.9x speedup), backward_pass is now the #1 bottleneck at 223ms (44.8%). However, since backward gradients depend on forward operations, we must ensure the forward pass is optimally implemented. This document outlines our investigation plan to identify and fix forward pass inefficiencies.

---

## Current Performance Snapshot (Vectorized Profiling)

### Iteration-Level Breakdown (480.34ms total)

| Operation | Time (ms) | % of Iteration | Status |
|-----------|-----------|----------------|--------|
| **backward_pass** | 223.14 | 46.4% | ⚠️ #1 Bottleneck |
| **predictor** | 111.52 | 23.2% | 🔍 Investigate |
| **forward_pass** | 106.49 | 22.2% | 🔍 Investigate |
| **context_encoder** | 28.81 | 6.0% | 🔍 Investigate |
| **mask_collation** | 17.47 | 3.6% | ✅ Optimized |
| **Other** | 10.91 | 2.3% | Minor |

**Total Forward-Related Time**: 246.82ms (51.4% of iteration!)
- If we optimize forward by 2x → backward also improves → total speedup ~1.6x

---

## Key Finding: Context Encoder Performance Anomaly

### Per-Layer Timing Analysis

|Encoder| Layers | Total Time | Time/Layer | Relative Speed |
|-------|--------|------------|------------|----------------|
| **Target Encoder** | 4 | 96.48ms | 24.12ms/layer | 1.0x (baseline) |
| **Predictor** | 2 | 90.22ms | 45.11ms/layer | 0.53x (2x slower!) |
| **Context Encoder** | 4 | 21.07ms | 5.27ms/layer | **4.6x faster!** |

### Why is Context Encoder 4.6x Faster Per Layer?

**Hypothesis 1: Reduced Sequence Length (Masking)**
- Context encoder only processes visible features (after masking)
- If masking removes ~75% of features → 4x speedup makes sense
- Attention is O(n²) where n = sequence length

**Hypothesis 2: Different Batch Sizes**
- Context encoder may process fewer samples due to masking strategy
- Need to check actual tensor shapes during forward pass

**Hypothesis 3: Implementation Differences**
- Are there code path differences between encoders?
- Different activation functions, dropout patterns, etc.?

### Why is Predictor 2x Slower Per Layer?

**Hypothesis 1: Mask Token Operations**
- Predictor creates and concatenates learnable mask tokens
- Extra operations: embedding, positional encoding, concatenation
- `post_transformer_processing`: 11.60ms (predictor) vs 3.95ms (target encoder)

**Hypothesis 2: Sequence Length**
- Predictor processes: [context_tokens + mask_tokens]
- Longer sequence → quadratic attention cost

**Hypothesis 3: Memory Bandwidth**
- Predictor shows 33GB GPU memory usage vs 1.6GB for target encoder
- High memory pressure → slower operations

---

## Investigation Plan (8 Tasks)

### ✅ Task 1: Analyze Current Profiling Data [COMPLETED]
**Findings:**
- Forward-related operations take 246.82ms (51.4% of iteration)
- Context encoder is suspiciously fast (5.27ms/layer)
- Predictor is suspiciously slow (45.11ms/layer)
- Need to understand sequence length impact on timing

---

### 🔄 Task 2: Read and Document Encoder Implementations [IN PROGRESS]

**Files to Analyze:**
- `src/encoder.py` - Context/Target encoder implementation
- `src/predictors.py` - Predictor implementation
- `src/tjepa_transformer.py` - Transformer layer details

**What to Document:**
1. **Encoder Flow**:
   ```python
   forward(x, mask):
       1. embedding:
          - feature_separation: split numerical/categorical
          - tokenizer: linear projection + categorical embeddings
          - positional_encoding: add learned positions
          - feature_index_embedding: (optional) per-feature embeddings
          - apply_mask: select visible features only (context encoder)
       2. transformer:
          - transformer_layers: N stacked layers
          - post_transformer_processing: dropout + layernorm + fc + dropout + layernorm
   ```

2. **Transformer Layer Flow**:
   ```python
   TransformerEncoderLayer:
       - MultiheadAttention (Q, K, V projections + scaled dot-product)
       - Dropout + Residual + LayerNorm
       - FFN: Linear(hidden → feedforward) → Activation → Linear(feedforward → hidden)
       - Dropout + Residual + LayerNorm
   ```

3. **Key Observations**:
   - Masking applied BEFORE transformer (reduces sequence length)
   - Position encoding uses learned embeddings (not sinusoidal)
   - `batch_first=True` for all operations
   - Custom TransformerEncoder (not torch.nn.TransformerEncoder)

---

### 📋 Task 3: Document Predictor Implementation [PENDING]

**Investigation Points:**
- How are mask tokens created and prepared?
- What's causing the 11.60ms `post_transformer_processing` time?
- Memory usage: Why 33GB vs 1.6GB for encoders?
- Sequence length calculation: context + mask tokens

---

### 🔍 Task 4: Identify PyTorch Anti-Patterns [PENDING]

**Common Inefficiencies to Check:**

1. **Unnecessary CPU-GPU Transfers**
   ```python
   # BAD: Detach + CPU + NumPy in hot path
   x_cat = x_cat.detach().cpu().numpy()  # encoder.py:269
   ```

2. **Repeated Tensor Creation**
   ```python
   # BAD: Creating zeros every forward pass
   torch.zeros(batch_size, n_tokens, hidden_dim).to(device)  # encoder.py:301, 325
   ```

3. **Non-Fused Operations**
   ```python
   # Could be fused?
   x = self.dropout1(x)
   x = self.layernorm1(x)
   x = self.fc(x)
   ```

4. **Sequential Operations That Could Be Parallel**
   ```python
   # Sequential embedding lookups
   x_cat_embedded = [
       self.category_embeddings[i](x_cat[i])
       for i in range(len(self.categories))
   ]
   ```

5. **Memory Layout Issues**
   - Are tensors contiguous?
   - Optimal strides for GPU access?

---

### ✅ Task 5: Check for Missing Optimizations [PENDING]

**Optimization Checklist:**

#### A. **Attention Optimizations**
- [ ] Flash Attention: O(n) memory vs O(n²)
- [ ] `torch.nn.functional.scaled_dot_product_attention()` (PyTorch 2.0+)
- [ ] Fused attention kernels from `xformers`
- [ ] Sparse attention patterns

#### B. **Kernel Fusion**
- [ ] Fused LayerNorm + Linear
- [ ] Fused Dropout + Add (residual connections)
- [ ] Fused activation functions (GELU, ReLU)
- [ ] Custom CUDA kernels for hot paths

#### C. **Mixed Precision**
- [ ] `torch.autocast()` enabled? Currently `model_amp=False`
- [ ] FP16 forward + backward
- [ ] Tensor Cores utilization (A100 GPU)

#### D. **Memory Optimizations**
- [ ] Gradient checkpointing (trade compute for memory)
- [ ] Activation recomputation
- [ ] In-place operations where safe
- [ ] Pre-allocated buffers for repeated operations

#### E. **Data Layout**
- [ ] Channels-last memory format for better cache locality
- [ ] Contiguous tensors (avoid unnecessary copies)
- [ ] Optimal batch sizes for GPU

---

### 🔬 Task 6: Create Fine-Grained Profiling Script [PENDING]

**Add Detailed Profiling Hooks:**

```python
# In transformer layer, add:
with profiler.profile("attention_qkv_projection"):
    q, k, v = ...

with profiler.profile("attention_scores"):
    scores = torch.matmul(q, k.transpose(-2, -1))

with profiler.profile("attention_softmax"):
    attn_weights = F.softmax(scores, dim=-1)

with profiler.profile("attention_output"):
    out = torch.matmul(attn_weights, v)

# In FFN, add:
with profiler.profile("ffn_linear1"):
    x = self.linear1(x)

with profiler.profile("ffn_activation"):
    x = self.activation(x)

with profiler.profile("ffn_linear2"):
    x = self.linear2(x)
```

**Measure:**
- Individual attention sub-operations
- FFN layer times
- LayerNorm times
- Dropout times (should be negligible)
- Memory allocations per operation

---

### 🚀 Task 7: Run Fine-Grained Profiling on SLURM [PENDING]

**Script**: `scripts/profile_forward_detailed.sh`

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

python run.py \
  --use_parquet_dataset=True \
  --parquet_data_files=chunk_10.parquet,...,chunk_14.parquet \
  --parquet_num_features=128 \
  --batch_size=4096 \
  --exp_train_total_epochs=1 \
  --use_vectorized_masking=True \
  --profiling_level=DETAILED \
  --profiling_output=profiling_forward_detailed.json \
  --project_name=forward_detailed
```

**Analysis:**
- Compare sub-operation times to expected values
- Identify specific slow operations
- Check for unexpected GPU synchronizations
- Memory allocation patterns

---

### 📝 Task 8: Document Findings and Propose Optimizations [PENDING]

**Deliverables:**
1. **Findings Document**: `FORWARD_PASS_FINDINGS.md`
   - List of identified inefficiencies
   - Root cause analysis for each
   - Profiling data showing impact

2. **Optimization Proposals**: `FORWARD_PASS_OPTIMIZATIONS.md`
   - Ranked list of optimizations by expected impact
   - Implementation complexity estimates
   - Risk assessment for each change

3. **Implementation Plan**:
   - Quick wins (< 1 hour, low risk)
   - Medium improvements (1-4 hours, moderate risk)
   - Major refactors (> 4 hours, higher risk)

---

## Expected Outcomes

### Best Case Scenario (All Optimizations)
- Enable Flash Attention: 1.3-1.5x speedup on attention
- Enable Mixed Precision (AMP): 1.5-2x overall speedup
- Fix Anti-Patterns: 1.1-1.2x speedup
- Kernel Fusion: 1.1-1.3x speedup
- **Total Potential**: 2.5-3.5x speedup on forward+backward

### Realistic Target
- Mixed Precision (AMP): 1.5x speedup (easiest, lowest risk)
- Flash Attention: 1.3x speedup (PyTorch 2.0+ builtin)
- Minor Fixes: 1.1x speedup
- **Total Realistic**: 2x speedup on forward+backward

### Impact on Full Training
- Current: 480ms/iteration → 96ms/epoch (805 iterations)
- After 2x forward+backward speedup: ~340ms/iteration → 68ms/epoch
- **100 epochs**: 9.6s → 6.8s (**~30% faster**)

---

## Next Steps

1. **Complete Task 2**: Finish documenting encoder implementation
2. **Complete Task 3**: Document predictor implementation
3. **Complete Task 4**: Identify anti-patterns in current code
4. **Complete Task 5**: Check against optimization checklist
5. **Run Task 6+7**: Add fine-grained profiling and run on SLURM
6. **Complete Task 8**: Document findings and create optimization plan

---

## Questions to Answer

1. **Sequence Length Impact**:
   - What are actual sequence lengths for target encoder, context encoder, predictor?
   - How much does masking reduce sequence length?

2. **Memory Usage**:
   - Why does predictor use 33GB vs 1.6GB for encoders?
   - Is this causing memory bandwidth bottlenecks?

3. **GPU Utilization**:
   - Are GPUs fully utilized during forward pass?
   - Any CPU bottlenecks or GPU stalls?

4. **Attention Implementation**:
   - Using PyTorch builtin MultiheadAttention?
   - Can we upgrade to fused scaled_dot_product_attention?

5. **Mixed Precision**:
   - Why is `model_amp=False`?
   - Was AMP tested and failed, or just not enabled?
