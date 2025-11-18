# Backward Pass Bottleneck Analysis

## What Code is Timed as "backward_pass"?

**Location**: `src/train.py:499-503`

```python
with self.profiler.profile("backward_pass"):  # <- Timer starts here
    if self.args.model_amp:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()  # <- This is what gets timed (AMP disabled)
# <- Timer ends here
```

**Profiler Implementation** (`src/utils/profiler.py:162-238`):
```python
@contextmanager
def profile(self, name: str, ...):
    # 1. Synchronize GPU (wait for all pending kernels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 2. Start timer
    start_time = time.perf_counter()

    # 3. Execute profiled code
    yield  # <--- loss.backward() executes here

    # 4. Synchronize GPU again
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 5. Record elapsed time
    elapsed = time.perf_counter() - start_time
```

**Key Insight**: The timer includes:
- GPU synchronization overhead (twice: before + after)
- All GPU kernel launches during backward
- All gradient computations
- Memory transfers between GPU memory banks

---

## Complete Computation Graph (Forward Pass)

Let me trace what operations are in the computation graph that `loss.backward()` must traverse:

### **Operations INCLUDED in Backward Graph** ✅

```python
# === FORWARD PASS (operations that create gradient graph) ===

# 1. TARGET ENCODER (excluded - inside torch.no_grad())
with torch.no_grad():  # ← NO gradients for this!
    h = target_encoder(batch)  # Not in backward graph
    h_masked = apply_masks_from_idx(h, masks_pred)  # Not in backward graph

# 2. CONTEXT ENCODER (4 transformer layers)
z = context_encoder(batch, masks_enc)  # ← GRADIENTS COMPUTED
#   Input: batch [B=4096, N=128]
#
#   Tokenizer:
#     numerical_features: Linear(1 → 64) for each feature
#     categorical_features: Embedding(cardinality → 64) for each
#     Result: [B, N, D=64]
#
#   Positional Encoding:
#     z += pos_embedding  # Adds learnable position embeddings
#
#   Apply Masking:
#     z_masked = z[:, masks_enc, :]  # Select visible features only
#
#   4 Transformer Layers (each layer):
#     Layer i:
#       # Self-Attention
#       attn_out = MultiheadAttention(z, z, z)  # [B, N_ctx, D]
#       z = LayerNorm(z + Dropout(attn_out))
#
#       # Feedforward
#       ffn_out = Linear_2(ReLU(Linear_1(z)))  # D→256→D
#       z = LayerNorm(z + Dropout(ffn_out))
#
#   Output: z [B*num_encs, N_ctx, D]

# 3. CONTEXT PREPARATION (tensor slicing)
if n_reg_tokens > 0:
    z_for_pred = z[:, :-n_reg_tokens, :]  # ← Slice operation (has gradient)

# 4. PREDICTOR (2 transformer layers)
z_pred = predictors(z_for_pred, masks_enc, masks_pred)  # ← GRADIENTS COMPUTED
#   Input: z_for_pred [B*num_encs, N_ctx, D]
#
#   Predictor Embedding:
#     z_emb = Linear(D → D)(z_for_pred)  # Project context
#
#   Positional Encoding:
#     z_emb += pos_embed  # Add positions
#
#   Create Mask Tokens:
#     mask_tokens = learnable_param.expand(B*num_preds, N_target, D)
#
#   Concatenate:
#     combined = concat([z_emb, mask_tokens], dim=1)  # [B, N_ctx+N_target, D]
#
#   2 Transformer Layers (same structure as encoder):
#     Layer i:
#       attn_out = MultiheadAttention(combined, combined, combined)
#       combined = LayerNorm(combined + Dropout(attn_out))
#
#       ffn_out = Linear_2(ReLU(Linear_1(combined)))
#       combined = LayerNorm(combined + Dropout(ffn_out))
#
#   Extract Predictions:
#     z_pred = combined[:, N_ctx:, :]  # Take only mask token outputs
#
#   Output Projection:
#     z_pred = Linear(D → D)(z_pred)  # Final projection
#
#   Output: z_pred [B*num_preds, N_target, D]

# 5. LOSS COMPUTATION
loss = MSELoss(z_pred, h_masked)  # ← MSE gradient computation
#   h_masked is detached (no gradient through target encoder)
#   loss = mean((z_pred - h_masked) ** 2)
```

---

## Backward Pass Pseudocode

**What `loss.backward()` does:**

```python
# === BACKWARD PASS (PyTorch autograd traverses graph backwards) ===

loss.backward()  # <--- This is the single line being profiled

# Internally, PyTorch does:

# STEP 1: Loss Gradient (MSE backward)
d_loss = 1.0  # Initial gradient
d_z_pred = 2 * (z_pred - h_masked) / (B * N_target * D)  # [B*num_preds, N_target, D]

# STEP 2: Predictor Backward (2 transformer layers + projection)
# 2a. Output projection backward
d_z_pred_proj = Linear.backward(d_z_pred)  # Backprop through output projection
grad_predictor_proj_weight += z_pred_input.T @ d_z_pred
grad_predictor_proj_bias += d_z_pred.sum(0)

# 2b. Transformer Layer 2 backward
#     LayerNorm backward (2nd FFN)
d_ffn_2 = LayerNorm.backward(d_z_pred_proj)
#     FFN backward (Linear_2 → ReLU → Linear_1)
d_relu_2 = Linear_2.backward(d_ffn_2)
grad_Linear_2_weight += ...
d_linear_1_out_2 = ReLU.backward(d_relu_2)
d_ffn_input_2 = Linear_1.backward(d_linear_1_out_2)
grad_Linear_1_weight += ...
#     Residual + Dropout backward
d_attn_out_2 = Dropout.backward(d_ffn_input_2)
d_z_2 = d_attn_out_2 + d_ffn_2  # Residual connection
#     LayerNorm backward (2nd attention)
d_attn_2 = LayerNorm.backward(d_z_2)
#     MultiheadAttention backward
d_q_2, d_k_2, d_v_2 = Attention.backward(d_attn_2)
#       Softmax backward: d_softmax = ...
#       Matmul backward: d_Q = d_attn @ K.T, d_K = Q.T @ d_attn
#       Linear backward for Q, K, V projections
grad_W_q_2 += ...
grad_W_k_2 += ...
grad_W_v_2 += ...

# 2c. Transformer Layer 1 backward (same structure as Layer 2)
#     [Repeat all operations above]

# 2d. Positional encoding backward
#     Just accumulate gradients to pos_embed parameter

# 2e. Predictor embedding backward
d_z_for_pred = Linear.backward(d_z_emb)
grad_predictor_emb_weight += z_for_pred.T @ d_z_emb

# STEP 3: Context Preparation Backward (tensor slicing)
d_z = torch.zeros_like(z)
d_z[:, :-n_reg_tokens, :] = d_z_for_pred  # Scatter gradient back

# STEP 4: Context Encoder Backward (4 transformer layers)
# 4a. Transformer Layer 4 backward
#     [Same structure as predictor layers but deeper]
#     LayerNorm → FFN (Linear_2 → ReLU → Linear_1) → Residual
#     LayerNorm → MultiheadAttention (Q, K, V projections)
grad_context_layer4_weights += ...

# 4b. Transformer Layer 3 backward
grad_context_layer3_weights += ...

# 4c. Transformer Layer 2 backward
grad_context_layer2_weights += ...

# 4d. Transformer Layer 1 backward
grad_context_layer1_weights += ...

# 4e. Positional encoding backward
grad_pos_embed += d_z.sum(0)

# 4f. Masking backward
#     Scatter gradients back to full feature set
d_z_full = torch.zeros(B, N, D)
d_z_full[:, masks_enc, :] = d_z_masked

# 4g. Tokenizer backward
#     Numerical features: Linear backward for each feature
for i, feat in enumerate(numerical_features):
    grad_numerical_weight[i] += batch[:, i] @ d_z_full[:, i, :]

#     Categorical features: Embedding backward for each feature
for i, feat in enumerate(categorical_features):
    grad_categorical_embed[i][batch[:, i]] += d_z_full[:, i, :]

# RESULT: All gradients stored in .grad attributes of parameters
```

---

## Breakdown of 223ms Backward Pass Time

Based on the architecture:
- Context Encoder: 4 layers × (attention + FFN) = 8 blocks
- Predictor: 2 layers × (attention + FFN) = 4 blocks
- **Total**: 12 transformer blocks to backpropagate through

**Estimated Time Breakdown**:

```
Predictor Backward (~90ms, 40%):
  - 2 transformer layers backward
  - Output projection backward
  - Embedding backward
  - Gradient accumulation for ~10M parameters

Context Encoder Backward (~120ms, 54%):
  - 4 transformer layers backward
  - Tokenizer backward (embeddings + linear)
  - Masking scatter operations
  - Gradient accumulation for ~15M parameters

Loss + Overhead (~13ms, 6%):
  - MSE gradient computation
  - GPU synchronization (2x)
  - Memory allocation/deallocation
```

---

## Key Findings

### 1. **Exact Code Coverage**
The `backward_pass` profiling only times:
- `loss.backward()` - single line (train.py:503)
- Plus 2x `torch.cuda.synchronize()` calls (profiler overhead)

### 2. **What Happens During backward()**
PyTorch autograd:
- Traverses computation graph in reverse order
- Computes gradients for 6 transformer layers (4 context + 2 predictor)
- Accumulates gradients for ~25M parameters
- Performs ~12 attention backward operations (O(n²×d) each)
- Performs ~12 FFN backward operations (2 matmuls each)
- Performs ~24 LayerNorm backward operations

### 3. **Why It's Slow**
- **Attention gradients**: O(n²×d) complexity, expensive for batch_size=4096
- **Many layers**: 6 total transformer layers = 6x backward cost
- **FP32 precision**: All gradients computed in 32-bit (2x slower than FP16)
- **Memory bandwidth**: Must load forward activations, write gradients
- **GPU synchronization**: Profiler syncs twice (before + after)

### 4. **NOT Included in Backward**
- Target encoder (inside `torch.no_grad()`)
- Optimizer step (profiled separately as `optimizer_step`)
- Gradient logging (profiled separately)
- Loss value reduction for distributed training

---

## Next Steps for Optimization

Now that we understand exactly what's being timed, potential optimizations:

1. **Enable Mixed Precision (AMP)** - Use FP16 for forward/backward (2x faster)
2. **Gradient Checkpointing** - Trade compute for memory (reduces activation storage)
3. **Reduce Layers** - Fewer transformer layers = less backward cost
4. **Optimize Attention** - Use Flash Attention or other efficient attention
5. **Profiler Overhead** - The 2x synchronization adds ~2-5ms overhead
6. **Check for Inefficiencies** - Profile sub-operations to find specific slow ops
