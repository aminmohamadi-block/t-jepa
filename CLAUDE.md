# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T-JEPA is a Joint Embedding Predictive Architecture for tabular data that learns representations without augmentations. The architecture implements a self-supervised learning strategy inspired by I-JEPA (Image JEPA).

### T-JEPA Training Strategy

**Core Architecture Components:**
1. **Context Encoder** - Trainable transformer that encodes visible (unmasked) features
2. **Target Encoder** - EMA copy of context encoder (frozen, momentum-updated) that encodes ALL features  
3. **Predictor Module** - Learns to predict target encoder representations from context encoder representations

**Detailed Training Process (src/train.py:377-497):**

1. **Mask Generation (src/mask.py:87-150):**
   - `MaskCollator` generates `masks_enc` (context) and `masks_pred` (target) per batch
   - Context masks: indices of features visible to context encoder
   - Target masks: indices of features to be predicted 
   - No overlap between context and target regions (enforced in mask creation)

2. **Target Forward Pass (train.py:389-395):**
   ```python
   with torch.no_grad():
       h = self.target_encoder(batch)  # Processes ALL features
       h = apply_masks_from_idx(h, masks_pred)  # Extract only target representations
   ```

3. **Context Forward Pass (train.py:397):**
   ```python
   z = self.context_encoder(batch, masks_enc)  # Processes only visible features
   ```
   - Masking applied inside encoder at embedding level (encoder.py:283-284)

4. **Prediction (train.py:400-410):**
   - **MLP Predictor**: Flattened context → separate MLPs per feature → predictions
   - **Transformer Predictor**: Context + mask tokens → transformer → target predictions

5. **Loss Computation:**
   - MSE loss between predicted and actual target representations
   - `loss = MSELoss(predicted_targets, actual_targets)`

6. **Momentum Update (train.py:491-497):**
   ```python
   with torch.no_grad():
       m = next(self.momentum_scheduler)
       for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
           param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
   ```

**Key Architectural Insight:** Target encoder provides stable target representations by seeing the complete feature set, while context encoder learns predictive patterns from partial observations. The EMA update prevents representation collapse.

**Masking Strategy Details (src/mask.py):**
- **Context masks (`masks_enc`):** Feature indices visible to context encoder
- **Target masks (`masks_pred`):** Feature indices to predict  
- **Sampling:** Random feature counts within min/max share bounds
- **Multiple predictions:** `num_preds=4` different target regions per sample
- **Constraint:** `num_encs * context_features + target_features ≤ total_features`

**Momentum Scheduling:**
- Starts at `model_ema_start` (0.996), evolves to `model_ema_end` (1.0)
- Gradual stabilization of target encoder during training

**Linear Probing Evaluation (train.py:200-362):**
- Periodic evaluation using frozen target encoder representations
- Trains linear probe on downstream task to measure representation quality
- Uses `OnlineDataset` with target encoder embeddings as input features

## Architecture

### Core Components

**Context/Target Encoder Architecture (src/encoder.py):**
- **Tokenizer (encoder.py:18-101):** Converts raw features to token embeddings
  - Numerical features: Linear projection with learnable weights
  - Categorical features: Embedding layers per category
  - CLS token: Prepended to sequence (`n_cls_tokens=1`)
- **Feature Embeddings:** Optional type and index embeddings
- **Positional Encoding:** Added to token embeddings
- **TabularEncoder (encoder.py:308-351):** Transformer with residual connections
  - Multi-head self-attention layers
  - Feedforward networks with layer normalization
  - Dropout regularization

**Predictor Architectures (src/predictors.py):**

1. **MLP Predictor (predictors.py:162-214):**
   - Separate MLP per feature position
   - Input: Flattened context representations (`hidden_dim * num_features`)
   - Architecture: `Linear → Dropout → LayerNorm → Activation → Linear`
   - Output: Per-feature predictions

2. **Transformer Predictor (predictors.py:217-339):**
   - **Context Processing:** Context embeddings with positional encoding
   - **Mask Tokens:** Learnable tokens for target positions
   - **Architecture:** `predictor_emb → pos_embed → transformer → predictor_proj`
   - **Key Innovation:** Concatenates context tokens with mask tokens, then extracts predictions
   - **Flow:** `[context_tokens, mask_tokens] → transformer → slice target outputs`

**Feature Processing Pipeline:**
1. **Raw Input:** `[batch_size, num_features]`
2. **Tokenization:** Numerical + categorical → embeddings `[batch_size, num_tokens, hidden_dim]`
3. **Positional Encoding:** Added to all tokens
4. **Masking:** Applied at token level using `apply_masks_from_idx()`
5. **Transformer:** Multi-layer self-attention processing
6. **Output:** Feature representations `[batch_size, selected_tokens, hidden_dim]`

### Models and Benchmarks
- **src/models/** - Deep learning baseline models (MLP, ResNet, AutoInt, DCNv2, FT-Transformer)
- **src/benchmark/** - Benchmarking infrastructure and tuned configurations
- **results/** - Stores benchmark results organized by dataset and model

### Datasets
- **src/datasets/** - Dataset implementations for 6 tabular datasets:
  - Adult Income (adult), Helena (helena), Jannis (jannis)
  - ALOI (aloi), California Housing (california), Higgs (higgs)
- Each dataset has both regular and embedded versions (*_embedded.py)

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (requires gdown)
./download_data.sh
```

### T-JEPA Training
```bash
# Launch T-JEPA pretraining with default settings (jannis dataset)
./scripts/launch_tjepa.sh

# Launch with specific dataset
./scripts/launch_tjepa.sh --data_path ./datasets --data_set helena

# Direct python execution with full control
python run.py --data_path ./datasets --data_set jannis --exp_train_total_epochs 300
```

### Benchmarking Deep Learning Models
```bash
# Run single benchmark with tuned config
python benchmark.py --config_file=src/benchmark/tuned_config/jannis/mlp_jannis_tuned.json --num_runs=1

# Run multiple experiments across datasets
python run_benchmark.py

# Run gradient boosted decision trees benchmark
python run_gbd.py
```

### Model Configuration
- Tuned model configs: `src/benchmark/tuned_config/<dataset>/<model>_<dataset>_tuned.json`
- T-JEPA tuning configs: `scripts/tjepa_tuning/config_tjepa_linear_<dataset>.yaml`
- Hyperparameter tuning configs: `src/benchmark/tuning_config/<dataset>/`

## Key Configuration Parameters

### T-JEPA Specific
- `--mask_max_ctx_share` / `--mask_min_ctx_share` - Context masking ratios
- `--mask_max_trgt_share` / `--mask_min_trgt_share` - Target masking ratios  
- `--mask_num_preds` - Number of prediction targets
- `--model_dim_hidden` - Hidden dimension size
- `--model_dropout_prob` - Dropout probability

### Training
- `--exp_train_total_epochs` - Total training epochs
- `--exp_lr` - Learning rate
- `--exp_weight_decay` - Weight decay
- `--batch_size` - Batch size
- `--mp_distributed` - Enable multi-process distributed training

### Data
- `--data_set` - Dataset name (adult, helena, jannis, aloi, california, higgs)
- `--data_path` - Path to dataset directory
- `--full_dataset_cuda` - Load entire dataset to GPU (for small datasets)

## Distributed Training & Scaling

### Multi-GPU Setup
The codebase has been optimized for distributed training with several key improvements:

**SLURM Integration:**
- `scripts/slurm_test.sh` - Production SLURM script with resource allocation
- `scripts/launch_tjepa.sh` - Smart launcher that detects GPU count and configures torchrun
- Automatic detection: `NUM_GPUS=${SLURM_GPUS:-1}`
- Uses `srun --mpi=pmi2 torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS` for multi-GPU

**Distributed Implementation:**
- NCCL backend for efficient GPU communication
- Custom SLURM environment detection in `src/utils/idr_torch.py`
- DistributedDataParallel (DDP) wrapping for context_encoder and predictors
- DistributedSampler for proper data sharding across ranks
- AllReduce custom autograd function for gradient synchronization

**Key Environment Variables:**
- RANK, WORLD_SIZE, LOCAL_RANK - Set by SLURM/torchrun  
- MASTER_ADDR, MASTER_PORT - Auto-configured from SLURM or defaults
- SLURM_GPUS, SLURM_STEP_GPUS - GPU allocation detection

### Foundation Model Scaling Considerations

**Memory Optimization:**
- `--full_dataset_cuda=True` - Load entire dataset to GPU memory (for smaller datasets)
- `--pin_memory=True` - Pin memory for faster CPU-GPU transfers
- `--model_amp=True` - Mixed precision training with GradScaler (currently disabled)
- Gradient clipping via `--exp_gradient_clipping` for training stability

**Data Loading Efficiency:**
- Custom `drop_single_sample_collate_fn` prevents single-sample batches
- Distributed dataloader with proper sharding: `get_distributed_dataloader()`
- Configurable worker processes: `--data_loader_nprocs`
- Masking collator for efficient T-JEPA batch processing

**Checkpointing & Resumption:**
- Robust checkpointing system in `src/utils/checkpointer.py`
- Resume from checkpoint: `--load_from_checkpoint=True --load_path=<path>`
- Target encoder momentum scheduling: `--model_ema_start/end` (EMA coefficient evolution)
- Early stopping with configurable patience: `--exp_patience`

**Scaling Parameters for Large Datasets:**
- `--batch_size` - Scale based on available GPU memory
- `--model_dim_hidden` - Hidden dimension size (64-512+ for larger models)
- `--model_num_layers` - Transformer layers (16+ for deeper models)
- `--exp_cache_cadence` - Checkpoint frequency (lower for long runs)
- `--mask_num_preds` - Number of prediction targets (affects memory)

### MLflow Integration
Production experiment tracking configured for Databricks:
- Automatic MLflow setup with proper authentication
- Experiment organization under `/groups/block-aird-team/t-jepa-test`
- Token-based authentication with fallback handling

### Performance Monitoring

**Training Metrics (train.py:524-541):**
- `tjepa_train_loss`: MSE reconstruction loss
- `tjepa_lr`: Current learning rate from scheduler
- `tjepa_momentum`: Current EMA coefficient for target encoder
- `tjepa_weight_decay`: Current weight decay value
- `linear_probe_metric`: Downstream task performance

**Gradient Monitoring (train.py:431-481):**
- Context encoder gradient statistics (mean, L2 norm, std)
- Target encoder gradient statistics (should be zero - frozen)
- Predictor gradient statistics
- Logged per iteration for training diagnostics

**Representation Collapse Detection (train.py:592-607):**
- **KL Divergence:** Between random sample representations
- **Euclidean Distance:** Between random sample representations  
- **Variance Metrics:** Intra-feature and inter-feature variance
- Computed periodically to detect representation collapse

**MLflow Integration (train.py:183-195, 537-541):**
- Automatic experiment tracking with run naming
- Parameter logging for reproducibility
- Metric logging with step-based tracking
- Artifact logging (embedding visualizations)

**Debug Logging:**
- Distributed setup verification with rank/device info
- Tensor value debugging with `_debug_values()` throughout pipeline
- Memory and timing statistics per epoch

## Output Structure

Results are organized as:
```
results/<dataset_code>/
├── <model>/
│   ├── output_0.json, output_1.json, ...  # Individual runs
│   └── summary.json                        # Aggregated results
```

Dataset codes: adult_AD, helena_HE, jannis_JA, aloi_AL, california_CA, higgs_HI

## Complete Training Pipeline (run.py)

### **Initialization Sequence (run.py:32-381)**

1. **Distributed Setup (run.py:34-96):**
   ```python
   # Environment variable detection and process group initialization
   dist.init_process_group(backend="nccl", init_method="env://")
   local_rank = idr_torch.local_rank or int(os.environ.get("LOCAL_RANK", 0))
   torch.cuda.set_device(local_rank)
   ```

2. **Data Preprocessing Pipeline (run.py:103-141):**
   ```python
   # Dataset loading and encoding
   dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
   dataset.load()
   
   # Feature encoding: MinMaxScaler for numerical, categorical as-is
   train_torchdataset = TorchDataset(dataset=dataset, preprocessing=encode_data)
   ```

3. **Model Architecture Instantiation (run.py:143-186):**
   ```python
   # Context encoder (trainable)
   context_encoder = Encoder(...)
   
   # Predictor module (trainable)  
   predictors = Predictors(pred_type=args.pred_type, ...)
   
   # Target encoder (EMA copy, frozen)
   target_encoder = copy.deepcopy(context_encoder)
   for p in target_encoder.parameters(): p.requires_grad = False
   ```

4. **Distributed Data Parallel Wrapping (run.py:192-205):**
   ```python
   # Only context_encoder and predictors wrapped (target_encoder stays unwrapped)
   context_encoder = DistributedDataParallel(context_encoder, device_ids=[local_rank])
   predictors = DistributedDataParallel(predictors, device_ids=[local_rank])
   ```

### **Optimization Setup (run.py:254-273)**

**AdamW Optimizer with Parameter Groups (optim_utils.py:57-77):**
- **Group 1**: Context encoder weights (excluding bias/1D params)
- **Group 2**: Predictor weights (excluding bias/1D params)  
- **Group 3**: Context encoder bias/1D params (weight_decay=0)
- **Group 4**: Predictor bias/1D params (weight_decay=0)

**Scheduler Configuration:**
- **Learning Rate**: WarmupCosineSchedule (scheduler.py:4-65)
  - Warmup: Linear increase from `start_lr` to `ref_lr`
  - Cosine decay: From `ref_lr` to `final_lr`
- **Weight Decay**: CosineWDSchedule (scheduler.py:67-109)
  - Cosine decay from `ref_wd` to `final_wd`

**EMA Momentum Scheduler (run.py:270-273):**
```python
momentum_scheduler = (
    ema_start + i * (ema_end - ema_start) / (ipe * num_epochs * ipe_scale)
    for i in range(int(ipe * num_epochs * ipe_scale) + 1)
)
```

### **Data Loading Architecture**

**Masking Collator (run.py:215-225):**
- Generates `masks_enc` (context indices) and `masks_pred` (target indices)
- Ensures no overlap between context and target regions
- Constraint: `num_encs * context_features + target_features ≤ total_features`

**Distributed DataLoader (run.py:229-248):**
- DistributedSampler for multi-GPU data sharding
- Custom collate function with masking logic
- Pin memory for efficient CPU-GPU transfers

### **Checkpoint and Resume Logic (run.py:275-310)**

**Checkpoint Loading:**
- Loads context_encoder, predictors, target_encoder, optimizer, schedulers
- Restores epoch count and advances momentum scheduler accordingly
- Re-freezes target encoder parameters after loading

**MLflow Integration (run.py:338-374):**
- Databricks authentication with token fallback
- Experiment organization under `/groups/block-aird-team/t-jepa-test`
- Parameter and metric logging for reproducibility

## Foundation Model Training Workflow

**Complete Training Command:**
```bash
./scripts/launch_tjepa.sh \
  --data_path ./datasets \
  --data_set your_large_dataset \
  --batch_size 1024 \
  --model_dim_hidden 512 \
  --model_num_layers 24 \
  --exp_train_total_epochs 1000 \
  --exp_lr 0.0003 \
  --exp_warmup 10 \
  --model_ema_start 0.996 \
  --model_ema_end 1.0 \
  --mask_min_ctx_share 0.15 \
  --mask_max_ctx_share 0.40 \
  --mask_min_trgt_share 0.15 \
  --mask_max_trgt_share 0.65 \
  --mask_num_preds 4 \
  --full_dataset_cuda=False \
  --pin_memory=True \
  --load_from_checkpoint=True \
  --load_path=checkpoints/.../epoch_X.pth
```

**Key Training Parameters:**
- `ipe_scale=1.25`: Extends scheduler beyond training epochs for stability
- `probe_cadence=20`: Linear probe evaluation every N epochs
- `exp_cache_cadence=20`: Checkpoint saving frequency

## Main Training Loop Architecture (src/train.py)

### **Training Loop Overview (train.py:168-590)**

The training loop implements the core T-JEPA self-supervised learning algorithm with comprehensive monitoring and evaluation.

### **Epoch-Level Operations (train.py:197-559)**

**1. Linear Probe Evaluation (train.py:200-362):**
```python
if self.probe_cadence > 0 and self.epoch % self.probe_cadence == 0:
    # Create OnlineDataset with target encoder embeddings
    online_dataset = OnlineDataset(args, self.target_encoder)
    
    # Train linear probe on downstream task
    model = model_class(loss=loss_fn, **dataset_args)
    trainer = pl.Trainer(...)
    trainer.fit(model, datamodule=datamodule)
    
    # Extract validation score for representation quality
    linear_probe_metric = val_metrics[0][f"{dataset}_val_score"]
```

**2. Representation Collapse Monitoring (train.py:265-278):**
```python
collapse_metrics = {
    "KL": [],  # KL divergence between random sample representations
    "euclidean": [],  # Euclidean distance between random samples
    "intra_feature_variance": np.mean(np.var(X, axis=0)),  # Feature variance
    "inter_feature_variance": np.var(X.mean(axis=1)),  # Sample variance
}
```

### **Batch-Level Training Loop (train.py:377-504)**

**Step 1: Data Loading and Device Transfer (train.py:377-385)**
```python
for itr, (batch, masks_enc, masks_pred) in enumerate(tqdm(self.dataloader)):
    batch = batch.to(self.device, non_blocking=True)
    masks_enc = [mask.to(self.device, non_blocking=True) for mask in masks_enc]
    masks_pred = [mask.to(self.device, non_blocking=True) for mask in masks_pred]
```

**Step 2: Target Encoder Forward Pass (train.py:388-395)**
```python
with torch.cuda.amp.autocast(enabled=self.args.model_amp):
    # Target encoder sees ALL features (no masking in forward pass)
    with torch.no_grad():
        h = self.target_encoder(batch)  # Shape: [B, N, D]
        
        # Extract only target region representations
        h = apply_masks_from_idx(h, masks_pred)  # Shape: [B*num_preds, N_target, D]
```

**Step 3: Context Encoder Forward Pass (train.py:397)**
```python
# Context encoder sees only visible features (masking applied internally)
z = self.context_encoder(batch, masks_enc)  # Shape: [B*num_encs, N_context, D]
```

**Step 4: Prediction (train.py:400-410)**
```python
if self.args.pred_type == "mlp":
    # MLP Predictor: Flatten context and predict per feature
    z = z.view(z.size(0), -1)  # Flatten: [B*num_encs, N_context*D]
    z = self.predictors(z, masks_pred.transpose(0, 1))
    
    # Compute loss for each prediction target
    loss = torch.zeros(1, device=self.device)
    for z_, h_ in zip(z, h):
        loss += self.loss_fn(z_, h_)  # MSE loss
        
else:  # Transformer Predictor
    # Transformer with mask tokens for target positions
    z = self.predictors(z, masks_enc, masks_pred)
    loss = self.loss_fn(z, h)  # Direct MSE between predictions and targets
```

**Step 5: Gradient Computation and Optimization (train.py:412-427)**
```python
# Distributed loss reduction for logging
if self.is_distributed:
    dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
    loss_value = loss_value / self.world_size

# Mixed precision training
if self.args.model_amp:
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    loss.backward()
    self.optimizer.step()
```

**Step 6: Gradient Monitoring (train.py:431-481)**
```python
if itr == 0:  # Log gradients for first iteration of each epoch
    # Context encoder gradients (should be non-zero)
    ctx_grads = [param.grad.flatten() for param in self.context_encoder.parameters() 
                 if param.grad is not None]
    
    # Target encoder gradients (should be zero - frozen)
    trgt_grads = [param.grad.flatten() for param in self.target_encoder.parameters() 
                  if param.grad is not None]
    
    # Log gradient statistics to MLflow
    grad_metrics = {
        "context_encoder_grad_mean": float(np.mean(ctx_grads)),
        "target_encoder_grad_mean": float(np.mean(trgt_grads)),  # Should be 0
        # ... additional gradient statistics
    }
```

**Step 7: EMA Target Encoder Update (train.py:490-497)**
```python
# Critical step: Update target encoder via exponential moving average
with torch.no_grad():
    m = next(self.momentum_scheduler)  # Get current momentum coefficient
    
    for param_q, param_k in zip(self.context_encoder.parameters(), 
                                self.target_encoder.parameters()):
        # EMA update: target = m * target + (1-m) * context
        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
```

**Step 8: Scheduler Updates (train.py:499-503)**
```python
# Step learning rate and weight decay schedulers
if self.scheduler is not None:
    self.scheduler.step()
if self.weight_decay_scheduler is not None:
    self.weight_decay_scheduler.step()
```

### **End-of-Epoch Processing (train.py:510-559)**

**Early Stopping and Checkpointing (train.py:510-552)**
```python
args_early_stop = {
    "train_loss": total_loss.item(),
    "context_encoder": self.context_encoder,
    "target_encoder": self.target_encoder,
    "predictor": self.predictors,
    "optimizer": self.optimizer,
    "scheduler": self.scheduler,
    "epoch": self.epoch,
    "val_score": linear_probe_metric,
}

# Update early stopping counter and save checkpoints
early_stop_signal = self.early_stop_counter.update(**args_early_stop)
```

**MLflow Logging (train.py:524-541)**
```python
log_dict = {
    "tjepa_train_loss": total_loss.item(),
    "tjepa_epoch": self.epoch,
    "tjepa_lr": self.scheduler.get_last_lr()[0],
    "tjepa_momentum": m,  # Current EMA coefficient
    "tjepa_weight_decay": self.weight_decay_scheduler.get_last_wd()[0],
    "linear_probe_metric": linear_probe_metric,
}
# Add collapse metrics if computed
if collapse_metrics is not None:
    log_dict.update(collapse_metrics)
```

### **Key Training Loop Properties**

**Data Flow Pattern:**
1. **Batch → Target Encoder (all features) → Extract targets**
2. **Batch → Context Encoder (visible features only) → Context representations** 
3. **Context representations → Predictor → Target predictions**
4. **MSE Loss(predictions, actual_targets)**
5. **EMA update of target encoder**

**Critical Timing:**
- **Target encoder forward**: Before context encoder (provides stable targets)
- **EMA update**: After gradients computed (uses updated context encoder)
- **Scheduler steps**: After EMA update (maintains training schedule)

**Memory Efficiency:**
- **Target encoder**: `torch.no_grad()` context (no gradient computation)
- **Mixed precision**: Optional AMP support for memory/speed
- **Gradient accumulation**: Single-step optimization per batch