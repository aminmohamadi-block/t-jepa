#!/usr/bin/env python3
"""
Standalone linear probe testing script for T-JEPA checkpoints.
Tests whether linear probe optimization issues are due to training or representations.
"""

import argparse
import os
import json
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import mlflow

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.encoder import Encoder
from src.models.linear_probing import LinearProbe
from src.datasets.online_dataset import OnlineDataset, OnlineDatasetArgs
from src.torch_dataset import DataModule
from src.benchmark.utils import get_loss_from_task, MODEL_NAME_TO_MODEL_MAP, MODEL_CONFIG_BASE_PATH
from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
from src.utils.encode_utils import encode_data
from src.utils.models_utils import TASK_TYPE



def generate_embeddings(encoder, dataset, batch_size: int = 512, device: str = "cuda"):
    """Generate embeddings for entire dataset using the encoder"""
    print(f"Generating embeddings using: {encoder.__class__.__name__}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Generating embeddings"):
            batch_x = batch_x.to(device)
            
            # Get embeddings from encoder
            z = encoder(batch_x)
            
            embeddings.append(z.cpu().numpy())
            labels.append(batch_y.numpy())
    
    X = np.concatenate(embeddings, axis=0)
    y = np.concatenate(labels, axis=0)
    
    print(f"Generated embeddings shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y


def create_probe_datamodule(X, y, test_size: float = 0.1, val_size: float = 0.1, 
                           batch_size: int = 128, random_state: int = 42):
    """Create PyTorch Lightning DataModule from embeddings"""
    
    # Simple train/val/test split
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    # Random permutation
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X[train_idx]),
        torch.LongTensor(y[train_idx])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X[val_idx]),
        torch.LongTensor(y[val_idx])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X[test_idx]),
        torch.LongTensor(y[test_idx])
    )
    
    # Create data module
    class EmbeddingDataModule(pl.LightningDataModule):
        def __init__(self, train_ds, val_ds, test_ds, batch_size):
            super().__init__()
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.test_ds = test_ds
            self.batch_size = batch_size
            
        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        def test_dataloader(self):
            return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    return EmbeddingDataModule(train_dataset, val_dataset, test_dataset, batch_size)


def train_linear_probe_with_datamodule(datamodule, online_dataset, online_dataset_args, args):
    """Train linear probe exactly like train.py does"""
    
    # Follow train.py pattern exactly (lines 284-349)
    model_class = MODEL_NAME_TO_MODEL_MAP["linear_probe"]
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_args = vars(online_dataset_args).copy()
    dataset_args.update(
        {
            "test_size_ratio": args.test_size,
            "val_size_ratio": args.val_size,
            "batch_size": args.batch_size,
            "task_type": online_dataset.task_type,
            "using_embedding": True,
            "exp_train_total_epochs": args.max_epochs,
            "model_name": "linear_probe",
            "dataset_name": online_dataset_args.data_set,
            "exp_patience": args.patience,
            "n_cls_tokens": 1,  # Default from args
            "data_loader_nprocs": 2,
            "pin_memory": True,
            "full_dataset_cuda": False,
            "mock": False,
            "random_state": args.random_state,
        }
    )
    dataset_args = Namespace(**dataset_args)

    # Create DataModule with model_class.preprocessing (like train.py line 314)
    datamodule = DataModule(
        dataset=online_dataset,
        test_size_ratio=dataset_args.test_size_ratio,
        val_size_ratio=dataset_args.val_size_ratio,
        random_state=dataset_args.random_state,
        device=device,
        batch_size=dataset_args.batch_size,
        workers=dataset_args.data_loader_nprocs,
        pin_memory=dataset_args.pin_memory,
        full_dataset_cuda=dataset_args.full_dataset_cuda,
        preprocessing=model_class.preprocessing,  # This was missing!
        mock=dataset_args.mock,
        using_embedding=True,
    )

    # Load model config from JSON (like train.py lines 323-332)
    base_config = {
        "dataset_name": args.dataset_name,
        "encoder_type": "linear_flatten",
    }
    
    from src.benchmark.utils import MODEL_CONFIG_BASE_PATH
    import json
    model_args = json.load(
        open(
            MODEL_CONFIG_BASE_PATH.format(
                dataset_name=args.dataset_name,
                model_name="linear_probe",
            )
        )
    )
    model_args.update(base_config)
    model_args = Namespace(**model_args)

    # Use model_class.get_model_args (like train.py lines 334-338)
    model_args = model_class.get_model_args(
        datamodule,
        dataset_args,
        model_args,
    )
    
    print(f"Loading linear_probe")
    from tabulate import tabulate
    print(
        tabulate(
            sorted(list(vars(model_args).items()), key=lambda x: x[0]),
            tablefmt="fancy_grid",
        )
    )

    # Create model exactly like train.py (lines 347-351)
    loss_fn = get_loss_from_task(dataset_args.task_type)
    dataset_args = {**vars(dataset_args), **vars(model_args)}
    model = model_class(loss=loss_fn, **dataset_args)
    model = model.float()
    
    # MLflow parameters for logging
    extra_params = {
        "task_type": online_dataset.task_type,
        "checkpoint_path": args.checkpoint_path,
        "model_name": "linear_probe"
    }
    
    # Setup callbacks
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=f"{args.dataset_name}_val_loss",
            patience=args.patience,
            verbose=True,
            mode="min"
        ),
        pl.callbacks.ModelCheckpoint(
            monitor=f"{args.dataset_name}_val_loss",
            save_top_k=1,
            verbose=True,
            mode="min"
        )
    ]
    
    # Add MLflow logger if enabled
    logger = None
    if args.use_mlflow:
        from pytorch_lightning.loggers import MLFlowLogger
        logger = MLFlowLogger(
            experiment_name=f"/groups/block-aird-team/{args.mlflow_experiment}",
            run_name=args.run_name,
            log_model=True
        )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )
    
    # Train model
    print("Starting linear probe training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Test model
    test_results = trainer.test(model, datamodule=datamodule)
    
    if args.use_mlflow and logger:
        # Log additional parameters that the logger doesn't automatically capture
        logger.experiment.log_params(logger.run_id, extra_params)
        
        # Log final test metrics (train/val metrics are logged automatically)
        logger.experiment.log_metrics(logger.run_id, {
            "test_loss": test_results[0].get("test_loss", 0),
            "test_accuracy": test_results[0].get("test_accuracy", 0),
        })
    
    return trainer, model, test_results


def main():
    parser = argparse.ArgumentParser(description="Test linear probe optimization with T-JEPA checkpoints")

    # Required arguments
    parser.add_argument("--checkpoint_path", required=True, help="Path to T-JEPA checkpoint")
    parser.add_argument("--dataset_name", default="higgs", help="Dataset name")
    parser.add_argument("--data_path", default="./datasets", help="Path to dataset")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")

    # Data splitting
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    # MLflow settings
    parser.add_argument("--use_mlflow", action="store_true", help="Use MLflow logging")
    parser.add_argument("--mlflow_experiment", default="linear_probe_test", help="MLflow experiment name")
    parser.add_argument("--run_name", default="probe_test", help="MLflow run name")

    # Other settings
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--embedding_batch_size", type=int, default=512, help="Batch size for embedding generation")

    # Parquet dataset arguments
    parser.add_argument("--use_parquet_dataset", action="store_true", help="If True, use parquet dataset")
    parser.add_argument("--parquet_data_dir", type=str, default=None, help="Directory containing parquet chunk files")
    parser.add_argument("--parquet_data_files", type=str, default=None, help="Comma-separated list of specific parquet files to use")
    parser.add_argument("--parquet_scaling_stats_file", type=str, default=None, help="Path to the scaling statistics parquet file")
    parser.add_argument("--parquet_feature_names_file", type=str, default=None, help="Path to feature importance CSV file")
    parser.add_argument("--parquet_num_features", type=int, default=None, help="Number of top features to use from feature_names_file")
    parser.add_argument("--parquet_target_col", type=str, default="target", help="Name of the target column in parquet files")
    parser.add_argument("--parquet_task_type", type=str, default="binary_class", choices=["binary_class", "multi_class", "regression"], help="Task type for parquet dataset")
    parser.add_argument("--parquet_id_col", type=str, default="ID", help="Name of the ID column in parquet files")
    parser.add_argument("--parquet_preload_data", action="store_true", default=True, help="If True, preload all data into memory")
    parser.add_argument("--parquet_shuffle", action="store_true", default=True, help="If True, shuffle data during training")
    parser.add_argument("--parquet_scaling_method", type=str, default="IQR", choices=["mean_std", "min_max", "IQR", "1_percentile", "5_percentile", "none"], help="Scaling method for features")
    parser.add_argument("--parquet_infill_value", type=str, default="zero", help="NaN infill strategy: 'zero', 'global_mean', 'previous_mean', or None")
    parser.add_argument("--parquet_transform", type=str, default=None, help="Transform to apply after scaling: 'asinh' or None")
    parser.add_argument("--parquet_categorize_nan", action="store_true", default=False, help="If True, add binary NaN indicator features")
    parser.add_argument("--parquet_clip_min", type=float, default=None, help="Minimum value for clipping features after scaling")
    parser.add_argument("--parquet_clip_max", type=float, default=None, help="Maximum value for clipping features after scaling")
    parser.add_argument("--probe_sample_fraction", type=float, default=0.1, help="Fraction of data to use for linear probe (takes last N%% from unshuffled dataset)")

    args = parser.parse_args()
    
    # Extract architecture from checkpoint filename first
    import re
    match = re.search(r'nlyrs_(\d+)_nheads_(\d+)_hdim_(\d+)', args.checkpoint_path)
    if match:
        num_layers_from_filename = int(match.group(1))
        num_heads_from_filename = int(match.group(2))
        hidden_dim_from_filename = int(match.group(3))
        print(f"Extracted from filename: layers={num_layers_from_filename}, heads={num_heads_from_filename}, hidden_dim={hidden_dim_from_filename}")
    else:
        num_layers_from_filename = None
        num_heads_from_filename = None
        hidden_dim_from_filename = None
        print("⚠️  WARNING: Could not extract architecture from filename")

    # Extract hidden_dim from checkpoint to verify
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    target_encoder_state = checkpoint['target_encoder']

    # Get hidden_dim and n_reg_tokens from tokenizer weights
    # tokenizer.weight shape is [num_tokens, hidden_dim]
    # num_tokens = num_features + n_cls_tokens + n_reg_tokens
    hidden_dim = None
    num_tokens_in_checkpoint = None
    for key, tensor in target_encoder_state.items():
        if 'tokenizer.weight' in key:
            num_tokens_in_checkpoint, hidden_dim = tensor.shape
            break

    if hidden_dim is None:
        hidden_dim = hidden_dim_from_filename or 64  # fallback

    # Infer n_reg_tokens from tokenizer shape
    # num_tokens = num_features (256) + n_cls_tokens (1) + n_reg_tokens
    # So n_reg_tokens = num_tokens - 256 - 1
    n_reg_tokens_detected = None
    if num_tokens_in_checkpoint is not None:
        # Assuming 256 features and 1 CLS token
        n_reg_tokens_detected = num_tokens_in_checkpoint - 256 - 1
        print(f"Detected from tokenizer shape [{num_tokens_in_checkpoint}, {hidden_dim}]: n_reg_tokens={n_reg_tokens_detected}")

    print(f"Detected hidden_dim from checkpoint: {hidden_dim}")
    if hidden_dim_from_filename and hidden_dim != hidden_dim_from_filename:
        print(f"⚠️  WARNING: Filename says {hidden_dim_from_filename} but checkpoint has {hidden_dim}")

    # Load dataset args - create as dict first to include parquet args
    dataset_args_dict = {
        'data_set': args.dataset_name,
        'data_path': args.data_path,
        'batch_size': args.embedding_batch_size,
        'data_loader_nprocs': 4,
        'pin_memory': True,
        'mock': False,
        'test_size_ratio': 0.0,
        'random_state': args.random_state,
        'val_size_ratio': 0.0,
        'full_dataset_cuda': False,
        'val_batch_size': args.batch_size,
        'input_embed_dim': hidden_dim,  # Set from checkpoint like train.py
        'probe_sample_fraction': args.probe_sample_fraction,
        'n_reg_tokens': 1,  # Match encoder configuration
        'n_cls_tokens': 1,  # Match encoder configuration
    }

    # Add parquet-specific arguments if using parquet dataset
    if args.use_parquet_dataset or args.dataset_name == 'parquet_dataset':
        dataset_args_dict.update({
            'use_parquet_dataset': True,
            'parquet_data_dir': args.parquet_data_dir,
            'parquet_data_files': args.parquet_data_files,
            'parquet_scaling_stats_file': args.parquet_scaling_stats_file,
            'parquet_feature_names_file': args.parquet_feature_names_file,
            'parquet_num_features': args.parquet_num_features,
            'parquet_target_col': args.parquet_target_col,
            'parquet_task_type': args.parquet_task_type,
            'parquet_id_col': args.parquet_id_col,
            'parquet_preload_data': args.parquet_preload_data,
            'parquet_shuffle': args.parquet_shuffle,
            'parquet_scaling_method': args.parquet_scaling_method,
            'parquet_infill_value': args.parquet_infill_value,
            'parquet_transform': args.parquet_transform,
            'parquet_categorize_nan': args.parquet_categorize_nan,
            'parquet_clip_min': args.parquet_clip_min,
            'parquet_clip_max': args.parquet_clip_max,
        })

    # Convert to namespace for compatibility
    dataset_args = Namespace(**dataset_args_dict)
    
    # Use a simpler approach: let OnlineDataset handle the encoder loading
    print(f"Loading checkpoint using OnlineDataset approach...")
    
    # checkpoint already loaded above
    if 'target_encoder' not in checkpoint:
        raise ValueError(f"target_encoder not found in checkpoint. Keys: {list(checkpoint.keys())}")
    
    target_encoder_state = checkpoint['target_encoder']
    print("Extracting architecture from checkpoint...")
    
    # Extract key dimensions from checkpoint (hidden_dim already extracted above)
    dim_feedforward = None
    num_layers = 0

    for key, tensor in target_encoder_state.items():
        # Detect feedforward dimension from any layer's linear1 weight
        # Pattern: 'encoder.transformer.layers.X.linear1.weight' has shape [dim_feedforward, hidden_dim]
        if '.linear1.weight' in key:
            dim_feedforward, _ = tensor.shape
        # Detect number of layers from transformer layer indices
        # Pattern: 'encoder.transformer.layers.X.' where X is the layer index
        elif 'encoder.transformer.layers.' in key:
            parts = key.split('.')
            try:
                layer_idx = parts.index('layers') + 1
                if layer_idx < len(parts):
                    layer_num = int(parts[layer_idx])
                    num_layers = max(num_layers, layer_num + 1)
            except (ValueError, IndexError):
                pass

    # Use num_heads from filename (most reliable source)
    num_heads = num_heads_from_filename
    # Use num_layers from filename if checkpoint extraction failed
    if num_layers == 0 and num_layers_from_filename:
        num_layers = num_layers_from_filename

    print(f"Final architecture: hidden_dim={hidden_dim}, feedforward={dim_feedforward}, layers={num_layers}, heads={num_heads}")

    # Check for feature embeddings in checkpoint
    has_feature_type_emb = any('feature_type_embedding' in key for key in target_encoder_state.keys())
    has_feature_index_emb = any('feature_index_embedding' in key for key in target_encoder_state.keys())
    print(f"Feature embeddings: type={has_feature_type_emb}, index={has_feature_index_emb}")

    # Add warning if architecture detection incomplete or used fallbacks
    if num_layers == 0 or dim_feedforward is None or num_heads is None or hidden_dim is None:
        print(f"⚠️  WARNING: Architecture detection incomplete. Using fallback values.")
        print(f"   Detected: layers={num_layers}, hidden={hidden_dim}, feedforward={dim_feedforward}, heads={num_heads}")
        print(f"   This may indicate checkpoint structure has changed or is incompatible.")

    # Load the dataset to get proper model architecture
    # dataset_args is already a Namespace with all necessary arguments
    dataset_class = DATASET_NAME_TO_DATASET_MAP[args.dataset_name]
    dataset = dataset_class(dataset_args)
    dataset.load()

    # Use detected n_reg_tokens from tokenizer shape
    # n_reg_tokens was detected earlier from: num_tokens = num_features + n_cls_tokens + n_reg_tokens
    if n_reg_tokens_detected is not None and n_reg_tokens_detected >= 0:
        best_config = n_reg_tokens_detected
        print(f"\n✓ Using n_reg_tokens={best_config} (detected from tokenizer shape)")
    else:
        # Fallback: try both values if detection failed
        print("\n" + "="*80)
        print("TESTING DIFFERENT n_reg_tokens VALUES (detection failed)")
        print("="*80)

        best_config = None
        best_errors = float('inf')

        for n_reg_val in [0, 1]:
            print(f"\nTesting n_reg_tokens={n_reg_val}...")

            test_encoder = Encoder(
                idx_num_features=dataset.num_features,
                cardinalities=dataset.cardinalities,
                hidden_dim=hidden_dim or 64,
                num_layers=num_layers or 4,
                num_heads=num_heads or 4,
                p_dropout=0.0,
                layer_norm_eps=1e-5,
                gradient_clipping=1.0,
                feature_type_embedding=has_feature_type_emb,
                feature_index_embedding=has_feature_index_emb,
                dim_feedforward=dim_feedforward or 256,
                device=args.device,
                args=Namespace(n_cls_tokens=1, n_reg_tokens=n_reg_val, model_act_func='relu')
            )

            try:
                incompatible = test_encoder.load_state_dict(checkpoint['target_encoder'], strict=False)
                total_errors = len(incompatible.missing_keys) + len(incompatible.unexpected_keys)

                print(f"  Missing keys: {len(incompatible.missing_keys)}")
                print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
                print(f"  Total errors: {total_errors}")

                if total_errors < best_errors:
                    best_errors = total_errors
                    best_config = n_reg_val
            except RuntimeError as e:
                print(f"  Failed to load: {str(e)[:100]}...")

        print(f"\n✓ Best config: n_reg_tokens={best_config}")

    # Create encoder with best configuration
    encoder = Encoder(
        idx_num_features=dataset.num_features,
        cardinalities=dataset.cardinalities,
        hidden_dim=hidden_dim or 64,
        num_layers=num_layers or 4,
        num_heads=num_heads or 4,
        p_dropout=0.0,
        layer_norm_eps=1e-5,
        gradient_clipping=1.0,
        feature_type_embedding=has_feature_type_emb,
        feature_index_embedding=has_feature_index_emb,
        dim_feedforward=dim_feedforward or 256,
        device=args.device,
        args=Namespace(n_cls_tokens=1, n_reg_tokens=best_config, model_act_func='relu')
    )

    # Update dataset_args with the correct n_reg_tokens value
    dataset_args.n_reg_tokens = best_config

    # Load the checkpoint state dict with validation
    incompatible_keys = encoder.load_state_dict(checkpoint['target_encoder'], strict=False)

    # Report any missing or unexpected keys
    if incompatible_keys.missing_keys:
        print(f"⚠️  Missing keys when loading checkpoint: {len(incompatible_keys.missing_keys)} keys")
        if len(incompatible_keys.missing_keys) <= 10:
            for key in incompatible_keys.missing_keys:
                print(f"   - {key}")
        else:
            print(f"   (showing first 10 of {len(incompatible_keys.missing_keys)})")
            for key in incompatible_keys.missing_keys[:10]:
                print(f"   - {key}")

    if incompatible_keys.unexpected_keys:
        print(f"⚠️  Unexpected keys in checkpoint: {len(incompatible_keys.unexpected_keys)} keys")
        if len(incompatible_keys.unexpected_keys) <= 10:
            for key in incompatible_keys.unexpected_keys:
                print(f"   - {key}")
        else:
            print(f"   (showing first 10 of {len(incompatible_keys.unexpected_keys)})")
            for key in incompatible_keys.unexpected_keys[:10]:
                print(f"   - {key}")

    if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
        print("✓ All checkpoint keys loaded successfully (exact match)")

    encoder.eval()
    encoder = encoder.to(args.device)
    
    # Freeze parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    print("✓ Checkpoint loaded successfully")

    # Create online dataset with loaded encoder
    online_dataset = OnlineDataset(dataset_args, encoder)
    online_dataset.load()  # This generates embeddings and stores them in online_dataset.X

    print(f"✓ Embeddings generated: {online_dataset.X.shape}, Labels: {online_dataset.y.shape}")

    # Train linear probe using DataModule (DataModule will be created inside the function)
    trainer, model, test_results = train_linear_probe_with_datamodule(None, online_dataset, dataset_args, args)
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Final validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
    print(f"Test results: {test_results}")
    print(f"Embedding shape: {online_dataset.X.shape}")
    print(f"Unique labels: {np.unique(online_dataset.y)}")
    
    # Check if optimization worked
    val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
    if val_loss < 0.5:
        print("✅ Linear probe optimization SUCCESS - Loss < 0.5")
    else:
        print("❌ Linear probe optimization FAILED - Loss >= 0.5")
        print("   Consider adjusting hyperparameters or checking representations")


if __name__ == "__main__":
    main()