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
    
    args = parser.parse_args()
    
    # Extract hidden_dim from checkpoint first to set input_embed_dim
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    target_encoder_state = checkpoint['target_encoder']
    
    # Get hidden_dim from tokenizer weights
    hidden_dim = None
    for key, tensor in target_encoder_state.items():
        if 'tokenizer.weight' in key:
            _, hidden_dim = tensor.shape
            break
    
    if hidden_dim is None:
        hidden_dim = 64  # fallback
    
    print(f"Detected hidden_dim from checkpoint: {hidden_dim}")
    
    # Load dataset args
    dataset_args = OnlineDatasetArgs(
        data_set=args.dataset_name,
        data_path=args.data_path,
        batch_size=args.embedding_batch_size,
        data_loader_nprocs=4,
        pin_memory=True,
        mock=False,
        test_size_ratio=0.0,
        random_state=args.random_state,
        val_size_ratio=0.0,
        full_dataset_cuda=False,
        val_batch_size=args.batch_size,
        input_embed_dim=hidden_dim  # Set from checkpoint like train.py
    )
    
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
    num_heads = None
    
    for key, tensor in target_encoder_state.items():
        if 'encoder.transformer_layers.linear1.weight' in key:
            dim_feedforward, _ = tensor.shape
        elif 'blocks.' in key:
            layer_num = int(key.split('.')[1])
            num_layers = max(num_layers, layer_num + 1)
        elif '.attn.q_proj.weight' in key and num_heads is None:
            attn_dim = tensor.shape[0]
            num_heads = attn_dim // hidden_dim if hidden_dim else 2
    
    print(f"Extracted from checkpoint: hidden_dim={hidden_dim}, feedforward={dim_feedforward}, layers={num_layers}, heads={num_heads}")
    
    # Load the dataset to get proper model architecture
    # Convert OnlineDatasetArgs to a namespace the dataset expects
    dataset_namespace = Namespace(**dataset_args)
    dataset_class = DATASET_NAME_TO_DATASET_MAP[args.dataset_name]
    dataset = dataset_class(dataset_namespace)
    dataset.load()
    
    # Create encoder using extracted architecture
    encoder = Encoder(
        idx_num_features=dataset.num_features,
        cardinalities=dataset.cardinalities,
        hidden_dim=hidden_dim or 64,
        num_layers=num_layers or 16,
        num_heads=num_heads or 2,
        p_dropout=0.0,
        layer_norm_eps=1e-5,
        gradient_clipping=1.0,
        feature_type_embedding=False,
        feature_index_embedding=False,
        dim_feedforward=dim_feedforward or 256,
        device=args.device,
        args=Namespace(n_cls_tokens=1, model_act_func='relu')
    )
    
    # Load the checkpoint state dict  
    encoder.load_state_dict(checkpoint['target_encoder'], strict=False)
    encoder.eval()
    encoder = encoder.to(args.device)
    
    # Freeze parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    print("✓ Checkpoint loaded successfully")
    
    # Create online dataset with loaded encoder
    online_dataset = OnlineDataset(dataset_namespace, encoder)
    online_dataset.load()  # This generates embeddings and stores them in online_dataset.X
    
    print(f"✓ Embeddings generated: {online_dataset.X.shape}, Labels: {online_dataset.y.shape}")
    
    # Train linear probe using DataModule (DataModule will be created inside the function)
    trainer, model, test_results = train_linear_probe_with_datamodule(None, online_dataset, dataset_namespace, args)
    
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