import os, copy
import random


import numpy as np
from tabulate import tabulate
import mlflow
import torch
from typing import cast
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.utils.encode_utils import encode_data
import src.utils.idr_torch as idr_torch  # JEAN-ZAY

from src.encoder import Encoder
from src.predictors import Predictors
from src.torch_dataset import TorchDataset
from src.train import Trainer
from src.mask import MaskCollator
from src.mask_vectorized import UltraVectorizedMaskCollator
from src.configs import build_parser
from src.utils.log_utils import make_job_name
from src.utils.log_utils import print_args
from src.utils.checkpointer import EarlyStopCounter
from src.utils.train_utils import init_weights, get_distributed_dataloader
from src.utils.optim_utils import init_optim

from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
from src.datasets.parquet_dataset import create_parquet_dataset_from_args
from src.utils.profiler import get_profiler, init_profiler, ProfilingLevel


def main(args):

    # Initialize profiler based on command line args
    # Use init_profiler() which returns NoOpProfiler for DISABLED level (zero overhead)
    if hasattr(args, 'profiling_level'):
        try:
            profiler = init_profiler(ProfilingLevel[args.profiling_level])
        except KeyError:
            print(f"[Profiling] Warning: Invalid profiling level '{args.profiling_level}', using DISABLED")
            profiler = init_profiler(ProfilingLevel.DISABLED)
    else:
        profiler = init_profiler(ProfilingLevel.DISABLED)

    if args.mp_distributed:
        # ------------------------------------------------------------------
        # Debugging information BEFORE initializing the process group
        # ------------------------------------------------------------------
        print(
            "[Distributed pre-init] RANK={} WORLD_SIZE={} LOCAL_RANK={} MASTER_ADDR={} MASTER_PORT={}".format(
                os.environ.get("RANK", "unset"),
                os.environ.get("WORLD_SIZE", "unset"),
                os.environ.get("LOCAL_RANK", "unset"),
                os.environ.get("MASTER_ADDR", "unset"),
                os.environ.get("MASTER_PORT", "unset"),
            )
        )

        torch.cuda.set_device(idr_torch.local_rank)

        # Initialize the default process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=idr_torch.size,
            rank=idr_torch.rank,
        )

        print(f"[Debug] idr_torch.world_size={idr_torch.size}", flush=True)
        print(f"[Debug] idr_torch.rank={idr_torch.rank}", flush=True)
        print(f"[Debug] idr_torch.local_rank={idr_torch.local_rank}", flush=True)
        print(f"[Debug] idr_torch.gpu_ids={idr_torch.gpu_ids}", flush=True)
        print(f"[Debug] idr_torch.cpus_per_task={idr_torch.cpus_per_task}", flush=True)

        # ------------------------------------------------------------------
        # Debugging information AFTER initializing the process group
        # ------------------------------------------------------------------
        print(
            "[Distributed post-init] rank {} / {} | backend={} | current_device={}".format(
                dist.get_rank(),
                dist.get_world_size(),
                dist.get_backend(),
                torch.cuda.current_device(),
            ),
            flush=True,
        )

        # Determine the local GPU for this rank and bind the process to it
        # Prefer idr_torch.local_rank if available, otherwise fall back to the
        # LOCAL_RANK environment variable set by torchrun.
        local_rank = (
            idr_torch.local_rank if hasattr(idr_torch, "local_rank") else int(os.environ.get("LOCAL_RANK", 0))
        )
        torch.cuda.set_device(local_rank)

        print(
            f"[Debug] torch.cuda.set_device({local_rank}) called on PID {os.getpid()}",
            flush=True,
        )

        distributed_args = {
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
            "gpu": local_rank,
        }

        print(f"[Debug] distributed_args={distributed_args}", flush=True)
    else:
        distributed_args = None

    ema_start = args.model_ema_start
    ema_end = args.model_ema_end
    num_epochs = args.exp_train_total_epochs
    ipe_scale = args.exp_ipe_scale

    # ------------------------------------------------------------------
    # Dataset Loading: Conditional based on use_parquet_dataset flag
    # ------------------------------------------------------------------
    if args.use_parquet_dataset:
        print("[Debug] Using parquet dataset from LocalFilesDataset", flush=True)
        dataset = create_parquet_dataset_from_args(args)
        print(f"[Debug] Parquet dataset created: N={dataset.N}, D={dataset.D}", flush=True)
        # For parquet datasets, we use the dataset directly as an iterable
        # No need for TorchDataset wrapper since LocalFilesDataset already handles batching
        use_torch_dataset_wrapper = False
    else:
        print("[Debug] Using benchmark dataset (CSV/ARFF)", flush=True)
        dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
        use_torch_dataset_wrapper = True

    args.is_batchlearning = args.batch_size != -1
    args.iteration = 0
    start_epoch = 0
    if args.test:
        args.mock = True

    if (not args.mp_distributed) or (args.mp_distributed and idr_torch.local_rank == 0):
        if args.verbose:
            print_args(args)

    if args.random:
        args.torch_seed = np.random.randint(0, 100000)
        args.np_seed = np.random.randint(0, 100000)

    torch.manual_seed(args.torch_seed)
    np.random.seed(args.np_seed)
    random.seed(args.np_seed)

    jobname = make_job_name(args)

    print(tabulate(vars(args).items(), tablefmt="fancy_grid"))

    if args.mp_distributed:
        # We have already set the appropriate device above via torch.cuda.set_device
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[Debug] Loading dataset …", flush=True)
    if use_torch_dataset_wrapper:
        # Benchmark datasets: load and wrap with TorchDataset
        dataset.load()
        print("[Debug] Dataset loaded", flush=True)
        args.test_size = 0
        train_torchdataset = TorchDataset(
            dataset=dataset,
            mode="train",
            kwargs=args,
            device=device,
            preprocessing=encode_data,
        )
    else:
        # Parquet datasets: already loaded and batched in LocalFilesDataset
        print("[Debug] Parquet dataset ready (preprocessing handled by LocalFilesDataset)", flush=True)
        train_torchdataset = dataset  # Use directly as iterable

    context_encoder = Encoder(
        idx_num_features=dataset.num_features,
        cardinalities=dataset.cardinalities,
        hidden_dim=args.model_dim_hidden,
        num_layers=args.model_num_layers,
        num_heads=args.model_num_heads,
        p_dropout=args.model_dropout_prob,
        layer_norm_eps=args.model_layer_norm_eps,
        gradient_clipping=args.exp_gradient_clipping,
        feature_type_embedding=args.model_feature_type_embedding,
        feature_index_embedding=args.model_feature_index_embedding,
        dim_feedforward=args.model_dim_feedforward,
        device=device,
        args=args,
    )

    predictors = Predictors(
        pred_type=args.pred_type,
        hidden_dim=args.model_dim_hidden,
        pred_embed_dim=args.pred_embed_dim,
        num_features=dataset.D,
        num_layers=args.pred_num_layers,
        num_heads=args.pred_num_heads,
        p_dropout=args.pred_p_dropout,
        layer_norm_eps=args.pred_layer_norm_eps,
        activation=args.pred_activation,
        device=device,
        cardinalities=dataset.cardinalities,
        pred_dim_feedforward=args.pred_dim_feedforward,
        n_cls_tokens=args.n_cls_tokens,
    )

    for m in context_encoder.modules():
        init_weights(m, init_type=args.init_type)

    if args.pred_type == "mlp":
        for pred in predictors.predictors:
            for m in pred.modules():
                init_weights(m, init_type=args.init_type)
    else:
        for m in predictors.predictors.modules():
            init_weights(m, init_type=args.init_type)

    target_encoder = copy.deepcopy(context_encoder)

    context_encoder.to(device)
    target_encoder.to(device)
    predictors.to(device)

    # ------------------------------------------------------------------
    # Wrap trainable modules with DistributedDataParallel so that gradients
    # are averaged across GPUs. Only needed when mp_distributed is enabled.
    # ------------------------------------------------------------------
    if args.mp_distributed:
        context_encoder = torch.nn.parallel.DistributedDataParallel(
            context_encoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        predictors = torch.nn.parallel.DistributedDataParallel(
            predictors,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        print("[Debug] Wrapped models in DistributedDataParallel", flush=True)

    scaler = GradScaler(enabled=args.model_amp)
    if args.model_amp:
        print(f"Initialized gradient scaler for Automatic Mixed Precision.")

    early_stop_counter = EarlyStopCounter(
        args, jobname, args.data_set, device=device, is_distributed=False
    )

    # Choose mask collator based on configuration
    # For Parquet datasets, use GPU mask generation (~100x faster)
    mask_device = device if args.use_parquet_dataset else None

    if args.use_vectorized_masking:
        if mask_device and 'cuda' in str(mask_device):
            print("[Optimization] Using UltraVectorizedMaskCollator with GPU mask generation (~100x faster)")
        else:
            print("[Optimization] Using UltraVectorizedMaskCollator (16.9x faster)")
        mask_collator = UltraVectorizedMaskCollator(
            args.mask_allow_overlap,
            args.mask_min_ctx_share,
            args.mask_max_ctx_share,
            args.mask_min_trgt_share,
            args.mask_max_trgt_share,
            args.mask_num_preds,
            args.mask_num_encs,
            dataset.D,
            dataset.cardinalities,
            args.n_cls_tokens,
            device=mask_device,  # NEW: Pass device for GPU mask generation
        )
    else:
        print("[Optimization] Using original MaskCollator")
        mask_collator = MaskCollator(
            args.mask_allow_overlap,
            args.mask_min_ctx_share,
            args.mask_max_ctx_share,
            args.mask_min_trgt_share,
            args.mask_max_trgt_share,
            args.mask_num_preds,
            args.mask_num_encs,
            dataset.D,
            dataset.cardinalities,
            args.n_cls_tokens,
        )

    print("[Debug] Building DataLoader …", flush=True)

    if args.use_parquet_dataset:
        # Parquet datasets: LocalFilesDataset already batches data
        # OPTIMIZATION: Use generate_masks_only() to avoid expensive tensor split/restack
        class ParquetDataLoaderWrapper:
            """
            Optimized wrapper for pre-batched Parquet data.

            Key optimization: The batch is already a single tensor [batch_size, num_features].
            We generate masks directly without splitting the batch into individual samples
            and re-stacking them. This saves ~20-30ms per iteration.

            OLD (slow):
                batch_list = [(batch[i], None) for i in range(4096)]  # Split into 4096 tensors
                batch, masks = collator(batch_list)  # Re-stack via default_collate

            NEW (fast):
                masks = collator.generate_masks_only(batch_size)  # Just generate masks
                # Batch stays as-is
            """

            def __init__(self, dataset, mask_collator, device, num_features):
                self.dataset = dataset
                self.mask_collator = mask_collator
                self.device = device
                self.num_features = num_features
                # Check if mask_collator supports optimized path
                self.use_optimized = hasattr(mask_collator, 'generate_masks_only')
                if self.use_optimized:
                    print("[Optimization] ParquetDataLoaderWrapper using generate_masks_only()")
                else:
                    print("[Warning] MaskCollator doesn't support generate_masks_only(), using slow path")

            def __iter__(self):
                for batch in self.dataset:
                    # batch is already a tensor [batch_size, num_features]
                    batch_size = len(batch)

                    if self.use_optimized:
                        # FAST PATH: Generate masks without touching the batch
                        # Saves ~20-30ms per iteration by avoiding tensor split/restack
                        masks_enc, masks_pred = self.mask_collator.generate_masks_only(
                            batch_size, self.num_features
                        )
                        # Batch stays as a single tensor - no conversion needed
                    else:
                        # SLOW PATH: For backwards compatibility with old mask collators
                        batch_list = [(batch[i], None) for i in range(batch_size)]
                        batch, masks_enc, masks_pred = self.mask_collator(batch_list)

                    yield batch, masks_enc, masks_pred

            def __len__(self):
                return len(self.dataset)

        dataloader = ParquetDataLoaderWrapper(
            train_torchdataset, mask_collator, device, num_features=dataset.D
        )
        print(f"[Debug] Parquet DataLoader created (pre-batched, batch_size={args.batch_size})")

    elif args.mp_distributed:
        # Use a DistributedSampler-backed DataLoader so that each rank gets a shard
        # Divide batch size by world size to maintain consistent effective batch size
        per_gpu_batch_size = args.batch_size // distributed_args["world_size"]
        print(f"[Debug] Per-GPU batch size: {per_gpu_batch_size}, Total effective batch size: {args.batch_size}")
        dataloader = get_distributed_dataloader(
            batchsize=per_gpu_batch_size,
            dataset=train_torchdataset,
            distributed_args=cast(dict, distributed_args),
            data_loader_nprocs=args.data_loader_nprocs,
            mask_collator=mask_collator,
            pin_memory=args.pin_memory,
        )
    else:
        dataloader = DataLoader(
            dataset=train_torchdataset,
            batch_size=args.batch_size,
            num_workers=args.data_loader_nprocs,
            collate_fn=mask_collator,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

    print("[Debug] DataLoader built", flush=True)

    ipe = len(dataloader)
    print(f"[Debug] ipe (iterations per epoch) = {ipe}", flush=True)

    (optimizer, scheduler, weightdecay_scheduler) = init_optim(
        context_encoder,
        predictors,
        ipe,
        args.exp_start_lr,
        args.exp_lr,
        args.exp_warmup,
        args.exp_train_total_epochs,
        args.exp_weight_decay,
        args.exp_final_weight_decay,
        args.exp_final_lr,
        args.exp_ipe_scale,
        args.exp_scheduler,
        args.exp_weight_decay_scheduler,
    )

    momentum_scheduler = (
        ema_start + i * (ema_end - ema_start) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    if args.load_from_checkpoint:
        if os.path.isfile(args.load_path):
            (
                context_encoder,
                predictors,
                target_encoder,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
            ) = early_stop_counter.load_model(
                load_pth=args.load_path,
                context_encoder=context_encoder,
                predictor=predictors,
                target_encoder=target_encoder,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                weightdecay_scheduler=weightdecay_scheduler,
            )
            # Retrieve the epoch we resumed from
            checkpoint_state = torch.load(args.load_path, map_location="cpu")
            start_epoch = int(checkpoint_state.get("epoch", 0))
            for _ in range(start_epoch * ipe):
                next(momentum_scheduler)
                mask_collator.step()
        else:
            print(
                "Tried loading from checkpoint,"
                " but provided path does not exist."
                " Starting training from scratch."
            )

    # Always freeze target encoder parameters (critical for T-JEPA training)
    for p in target_encoder.parameters():
        p.requires_grad = False

    print("[Debug] Instantiating Trainer", flush=True)

    trainer = Trainer(
        args=args,
        start_epoch=start_epoch,
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        scheduler=scheduler,
        weightdecay_scheduler=weightdecay_scheduler,
        early_stop_counter=early_stop_counter,
        momentum_scheduler=momentum_scheduler,
        optimizer=optimizer,
        scaler=scaler,
        torch_dataset=train_torchdataset,
        dataloader=dataloader,
        distributed_args=cast(dict, distributed_args),
        device=device,
        probe_cadence=args.probe_cadence,
        probe_model=args.probe_model,
    )

    print("[Debug] Trainer instantiated", flush=True)

    print("Starting training…", flush=True)
    trainer.train()
    
    # Output final validation score for Optuna
    if hasattr(trainer, 'early_stop_counter') and hasattr(trainer.early_stop_counter, 'best_val_score'):
        best_score = trainer.early_stop_counter.best_val_score
        print(f"OPTUNA_SCORE: {best_score}")
        print(f"Best validation score: {best_score}")
        
        # Also save to file for SLURM jobs
        if hasattr(args, 'optuna_output_dir') and args.optuna_output_dir:
            score_file = os.path.join(args.optuna_output_dir, f"trial_{args.optuna_trial_number}_score.txt")
            with open(score_file, 'w') as f:
                f.write(f"OPTUNA_SCORE: {best_score}\n")

def setup_mlflow_logging(args) -> None:
    """
    Setup MLflow logging for experiment tracking.

    Configures Databricks MLflow integration with proper authentication
    and experiment organization.

    Set SKIP_MLFLOW=1 to disable MLflow setup entirely.
    """
    # Check if MLflow should be skipped
    if os.environ.get("SKIP_MLFLOW") == "1":
        print("⚠️  SKIP_MLFLOW=1 - skipping MLflow setup")
        return

    try:
        import mlflow
        from mlflow import MlflowClient

        # Setup Databricks connection
        os.environ["DATABRICKS_HOST"] = "https://block-lakehouse-production.cloud.databricks.com"
        
        # Handle authentication
        if os.environ.get("DATABRICKS_TOKEN") is None:
            if os.environ.get("DATABRICKS_TOKEN_MINE"):
                os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_TOKEN_MINE"]
                print("✓ Using DATABRICKS_TOKEN_MINE for authentication")
            else:
                print("⚠️  Warning: DATABRICKS_TOKEN not set - MLflow logging may fail")
        
        # Configure MLflow
        mlflow.set_tracking_uri(uri="databricks")
        
        # Set experiment using provided project name
        project_name = args.project_name
        mlflow.set_experiment(f"/groups/block-aird-team/{project_name}")
        
        print(f"✓ MLflow logging configured for project: {project_name}")
        
    except ImportError:
        print("⚠️  MLflow not available - skipping experiment tracking setup")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    setup_mlflow_logging(args)
    main(args)
