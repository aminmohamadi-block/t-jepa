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
from src.configs import build_parser
from src.utils.log_utils import make_job_name
from src.utils.log_utils import print_args
from src.utils.checkpointer import EarlyStopCounter
from src.utils.train_utils import init_weights, get_distributed_dataloader
from src.utils.optim_utils import init_optim

from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP


def main(args):

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

    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
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
    )

    for m in context_encoder.modules():
        init_weights(m, init_type=args.init_type)

    if args.pred_type == "mlp":
        for pred in predictors.predictors:
            for m in pred.modules():
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
    )

    print("[Debug] Building DataLoader …", flush=True)

    if args.mp_distributed:
        # Use a DistributedSampler-backed DataLoader so that each rank gets a shard
        dataloader = get_distributed_dataloader(
            batchsize=args.batch_size,
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

def setup_mlflow_logging() -> None:
    """
    Setup MLflow logging for experiment tracking.
    
    Configures Databricks MLflow integration with proper authentication
    and experiment organization.
    
    Args:
        config_dict: Configuration dictionary containing run settings
    """
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
        
        # Set experiment
        project_name = "t-jepa-test"
        mlflow.set_experiment(f"/groups/block-aird-team/{project_name}")
        
        print(f"✓ MLflow logging configured for project: {project_name}")
        
    except ImportError:
        print("⚠️  MLflow not available - skipping experiment tracking setup")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    setup_mlflow_logging()
    main(args)
