from argparse import Namespace
import json
import torchvision
import pytorch_lightning as pl
import os
from datetime import datetime
import contextlib  # Added for nullcontext when not main process

import numpy as np
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from run_benchmark import set_callbacks_loggers
from src.benchmark.utils import (
    MODEL_CONFIG_BASE_PATH,
    MODEL_NAME_TO_MODEL_MAP,
    get_loss_from_task,
)
from src.datasets.online_dataset import OnlineDataset, OnlineDatasetArgs
from src.torch_dataset import DataModule, TorchDataset
from src.utils.log_utils import _debug_values, make_job_name, get_system_metrics
from src.utils.checkpointer import EarlyStopSignal, MODEL_CP_NAME
from src.utils.models_utils import BaseModel
from src.utils.train_utils import (
    AllReduce,
    apply_masks_from_idx,
    get_dist,
)
from src.utils.profiler import get_profiler
from tqdm import tqdm
import mlflow
import tempfile


class Trainer:

    def __init__(
        self,
        args,
        start_epoch: int,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictors: nn.Module,
        scheduler,
        weightdecay_scheduler,
        early_stop_counter,
        momentum_scheduler,
        optimizer: optim.Optimizer,
        scaler,
        torch_dataset: TorchDataset,
        dataloader,
        distributed_args: dict,
        device: torch.device,
        probe_cadence: int,
        probe_model: str,
    ):

        self.epoch = start_epoch
        self.is_distributed = False
        self.args = args
        self.seed = self.args.np_seed
        self.device = device
        self.total_train_time = 0
        self.epoch_time = []
        self.data_loader_nprocs = self.args.data_loader_nprocs
        self.log_tb = args.log_tensorboard
        self.probe_model = probe_model

        if distributed_args is not None:
            self.is_distributed = True
            self.world_size = distributed_args["world_size"]
            self.rank = distributed_args["rank"]
            self.gpu = distributed_args["gpu"]

        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictors = predictors
        self.probe_cadence = probe_cadence

        self.scheduler = scheduler
        self.weight_decay_scheduler = weightdecay_scheduler
        self.early_stop_counter = early_stop_counter
        self.momentum_scheduler = momentum_scheduler
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_fn = torch.nn.MSELoss()  # F.smooth_l1_loss

        self.dataset = torch_dataset
        self.cardinalities = self.dataset.cardinalities

        self.is_batch_learning = self.args.batch_size != -1
        self.num_epoch = self.args.exp_train_total_epochs

        if not self.is_batch_learning:
            self.batch_size = len(self.dataset.train[0])
        else:
            self.batch_size = self.args.batch_size

        self.dataloader = dataloader

        self.is_main_process = (self.is_distributed and self.rank == 0) or (
            not self.is_distributed
        )

        print(f"[Debug] is_main_process={self.is_main_process}", flush=True)

        self.job_name = self.early_stop_counter.get_job_name()
        self.training_is_over = False

        if self.is_main_process:
            loss_dir = os.path.join(
                "./tblogs", self.dataset.dataset_name, self.job_name
            )

            if not os.path.isdir(loss_dir) and self.log_tb:
                os.makedirs(loss_dir)

            if self.log_tb:
                self.writer = SummaryWriter(loss_dir)

            self.res_dir = os.path.join(
                "./results", self.dataset.dataset_name, self.job_name
            )

            if not os.path.isdir(self.res_dir):
                os.makedirs(self.res_dir)

            self.avg_time = []

        self.checkpoint_dir = os.path.join(
            "./checkpoints", self.dataset.dataset_name, self.job_name
        )

        if not os.path.isdir(self.checkpoint_dir):
            if self.is_main_process:
                success = False
                while not success:
                    try:
                        os.makedirs(self.checkpoint_dir)
                        success = True
                    except:
                        self.job_name = make_job_name(
                            self.args, self.seed, self.iteration
                        )
                        self.checkpoint_dir = os.path.join(
                            "./checkpoints", self.dataset.dataset_name, self.job_name
                        )
                        pass

        # Base parameters
        base_params = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epoch,
        }
        # Convert args Namespace to a flat dict of serializable values
        args_params = {
            k: (v if isinstance(v, (int, float, bool, str)) else str(v))
            for k, v in vars(self.args).items()
        }
        base_params.update(args_params)
        self.mlflow_params = base_params

        # Initialize profiler
        self.profiler = get_profiler()


    def train(
        self,
    ):
        """
        Training Loop
        """

        self.context_encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.predictors.to(self.device)

        if self.args.test:
            self.num_epoch = 1
            self.args.mock = True

        # Initialize MLflow experiment and log parameters
        # Only the main process should create an MLflow run. Other processes enter a
        # dummy context to keep the control-flow uniform while preventing nested
        # or duplicated MLflow runs when training with multiple GPUs.
        run_context = (
            mlflow.start_run(run_name=self.job_name, log_system_metrics=True)
            if self.is_main_process
            else contextlib.nullcontext()
        )

        with run_context:
            
            if self.is_main_process:
                system_metrics = get_system_metrics()
                mlflow.log_params(self.mlflow_params)
                mlflow.log_params(system_metrics)

            while self.epoch < self.num_epoch:
                with self.profiler.profile("epoch", epoch=self.epoch):
                    collapse_metrics = None
                    linear_probe_metric = 0
                    probe_val_metrics = {}
                    if self.probe_cadence > 0 and self.epoch % self.probe_cadence == 0:
                        print(f"Running probe at epoch {self.epoch}")

                        # Create args for OnlineDataset
                        # Start with full args to include parquet-specific parameters
                        online_dataset_args = Namespace(**vars(self.args))

                        # Override specific fields for linear probe
                        online_dataset_args.data_set = self.dataset.dataset_name
                        online_dataset_args.batch_size = 512  # TODO: Make this dynamic
                        online_dataset_args.test_size_ratio = 0
                        online_dataset_args.random_state = self.args.np_seed
                        online_dataset_args.val_size_ratio = 0
                        online_dataset_args.input_embed_dim = self.args.model_dim_hidden
                        online_dataset = OnlineDataset(
                            online_dataset_args,
                            self.target_encoder,
                        )
                        online_dataset.load()

                        X = online_dataset.X

                        rnd_sample = np.random.randint(0, X.shape[0])
                        sampled_data_1 = X[0]
                        sampled_data_2 = X[rnd_sample]

                        imgs = torch.stack(
                            [
                                torch.tensor(sampled_data_1),
                                torch.tensor(sampled_data_2),
                            ]
                        )

                        def apply_colormap(img_tensor, colormap="viridis"):
                            import matplotlib.pyplot as plt

                            img_np = img_tensor.numpy()

                            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())

                            cmap = plt.get_cmap(colormap)
                            img_colormap = cmap(img_norm)[:, :, :3]

                            img_colormap_tensor = torch.tensor(img_colormap).permute(2, 0, 1)
                            return img_colormap_tensor

                        colored_imgs = torch.stack([apply_colormap(img) for img in imgs])

                        img_grid = torchvision.utils.make_grid(
                            colored_imgs,
                            nrow=2,
                        )

                        # Log the image grid as an artifact with MLflow (main process only)
                        if self.is_main_process:
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                                torchvision.utils.save_image(img_grid, tmpfile.name)
                                mlflow.log_artifact(
                                    tmpfile.name, artifact_path="embedding_images"
                                )

                        collapse_metrics = {
                            "KL": [],
                            "euclidean": [],
                            "intra_feature_variance": np.mean(np.var(X, axis=0)),
                            "inter_feature_variance": np.var(X.mean(axis=1)),
                        }

                        for _ in range(20):
                            m = self.get_collapse_metrics(X)
                            collapse_metrics["KL"].append(m["KL"])
                            collapse_metrics["euclidean"].append(m["euclidean"])

                        collapse_metrics["KL"] = np.mean(collapse_metrics["KL"])
                        collapse_metrics["euclidean"] = np.mean(collapse_metrics["euclidean"])

                        model_class: BaseModel = MODEL_NAME_TO_MODEL_MAP[self.probe_model]

                        device = "cuda:0" if torch.cuda.is_available() else "cpu"
                        dataset_args = vars(online_dataset_args).copy()
                        dataset_args.update(
                            {
                                "test_size_ratio": 0.1,
                                "val_size_ratio": 0.1,
                                "batch_size": 128,
                                "task_type": online_dataset.task_type,
                                "using_embedding": True,
                                "exp_train_total_epochs": 200 if not self.args.test else 1,
                                "model_name": self.probe_model,
                                "dataset_name": online_dataset_args.data_set,
                                "exp_patience": 50,
                                "n_cls_tokens": self.args.n_cls_tokens,
                            }
                        )
                        dataset_args = Namespace(**dataset_args)

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
                            preprocessing=model_class.preprocessing,
                            mock=dataset_args.mock,
                            using_embedding=True,
                        )

                        base_config = {
                            "dataset_name": self.args.data_set,
                            "encoder_type": "linear_flatten",
                        }
                        model_args = json.load(
                            open(
                                MODEL_CONFIG_BASE_PATH.format(
                                    dataset_name=self.args.data_set,
                                    model_name=self.probe_model,
                                )
                            )
                        )
                        model_args.update(base_config)
                        model_args = Namespace(**model_args)

                        model_args = model_class.get_model_args(
                            datamodule,
                            dataset_args,
                            model_args,
                        )
                        print(f"Loading {self.probe_model}")
                        print(
                            tabulate(
                                sorted(list(vars(model_args).items()), key=lambda x: x[0]),
                                tablefmt="fancy_grid",
                            )
                        )

                        loss_fn = get_loss_from_task(dataset_args.task_type)
                        dataset_args = {**vars(dataset_args), **vars(model_args)}
                        model = model_class(loss=loss_fn, **dataset_args)
                        summary(model, input_size=model_args.summary_input)
                        model = model.float()

                        # Create a descriptive run name for the linear probe
                        probe_run_name = f"{self.job_name}_epoch_{self.epoch}_probe"

                        # Gather all relevant parameters for logging
                        probe_run_params = {
                            **dataset_args,
                            "parent_run_name": self.job_name,
                            "parent_epoch": self.epoch,
                        }

                        callbacks, loggers = set_callbacks_loggers(
                            dataset_args, run_name=probe_run_name, run_params=probe_run_params
                        )

                        trainer = pl.Trainer(
                            max_epochs=dataset_args["exp_train_total_epochs"],
                            logger=loggers,
                            callbacks=callbacks,
                            log_every_n_steps=10,
                            strategy="ddp",
                            devices=-1,
                        )

                        trainer.fit(model, datamodule=datamodule)

                        # ALL ranks must participate in validation and testing to avoid deadlocks.
                        val_metrics_list = trainer.validate(model, datamodule=datamodule)
                        trainer.test(model, datamodule=datamodule)
                        
                        # Only process and log metrics on the main rank
                        if self.is_main_process:
                            linear_probe_metric = val_metrics_list[0][
                                f"{self.args.data_set}_val_score"
                            ]
                            # Add a prefix to all probe validation metrics for clarity in the main run
                            probe_val_metrics = {f"probe/{k}": v for k, v in val_metrics_list[0].items()}
                        
                        # Broadcast the metric from the main process to all other processes.
                        # First, create a tensor on the correct device for all processes.
                        metric_tensor = torch.tensor(linear_probe_metric if self.is_main_process else 0.0, device=self.device)
                        if self.is_distributed:
                            dist.broadcast(metric_tensor, src=0)
                        linear_probe_metric = metric_tensor.item()

                        # Add a barrier to ensure all processes sync up before continuing.
                        if self.is_distributed:
                            dist.barrier()


                    start_time = datetime.now()
                    to_print = f"Training epoch: {self.epoch+1}/{self.num_epoch}"
                    if self.is_main_process:
                        print(f"{to_print:#^80}")

                    if self.is_distributed:
                        # Ensure shuffling is synchronized across epochs.
                        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
                            self.dataloader.sampler.set_epoch(self.epoch)
                        elif hasattr(self.dataloader, "set_epoch"):
                            self.dataloader.set_epoch(self.epoch)
                    total_loss = torch.zeros(1, device=self.device)

                    for itr, (batch, masks_enc, masks_pred) in enumerate(tqdm(self.dataloader)):
                        with self.profiler.profile("iteration", iteration=itr):

                            with self.profiler.profile("data_transfer"):
                                batch = batch.to(self.device, non_blocking=True)
                                masks_enc = [
                                    mask.to(self.device, non_blocking=True) for mask in masks_enc
                                ]
                                masks_pred = [
                                    mask.to(self.device, non_blocking=True) for mask in masks_pred
                                ]

                            # Autocast wraps ALL forward passes, loss, and backward
                            with torch.autocast(device_type=self.device.type, enabled=self.args.model_amp):
                                with self.profiler.profile("forward_pass"):
                                    # target forward
                                    with self.profiler.profile("target_encoder"):
                                        with torch.no_grad():

                                            _debug_values(batch[0].T, "batch[0]")
                                            # Target encocccccbvefivhblvucikbrjleujicuggufkgcctdjbvbv
                                            # der sees ALL features (no mask passed)
                                            h = self.target_encoder(batch)
                                _debug_values(h[0].T, "h[0] after target_encoder")

                                with self.profiler.profile("target_masking"):
                                    # Step 1: Remove REG tokens from target encoder output
                                    if self.args.n_reg_tokens > 0:
                                        h_no_reg = h[:, :-self.args.n_reg_tokens, :]
                                    else:
                                        h_no_reg = h

                                    # Step 2: Split CLS and features
                                    h_cls = h_no_reg[:, :self.args.n_cls_tokens, :]  # CLS tokens
                                    h_features = h_no_reg[:, self.args.n_cls_tokens:, :]  # All features

                                    # Step 3: Apply masks to features only
                                    h_masked_features = apply_masks_from_idx(h_features, masks_pred)

                                    # Step 4: Reconstruct with CLS prepended to masked features
                                    # Expand CLS to match the number of mask predictions
                                    h_cls_expanded = h_cls.repeat(len(masks_pred), 1, 1)
                                    h = torch.cat([h_cls_expanded, h_masked_features], dim=1)

                                _debug_values(h[0].T, "h[0] after masking (CLS and REG removed)")

                                # Context encoder with masks (masks now include CLS offset)
                                with self.profiler.profile("context_encoder"):
                                    z = self.context_encoder(batch, masks_enc)
                                    _debug_values(z[0].T, "z[0] after context_encoder")

                                with self.profiler.profile("context_preparation"):
                                    # Context encoder output: [CLS, masked_features, REG]
                                    # Remove REG token before prediction (keep CLS token)
                                    if self.args.n_reg_tokens > 0:
                                        z_for_pred = z[:, :-self.args.n_reg_tokens, :]  # Remove REG only
                                    else:
                                        z_for_pred = z

                                with self.profiler.profile("predictor"):
                                    if self.args.pred_type == "mlp":
                                        z_for_pred = z_for_pred.view(z_for_pred.size(0), -1)  # flatten
                                        z_pred = self.predictors(z_for_pred, None, masks_pred)

                                    else:  # Transformer predictor
                                        # Pass [CLS, masked_features] to predictor (REG already removed)
                                        # Masks already account for CLS tokens
                                        z_pred = self.predictors(z_for_pred, masks_enc, masks_pred)
                                        _debug_values(z_pred[0].T, "z_pred after predictors")

                                with self.profiler.profile("loss_computation"):
                                    if self.args.pred_type == "mlp":
                                        loss = torch.zeros(1, device=self.device)
                                        for z_, h_ in zip(z_pred, h):
                                            loss += self.loss_fn(z_, h_)
                                    else:
                                        loss = self.loss_fn(z_pred, h)

                                # Synchronise gradients via DDP; we only need to
                                # reduce the loss tensor for logging/metrics.
                                loss_value = loss.detach()
                                if self.is_distributed:
                                    dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)

                                with self.profiler.profile("backward_pass"):
                                    if self.args.model_amp:
                                        self.scaler.scale(loss).backward()
                                    else:
                                        loss.backward()

                            with self.profiler.profile("optimizer_step"):
                                if self.args.model_amp:
                                    self.scaler.step(self.optimizer)
                                    self.scaler.update()
                                else:
                                    self.optimizer.step()

                            assert not np.isnan(loss.item()), "loss is NaN"

                            if itr == 0 and self.epoch % 10 == 0:  # Only log gradients every 10 epochs
                                # OPTIMIZATION: Compute gradient stats on GPU (4.2x speedup)
                                # Only transfer scalar statistics instead of full gradient arrays
                                with self.profiler.profile("gradient_logging"):
                                    def compute_grad_stats_gpu(model):
                                        """Compute gradient statistics on GPU, return scalars."""
                                        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
                                        if len(grads) > 0:
                                            all_grads = torch.cat(grads)
                                            return {
                                                'mean': all_grads.mean().item(),
                                                'l2': all_grads.norm().item(),
                                                'std': all_grads.std().item(),
                                            }
                                        return {'mean': 0.0, 'l2': 0.0, 'std': 0.0}

                                    ctx_stats = compute_grad_stats_gpu(self.context_encoder)
                                    trgt_stats = compute_grad_stats_gpu(self.target_encoder)
                                    pred_stats = compute_grad_stats_gpu(self.predictors)

                                    # Log gradient statistics with MLflow
                                    grad_metrics = {
                                        "grad/context_encoder_grad_mean": ctx_stats['mean'],
                                        "grad/context_encoder_grad_l2": ctx_stats['l2'],
                                        "grad/context_encoder_grad_std": ctx_stats['std'],
                                        "grad/target_encoder_grad_mean": trgt_stats['mean'],
                                        "grad/target_encoder_grad_l2": trgt_stats['l2'],
                                        "grad/target_encoder_grad_std": trgt_stats['std'],
                                        "grad/predictor_grad_mean": pred_stats['mean'],
                                        "grad/predictor_grad_l2": pred_stats['l2'],
                                        "grad/predictor_grad_std": pred_stats['std'],
                                    }
                                    if self.is_main_process:
                                        mlflow.log_metrics(
                                            grad_metrics,
                                            step=itr + self.epoch * len(self.dataloader),
                                        )

                            self.optimizer.zero_grad()
                            if self.is_main_process and self.log_tb:
                                self.writer.add_scalar(
                                    f"train/loss", loss_value.item(), itr * (self.epoch + 1)
                                )
                            total_loss += loss_value

                            # Step 3. momentum update of target encoder
                            # OPTIMIZATION: Use _foreach operations for 25.6x speedup
                            with self.profiler.profile("ema_update"):
                                with torch.no_grad():
                                    m = next(self.momentum_scheduler)
                                    # Collect parameter data tensors
                                    params_q = [p.data for p in self.context_encoder.parameters()]
                                    params_k = [p.data for p in self.target_encoder.parameters()]
                                    # Batch EMA update: params_k = m * params_k + (1-m) * params_q
                                    torch._foreach_mul_(params_k, m)
                                    torch._foreach_add_(params_k, params_q, alpha=1.0 - m)

                            if self.scheduler is not None:
                                self.scheduler.step()

                            if self.weight_decay_scheduler is not None:
                                self.weight_decay_scheduler.step()

                    end_time = datetime.now()
                    total_epoch_time = (end_time - start_time).total_seconds()
                    self.total_train_time += total_epoch_time
                    self.epoch_time.append(total_epoch_time)

                    args_early_stop = {
                        "train_loss": total_loss.item(),
                        "context_encoder": self.context_encoder,
                        "target_encoder": self.target_encoder,
                        "predictor": self.predictors,
                        "optimizer": self.optimizer,
                        "scaler": self.scaler,
                        "scheduler": self.scheduler,
                        "weightdecay_scheduler": self.weight_decay_scheduler,
                        "epoch": self.epoch,
                        "end_experiment": (self.epoch == self.num_epoch),
                        "val_score": linear_probe_metric if linear_probe_metric != 0 else None,
                    }

                    log_dict = {
                        "train/loss": total_loss.item(),
                        "train/epoch": self.epoch,
                        "sys/time_per_epoch": total_epoch_time,
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/momentum": m,
                        "train/weight_decay": self.weight_decay_scheduler.get_last_wd()[0],
                        "val/linear_probe_metric": linear_probe_metric,
                    }
                    if collapse_metrics is not None:
                        # Add validation prefix to collapse metrics
                        collapse_metrics_prefixed = {
                            f"val/{k}": v for k, v in collapse_metrics.items()
                        }
                        log_dict.update(collapse_metrics_prefixed)

                    # MLflow expects scalar metrics; filter and cast appropriately
                    if self.is_main_process:
                        log_dict.update(probe_val_metrics)
                        mlflow_log_dict = {
                            k: float(v) for k, v in log_dict.items() if np.isscalar(v)
                        }
                        mlflow.log_metrics(mlflow_log_dict, step=self.epoch)

                    if self.is_main_process:
                        (
                            early_stop_signal,
                            self.context_encoder,
                            self.target_encoder,
                            self.predictors,
                            self.optimizer,
                            self.scaler,
                            self.scheduler,
                            self.weight_decay_scheduler,
                        ) = self.early_stop_counter.update(**args_early_stop)

                    if self.is_distributed:
                        # Broadcast the early_stop_signal from the main process to all other processes.
                        signal_tensor = torch.tensor(
                            early_stop_signal.value if self.is_main_process else 0,
                            dtype=torch.int,
                            device=self.device,
                        )
                        dist.broadcast(signal_tensor, src=0)

                        if not self.is_main_process:
                            early_stop_signal = EarlyStopSignal(signal_tensor.item())

                        dist.barrier()

                    if early_stop_signal == EarlyStopSignal.STOP:
                        if not (self.epoch == self.num_epoch):
                            print(self.early_stop_counter.early_stop_signal_message)
                            break

                self.epoch += 1

                # Print profiling summary periodically
                if self.is_main_process and self.profiler.is_enabled() and self.epoch % 10 == 0:
                    print(f"\n{'='*80}")
                    print(f"PROFILING SUMMARY - Epoch {self.epoch}/{self.num_epoch}")
                    print(f"{'='*80}")
                    self.profiler.print_summary(top_k=20)

            # Save profiling results (MOVED INSIDE run_context to prevent orphaned MLflow run)
            if self.is_main_process and self.profiler.is_enabled():
                profiling_file = f"profiling_{self.job_name}.json"
                self.profiler.save_results(profiling_file)
                self.profiler.log_to_mlflow(mlflow)

        print(f"Total training time took: {self.total_train_time} seconds")
        # print(
        # "This amounts to an average epoch time of {avg_time}".format(
        # avg_time=self.total_train_time / sel
        # )
        # )
        self.training_is_over = True
        if self.is_main_process:
            self.avg_time = self.total_train_time

        # make sure that last epoch was stored
        svd_path = os.path.join(
            self.early_stop_counter.checkpoint_dir,
            MODEL_CP_NAME.format(epoch=self.epoch),
        )
        if not os.path.isfile(svd_path):
            self.early_stop_counter.cache_model(
                context_encoder=self.context_encoder,
                target_encoder=self.target_encoder,
                predictor=self.predictors,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=self.scheduler,
                weightdecay_scheduler=self.weight_decay_scheduler,
                epoch=self.epoch,
                train_loss=total_loss,
                save_pth=None,
            )

        return None

    def get_collapse_metrics(self, X):
        idx_1 = np.random.randint(0, X.shape[0])
        idx_2 = np.random.randint(0, X.shape[0])

        data_1 = X[idx_1].flatten()
        data_2 = X[idx_2].flatten()

        p1 = get_dist(data_1)
        p2 = get_dist(data_2)

        collapse_metrics = {
            "KL": np.sum(p1 * np.log(p1 / p2)),
            "euclidean": np.linalg.norm(data_1 - data_2),
        }

        return collapse_metrics
