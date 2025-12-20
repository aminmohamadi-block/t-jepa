from typing import TypedDict
from tqdm import tqdm
from src.datasets.base import ArgsDataset, BaseDataset
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch
import numpy as np

from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
from scipy.io import arff
import os
import numpy as np
import pandas as pd

from src.datasets.base import BaseDataset
from src.torch_dataset import TorchDataset
from src.utils.encode_utils import encode_data
from src.utils.models_utils import TASK_TYPE
from src.utils.train_utils import apply_masks


class OnlineDatasetArgs(TypedDict):
    data_set: str
    data_path: str
    batch_size: int
    data_loader_nprocs: int
    pin_memory: bool
    mock: bool
    test_size_ratio: float
    random_state: int
    val_size_ratio: float
    full_dataset_cuda: bool
    val_batch_size: int
    input_embed_dim: int


class OnlineDataset(BaseDataset):

    def __init__(
        self,
        args: OnlineDatasetArgs,
        encoder: nn.Module,
    ):
        super().__init__(args)

        self.dataset: BaseDataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
        self.encoder = encoder
        self.args = args

    def load(self):
        if self.is_data_loaded:
            return
        self.dataset.load()

        # For large datasets, use only last 10% for linear probe to reduce memory
        # Data is unshuffled, so this is a deterministic temporal split
        probe_sample_fraction = getattr(self.args, 'probe_sample_fraction', 0.1)
        if probe_sample_fraction < 1.0:
            n_total = self.dataset.N
            n_probe = int(n_total * probe_sample_fraction)
            start_idx = n_total - n_probe  # Take from the end

            print(f"Using last {probe_sample_fraction*100:.0f}% of data for linear probe: "
                  f"{n_probe:,} samples (from index {start_idx:,} to {n_total:,})")

            # Slice the last 10% from the unshuffled dataset
            self.dataset.X = self.dataset.X[start_idx:]
            self.dataset.y = self.dataset.y[start_idx:]
            self.dataset.N = n_probe

        self.target = self.dataset.y
        self.y = self.dataset.y
        self.task_type = self.dataset.task_type

        self.D = self.dataset.D
        self.N = self.dataset.N
        self.H = self.args.input_embed_dim
        self.cardinalities = self.dataset.cardinalities
        self.num_or_cat = self.dataset.num_or_cat
        self.cat_features = self.dataset.cat_features
        self.num_features = self.dataset.num_features
        self.is_data_loaded = True

        # Place the temporary TorchDataset and its batches on **the same device**
        # as the frozen encoder. This avoids device-mismatch errors when each
        # distributed rank runs on a different GPU.
        device = next(self.encoder.parameters()).device
        train_torchdataset = TorchDataset(
            dataset=self.dataset,
            mode="train",
            kwargs=self.args,
            device=device,
            preprocessing=encode_data,
        )

        dataloader = DataLoader(
            dataset=train_torchdataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.data_loader_nprocs,
            collate_fn=None,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )

        total_z = []
        print(f"Generating embedding using: {self.encoder.__class__.__name__}")
        for batch, _ in tqdm(dataloader):
            batch = batch.to(device)
            z = self.encoder(batch)

            # Remove REG tokens if present (they are at the end of the sequence)
            if hasattr(self.args, 'n_reg_tokens') and self.args.n_reg_tokens > 0:
                z = z[:, :-self.args.n_reg_tokens, :]

            # Flatten embeddings to 2D for linear probe: [batch, tokens * hidden_dim]
            z = z.reshape(z.size(0), -1)

            total_z.append(z.detach().cpu().numpy())

        self.X = np.concatenate(total_z, axis=0)
