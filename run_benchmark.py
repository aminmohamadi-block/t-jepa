import cProfile
import datetime
import json
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
import mlflow
from tabulate import tabulate
from torchinfo import summary

from src.benchmark.benchmark_configs import build_parser
from src.benchmark.utils import MODEL_NAME_TO_MODEL_MAP, get_loss_from_task
from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
from src.torch_dataset import DataModule
from src.utils.models_utils import BaseModel
from src.utils.log_utils import get_system_metrics


def train_benchmark_model(args, model_args, profile_name: str):
    if args.test:
        args.mock = True
        args.exp_train_total_epochs = 1

    
    if args.random:
        args.torch_seed = np.random.randint(0, 100000)
        args.np_seed = np.random.randint(0, 100000)
        print("Random seeds: ", args.torch_seed, args.np_seed)
    
    torch.manual_seed(args.torch_seed)
    torch.set_float32_matmul_precision("high")
    np.random.seed(args.np_seed)
    random.seed(args.np_seed)

    print("General args: ")
    print(
        tabulate(
            sorted(list(vars(args).items()), key=lambda x: x[0]),
            tablefmt="fancy_grid",
        )
    )

    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
    dataset.load()
    args.task_type = dataset.task_type
    args.is_batchlearning = args.batch_size != -1
    args.iteration = 0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_class: BaseModel = MODEL_NAME_TO_MODEL_MAP[args.model_name]

    datamodule = DataModule(
        dataset=dataset,
        test_size_ratio=args.test_size_ratio,
        val_size_ratio=args.val_size_ratio,
        random_state=args.random_state,
        device=device,
        batch_size=args.batch_size,
        workers=args.data_loader_nprocs,
        pin_memory=args.pin_memory,
        full_dataset_cuda=args.full_dataset_cuda,
        preprocessing=model_class.preprocessing,
        mock=args.mock,
        using_embedding=args.using_embedding,
    )

    model_args = model_class.get_model_args(datamodule, args, model_args)

    print("Loading model: ", args.model_name)
    print(
        tabulate(
            sorted(list(vars(model_args).items()), key=lambda x: x[0]),
            tablefmt="fancy_grid",
        )
    )

    args = {**vars(args), **vars(model_args)}
    loss_fn = get_loss_from_task(dataset.task_type)
    model = model_class(loss=loss_fn, **args)
    summary(model.cpu(), input_size=model_args.summary_input, depth=9, verbose=1, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
    model = model.float()

    callbacks, loggers = set_callbacks_loggers(args)

    trainer = pl.Trainer(
        max_epochs=args["exp_train_total_epochs"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=5,
    )

    cProfile.runctx("trainer.fit(model, datamodule=datamodule)", globals(), locals(), filename=profile_name)

    val_metrics = trainer.validate(model, datamodule=datamodule)
    print("Validation metrics: ", val_metrics)

    test_metrics = trainer.test(model, datamodule=datamodule)
    print("Test metrics: ", test_metrics)

    trainer.save_checkpoint(f"{args['model_name']}_{args['data_set']}.ckpt")

    return val_metrics, test_metrics, args


def set_callbacks_loggers(args: dict, run_name: str = None, run_params: dict = None):
    """
    Setup callbacks and loggers for pytorch-lightning Trainer.
    It will setup:
        - EarlyStopping
        - ModelCheckpoint
        - TensorBoardLogger
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # We can't know which metric to monitor, so we'll use a generic one
    # and hope it's available
    callbacks = [
        ModelCheckpoint(
            monitor=f"{args['data_set']}_val_loss",
            mode="min",
            save_top_k=1,
            dirpath="checkpoints/",
            filename="model-{epoch:02d}-{val_loss:.2f}",
        ),
        EarlyStopping(
            monitor=f"{args['data_set']}_val_loss",
            patience=args["exp_patience"],
            mode="min",
        ),
    ]

    current_tracking_uri = mlflow.get_tracking_uri()
    # Get the experiment name that was created before
    current_experiment = (
        mlflow.get_experiment(mlflow.active_run().info.experiment_id)
        if mlflow.active_run()
        else None
    )
    current_experiment_name = current_experiment.name if current_experiment else None

    # If a run_name is provided, use it, otherwise create a default one
    if run_name is None:
        run_name = f"{args['model_name']}_{timestamp}"

    mlf_logger = MLFlowLogger(
        tracking_uri=current_tracking_uri,
        experiment_name=current_experiment_name,
        run_name=run_name,
    )

    loggers = [
        mlf_logger,
        TensorBoardLogger(
            "lightning_logs",
            name="t-jepa-2",
            version=f"{args['model_name']}_{timestamp}",
        ),
    ]

    # If parameters are provided, log them.
    # We need to access the underlying experiment object to log params to the correct run.
    if run_params:
        mlf_logger.log_hyperparams(run_params)
        # Also log system metrics for completeness
        system_metrics = get_system_metrics()
        mlf_logger.log_hyperparams(system_metrics)

    return callbacks, loggers


# TODO: Update MLFlow

if __name__ == "__main__":

    a, model_args = build_parser()
    args, _ = a.parse_known_args()
    model_args, _ = model_args.parse_known_args()

    profile_name = f'profile_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.prof'

    mlflow.set_experiment("t-jepa")
    run_name = f"{args.model_name}_{args.data_set}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)

    # Log input arguments
    mlflow.log_params({k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in vars(args).items()})

    val_metrics, test_metrics, args = train_benchmark_model(args, model_args, profile_name)

    # Log evaluation metrics (use test metrics primarily)
    if isinstance(test_metrics, list) and len(test_metrics) > 0:
        for metric_dict in test_metrics:
            for k, v in metric_dict.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

    mlflow.end_run()

    import pstats
    from pstats import SortKey
    p = pstats.Stats(profile_name)

    profiling_info = []
    for func, (cc, nc, tt, ct, callers) in list(p.strip_dirs().sort_stats(SortKey.CUMULATIVE).stats.items())[:10]:
        func_info = {
            'function': f'{func[0]}:{func[1]}({func[2]})',
            'call_count': cc,
            'total_time': tt,
            'cumulative_time': ct,
        }
        profiling_info.append(func_info)
    
    total_elapsed_time = p.total_tt
    data = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'args': args,
        'profiling_info': profiling_info,
        'total_elapsed_time': total_elapsed_time
    }

    output_file = args['output_file']
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
