import datetime
import socket
import platform
import psutil
import subprocess
import os

import torch

PLOT_DEBUG_VALUES = False


def make_job_name(args):
    job_name = (
        "{dataset}__model_nlyrs_{n_layers}_nheads_{n_heads}_hdim_{hdim}"
        "__pred_ovrlap_{overlap}_npreds_{n_mask_preds}_"
        "_nlyrs_{n_pred_layers}_activ_{activation}"
        "nenc_{n_mask_ctx}"
        "__lr_{lr}_start_{start_lr}_final_{final_lr}_{datetime}"
    )
    job_name = job_name.format(
        dataset=args.data_set,
        n_layers=args.model_num_layers,
        n_heads=args.model_num_heads,
        hdim=args.model_dim_hidden,
        overlap="T" if args.mask_allow_overlap else "F",
        n_mask_preds=args.mask_num_preds,
        n_pred_layers=args.pred_num_layers,
        activation=args.pred_activation,
        n_mask_ctx=args.mask_num_encs,
        lr=args.exp_lr,
        start_lr=args.exp_start_lr,
        final_lr=args.exp_final_lr,
        datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    if hasattr(args, 'tag') and args.tag and args.tag.strip():
        job_name = f"{args.tag.strip()}-{job_name}"
    return job_name


def print_args(args):
    to_print_model = (
        "Encoder and Target Encoders:\n"
        "\t- Number of layers: {n_layers}\n"
        "\t- Number of att heads: {n_heads}\n"
        "\t- Hidden dimension: {hdim}\n"
        "\t- Dropout: {model_p_dropout}\n"
    )
    print(
        to_print_model.format(
            n_layers=args.model_num_layers,
            n_heads=args.model_num_heads,
            hdim=args.model_dim_hidden,
            model_p_dropout=args.model_dropout_prob,
        )
    )

    to_print_mask = (
        "Masking:\n"
        "\t- Allow overlap: {overlap}\n"
        "\t- Mask share interval for target: [{min_trgt},{max_trgt}]\n"
        "\t- Mask share interval for context: [{min_ctxt},{max_ctxt}]\n"
        "\t- Number of masks for target: {n_preds}\n"
        "\t- Number of masks for context: {n_ctx}\n"
    )
    print(
        to_print_mask.format(
            overlap=("True" if args.mask_allow_overlap else "False"),
            min_trgt=args.mask_min_trgt_share,
            max_trgt=args.mask_max_trgt_share,
            min_ctxt=args.mask_min_ctx_share,
            max_ctxt=args.mask_max_ctx_share,
            n_preds=args.mask_num_preds,
            n_ctx=args.mask_num_encs,
        )
    )

    to_print_pred = (
        "Predictors:\n"
        "\t- Number of layers: {pred_num_layers}\n"
        "\t- Dropout: {pred_p_dropout}\n"
        "\t- Layer norm epsilon: {pred_layer_norm_eps}\n"
        "\t- Activation function: {pred_activation}\n"
    )
    print(
        to_print_pred.format(
            pred_num_layers=args.pred_num_layers,
            pred_p_dropout=args.pred_p_dropout,
            pred_layer_norm_eps=args.pred_layer_norm_eps,
            pred_activation=args.pred_activation,
        )
    )

    to_print_optimization = (
        "Optimization details:\n"
        "\t- Optimizer: AdamW (default)\n"
        "\t- Learning rate scheduler: {scheduler}\n"
        "\t- Reference learning rate: {ref_lr}\n"
        "\t- Start learning rate: {start_lr}\n"
        "\t- Final learning rate: {final_lr}\n"
        "\t- Weight decay scheduler: {wd_scheduler}\n"
        "\t- Weight decay: {wd}\n"
        "\t- Final weight decay: {final_wd}\n"
        "\t- Gradient clipping: {gradient_clipping}\n"
    )

    print(
        to_print_optimization.format(
            scheduler=("True" if args.exp_scheduler else "False"),
            ref_lr=args.exp_lr,
            start_lr=args.exp_start_lr,
            final_lr=args.exp_final_lr,
            wd_scheduler=("True" if args.exp_weight_decay_scheduler else "False"),
            wd=args.exp_weight_decay,
            final_wd=args.exp_final_weight_decay,
            gradient_clipping=args.exp_gradient_clipping,
        )
    )


def _debug_values(data: torch.Tensor, title="Data", skip=False):
    if skip or not PLOT_DEBUG_VALUES:
        return

    if "plt" not in globals():
        import matplotlib.pyplot as plt

    _data = data
    if len(data.shape) == 1:
        _data = data.unsqueeze(0)

    plt.imshow(_data.cpu().detach().numpy(), aspect="auto")

    if len(data.shape) == 1:
        for i in range(data.shape[0]):
            # make the value in scientific notation if it is less than 0.01
            value = (
                "{:.2f}".format(data[i].item())
                if abs(data[i].item()) > 0.01
                else "{:.1e}".format(data[i].item())
            )
            plt.text(
                i,
                0,
                s=value,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = (
                    "{:.2f}".format(data[i, j].item())
                    if abs(data[i, j].item()) > 0.01
                    else "{:.1e}".format(data[i, j].item())
                )
                plt.text(
                    j,
                    i,
                    s=value,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )
    plt.title(title)
    plt.show()


def get_system_metrics():
    """Collect comprehensive system metrics for MLflow logging."""
    metrics = {}
    
    # Basic system info
    metrics['sys/hostname'] = socket.gethostname()
    metrics['sys/platform'] = platform.platform()
    metrics['sys/python_version'] = platform.python_version()
    
    # CPU info
    metrics['sys/cpu_count'] = psutil.cpu_count()
    metrics['sys/cpu_count_logical'] = psutil.cpu_count(logical=True)
    metrics['sys/cpu_count_physical'] = psutil.cpu_count(logical=False)
    
    # Memory info (in GB)
    memory = psutil.virtual_memory()
    metrics['sys/memory_total_gb'] = memory.total / (1024**3)
    metrics['sys/memory_available_gb'] = memory.available / (1024**3)
    metrics['sys/memory_percent'] = memory.percent
    
    # GPU info
    if torch.cuda.is_available():
        metrics['sys/gpu_count'] = torch.cuda.device_count()
        metrics['sys/cuda_version'] = torch.version.cuda
        
        # Per-GPU memory info
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            metrics[f'sys/gpu_{i}_name'] = gpu_name
            metrics[f'sys/gpu_{i}_memory_gb'] = gpu_memory
    else:
        metrics['sys/gpu_count'] = 0
        metrics['sys/cuda_version'] = 'N/A'
    
    # SLURM info (if available)
    slurm_vars = [
        'SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_GPUS', 'SLURM_CPUS_PER_TASK',
        'SLURM_MEM_PER_NODE', 'SLURM_NNODES', 'SLURM_NTASKS', 'SLURM_PARTITION'
    ]
    for var in slurm_vars:
        value = os.environ.get(var)
        if value:
            metrics[f'slurm/{var.lower()}'] = value
    
    # PyTorch info
    metrics['sys/pytorch_version'] = torch.__version__
    
    # Distributed training info
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        metrics['sys/world_size'] = torch.distributed.get_world_size()
        metrics['sys/rank'] = torch.distributed.get_rank()
        metrics['sys/backend'] = torch.distributed.get_backend()
        
    return metrics
