import torch

from src.utils.scheduler import WarmupCosineSchedule, CosineWDSchedule


def init_optim(
    context_encoder,
    predictors,
    iter_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    ipe_scale=1.25,
    use_scheduler=True,
    use_wd_scheduler=True,
):

    # Allow for DistributedDataParallel-wrapped modules
    if hasattr(predictors, "module"):
        predictors_inner = predictors.module  # type: ignore[attr-defined]
    else:
        predictors_inner = predictors

    if hasattr(context_encoder, "module"):
        context_inner = context_encoder.module  # type: ignore[attr-defined]
    else:
        context_inner = context_encoder

    if isinstance(predictors_inner.predictors, list):
        predictor_params1 = (
            p
            for pred in predictors_inner.predictors
            for n, p in pred.named_parameters()
            if ("bias" not in n) and (len(p.shape) != 1)
        )
        predictor_params2 = (
            p
            for pred in predictors_inner.predictors
            for n, p in pred.named_parameters()
            if ("bias" in n) or (len(p.shape) == 1)
        )
    else:
        predictor_params1 = (
            p
            for n, p in predictors_inner.predictors.named_parameters()
            if ("bias" not in n) and (len(p.shape) != 1)
        )
        predictor_params2 = (
            p
            for n, p in predictors_inner.predictors.named_parameters()
            if ("bias" in n) or (len(p.shape) == 1)
        )
    param_groups = [
        {
            "params": (
                p
                for n, p in context_inner.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {"params": predictor_params1},
        {
            "params": (
                p
                for n, p in context_inner.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {"params": predictor_params2, "WD_exclude": True, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    if use_scheduler:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * num_epochs * iter_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iter_per_epoch),
        )
    else:
        scheduler = None

    if use_wd_scheduler:
        wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=wd,
            final_wd=final_wd,
            T_max=int(ipe_scale * num_epochs * iter_per_epoch),
        )
    else:
        wd_scheduler = None

    return (optimizer, scheduler, wd_scheduler)
