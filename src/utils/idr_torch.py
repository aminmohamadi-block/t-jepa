#!/usr/bin/env python
# coding: utf-8

import os

# import hostlist

try:
    # get SLURM variables
    rank = int(os.environ.get("RANK", "unset"))
    local_rank = int(os.environ.get("LOCAL_RANK", "unset"))
    size = int(os.environ.get("WORLD_SIZE", "unset"))
    cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])

    # get node list from slurm
    # hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    # get IDs of reserved GPU
    gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
    # Define MASTER_ADDR & MASTER_PORT only if they are not already provided
    # (they will be set automatically when using torchrun --standalone).

    if "MASTER_ADDR" not in os.environ:
        # Fallback to the current node hostname if torchrun has not set it.
        os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", "127.0.0.1")

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(24567 + int(min(gpu_ids)))

    # Debug prints to verify distributed environment detection.
    print(
        f"[idr_torch] rank={rank} local_rank={local_rank} size={size} | "
        f"MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}",
        flush=True,
    )
except:
    print("Not SLURM job.")
