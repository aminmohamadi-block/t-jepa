#!/usr/bin/env python3
"""
Verify Phase 5 optimizations by comparing old vs new implementation directly.
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark(func, *args, warmup=5, iterations=50, **kwargs):
    """Run benchmark with warmup."""
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 70)
    print("PHASE 5 OPTIMIZATION VERIFICATION")
    print("=" * 70)

    # Create realistic parameter sizes (similar to actual model)
    param_shapes = [
        (64, 64), (64,), (64, 64), (64,),  # Attention params
        (256, 64), (256,), (64, 256), (64,),  # FFN params
        (64, 64), (64,),  # Layer norm
    ] * 4  # 4 transformer layers

    params_q = [torch.randn(shape, device=device, requires_grad=False) for shape in param_shapes]
    params_k = [torch.randn(shape, device=device, requires_grad=False) for shape in param_shapes]

    m = 0.996

    print(f"\n1. EMA UPDATE BENCHMARK ({len(params_q)} parameters)")
    print("-" * 70)

    # OLD: Loop-based EMA
    def ema_loop():
        for pq, pk in zip(params_q, params_k):
            pk.mul_(m).add_((1.0 - m) * pq.detach())

    # NEW: _foreach EMA
    def ema_foreach():
        pk_data = [p for p in params_k]
        pq_data = [p for p in params_q]
        torch._foreach_mul_(pk_data, m)
        torch._foreach_add_(pk_data, pq_data, alpha=1.0 - m)

    loop_mean, loop_std = benchmark(ema_loop)
    foreach_mean, foreach_std = benchmark(ema_foreach)

    print(f"  Loop-based:      {loop_mean:.3f} +/- {loop_std:.3f} ms")
    print(f"  _foreach:        {foreach_mean:.3f} +/- {foreach_std:.3f} ms")
    print(f"  Speedup:         {loop_mean/foreach_mean:.1f}x")

    print(f"\n2. GRADIENT STATS BENCHMARK")
    print("-" * 70)

    # Simulate gradients
    grads = [torch.randn(shape, device=device) for shape in param_shapes]

    # OLD: CPU transfer
    def grad_stats_cpu():
        all_grads = torch.cat([g.flatten() for g in grads])
        all_grads_np = all_grads.cpu().detach().numpy()
        return {
            'mean': float(np.mean(all_grads_np)),
            'l2': float(np.linalg.norm(all_grads_np)),
            'std': float(np.std(all_grads_np)),
        }

    # NEW: GPU computation
    def grad_stats_gpu():
        all_grads = torch.cat([g.flatten() for g in grads])
        return {
            'mean': all_grads.mean().item(),
            'l2': all_grads.norm().item(),
            'std': all_grads.std().item(),
        }

    cpu_mean, cpu_std = benchmark(grad_stats_cpu)
    gpu_mean, gpu_std = benchmark(grad_stats_gpu)

    print(f"  CPU transfer:    {cpu_mean:.3f} +/- {cpu_std:.3f} ms")
    print(f"  GPU compute:     {gpu_mean:.3f} +/- {gpu_std:.3f} ms")
    print(f"  Speedup:         {cpu_mean/gpu_mean:.1f}x")

    print(f"\n3. EXPAND VS REPEAT BENCHMARK")
    print("-" * 70)

    B = 2048
    pos_embed = torch.randn(1, 56, 64, device=device)

    def use_repeat():
        return pos_embed.repeat(B, 1, 1)

    def use_expand():
        return pos_embed.expand(B, -1, -1)

    repeat_mean, repeat_std = benchmark(use_repeat)
    expand_mean, expand_std = benchmark(use_expand)

    print(f"  .repeat():       {repeat_mean:.3f} +/- {repeat_std:.3f} ms")
    print(f"  .expand():       {expand_mean:.3f} +/- {expand_std:.3f} ms")
    print(f"  Speedup:         {repeat_mean/expand_mean:.1f}x")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_old = loop_mean + cpu_mean + repeat_mean
    total_new = foreach_mean + gpu_mean + expand_mean
    print(f"\nTotal time (old operations): {total_old:.3f} ms")
    print(f"Total time (new operations): {total_new:.3f} ms")
    print(f"Combined speedup: {total_old/total_new:.1f}x")
    print(f"Time saved per iteration: {total_old - total_new:.3f} ms")


if __name__ == "__main__":
    main()
