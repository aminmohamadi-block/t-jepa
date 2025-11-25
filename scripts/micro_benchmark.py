#!/usr/bin/env python3
"""
Micro-benchmarks for specific bottlenecks identified in the deep analysis.

This script validates the performance impact of specific operations and tests
proposed optimizations.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark(func, *args, warmup=5, iterations=100, **kwargs):
    """Run benchmark with warmup and return statistics."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def print_comparison(name, original, optimized):
    """Print comparison between original and optimized."""
    speedup = original['mean_ms'] / optimized['mean_ms']
    print(f"\n{name}")
    print("-" * 60)
    print(f"  Original:  {original['mean_ms']:.4f} ms (+/- {original['std_ms']:.4f})")
    print(f"  Optimized: {optimized['mean_ms']:.4f} ms (+/- {optimized['std_ms']:.4f})")
    print(f"  Speedup:   {speedup:.2f}x")


# =============================================================================
# BENCHMARK 1: apply_masks_from_idx - batch_idx creation
# =============================================================================

def apply_masks_original(x, masks):
    """Original implementation - creates batch_idx in loop."""
    all_x = []
    B = x.size(0)

    for m in masks:
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
        batch_idx = batch_idx.expand(-1, m.size(1))
        all_x.append(x[batch_idx, m])

    return torch.cat(all_x, dim=0)


def apply_masks_optimized(x, masks):
    """Optimized - creates batch_idx once."""
    all_x = []
    B = x.size(0)
    batch_idx_base = torch.arange(B, device=x.device).unsqueeze(1)  # Create ONCE

    for m in masks:
        batch_idx = batch_idx_base.expand(-1, m.size(1))  # View only
        all_x.append(x[batch_idx, m])

    return torch.cat(all_x, dim=0)


def apply_masks_vectorized(x, masks):
    """Fully vectorized - single indexing operation (if masks same size)."""
    if len(masks) == 0:
        return x

    B, N, D = x.size()
    mask_size = masks[0].size(1)

    # Stack all masks
    masks_stacked = torch.stack(masks, dim=0)  # [num_masks, B, mask_size]
    num_masks = len(masks)

    # Create batch indices
    batch_idx = torch.arange(B, device=x.device).view(1, B, 1).expand(num_masks, -1, mask_size)

    # Index directly using advanced indexing with reshape
    # x[batch_idx, masks_stacked] would work if shapes align
    # Instead, use gather with proper expansion
    results = []
    for i in range(num_masks):
        bi = batch_idx[i]
        mi = masks_stacked[i]
        results.append(x[bi, mi])

    return torch.cat(results, dim=0)


def benchmark_apply_masks(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: apply_masks_from_idx")
    print("=" * 70)

    B, N, D = 2048, 256, 64
    x = torch.randn(B, N, D, device=device)
    masks = [torch.randint(0, N, (B, 100), device=device) for _ in range(4)]

    original = benchmark(apply_masks_original, x, masks)
    optimized = benchmark(apply_masks_optimized, x, masks)
    vectorized = benchmark(apply_masks_vectorized, x, masks)

    print_comparison("apply_masks (batch_idx caching)", original, optimized)
    print_comparison("apply_masks (vectorized vs original)", original, vectorized)


# =============================================================================
# BENCHMARK 2: repeat vs expand
# =============================================================================

def benchmark_repeat_expand(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: repeat vs expand")
    print("=" * 70)

    B = 2048
    template = torch.randn(1, 257, 64, device=device)

    def use_repeat():
        return template.repeat(B, 1, 1)

    def use_expand_only():
        return template.expand(B, -1, -1)

    def use_expand_contiguous():
        return template.expand(B, -1, -1).contiguous()

    repeat_result = benchmark(use_repeat)
    expand_result = benchmark(use_expand_only)
    expand_contig_result = benchmark(use_expand_contiguous)

    print(f"\nTensor shape: {template.shape} -> expanded to ({B}, 257, 64)")
    print("-" * 60)
    print(f"  .repeat():                {repeat_result['mean_ms']:.4f} ms")
    print(f"  .expand():                {expand_result['mean_ms']:.4f} ms")
    print(f"  .expand().contiguous():   {expand_contig_result['mean_ms']:.4f} ms")
    print(f"  repeat/expand ratio:      {repeat_result['mean_ms']/expand_result['mean_ms']:.1f}x")


# =============================================================================
# BENCHMARK 3: Index shifting operations
# =============================================================================

def benchmark_index_shift(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: Index shifting (mask + n_cls_tokens)")
    print("=" * 70)

    B = 2048
    mask_size = 100
    n_cls_tokens = 1
    masks = [torch.randint(0, 256, (B, mask_size), device=device) for _ in range(4)]

    def shift_with_list_comp():
        return [mask + n_cls_tokens for mask in masks]

    def shift_with_clone_inplace():
        shifted = []
        for mask in masks:
            m = mask.clone()
            m.add_(n_cls_tokens)
            shifted.append(m)
        return shifted

    def shift_stacked():
        stacked = torch.stack(masks, dim=0)  # [4, B, mask_size]
        shifted = stacked + n_cls_tokens
        return [shifted[i] for i in range(shifted.size(0))]

    list_comp = benchmark(shift_with_list_comp)
    clone_inplace = benchmark(shift_with_clone_inplace)
    stacked = benchmark(shift_stacked)

    print(f"\n4 masks, each ({B}, {mask_size})")
    print("-" * 60)
    print(f"  List comprehension:       {list_comp['mean_ms']:.4f} ms")
    print(f"  Clone + inplace:          {clone_inplace['mean_ms']:.4f} ms")
    print(f"  Stack + add:              {stacked['mean_ms']:.4f} ms")


# =============================================================================
# BENCHMARK 4: Bias concatenation optimization
# =============================================================================

def benchmark_bias_concat(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: Bias concatenation")
    print("=" * 70)

    n_cls_tokens = 1
    d_token = 64
    d_numerical = 256

    bias_cls_zeros = torch.zeros(n_cls_tokens, d_token, device=device)
    bias = torch.randn(d_numerical, d_token, device=device)

    # Pre-concatenated version
    bias_precat = torch.cat([bias_cls_zeros, bias], dim=0)

    def concat_every_time():
        return torch.cat([bias_cls_zeros, bias], dim=0)

    def use_precat():
        return bias_precat

    concat = benchmark(concat_every_time)
    precat = benchmark(use_precat)

    print(f"\nBias shape: ({n_cls_tokens + d_numerical}, {d_token})")
    print("-" * 60)
    print(f"  Concat each forward:      {concat['mean_ms']:.4f} ms")
    print(f"  Pre-concatenated:         {precat['mean_ms']:.4f} ms")
    print(f"  Speedup:                  {concat['mean_ms']/precat['mean_ms']:.1f}x")


# =============================================================================
# BENCHMARK 5: Gradient statistics computation
# =============================================================================

def benchmark_gradient_stats(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: Gradient statistics computation")
    print("=" * 70)

    # Simulate gradients from a transformer model
    grad_shapes = [
        (64, 64),     # Embedding
        (64, 64),     # Q, K, V projections
        (256, 64),    # FFN
        (64, 256),    # FFN
    ] * 4  # 4 layers

    grads = [torch.randn(shape, device=device) for shape in grad_shapes]

    def cpu_stats():
        all_grads = torch.cat([g.flatten() for g in grads])
        all_grads_np = all_grads.cpu().detach().numpy()
        return {
            'mean': float(np.mean(all_grads_np)),
            'std': float(np.std(all_grads_np)),
            'norm': float(np.linalg.norm(all_grads_np)),
        }

    def gpu_stats():
        all_grads = torch.cat([g.flatten() for g in grads])
        return {
            'mean': all_grads.mean().item(),
            'std': all_grads.std().item(),
            'norm': all_grads.norm().item(),
        }

    cpu_result = benchmark(cpu_stats)
    gpu_result = benchmark(gpu_stats)

    print(f"\nTotal gradient elements: {sum(np.prod(s) for s in grad_shapes)}")
    print("-" * 60)
    print(f"  CPU transfer + numpy:     {cpu_result['mean_ms']:.4f} ms")
    print(f"  GPU computation:          {gpu_result['mean_ms']:.4f} ms")
    print(f"  Speedup:                  {cpu_result['mean_ms']/gpu_result['mean_ms']:.1f}x")


# =============================================================================
# BENCHMARK 6: EMA update methods
# =============================================================================

def benchmark_ema_update(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: EMA update methods")
    print("=" * 70)

    # Create parameters to update
    n_params = 50
    param_shapes = [(64, 64), (256, 64), (64, 256)] * (n_params // 3 + 1)
    param_shapes = param_shapes[:n_params]

    params_q = [torch.randn(shape, device=device, requires_grad=False) for shape in param_shapes]
    params_k = [torch.randn(shape, device=device, requires_grad=False) for shape in param_shapes]

    m = 0.996

    def ema_loop():
        for param_q, param_k in zip(params_q, params_k):
            param_k.mul_(m).add_((1.0 - m) * param_q.detach())

    def ema_foreach():
        torch._foreach_mul_(params_k, m)
        torch._foreach_add_(params_k, params_q, alpha=1.0 - m)

    loop_result = benchmark(ema_loop)

    try:
        foreach_result = benchmark(ema_foreach)
        print(f"\n{n_params} parameters")
        print("-" * 60)
        print(f"  Loop-based:               {loop_result['mean_ms']:.4f} ms")
        print(f"  _foreach operations:      {foreach_result['mean_ms']:.4f} ms")
        print(f"  Speedup:                  {loop_result['mean_ms']/foreach_result['mean_ms']:.1f}x")
    except:
        print(f"\n{n_params} parameters")
        print("-" * 60)
        print(f"  Loop-based:               {loop_result['mean_ms']:.4f} ms")
        print(f"  _foreach operations:      Not available in this PyTorch version")


# =============================================================================
# BENCHMARK 7: torch.cat alternatives
# =============================================================================

def benchmark_cat_alternatives(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: torch.cat alternatives")
    print("=" * 70)

    B = 2048
    seq_len = 51
    hidden = 64
    num_pred = 4

    context = torch.randn(B * num_pred, seq_len, hidden, device=device)
    pred_tokens = torch.randn(B * num_pred, 101, hidden, device=device)

    def use_cat():
        return torch.cat([context, pred_tokens], dim=1)

    def use_preallocated():
        result = torch.empty(B * num_pred, seq_len + 101, hidden, device=device)
        result[:, :seq_len, :] = context
        result[:, seq_len:, :] = pred_tokens
        return result

    cat_result = benchmark(use_cat)
    prealloc_result = benchmark(use_preallocated)

    print(f"\nConcatenating ({B*num_pred}, {seq_len}, {hidden}) + ({B*num_pred}, 101, {hidden})")
    print("-" * 60)
    print(f"  torch.cat:                {cat_result['mean_ms']:.4f} ms")
    print(f"  Pre-allocated copy:       {prealloc_result['mean_ms']:.4f} ms")


# =============================================================================
# BENCHMARK 8: Linear projection patterns
# =============================================================================

def benchmark_linear_patterns(device):
    print("\n" + "=" * 70)
    print("BENCHMARK: Linear projection patterns (Tokenizer)")
    print("=" * 70)

    B = 2048
    N = 257  # cls + features
    D = 64

    x = torch.randn(B, N, device=device)
    weight = torch.randn(N, D, device=device)

    def broadcast_mul():
        # Current implementation
        return weight[None] * x[:, :, None]

    def einsum():
        return torch.einsum('nd,bn->bnd', weight, x)

    def manual_expand():
        return weight.unsqueeze(0).expand(B, -1, -1) * x.unsqueeze(-1).expand(-1, -1, D)

    broadcast_result = benchmark(broadcast_mul)
    einsum_result = benchmark(einsum)
    expand_result = benchmark(manual_expand)

    print(f"\nLinear projection: ({B}, {N}) -> ({B}, {N}, {D})")
    print("-" * 60)
    print(f"  Broadcasting:             {broadcast_result['mean_ms']:.4f} ms")
    print(f"  torch.einsum:             {einsum_result['mean_ms']:.4f} ms")
    print(f"  Manual expand:            {expand_result['mean_ms']:.4f} ms")


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running micro-benchmarks on: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run all benchmarks
    benchmark_apply_masks(device)
    benchmark_repeat_expand(device)
    benchmark_index_shift(device)
    benchmark_bias_concat(device)
    benchmark_gradient_stats(device)
    benchmark_ema_update(device)
    benchmark_cat_alternatives(device)
    benchmark_linear_patterns(device)

    print("\n" + "=" * 70)
    print("MICRO-BENCHMARKS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
