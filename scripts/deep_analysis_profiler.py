#!/usr/bin/env python3
"""
Deep Analysis Profiler for T-JEPA

This script performs an extremely detailed profiling of:
1. Encoder forward pass (Tokenizer, embeddings, transformer)
2. Predictor forward pass (TransformerPredictor)
3. Mask operations (apply_masks_from_idx)
4. Training loop operations (data transfer, backward pass, EMA update)

The goal is to identify bottlenecks at the finest level of granularity.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import Encoder, Tokenizer, TabularEncoder
from src.predictors import Predictors, TransformerPredictor
from src.mask import MaskCollator
from src.mask_vectorized import UltraVectorizedMaskCollator
from src.utils.train_utils import apply_masks_from_idx


class DetailedProfiler:
    """Custom detailed profiler for fine-grained analysis."""

    def __init__(self, warmup_iters=5, profile_iters=50):
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.timings = defaultdict(list)
        self.memory = defaultdict(list)
        self.current_stack = []

    @contextmanager
    def profile(self, name):
        """Profile a code block."""
        # CUDA sync before
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        mem_before = 0
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**2

        start = time.perf_counter()

        # Track hierarchy
        full_name = "/".join(self.current_stack + [name])
        self.current_stack.append(name)

        try:
            yield
        finally:
            # CUDA sync after
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000

            mem_after = 0
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2

            self.timings[full_name].append(elapsed_ms)
            self.memory[full_name].append(mem_after - mem_before)

            self.current_stack.pop()

    def reset(self):
        """Reset collected timings."""
        self.timings.clear()
        self.memory.clear()

    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        for name, times in self.timings.items():
            # Skip warmup
            times = times[self.warmup_iters:]
            if len(times) == 0:
                continue

            mem = self.memory[name][self.warmup_iters:]

            summary[name] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'p50_ms': np.percentile(times, 50),
                'p95_ms': np.percentile(times, 95),
                'count': len(times),
                'total_ms': np.sum(times),
                'mem_delta_mb': np.mean(mem) if mem else 0,
            }
        return summary

    def print_summary(self, sort_by='total_ms'):
        """Print formatted summary."""
        summary = self.get_summary()

        if not summary:
            print("No profiling data collected.")
            return

        # Sort by specified metric
        sorted_items = sorted(summary.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

        print("\n" + "=" * 120)
        print("DETAILED PROFILING SUMMARY")
        print("=" * 120)
        print(f"{'Operation':<55} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'P95':>10} {'Total':>12}")
        print(f"{'':55} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>12}")
        print("-" * 120)

        for name, stats in sorted_items:
            # Indent based on depth
            depth = name.count('/')
            indent = "  " * depth
            display_name = indent + name.split('/')[-1]

            print(f"{display_name:<55} {stats['mean_ms']:>10.3f} {stats['std_ms']:>10.3f} "
                  f"{stats['min_ms']:>10.3f} {stats['max_ms']:>10.3f} {stats['p95_ms']:>10.3f} "
                  f"{stats['total_ms']:>12.2f}")

        print("=" * 120)
        return summary


def profile_tokenizer(profiler, device, batch_size=2048, n_features=256, hidden_dim=64, n_iterations=50):
    """Profile the Tokenizer in detail."""
    print("\n" + "=" * 80)
    print("PROFILING: Tokenizer")
    print("=" * 80)

    tokenizer = Tokenizer(
        d_numerical=n_features,
        categories=None,  # No categorical features for this test
        d_token=hidden_dim,
        bias=True,
        n_cls_tokens=1,
        n_reg_tokens=0,
    ).to(device)

    # Create test input
    x_num = torch.randn(batch_size, n_features, device=device)

    # Warmup + Profile
    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("tokenizer_total"):
            # Profile individual operations
            with profiler.profile("cls_token_expand"):
                if hasattr(tokenizer, 'cls_token_template'):
                    cls_tokens = tokenizer.cls_token_template.expand(batch_size, -1)

            with profiler.profile("cat_special_tokens"):
                # Concatenate CLS + x_num
                special_tokens = [tokenizer.cls_token_template.expand(batch_size, -1), x_num]
                x_concat = torch.cat(special_tokens, dim=1)

            with profiler.profile("linear_projection"):
                # This is the key operation: weight[None] * x[:, :, None]
                x = tokenizer.weight[None] * x_concat[:, :, None]

            with profiler.profile("bias_addition"):
                if tokenizer.bias is not None:
                    bias_parts = []
                    if tokenizer.n_cls_tokens > 0:
                        bias_parts.append(tokenizer.bias_cls_zeros)
                    bias_parts.append(tokenizer.bias)
                    bias = torch.cat(bias_parts, dim=0)
                    x = x + bias[None]

    return profiler.print_summary()


def profile_encoder_embedding(profiler, device, batch_size=2048, n_features=256, hidden_dim=64, n_iterations=50):
    """Profile the encoder embedding phase in detail."""
    print("\n" + "=" * 80)
    print("PROFILING: Encoder Embedding (in_embed_sample)")
    print("=" * 80)

    # Create a mock args namespace
    class MockArgs:
        n_cls_tokens = 1
        n_reg_tokens = 0
        model_feature_type_embedding = False
        model_feature_index_embedding = False
        model_act_func = 'gelu'

    args = MockArgs()

    encoder = Encoder(
        idx_num_features=list(range(n_features)),
        cardinalities=[],
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=4,
        p_dropout=0.1,
        layer_norm_eps=1e-5,
        gradient_clipping=0,
        feature_type_embedding=False,
        feature_index_embedding=False,
        dim_feedforward=hidden_dim * 4,
        device=device,
        args=args,
    ).to(device)

    # Create test inputs
    x = torch.randn(batch_size, n_features, device=device)

    # Create masks (n_enc=1, mask_size=50)
    mask_size = 50
    masks = [torch.randint(0, n_features, (batch_size, mask_size), device=device)]

    # Warmup + Profile
    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("encoder_embedding_total"):

            with profiler.profile("feature_separation"):
                x_num = x[:, encoder.idx_num_features]

            with profiler.profile("tokenizer_forward"):
                tokens = encoder.tokenizer(x_num, None)

            with profiler.profile("positional_encoding"):
                tokens = encoder.pe(tokens)

            with profiler.profile("apply_mask"):
                # Split and mask
                cls_tokens = tokens[:, :1, :]
                feature_tokens = tokens[:, 1:, :]

                with profiler.profile("mask_indexing"):
                    masked_features = apply_masks_from_idx(feature_tokens, masks)

                with profiler.profile("cat_cls_masked"):
                    out = torch.cat([cls_tokens, masked_features], dim=1)

    return profiler.print_summary()


def profile_transformer(profiler, device, batch_size=2048, seq_len=51, hidden_dim=64,
                       num_layers=4, num_heads=4, n_iterations=50):
    """Profile the TabularEncoder (Transformer) in detail."""
    print("\n" + "=" * 80)
    print("PROFILING: TabularEncoder (Transformer)")
    print("=" * 80)

    encoder = TabularEncoder(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        p_dropout=0.1,
        layer_norm_eps=1e-5,
        activation='gelu',
        dim_feedforward=hidden_dim * 4,
    ).to(device)

    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup + Profile
    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("transformer_total"):

            with profiler.profile("transformer_layers"):
                out = encoder.transformer(x)

            with profiler.profile("dropout1"):
                out = encoder.dropout1(out)

            with profiler.profile("layernorm1"):
                out = encoder.layernorm1(out)

            with profiler.profile("fc"):
                out = encoder.fc(out)

            with profiler.profile("dropout2"):
                out = encoder.dropout2(out)

            with profiler.profile("layernorm2"):
                out = encoder.layernorm2(out)

    return profiler.print_summary()


def profile_predictor(profiler, device, batch_size=2048, n_features=256, hidden_dim=64,
                     ctx_mask_size=50, pred_mask_size=100, n_iterations=50):
    """Profile the TransformerPredictor in detail."""
    print("\n" + "=" * 80)
    print("PROFILING: TransformerPredictor")
    print("=" * 80)

    predictor = TransformerPredictor(
        num_features=n_features,
        model_hidden_dim=hidden_dim,
        pred_embed_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        p_dropout=0.1,
        layer_norm_eps=1e-5,
        activation='gelu',
        dim_feedforward=hidden_dim * 4,
        n_cls_tokens=1,
    ).to(device)

    # Create test inputs
    # Context: [batch, n_cls + ctx_mask_size, hidden_dim]
    x = torch.randn(batch_size, 1 + ctx_mask_size, hidden_dim, device=device)

    # Masks
    masks_enc = [torch.randint(0, n_features, (batch_size, ctx_mask_size), device=device)]
    masks_pred = [torch.randint(0, n_features, (batch_size, pred_mask_size), device=device)
                  for _ in range(4)]  # 4 prediction targets

    # Warmup + Profile
    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("predictor_total"):
            B = len(x)

            with profiler.profile("predictor_embedding"):
                x_emb = predictor.predictor_emb(x)

            with profiler.profile("positional_embedding_context"):
                pos_embed_expanded = predictor.predictor_pos_embed.repeat(B, 1, 1)

                with profiler.profile("cls_pos_extract"):
                    cls_pos = pos_embed_expanded[:, :1, :]

                with profiler.profile("feature_pos_indexing"):
                    feature_indices = [mask + 1 for mask in masks_enc]
                    feature_pos = apply_masks_from_idx(pos_embed_expanded, feature_indices)

                with profiler.profile("cat_pos_embed"):
                    x_pos_embed = torch.cat([cls_pos, feature_pos], dim=1)

                with profiler.profile("add_pos_embed"):
                    x_with_pos = x_emb + x_pos_embed

            with profiler.profile("mask_token_preparation"):
                pos_embs = pos_embed_expanded

                with profiler.profile("pred_pos_indexing"):
                    cls_pos_embs = pos_embs[:, :1, :]
                    pred_indices = [mask + 1 for mask in masks_pred]
                    feature_pos_embs = apply_masks_from_idx(pos_embs, pred_indices)

                with profiler.profile("cls_pos_repeat"):
                    cls_pos_embs = cls_pos_embs.repeat(len(masks_pred), 1, 1)

                with profiler.profile("cat_pred_pos"):
                    pos_embs_final = torch.cat([cls_pos_embs, feature_pos_embs], dim=1)

                with profiler.profile("mask_token_repeat"):
                    pred_tokens = predictor.mask_token.repeat(pos_embs_final.size(0), pos_embs_final.size(1), 1)

                with profiler.profile("add_pred_pos"):
                    pred_tokens = pred_tokens + pos_embs_final

                with profiler.profile("context_repeat"):
                    x_repeated = x_with_pos.repeat(len(masks_pred), 1, 1)

                with profiler.profile("cat_ctx_pred"):
                    x_final = torch.cat([x_repeated, pred_tokens], dim=1)

            with profiler.profile("predictor_transformer"):
                x_out = predictor.transformer(x_final)

            with profiler.profile("predictor_norm"):
                x_normed = predictor.predictor_norm(x_out)

            with profiler.profile("predictor_slice"):
                N_ctxt = 1 + ctx_mask_size
                x_sliced = x_normed[:, N_ctxt:]

            with profiler.profile("predictor_proj"):
                output = predictor.predictor_proj(x_sliced)

    return profiler.print_summary()


def profile_apply_masks(profiler, device, batch_size=2048, n_features=256, hidden_dim=64, n_iterations=100):
    """Profile the apply_masks_from_idx function in detail."""
    print("\n" + "=" * 80)
    print("PROFILING: apply_masks_from_idx")
    print("=" * 80)

    # Create test inputs
    x = torch.randn(batch_size, n_features, hidden_dim, device=device)

    # Different mask configurations
    mask_sizes = [50, 100, 150]

    for mask_size in mask_sizes:
        profiler.reset()
        masks = [torch.randint(0, n_features, (batch_size, mask_size), device=device)
                 for _ in range(4)]

        for i in range(profiler.warmup_iters + n_iterations):
            with profiler.profile(f"apply_masks_size_{mask_size}"):

                with profiler.profile("all_masks_concat"):
                    all_x = []
                    for m in masks:
                        with profiler.profile("single_mask"):
                            with profiler.profile("batch_idx_create"):
                                batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)

                            with profiler.profile("batch_idx_expand"):
                                batch_idx = batch_idx.expand(-1, m.size(1))

                            with profiler.profile("advanced_indexing"):
                                result = x[batch_idx, m]

                            all_x.append(result)

                with profiler.profile("final_concat"):
                    output = torch.cat(all_x, dim=0)

        profiler.print_summary()


def profile_mask_collation(profiler, device, batch_size=2048, n_features=256, n_iterations=50):
    """Profile mask collation (original vs vectorized)."""
    print("\n" + "=" * 80)
    print("PROFILING: Mask Collation")
    print("=" * 80)

    # Create test batch
    batch = [(torch.randn(n_features),) for _ in range(batch_size)]

    # Original MaskCollator
    original = MaskCollator(
        allow_overlap=False,
        min_context_share=0.15,
        max_context_share=0.85,
        min_target_share=0.15,
        max_target_share=0.85,
        num_preds=4,
        num_encs=1,
        num_features=n_features,
        cardinalities=[],
        n_cls_tokens=1,
    )

    # Vectorized
    vectorized = UltraVectorizedMaskCollator(
        allow_overlap=False,
        min_context_share=0.15,
        max_context_share=0.85,
        min_target_share=0.15,
        max_target_share=0.85,
        num_preds=4,
        num_encs=1,
        num_features=n_features,
        cardinalities=[],
        n_cls_tokens=1,
    )

    # Profile original
    print("\nOriginal MaskCollator:")
    profiler.reset()
    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("original_mask_collator"):
            _ = original(batch)
    profiler.print_summary()

    # Profile vectorized
    print("\nVectorized MaskCollator:")
    profiler.reset()
    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("vectorized_mask_collator"):
            _ = vectorized(batch)
    profiler.print_summary()


def profile_full_forward_pass(profiler, device, batch_size=2048, n_features=256, hidden_dim=64):
    """Profile a complete forward pass as done in training."""
    print("\n" + "=" * 80)
    print("PROFILING: Full Forward Pass (Training Iteration)")
    print("=" * 80)

    # Create mock args
    class MockArgs:
        n_cls_tokens = 1
        n_reg_tokens = 0
        model_feature_type_embedding = False
        model_feature_index_embedding = False
        model_act_func = 'gelu'
        pred_type = 'transformer'

    args = MockArgs()

    # Create models
    context_encoder = Encoder(
        idx_num_features=list(range(n_features)),
        cardinalities=[],
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=4,
        p_dropout=0.1,
        layer_norm_eps=1e-5,
        gradient_clipping=0,
        feature_type_embedding=False,
        feature_index_embedding=False,
        dim_feedforward=hidden_dim * 4,
        device=device,
        args=args,
    ).to(device)

    target_encoder = Encoder(
        idx_num_features=list(range(n_features)),
        cardinalities=[],
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=4,
        p_dropout=0.1,
        layer_norm_eps=1e-5,
        gradient_clipping=0,
        feature_type_embedding=False,
        feature_index_embedding=False,
        dim_feedforward=hidden_dim * 4,
        device=device,
        args=args,
    ).to(device)

    predictor = Predictors(
        pred_type='transformer',
        hidden_dim=hidden_dim,
        pred_embed_dim=hidden_dim,
        num_features=n_features,
        num_layers=2,
        num_heads=4,
        p_dropout=0.1,
        layer_norm_eps=1e-5,
        activation='gelu',
        device=device,
        cardinalities=[],
        pred_dim_feedforward=hidden_dim * 4,
        n_cls_tokens=1,
    ).to(device)

    # Freeze target encoder
    for p in target_encoder.parameters():
        p.requires_grad = False

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=1e-4)

    # Create test inputs
    batch = torch.randn(batch_size, n_features, device=device)

    ctx_mask_size = 50
    pred_mask_size = 100

    masks_enc = [torch.randint(0, n_features, (batch_size, ctx_mask_size), device=device)]
    masks_pred = [torch.randint(0, n_features, (batch_size, pred_mask_size), device=device)
                  for _ in range(4)]

    n_iterations = 50

    # Warmup + Profile
    for i in range(profiler.warmup_iters + n_iterations):
        optimizer.zero_grad()

        with profiler.profile("training_iteration"):

            with profiler.profile("target_encoder_forward"):
                with torch.no_grad():
                    h = target_encoder(batch)

            with profiler.profile("target_masking"):
                h_cls = h[:, :1, :]
                h_features = h[:, 1:, :]

                with profiler.profile("apply_masks_target"):
                    h_masked_features = apply_masks_from_idx(h_features, masks_pred)

                with profiler.profile("cls_expand_cat"):
                    h_cls_expanded = h_cls.repeat(len(masks_pred), 1, 1)
                    h_final = torch.cat([h_cls_expanded, h_masked_features], dim=1)

            with profiler.profile("context_encoder_forward"):
                z = context_encoder(batch, masks_enc)

            with profiler.profile("predictor_forward"):
                z_pred = predictor(z, masks_enc, masks_pred)

            with profiler.profile("loss_computation"):
                loss = loss_fn(z_pred, h_final)

            with profiler.profile("backward_pass"):
                loss.backward()

            with profiler.profile("optimizer_step"):
                optimizer.step()

            with profiler.profile("ema_update"):
                with torch.no_grad():
                    m = 0.996
                    for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    return profiler.print_summary()


def profile_tensor_operations(profiler, device, batch_size=2048, n_features=256, hidden_dim=64):
    """Profile common tensor operations to identify micro-bottlenecks."""
    print("\n" + "=" * 80)
    print("PROFILING: Common Tensor Operations")
    print("=" * 80)

    # Test tensors
    x = torch.randn(batch_size, n_features, hidden_dim, device=device)
    w = torch.randn(hidden_dim, hidden_dim, device=device)

    n_iterations = 100

    # 1. torch.cat operations
    profiler.reset()
    a = torch.randn(batch_size, 50, hidden_dim, device=device)
    b = torch.randn(batch_size, 100, hidden_dim, device=device)

    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("torch_cat_dim1"):
            _ = torch.cat([a, b], dim=1)
    print("\ntorch.cat dim=1:")
    profiler.print_summary()

    # 2. repeat operations
    profiler.reset()
    c = torch.randn(batch_size, 10, hidden_dim, device=device)

    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("repeat_4x"):
            _ = c.repeat(4, 1, 1)
    print("\n.repeat(4, 1, 1):")
    profiler.print_summary()

    # 3. expand vs repeat
    profiler.reset()
    d = torch.randn(1, 10, hidden_dim, device=device)

    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("expand"):
            _ = d.expand(batch_size, -1, -1)
    print("\n.expand():")
    profiler.print_summary()

    profiler.reset()
    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("repeat_from_1"):
            _ = d.repeat(batch_size, 1, 1)
    print("\n.repeat() from size 1:")
    profiler.print_summary()

    # 4. Advanced indexing
    profiler.reset()
    indices = torch.randint(0, n_features, (batch_size, 50), device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, 50)

    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("advanced_indexing"):
            _ = x[batch_idx, indices]
    print("\nAdvanced indexing:")
    profiler.print_summary()

    # 5. Linear layer
    profiler.reset()
    linear = nn.Linear(hidden_dim, hidden_dim).to(device)
    x_flat = torch.randn(batch_size * n_features, hidden_dim, device=device)

    for i in range(profiler.warmup_iters + n_iterations):
        with profiler.profile("linear_layer"):
            _ = linear(x_flat)
    print("\nLinear layer (batch*seq, hidden):")
    profiler.print_summary()


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Configuration
    batch_size = 2048
    n_features = 256
    hidden_dim = 64

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num features: {n_features}")
    print(f"  Hidden dim: {hidden_dim}")

    profiler = DetailedProfiler(warmup_iters=5, profile_iters=50)
    all_results = {}

    # Run all profiling tests
    print("\n" + "=" * 80)
    print("STARTING DETAILED PROFILING")
    print("=" * 80)

    # 1. Tokenizer
    profiler.reset()
    results = profile_tokenizer(profiler, device, batch_size, n_features, hidden_dim)
    all_results['tokenizer'] = results

    # 2. Encoder embedding
    profiler.reset()
    results = profile_encoder_embedding(profiler, device, batch_size, n_features, hidden_dim)
    all_results['encoder_embedding'] = results

    # 3. Transformer
    profiler.reset()
    results = profile_transformer(profiler, device, batch_size, 51, hidden_dim)
    all_results['transformer'] = results

    # 4. Predictor
    profiler.reset()
    results = profile_predictor(profiler, device, batch_size, n_features, hidden_dim)
    all_results['predictor'] = results

    # 5. apply_masks
    profile_apply_masks(profiler, device, batch_size, n_features, hidden_dim)

    # 6. Mask collation
    profile_mask_collation(profiler, device, batch_size, n_features)

    # 7. Full forward pass
    profiler.reset()
    results = profile_full_forward_pass(profiler, device, batch_size, n_features, hidden_dim)
    all_results['full_forward_pass'] = results

    # 8. Tensor operations
    profile_tensor_operations(profiler, device, batch_size, n_features, hidden_dim)

    # Save results
    output_file = f"deep_profiling_results_{device.type}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
