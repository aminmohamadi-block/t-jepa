"""
Comprehensive Performance Profiling System for T-JEPA

This module provides a hierarchical, low-overhead profiling system that tracks:
- Execution time (hierarchical, with nesting support)
- GPU/CPU memory usage
- Operation counts and statistics
- Distributed training metrics

Usage:
    from src.utils.profiler import get_profiler, ProfilingLevel

    profiler = get_profiler()
    profiler.set_level(ProfilingLevel.DETAILED)

    # In training loop:
    with profiler.profile("iteration", iteration=i, epoch=epoch):
        with profiler.profile("data_loading"):
            batch = next(dataloader)

        with profiler.profile("forward_pass"):
            output = model(batch)

    # Get results
    profiler.print_summary()
    profiler.save_results("profiling_results.json")
"""

import time
import json
import torch
import psutil
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Any
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
import torch.distributed as dist


class ProfilingLevel(Enum):
    """Profiling verbosity levels."""
    DISABLED = 0      # No profiling
    LIGHTWEIGHT = 1   # Only major operations (epoch, iteration, forward/backward)
    DETAILED = 2      # All instrumented operations
    TRACE = 3         # Full PyTorch profiler with memory tracking


@dataclass
class TimingStats:
    """Statistics for a single operation type."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    # Memory stats (in MB)
    gpu_mem_allocated: List[float] = field(default_factory=list)
    gpu_mem_reserved: List[float] = field(default_factory=list)

    # Nesting level for hierarchical display
    level: int = 0

    # Parent operation (for hierarchy)
    parent: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def std_time(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(np.std(self.times))

    @property
    def mean_gpu_mem_allocated(self) -> float:
        if not self.gpu_mem_allocated:
            return 0.0
        return float(np.mean(self.gpu_mem_allocated))

    def update(self, elapsed: float, gpu_mem_alloc: float = 0.0, gpu_mem_res: float = 0.0):
        """Update statistics with a new measurement."""
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

        if gpu_mem_alloc > 0:
            self.gpu_mem_allocated.append(gpu_mem_alloc)
        if gpu_mem_res > 0:
            self.gpu_mem_reserved.append(gpu_mem_res)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'count': self.count,
            'total_time': self.total_time,
            'mean_time': self.mean_time,
            'std_time': self.std_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'mean_gpu_mem_allocated_mb': self.mean_gpu_mem_allocated,
            'level': self.level,
            'parent': self.parent,
            'metadata': self.metadata,
        }


class PerformanceProfiler:
    """
    Hierarchical performance profiler with minimal overhead.

    Key features:
    - Nested timing with automatic hierarchy tracking
    - GPU/CPU memory profiling
    - Statistical aggregation across iterations
    - Distributed training support (per-rank and aggregated stats)
    - Export to JSON/MLflow
    """

    def __init__(self, level: ProfilingLevel = ProfilingLevel.DISABLED):
        self.level = level
        self.stats: Dict[str, TimingStats] = {}
        self.current_stack: List[str] = []  # Track nesting
        self.start_times: Dict[str, float] = {}

        # For distributed training
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        # PyTorch profiler (for TRACE level)
        self.torch_profiler: Optional[torch.profiler.profile] = None
        self.trace_dir: Optional[Path] = None

        # Track iteration/epoch for contextualized results
        self.current_epoch = 0
        self.current_iteration = 0

        # Store detailed traces for later analysis
        self.iteration_traces: List[Dict] = []

    def set_level(self, level: ProfilingLevel):
        """Change profiling level."""
        self.level = level

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.level != ProfilingLevel.DISABLED

    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict] = None, **context):
        """
        Profile a code block.

        Args:
            name: Operation name (e.g., "forward_pass", "data_loading")
            metadata: Additional metadata to store
            **context: Context variables (e.g., iteration=5, epoch=10)

        Example:
            with profiler.profile("forward_pass", metadata={"batch_size": 256}):
                output = model(input)
        """
        if not self.is_enabled():
            yield
            return

        # Update context
        if 'epoch' in context:
            self.current_epoch = context['epoch']
        if 'iteration' in context:
            self.current_iteration = context['iteration']

        # Determine full name with hierarchy
        parent = self.current_stack[-1] if self.current_stack else None
        full_name = f"{parent}/{name}" if parent else name

        # Initialize stats if needed
        if full_name not in self.stats:
            self.stats[full_name] = TimingStats(
                name=name,
                level=len(self.current_stack),
                parent=parent,
                metadata=metadata or {}
            )

        # Push to stack
        self.current_stack.append(full_name)

        # Synchronize CUDA for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # Record GPU memory before
        gpu_mem_before = 0.0
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**2  # MB

        try:
            yield
        finally:
            # Synchronize CUDA for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time

            # Record GPU memory after
            gpu_mem_after = 0.0
            gpu_mem_reserved = 0.0
            if torch.cuda.is_available():
                gpu_mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB

            # Update statistics
            self.stats[full_name].update(
                elapsed=elapsed,
                gpu_mem_alloc=gpu_mem_after,
                gpu_mem_res=gpu_mem_reserved
            )

            # Store detailed trace for this iteration
            if self.level == ProfilingLevel.DETAILED or self.level == ProfilingLevel.TRACE:
                self.iteration_traces.append({
                    'epoch': self.current_epoch,
                    'iteration': self.current_iteration,
                    'operation': full_name,
                    'elapsed_ms': elapsed * 1000,
                    'gpu_mem_allocated_mb': gpu_mem_after,
                    'gpu_mem_delta_mb': gpu_mem_after - gpu_mem_before,
                    'rank': self.rank,
                })

            # Pop from stack
            self.current_stack.pop()

    @contextmanager
    def trace_mode(self, trace_dir: str = "./profiler_traces"):
        """
        Enable PyTorch profiler for detailed tracing.

        This captures kernel-level information, memory allocation stack traces, etc.
        Use sparingly as it has significant overhead.

        Example:
            with profiler.trace_mode("./traces"):
                with profiler.profile("training"):
                    train_one_epoch()
        """
        if self.level != ProfilingLevel.TRACE:
            yield
            return

        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        trace_file = self.trace_dir / f"trace_rank{self.rank}.json"

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.trace_dir)),
        ) as prof:
            self.torch_profiler = prof
            yield prof
            self.torch_profiler = None

        # Save text summary
        summary_file = self.trace_dir / f"summary_rank{self.rank}.txt"
        with open(summary_file, 'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

    def step_profiler(self):
        """Step the PyTorch profiler (if active)."""
        if self.torch_profiler is not None:
            self.torch_profiler.step()

    def get_summary(self, sort_by: str = "total_time") -> List[Dict]:
        """
        Get summary statistics sorted by specified metric.

        Args:
            sort_by: Metric to sort by ('total_time', 'mean_time', 'count', 'max_time')

        Returns:
            List of operation statistics dictionaries
        """
        results = [stat.to_dict() for stat in self.stats.values()]

        if sort_by in ['total_time', 'mean_time', 'count', 'max_time']:
            results.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

        return results

    def print_summary(self, top_k: int = 30, sort_by: str = "total_time"):
        """
        Print formatted summary of profiling results.

        Args:
            top_k: Number of top operations to display
            sort_by: Metric to sort by
        """
        if not self.is_enabled():
            print("Profiler is disabled.")
            return

        summary = self.get_summary(sort_by=sort_by)[:top_k]

        print("\n" + "=" * 120)
        print(f"PERFORMANCE PROFILING SUMMARY (Rank {self.rank})")
        print("=" * 120)
        print(f"{'Operation':<40} {'Count':>8} {'Total (s)':>12} {'Mean (ms)':>12} "
              f"{'Std (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'GPU Mem (MB)':>15}")
        print("-" * 120)

        for stat in summary:
            indent = "  " * stat['level']
            name = indent + stat['name']

            print(f"{name:<40} "
                  f"{stat['count']:>8} "
                  f"{stat['total_time']:>12.3f} "
                  f"{stat['mean_time']*1000:>12.2f} "
                  f"{stat['std_time']*1000:>12.2f} "
                  f"{stat['min_time']*1000:>12.2f} "
                  f"{stat['max_time']*1000:>12.2f} "
                  f"{stat['mean_gpu_mem_allocated_mb']:>15.1f}")

        print("=" * 120)

        # Print total iteration time if available
        iteration_stats = [s for s in summary if 'iteration' in s['name'].lower()]
        if iteration_stats:
            iter_stat = iteration_stats[0]
            print(f"\nAverage iteration time: {iter_stat['mean_time']*1000:.2f} ms "
                  f"(±{iter_stat['std_time']*1000:.2f} ms)")

        # Print GPU memory info
        if torch.cuda.is_available():
            print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, "
                  f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved")

        print()

    def get_breakdown(self, parent: Optional[str] = None) -> Dict[str, float]:
        """
        Get time breakdown as percentages for operations under a parent.

        Args:
            parent: Parent operation name (None for top-level)

        Returns:
            Dictionary mapping operation names to percentage of parent time
        """
        children = [s for s in self.stats.values() if s.parent == parent]

        if not children:
            return {}

        total_time = sum(s.total_time for s in children)

        if total_time == 0:
            return {}

        return {s.name: (s.total_time / total_time) * 100 for s in children}

    def print_breakdown(self, parent: Optional[str] = None):
        """Print time breakdown for operations under a parent."""
        breakdown = self.get_breakdown(parent)

        if not breakdown:
            print(f"No breakdown available for parent: {parent}")
            return

        parent_name = parent or "TOP-LEVEL"
        print(f"\n{'='*60}")
        print(f"TIME BREAKDOWN: {parent_name}")
        print(f"{'='*60}")

        for name, percentage in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(percentage / 2)  # 50 chars = 100%
            bar = '█' * bar_length
            print(f"{name:<30} {percentage:>6.2f}% {bar}")

        print(f"{'='*60}\n")

    def save_results(self, filepath: str):
        """
        Save profiling results to JSON file.

        Args:
            filepath: Path to save results
        """
        results = {
            'profiling_level': self.level.name,
            'rank': self.rank,
            'world_size': self.world_size,
            'summary': self.get_summary(),
            'detailed_traces': self.iteration_traces[-1000:],  # Last 1000 traces
            'system_info': self._get_system_info(),
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Profiling results saved to: {filepath}")

    def log_to_mlflow(self, mlflow):
        """
        Log profiling summary to MLflow.

        Args:
            mlflow: MLflow module (pass mlflow to avoid import issues)
        """
        if not self.is_enabled():
            return

        summary = self.get_summary()

        # Log top-level metrics
        for stat in summary:
            if stat['level'] == 0:  # Only top-level operations
                mlflow.log_metric(f"perf/{stat['name']}_mean_ms", stat['mean_time'] * 1000)
                mlflow.log_metric(f"perf/{stat['name']}_total_s", stat['total_time'])

        # Log breakdown percentages
        breakdown = self.get_breakdown(None)
        for name, percentage in breakdown.items():
            mlflow.log_metric(f"perf/breakdown_{name}_pct", percentage)

    def compare_with(self, other: 'PerformanceProfiler', output_file: Optional[str] = None):
        """
        Compare this profiler's results with another (e.g., before/after optimization).

        Args:
            other: Another PerformanceProfiler instance
            output_file: Optional file to save comparison
        """
        print("\n" + "=" * 140)
        print("PERFORMANCE COMPARISON")
        print("=" * 140)
        print(f"{'Operation':<40} {'Before (ms)':>15} {'After (ms)':>15} {'Speedup':>12} {'Change':>12}")
        print("-" * 140)

        # Find common operations
        common_ops = set(self.stats.keys()) & set(other.stats.keys())

        comparisons = []
        for op_name in sorted(common_ops):
            before_stat = self.stats[op_name]
            after_stat = other.stats[op_name]

            before_mean = before_stat.mean_time * 1000
            after_mean = after_stat.mean_time * 1000

            if before_mean > 0:
                speedup = before_mean / after_mean if after_mean > 0 else float('inf')
                change_pct = ((after_mean - before_mean) / before_mean) * 100
            else:
                speedup = 1.0
                change_pct = 0.0

            comparisons.append({
                'operation': op_name,
                'before_ms': before_mean,
                'after_ms': after_mean,
                'speedup': speedup,
                'change_pct': change_pct,
            })

            indent = "  " * before_stat.level
            display_name = indent + before_stat.name

            change_str = f"{change_pct:+.1f}%"
            if change_pct < -5:  # Improvement
                change_str = f"\033[92m{change_str}\033[0m"  # Green
            elif change_pct > 5:  # Regression
                change_str = f"\033[91m{change_str}\033[0m"  # Red

            print(f"{display_name:<40} {before_mean:>15.2f} {after_mean:>15.2f} "
                  f"{speedup:>12.2f}x {change_str:>12}")

        print("=" * 140)

        # Calculate overall speedup
        before_total = sum(s.total_time for s in self.stats.values() if s.level == 0)
        after_total = sum(s.total_time for s in other.stats.values() if s.level == 0)

        if before_total > 0 and after_total > 0:
            overall_speedup = before_total / after_total
            print(f"\nOverall speedup: {overall_speedup:.2f}x ({(overall_speedup-1)*100:+.1f}%)")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'comparisons': comparisons,
                    'before_total_s': before_total,
                    'after_total_s': after_total,
                    'overall_speedup': overall_speedup if before_total > 0 and after_total > 0 else None,
                }, f, indent=2)
            print(f"\nComparison saved to: {output_file}")

    def reset(self):
        """Reset all profiling statistics."""
        self.stats.clear()
        self.current_stack.clear()
        self.start_times.clear()
        self.iteration_traces.clear()

    def _get_system_info(self) -> Dict:
        """Get system information for context."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3,
        }

        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            })

        return info


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def set_profiling_level(level: ProfilingLevel):
    """Set the global profiling level."""
    get_profiler().set_level(level)


# Convenience decorators
def profile_function(name: Optional[str] = None):
    """
    Decorator to profile a function.

    Example:
        @profile_function("my_expensive_function")
        def my_function():
            ...
    """
    def decorator(func):
        func_name = name or func.__name__

        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile(func_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator
