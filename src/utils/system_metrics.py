"""
System metrics collection for MLflow tracking.

This module provides functionality to collect and log system metrics including:
- GPU utilization and memory usage (per GPU)
- CPU utilization and memory usage
- Disk I/O metrics
- Network I/O metrics

Designed for distributed training on SLURM clusters.
"""

import time
from typing import Dict, Optional
import logging
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system metrics collection will be limited")

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("nvidia-ml-py not available - GPU metrics collection will be disabled")


class SystemMetricsCollector:
    """
    Collects system metrics for MLflow logging during training.
    
    Features:
    - GPU metrics: utilization, memory usage, temperature per GPU
    - CPU metrics: utilization, memory usage, load average
    - System metrics: disk I/O, network I/O
    - Distributed training aware (collects metrics per rank)
    - On-demand collection synchronized with model metrics logging
    """
    
    def __init__(self, 
                 local_rank: Optional[int] = None,
                 world_size: Optional[int] = None):
        """
        Initialize system metrics collector.
        
        Args:
            local_rank: Local rank for distributed training
            world_size: World size for distributed training
        """
        self.local_rank = local_rank or 0
        self.world_size = world_size or 1
        
        # Initialize GPU monitoring
        self.gpu_count = 0
        self.gpu_handles = []
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
                self.gpu_handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
                logging.info(f"Initialized GPU monitoring for {self.gpu_count} GPUs")
            except Exception as e:
                logging.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_count = 0
                self.gpu_handles = []
        
        # Baseline metrics for delta calculations (initialized on first call)
        self._baseline_disk_io = None
        self._baseline_network_io = None
        self._last_collection_time = None
    
    def collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU metrics for all available GPUs."""
        metrics = {}
        
        if not NVML_AVAILABLE or self.gpu_count == 0:
            return metrics
        
        try:
            for gpu_idx, handle in enumerate(self.gpu_handles):
                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f"sys/gpu_{gpu_idx}_utilization"] = float(util.gpu)
                metrics[f"sys/gpu_{gpu_idx}_memory_utilization"] = float(util.memory)
                
                # GPU memory
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"sys/gpu_{gpu_idx}_memory_used_gb"] = float(mem_info.used) / (1024**3)
                metrics[f"sys/gpu_{gpu_idx}_memory_total_gb"] = float(mem_info.total) / (1024**3)
                metrics[f"sys/gpu_{gpu_idx}_memory_free_gb"] = float(mem_info.free) / (1024**3)
                metrics[f"sys/gpu_{gpu_idx}_memory_used_percent"] = float(mem_info.used) / float(mem_info.total) * 100
                
                # GPU temperature
                try:
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    metrics[f"sys/gpu_{gpu_idx}_temperature"] = float(temp)
                except Exception:
                    pass  # Temperature might not be available on all GPUs
                
                # GPU power consumption
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle)
                    metrics[f"sys/gpu_{gpu_idx}_power_watts"] = float(power) / 1000.0
                except Exception:
                    pass  # Power monitoring might not be available
        
        except Exception as e:
            logging.warning(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    def collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU and system memory metrics."""
        metrics = {}
        
        if not PSUTIL_AVAILABLE:
            return metrics
        
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics["sys/cpu_utilization"] = float(cpu_percent)
            
            # CPU per-core utilization
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            for i, cpu_usage in enumerate(cpu_per_core):
                metrics[f"sys/cpu_{i}_utilization"] = float(cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics["sys/memory_used_gb"] = float(memory.used) / (1024**3)
            metrics["sys/memory_total_gb"] = float(memory.total) / (1024**3)
            metrics["sys/memory_available_gb"] = float(memory.available) / (1024**3)
            metrics["sys/memory_used_percent"] = float(memory.percent)
            
            # Load average (Unix systems)
            try:
                load_avg = os.getloadavg()
                metrics["sys/load_avg_1min"] = float(load_avg[0])
                metrics["sys/load_avg_5min"] = float(load_avg[1])
                metrics["sys/load_avg_15min"] = float(load_avg[2])
            except (AttributeError, OSError):
                pass  # Not available on all systems
        
        except Exception as e:
            logging.warning(f"Error collecting CPU metrics: {e}")
        
        return metrics
    
    def collect_io_metrics(self) -> Dict[str, float]:
        """Collect disk and network I/O metrics."""
        metrics = {}
        
        if not PSUTIL_AVAILABLE:
            return metrics
        
        try:
            current_time = time.time()
            
            # Initialize baselines on first call
            if self._baseline_disk_io is None:
                self._baseline_disk_io = psutil.disk_io_counters()
                self._baseline_network_io = psutil.net_io_counters()
                self._last_collection_time = current_time
                # Return empty metrics on first call since we need deltas
                return metrics
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io and self._baseline_disk_io and self._last_collection_time:
                time_delta = current_time - self._last_collection_time
                
                if time_delta > 0:  # Avoid division by zero
                    read_bytes_delta = disk_io.read_bytes - self._baseline_disk_io.read_bytes
                    write_bytes_delta = disk_io.write_bytes - self._baseline_disk_io.write_bytes
                    
                    metrics["sys/disk_read_mb_per_sec"] = float(read_bytes_delta) / (1024**2) / time_delta
                    metrics["sys/disk_write_mb_per_sec"] = float(write_bytes_delta) / (1024**2) / time_delta
                
                self._baseline_disk_io = disk_io
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io and self._baseline_network_io and self._last_collection_time:
                time_delta = current_time - self._last_collection_time
                
                if time_delta > 0:  # Avoid division by zero
                    bytes_sent_delta = net_io.bytes_sent - self._baseline_network_io.bytes_sent
                    bytes_recv_delta = net_io.bytes_recv - self._baseline_network_io.bytes_recv
                    
                    metrics["sys/network_sent_mb_per_sec"] = float(bytes_sent_delta) / (1024**2) / time_delta
                    metrics["sys/network_recv_mb_per_sec"] = float(bytes_recv_delta) / (1024**2) / time_delta
                
                self._baseline_network_io = net_io
            
            self._last_collection_time = current_time
        
        except Exception as e:
            logging.warning(f"Error collecting I/O metrics: {e}")
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, float]:
        """Collect all available system metrics."""
        metrics = {}
        
        # Add rank info for distributed training
        if self.world_size > 1:
            metrics["sys/rank"] = float(self.local_rank)
            metrics["sys/world_size"] = float(self.world_size)
        
        # Collect all metric types
        metrics.update(self.collect_gpu_metrics())
        metrics.update(self.collect_cpu_metrics())
        metrics.update(self.collect_io_metrics())
        
        # Add timestamp
        metrics["sys/timestamp"] = time.time()
        
        return metrics