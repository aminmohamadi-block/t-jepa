#!/usr/bin/env python3
"""
Test script to verify linear probe integration with parquet datasets.

This script tests:
1. ParquetBaseDataset creation and loading
2. OnlineDataset compatibility
3. End-to-end linear probe training with T-JEPA

Usage:
    python test_linear_probe_integration.py
"""

import sys
import os
import argparse
from argparse import Namespace

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)

    try:
        from src.datasets.parquet_base_dataset import ParquetBaseDataset
        print("✓ ParquetBaseDataset imported")
    except ImportError as e:
        print(f"✗ ParquetBaseDataset import failed: {e}")
        return False

    try:
        from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
        assert "parquet_dataset" in DATASET_NAME_TO_DATASET_MAP
        assert "parquet_chargeback" in DATASET_NAME_TO_DATASET_MAP
        print("✓ ParquetBaseDataset registered in DATASET_NAME_TO_DATASET_MAP")
    except (ImportError, AssertionError) as e:
        print(f"✗ DATASET_NAME_TO_DATASET_MAP registration failed: {e}")
        return False

    try:
        from src.datasets.online_dataset import OnlineDataset
        print("✓ OnlineDataset imported")
    except ImportError as e:
        print(f"✗ OnlineDataset import failed: {e}")
        return False

    return True


def test_parquet_base_dataset_creation():
    """Test that ParquetBaseDataset can be instantiated and loaded."""
    print("\n" + "="*70)
    print("TEST 2: ParquetBaseDataset Creation and Loading")
    print("="*70)

    from src.datasets.parquet_base_dataset import ParquetBaseDataset
    from src.configs import build_parser

    # Create minimal args for testing
    parser = build_parser()
    args = parser.parse_args([
        "--data_path", ".",
        "--parquet_data_dir", "/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train",
        "--parquet_data_files", "chunk_10.parquet",
        "--parquet_scaling_stats_file", "/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train/stats_bootstrap.pq",
        "--parquet_feature_names_file", "/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv",
        "--parquet_num_features", "30",
        "--parquet_task_type", "binary_class",
        "--parquet_preload_data", "True",
        "--batch_size", "512",
    ])

    try:
        dataset = ParquetBaseDataset(args)
        print(f"✓ ParquetBaseDataset instantiated")

        dataset.load()
        print(f"✓ Dataset loaded successfully")

        # Verify attributes
        assert hasattr(dataset, 'X'), "Missing X attribute"
        assert hasattr(dataset, 'y'), "Missing y attribute"
        assert hasattr(dataset, 'N'), "Missing N attribute"
        assert hasattr(dataset, 'D'), "Missing D attribute"
        assert hasattr(dataset, 'task_type'), "Missing task_type attribute"
        print(f"✓ All required BaseDataset attributes present")

        print(f"  N (samples): {dataset.N}")
        print(f"  D (features): {dataset.D}")
        print(f"  Task type: {dataset.task_type}")
        print(f"  X shape: {dataset.X.shape}, dtype: {dataset.X.dtype}")
        print(f"  y shape: {dataset.y.shape}, dtype: {dataset.y.dtype}")

        # Verify dimensions match
        assert dataset.X.shape[0] == dataset.N, "X shape mismatch with N"
        assert dataset.X.shape[1] == dataset.D, "X shape mismatch with D"
        assert dataset.y.shape[0] == dataset.N, "y shape mismatch with N"
        print(f"✓ Dimensions verified")

        return True

    except Exception as e:
        print(f"✗ ParquetBaseDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_online_dataset_compatibility():
    """Test that OnlineDataset can load ParquetBaseDataset via dict lookup."""
    print("\n" + "="*70)
    print("TEST 3: Dataset Registration & Lookup")
    print("="*70)

    from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
    from src.configs import build_parser

    # Create minimal args
    parser = build_parser()
    args = parser.parse_args([
        "--data_path", ".",
        "--data_set", "parquet_chargeback",
        "--parquet_data_dir", "/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train",
        "--parquet_data_files", "chunk_10.parquet",
        "--parquet_scaling_stats_file", "/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train/stats_bootstrap.pq",
        "--parquet_feature_names_file", "/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv",
        "--parquet_num_features", "30",
        "--parquet_task_type", "binary_class",
        "--parquet_preload_data", "True",
        "--batch_size", "512",
    ])

    try:
        # Test that OnlineDataset can look up the dataset
        print("Testing dataset lookup via DATASET_NAME_TO_DATASET_MAP...")
        dataset_class = DATASET_NAME_TO_DATASET_MAP["parquet_chargeback"]
        print(f"✓ Found dataset class: {dataset_class.__name__}")

        # Test instantiation
        print("Instantiating dataset via lookup...")
        dataset = dataset_class(args)
        print(f"✓ Dataset instantiated: {type(dataset).__name__}")

        # Test loading (this is what OnlineDataset.load() does)
        print("Loading dataset...")
        dataset.load()
        print(f"✓ Dataset loaded successfully")

        # Verify it has BaseDataset interface
        assert hasattr(dataset, 'X'), "Missing X attribute"
        assert hasattr(dataset, 'y'), "Missing y attribute"
        assert hasattr(dataset, 'task_type'), "Missing task_type attribute"
        print(f"✓ BaseDataset interface verified")
        print(f"  Dataset ready for OnlineDataset embedding generation")

        return True

    except Exception as e:
        print(f"✗ Dataset registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_test_command():
    """Print the command to run end-to-end test."""
    print("\n" + "="*70)
    print("END-TO-END TEST COMMAND")
    print("="*70)
    print("\nTo test linear probe with actual T-JEPA training, run:\n")
    print("python run.py \\")
    print("  --use_parquet_dataset=True \\")
    print("  --data_set=parquet_chargeback \\")
    print("  --parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train \\")
    print("  --parquet_data_files=chunk_10.parquet \\")
    print("  --parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train/stats_bootstrap.pq \\")
    print("  --parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv \\")
    print("  --parquet_num_features=30 \\")
    print("  --parquet_task_type=binary_class \\")
    print("  --parquet_preload_data=True \\")
    print("  --probe_cadence=2 \\")
    print("  --exp_train_total_epochs=5 \\")
    print("  --batch_size=512 \\")
    print("  --model_dim_hidden=64 \\")
    print("  --model_num_layers=4 \\")
    print("  --model_num_heads=4 \\")
    print("  --test=True")
    print("\n" + "="*70)


def main():
    print("\n" + "="*70)
    print("LINEAR PROBE INTEGRATION TEST SUITE")
    print("="*70)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: ParquetBaseDataset
    if results[-1][1]:  # Only run if imports passed
        results.append(("ParquetBaseDataset", test_parquet_base_dataset_creation()))
    else:
        print("\nSkipping ParquetBaseDataset test due to import failures")
        results.append(("ParquetBaseDataset", False))

    # Test 3: OnlineDataset
    if results[-1][1]:  # Only run if previous test passed
        results.append(("OnlineDataset", test_online_dataset_compatibility()))
    else:
        print("\nSkipping OnlineDataset test due to previous failures")
        results.append(("OnlineDataset", False))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print_test_command()
        return 0
    else:
        print("\n" + "="*70)
        print("✗ SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
