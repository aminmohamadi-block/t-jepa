#!/usr/bin/env python
"""
Test script to verify parquet dataset integration.

This script tests that all components are correctly imported and configured.
"""

import sys
import argparse

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import polars as pl
        print("  ✓ polars imported successfully")
    except ImportError:
        print("  ✗ polars not found. Install with: pip install polars")
        return False

    try:
        import pyarrow.parquet as pq
        print("  ✓ pyarrow imported successfully")
    except ImportError:
        print("  ✗ pyarrow not found. Install with: pip install pyarrow")
        return False

    try:
        from src.datasets.local_dataset import LocalFilesDataset
        print("  ✓ LocalFilesDataset imported successfully")
    except ImportError as e:
        print(f"  ✗ LocalFilesDataset import failed: {e}")
        return False

    try:
        from src.datasets.parquet_dataset import create_parquet_dataset_from_args
        print("  ✓ parquet_dataset module imported successfully")
    except ImportError as e:
        print(f"  ✗ parquet_dataset import failed: {e}")
        return False

    try:
        from src.configs import build_parser
        print("  ✓ configs module imported successfully")
    except ImportError as e:
        print(f"  ✗ configs import failed: {e}")
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_config_parsing():
    """Test that parquet-related arguments are properly parsed."""
    print("Testing config parsing...")
    try:
        from src.configs import build_parser
        parser = build_parser()

        # Create test args
        test_args = [
            "--use_parquet_dataset=True",
            "--parquet_data_dir=/test/path",
            "--parquet_scaling_stats_file=/test/stats.pq",
            "--parquet_num_features=100",
            "--batch_size=256",
        ]

        args = parser.parse_args(test_args)

        # Verify parquet args are present
        assert args.use_parquet_dataset == True, "use_parquet_dataset not parsed"
        assert args.parquet_data_dir == "/test/path", "parquet_data_dir not parsed"
        assert args.parquet_scaling_stats_file == "/test/stats.pq", "parquet_scaling_stats_file not parsed"
        assert args.parquet_num_features == 100, "parquet_num_features not parsed"
        assert args.batch_size == 256, "batch_size not parsed"

        print("  ✓ All parquet arguments parsed correctly")
        print(f"  ✓ use_parquet_dataset: {args.use_parquet_dataset}")
        print(f"  ✓ parquet_data_dir: {args.parquet_data_dir}")
        print(f"  ✓ parquet_num_features: {args.parquet_num_features}")
        print(f"  ✓ parquet_scaling_method: {args.parquet_scaling_method}")

        print("\n✓ Config parsing successful!\n")
        return True

    except Exception as e:
        print(f"  ✗ Config parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_run_py_integration():
    """Test that run.py can import the parquet integration."""
    print("Testing run.py integration...")
    try:
        # Try importing run.py's main dependencies
        from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
        from src.datasets.parquet_dataset import create_parquet_dataset_from_args
        print("  ✓ run.py imports successful")
        print("\n✓ run.py integration successful!\n")
        return True
    except Exception as e:
        print(f"  ✗ run.py integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("T-JEPA Parquet Dataset Integration Test")
    print("="*70)
    print()

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Config parsing
    if results[0][1]:  # Only if imports succeeded
        results.append(("Config Parsing", test_config_parsing()))

    # Test 3: run.py integration
    if results[0][1]:  # Only if imports succeeded
        results.append(("run.py Integration", test_run_py_integration()))

    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests passed! Parquet integration is ready.")
        print("\nNext steps:")
        print("1. Prepare your parquet dataset with:")
        print("   - Parquet chunk files (chunk_0.parquet, chunk_1.parquet, ...)")
        print("   - Scaling statistics file (stats_bootstrap.pq)")
        print("   - Feature importance file (mi.csv) [optional]")
        print("\n2. Run T-JEPA with parquet dataset:")
        print("   python run.py \\")
        print("     --use_parquet_dataset=True \\")
        print("     --parquet_data_dir=/path/to/parquet/files \\")
        print("     --parquet_scaling_stats_file=/path/to/stats_bootstrap.pq \\")
        print("     --parquet_feature_names_file=/path/to/mi.csv \\")
        print("     --parquet_num_features=512 \\")
        print("     --batch_size=1024 \\")
        print("     --exp_train_total_epochs=100")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("- Missing dependencies: pip install polars pyarrow")
        print("- Incorrect file paths or module structure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
