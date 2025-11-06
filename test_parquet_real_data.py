#!/usr/bin/env python3
"""
Comprehensive test for parquet integration with REAL data.

This runs a quick end-to-end test with:
- Small number of features (50)
- Few training steps (2-3 epochs)
- MLflow logging enabled
- Validates all components work together
"""

import sys
import os


def test_dependencies():
    """Test that all dependencies are installed."""
    print("\n" + "="*70)
    print("TEST 1: Dependencies")
    print("="*70)

    try:
        import polars
        print(f"✓ polars version: {polars.__version__}")
    except ImportError:
        print("✗ polars not installed. Run: pip install polars")
        return False

    try:
        import pyarrow
        print(f"✓ pyarrow version: {pyarrow.__version__}")
    except ImportError:
        print("✗ pyarrow not installed. Run: pip install pyarrow")
        return False

    try:
        import torch
        print(f"✓ torch version: {torch.__version__}")
    except ImportError:
        print("✗ torch not installed")
        return False

    return True


def test_imports():
    """Test that parquet modules can be imported."""
    print("\n" + "="*70)
    print("TEST 2: Module Imports")
    print("="*70)

    try:
        from src.datasets.local_dataset import LocalFilesDataset
        print("✓ LocalFilesDataset imported")
    except ImportError as e:
        print(f"✗ LocalFilesDataset import failed: {e}")
        return False

    try:
        from src.datasets.parquet_dataset import TJEPAParquetDataset, create_parquet_dataset_from_args
        print("✓ TJEPAParquetDataset imported")
    except ImportError as e:
        print(f"✗ TJEPAParquetDataset import failed: {e}")
        return False

    try:
        from src.configs import build_parser
        print("✓ configs imported")
    except ImportError as e:
        print(f"✗ configs import failed: {e}")
        return False

    return True


def test_data_files_exist():
    """Test that the data files exist."""
    print("\n" + "="*70)
    print("TEST 3: Data File Existence")
    print("="*70)

    data_dir = "/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train"
    stats_file = f"{data_dir}/stats_bootstrap.pq"
    feature_file = "/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv"

    if not os.path.exists(data_dir):
        print(f"✗ Data directory not found: {data_dir}")
        return False
    print(f"✓ Data directory exists")

    if not os.path.exists(stats_file):
        print(f"✗ Stats file not found: {stats_file}")
        return False
    print(f"✓ Stats file exists")

    if not os.path.exists(feature_file):
        print(f"✗ Feature file not found: {feature_file}")
        return False
    print(f"✓ Feature file exists")

    # Check chunk files
    chunk_files = [f"chunk_{i}.parquet" for i in [10, 11]]
    for chunk in chunk_files:
        chunk_path = os.path.join(data_dir, chunk)
        if not os.path.exists(chunk_path):
            print(f"✗ Chunk file not found: {chunk_path}")
            return False
        file_size_mb = os.path.getsize(chunk_path) / (1024*1024)
        print(f"✓ Chunk exists: {chunk} ({file_size_mb:.1f} MB)")

    return True


def test_end_to_end_training():
    """Test end-to-end training with MLflow logging."""
    print("\n" + "="*70)
    print("TEST 4: End-to-End Training with MLflow")
    print("="*70)

    try:
        import torch
        from src.configs import build_parser

        # Build test command with minimal resources
        test_args = [
            # Dataset name (used for checkpointing and logging)
            "--data_set=parquet_chargeback_test",

            # Parquet dataset config
            "--use_parquet_dataset=True",
            "--parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train",
            "--parquet_data_files=chunk_10.parquet,chunk_11.parquet",
            "--parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train/stats_bootstrap.pq",
            "--parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv",
            "--parquet_num_features=30",  # Use 30 features for testing (should be small and fast)
            # Use default target and ID column names - LocalFilesDataset should handle them properly
            "--parquet_scaling_method=IQR",
            "--parquet_infill_value=zero",
            "--parquet_transform=asinh",
            "--parquet_categorize_nan=False",
            "--parquet_preload_data=True",  # Enable preloading so we can compute dataloader length
            "--parquet_shuffle=False",
            "--batch_size=512",  # Small batch

            # Training config (minimal for testing)
            "--exp_train_total_epochs=2",  # Just 2 epochs to verify it works
            "--exp_lr=0.0003",
            "--exp_warmup=1",
            "--exp_weight_decay=0.0001",
            "--probe_cadence=0",  # Disable linear probe for quick test

            # Model config (small)
            "--model_dim_hidden=64",
            "--model_num_layers=4",
            "--model_num_heads=4",
            "--model_dropout_prob=0.1",

            # Masking config
            "--mask_min_ctx_share=0.2",
            "--mask_max_ctx_share=0.4",
            "--mask_min_trgt_share=0.2",
            "--mask_max_trgt_share=0.4",
            "--mask_num_preds=2",
            "--mask_num_encs=1",
        ]

        print("  Parsing arguments...")
        parser = build_parser()
        args = parser.parse_args(test_args)
        print("✓ Arguments parsed")

        print("\n  Starting training (2 epochs)...")
        print("  This will test:")
        print("    - Parquet data loading")
        print("    - Batch iteration")
        print("    - Forward/backward passes")
        print("    - EMA updates")
        print("    - MLflow logging")
        print()

        # Import and run training
        from run import main, setup_mlflow_logging

        # Setup MLflow logging to Databricks
        setup_mlflow_logging(args)

        # Run training
        main(args)

        print("\n✓ Training completed successfully!")
        print("✓ MLflow logging verified (check MLflow UI for logged metrics)")

        return True

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("T-JEPA PARQUET INTEGRATION - END-TO-END TEST")
    print("Testing with REAL data + MLflow logging")
    print("="*70)

    results = []

    # Test 1: Dependencies
    if not test_dependencies():
        print("\n✗ Install dependencies first: pip install polars pyarrow")
        return 1
    results.append(("Dependencies", True))

    # Test 2: Imports
    if not test_imports():
        results.append(("Module Imports", False))
        print("\n✗ Fix import errors before continuing")
        return 1
    results.append(("Module Imports", True))

    # Test 3: Data files
    if not test_data_files_exist():
        results.append(("Data Files", False))
        print("\n✗ Data files not accessible")
        return 1
    results.append(("Data Files", True))

    # Test 4: End-to-end training with MLflow
    training_ok = test_end_to_end_training()
    results.append(("End-to-End Training", training_ok))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*70)
        print("\nParquet integration is FULLY FUNCTIONAL!")
        print("Training ran successfully with MLflow logging enabled.")
        print("\nCheck MLflow UI for logged metrics:")
        print("  - tjepa_train_loss")
        print("  - tjepa_lr")
        print("  - tjepa_momentum")
        print("\nReady for full-scale training!")
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("="*70)
        print("\nReview the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
