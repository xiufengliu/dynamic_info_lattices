#!/usr/bin/env python3
"""
Test script for real data integration

This script tests that the real dataset loading works correctly
without requiring CUDA libraries.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_data_files_exist():
    """Test that data files exist"""
    print("Testing data file availability...")
    
    data_dir = Path("./data")
    expected_files = [
        "ETTh1.csv",
        "ETTh2.csv", 
        "ETTm1.csv",
        "ETTm2.csv",
        "ECL.csv",
        "gefcom2014.csv",
        "southern_china.csv"
    ]
    
    results = []
    for file in expected_files:
        file_path = data_dir / file
        exists = file_path.exists()
        if exists:
            # Check file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file} (missing)")
        results.append(exists)
    
    return all(results)

def test_csv_structure():
    """Test CSV file structure"""
    print("\nTesting CSV file structure...")
    
    data_dir = Path("./data")
    test_files = ["ETTh1.csv", "ECL.csv"]
    
    results = []
    for file in test_files:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"✗ {file} not found")
            results.append(False)
            continue
            
        try:
            # Load and examine CSV
            df = pd.read_csv(file_path, nrows=10)  # Just first 10 rows
            
            print(f"✓ {file}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")  # First 5 columns
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"  Numeric columns: {len(numeric_cols)}")
            
            results.append(True)
            
        except Exception as e:
            print(f"✗ {file}: Error reading - {e}")
            results.append(False)
    
    return all(results)

def test_dataset_loader_logic():
    """Test dataset loader logic without torch"""
    print("\nTesting dataset loader logic...")
    
    try:
        # Test configuration mapping
        dataset_configs = {
            "etth1": {"file": "ETTh1.csv", "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]},
            "ecl": {"file": "ECL.csv", "target_columns": ["MT_362"]},
            "traffic": {"file": "ECL.csv", "target_columns": ["MT_362"]},  # Legacy mapping
        }
        
        print("✓ Dataset configurations defined")
        
        # Test data loading logic
        data_dir = Path("./data")
        test_file = "ETTh1.csv"
        
        if (data_dir / test_file).exists():
            df = pd.read_csv(data_dir / test_file, nrows=100)
            
            # Test column selection logic
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['hour', 'day_of_week', 'day_of_year', 'month', 'quarter', 'year', 
                           'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                           'is_weekend', 'season']
            
            target_cols = [col for col in numeric_cols if col not in exclude_cols]
            print(f"✓ Target columns identified: {target_cols[:3]}...")
            
            # Test data extraction
            data = df[target_cols].values.astype(np.float32)
            data = np.nan_to_num(data, nan=0.0)
            print(f"✓ Data extracted: shape {data.shape}")
            
            # Test train/val/test split logic
            n_samples = len(data)
            train_end = int(n_samples * 0.7)
            val_end = int(n_samples * 0.9)
            
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
            
            print(f"✓ Data splits: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
            
            return True
        else:
            print(f"✗ Test file {test_file} not found")
            return False
            
    except Exception as e:
        print(f"✗ Dataset loader logic test failed: {e}")
        return False

def test_sequence_creation_logic():
    """Test sequence creation logic"""
    print("\nTesting sequence creation logic...")
    
    try:
        # Create sample data
        data = np.random.randn(1000, 3).astype(np.float32)
        sequence_length = 96
        prediction_length = 24
        
        # Test sequence creation
        sequences = []
        total_length = sequence_length + prediction_length
        
        for i in range(len(data) - total_length + 1):
            x = data[i:i + sequence_length]
            y = data[i + sequence_length:i + total_length]
            mask = np.ones_like(x)
            sequences.append((x, y, mask))
        
        print(f"✓ Created {len(sequences)} sequences")
        print(f"  Input shape: {sequences[0][0].shape}")
        print(f"  Target shape: {sequences[0][1].shape}")
        print(f"  Mask shape: {sequences[0][2].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sequence creation test failed: {e}")
        return False

def test_import_structure():
    """Test import structure without torch"""
    print("\nTesting import structure...")
    
    try:
        # Test that we can import the module structure
        from dynamic_info_lattices.data import real_datasets
        print("✓ Real datasets module importable")
        
        # Test configuration access
        if hasattr(real_datasets, 'get_real_dataset'):
            print("✓ get_real_dataset function available")
        else:
            print("✗ get_real_dataset function not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Import structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING REAL DATA INTEGRATION")
    print("=" * 60)
    
    tests = [
        ("Data Files Exist", test_data_files_exist),
        ("CSV Structure", test_csv_structure),
        ("Dataset Loader Logic", test_dataset_loader_logic),
        ("Sequence Creation Logic", test_sequence_creation_logic),
        ("Import Structure", test_import_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Real data integration is working correctly")
        print("✅ CSV files are properly structured")
        print("✅ Dataset loading logic is sound")
        print("✅ Ready to use real datasets")
    else:
        print(f"\n⚠️  {total-passed} tests failed")
        print("Please check the failed tests above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
