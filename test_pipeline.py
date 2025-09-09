#!/usr/bin/env python3
"""
Test script for the response-level analysis pipeline.
Tests each component individually before running the full pipeline.
"""

import sys
from pathlib import Path

def test_csv_processor():
    """Test the CSV data processor."""
    print("="*50)
    print("TESTING CSV DATA PROCESSOR")
    print("="*50)
    
    from response_level_analysis.csv_data_processor import (
        load_and_filter_csv, 
        print_sample_statistics
    )
    
    csv_path = '/workspace/model-organisms-for-EM/em_organism_dir/results/5_epoch_rank1_lora_evaluation.csv'
    
    try:
        misaligned, aligned = load_and_filter_csv(
            csv_path,
            coherence_threshold=50.0,
            alignment_threshold_low=30.0, 
            alignment_threshold_high=80.0
        )
        
        print_sample_statistics(misaligned, aligned)
        print(f"✓ CSV processor test passed: {len(misaligned)} misaligned, {len(aligned)} aligned")
        return True, (misaligned, aligned)
        
    except Exception as e:
        print(f"✗ CSV processor test failed: {e}")
        return False, None


def test_shared_utilities():
    """Test shared utilities."""
    print("\n" + "="*50)
    print("TESTING SHARED UTILITIES") 
    print("="*50)
    
    try:
        from shared.feature_statistics_computer import (
            compute_group_feature_statistics,
            rank_features_by_difference,
            FeatureStats
        )
        
        # Create mock data
        import torch
        evil_features = [torch.randn(100), torch.randn(100), torch.randn(100)]
        safe_features = [torch.randn(100), torch.randn(100)]
        
        stats = compute_group_feature_statistics(evil_features, safe_features)
        ranked = rank_features_by_difference(stats, top_k=10)
        
        print(f"✓ Statistics computation passed: {len(ranked)} ranked features")
        return True, (stats, ranked)
        
    except Exception as e:
        print(f"✗ Shared utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_main_script_help():
    """Test that the main script shows help."""
    print("\n" + "="*50)
    print("TESTING MAIN SCRIPT")
    print("="*50)
    
    import subprocess
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, 'response_level_main.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'Response-level feature analysis' in result.stdout:
            print("✓ Main script help works correctly")
            return True
        else:
            print(f"✗ Main script help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Main script test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("RESPONSE-LEVEL ANALYSIS PIPELINE TESTS")
    print("="*60)
    
    all_passed = True
    
    # Test CSV processor
    csv_passed, csv_data = test_csv_processor()
    all_passed &= csv_passed
    
    # Test shared utilities  
    shared_passed, shared_data = test_shared_utilities()
    all_passed &= shared_passed
    
    # Test main script
    main_passed = test_main_script_help()
    all_passed &= main_passed
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("✓ All tests passed! Pipeline is ready for use.")
        print("\nTo run the full analysis:")
        print("python response_level_main.py /path/to/csv /path/to/model --output_dir results")
        
        if csv_data:
            misaligned, aligned = csv_data
            print(f"\nCurrent test data: {len(misaligned)} misaligned, {len(aligned)} aligned samples")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())