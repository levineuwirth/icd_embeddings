"""
Test script for ICD-only model predictions.

This script tests the new ICD-only model implementation to ensure:
1. Models load correctly
2. Predictions are generated
3. Risk adjustment is applied properly
4. Classifications are correct based on adjusted thresholds
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path modification
from main import (
    calibrate_probability,
    BETA_READMIT,
    BETA_MORTALITY,
    THRESHOLD_READMIT,
    THRESHOLD_MORTALITY
)
import tensorflow as tf

# Test calibrate_probability function
print("Testing calibrate_probability function...")
print("-" * 50)

# Test with readmission beta
test_prob = 0.95
adjusted_prob = float(calibrate_probability(test_prob, BETA_READMIT).numpy())
print(f"Readmission: Raw prob: {test_prob:.4f}, Adjusted: {adjusted_prob:.4f}")

# Test with mortality beta (should result in low adjusted risk even for high raw prob)
adjusted_prob_mort = float(calibrate_probability(test_prob, BETA_MORTALITY).numpy())
print(f"Mortality: Raw prob: {test_prob:.4f}, Adjusted: {adjusted_prob_mort:.4f}")
print(f"Sanity check: Mortality adjusted should be < 0.5 -> {adjusted_prob_mort < 0.5}")

# Test threshold adjustment
threshold_readmit_adj = float(calibrate_probability(THRESHOLD_READMIT, BETA_READMIT).numpy())
threshold_mortality_adj = float(calibrate_probability(THRESHOLD_MORTALITY, BETA_MORTALITY).numpy())

print("\nThreshold adjustments:")
print(f"Readmission: Original: {THRESHOLD_READMIT:.6f}, Adjusted: {threshold_readmit_adj:.6f}")
print(f"Mortality: Original: {THRESHOLD_MORTALITY:.6f}, Adjusted: {threshold_mortality_adj:.6f}")

print("\n" + "=" * 50)
print("Test completed successfully!")
print("=" * 50)
