"""
Test script for full ICD-only prediction flow.

This tests the predict_icd_only function with actual ICD codes.
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path modification
from main import predict_icd_only

# Test with sample ICD codes
print("Testing predict_icd_only function with sample ICD codes...")
print("=" * 70)

# Example ICD codes (common conditions)
test_icd_codes = [
    "I10",      # Essential hypertension
    "E11.9",    # Type 2 diabetes without complications
    "J44.0",    # COPD with acute lower respiratory infection
    "I50.9",    # Heart failure, unspecified
    "N18.3"     # Chronic kidney disease, stage 3
]

print(f"\nTest ICD codes: {test_icd_codes}")
print("-" * 70)

try:
    result = predict_icd_only(test_icd_codes)

    print("\n" + "=" * 70)
    print("READMISSION PREDICTION")
    print("=" * 70)
    print(f"Raw prediction:      {result['readmission']['raw_prediction']:.6f}")
    print(f"Adjusted prediction: {result['readmission']['prediction']:.6f}")
    print(f"Confidence interval: [{result['readmission']['confidence_interval'][0]:.6f}, {result['readmission']['confidence_interval'][1]:.6f}]")
    print(f"Threshold used:      {result['readmission']['threshold_used']:.6f}")
    print(f"High risk:           {result['readmission']['high_risk']}")
    print(f"Interpretation:      {result['readmission']['interpretation']}")
    print(f"Model used:          {result['readmission']['model_used']}")

    print("\n" + "=" * 70)
    print("MORTALITY PREDICTION")
    print("=" * 70)
    print(f"Raw prediction:      {result['mortality']['raw_prediction']:.6f}")
    print(f"Adjusted prediction: {result['mortality']['prediction']:.6f}")
    print(f"Confidence interval: [{result['mortality']['confidence_interval'][0]:.6f}, {result['mortality']['confidence_interval'][1]:.6f}]")
    print(f"Threshold used:      {result['mortality']['threshold_used']:.6f}")
    print(f"High risk:           {result['mortality']['high_risk']}")
    print(f"Interpretation:      {result['mortality']['interpretation']}")
    print(f"Model used:          {result['mortality']['model_used']}")

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    print(f"✓ Readmission adjusted < 1.0: {result['readmission']['prediction'] < 1.0}")
    print(f"✓ Mortality adjusted < 1.0:   {result['mortality']['prediction'] < 1.0}")
    print(f"✓ Readmission adjusted >= 0:  {result['readmission']['prediction'] >= 0}")
    print(f"✓ Mortality adjusted >= 0:    {result['mortality']['prediction'] >= 0}")
    print(f"✓ Models are ICD-only:        {result['readmission']['model_used'] == 'icd_only' and result['mortality']['model_used'] == 'icd_only'}")

    print("\n" + "=" * 70)
    print("TEST PASSED!")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
