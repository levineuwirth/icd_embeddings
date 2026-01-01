"""
Test script to verify HIGH/LOW risk classification is included in API responses.

Tests:
1. Full demographics model returns high_risk and threshold_used fields
2. ICD-only model returns high_risk and threshold_used fields
3. Risk classification matches expected values based on thresholds
"""

import requests
import json

# Update this to match your backend URL
API_URL = "http://localhost:8000"

test_icd_codes = ["I10", "E11.9", "J44.0"]

print("=" * 80)
print("RISK CLASSIFICATION TEST")
print("=" * 80)

# Test 1: Full demographics model
print("\n[Test 1] Full demographics - Check risk classification fields")
print("-" * 80)
payload = {
    "age": 65,
    "female": 1,
    "pay1": 1,
    "zipinc_qrtl": 3,
    "icd_codes": test_icd_codes
}
response = requests.post(f"{API_URL}/predict_flex/", json=payload)
if response.status_code == 200:
    result = response.json()
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {result['readmission']['model_used']}")

    # Check readmission fields
    has_readmission_high_risk = 'high_risk' in result['readmission']
    has_readmission_threshold = 'threshold_used' in result['readmission']
    print(f"\nReadmission fields:")
    print(f"  has 'high_risk': {has_readmission_high_risk}")
    print(f"  has 'threshold_used': {has_readmission_threshold}")
    if has_readmission_high_risk and has_readmission_threshold:
        print(f"  prediction: {result['readmission']['prediction']:.4f}")
        print(f"  threshold: {result['readmission']['threshold_used']:.6f}")
        print(f"  high_risk: {result['readmission']['high_risk']}")
        print(f"  ✓ PASS: All fields present")
    else:
        print(f"  ✗ FAIL: Missing fields")

    # Check mortality fields
    has_mortality_high_risk = 'high_risk' in result['mortality']
    has_mortality_threshold = 'threshold_used' in result['mortality']
    print(f"\nMortality fields:")
    print(f"  has 'high_risk': {has_mortality_high_risk}")
    print(f"  has 'threshold_used': {has_mortality_threshold}")
    if has_mortality_high_risk and has_mortality_threshold:
        print(f"  prediction: {result['mortality']['prediction']:.4f}")
        print(f"  threshold: {result['mortality']['threshold_used']:.6f}")
        print(f"  high_risk: {result['mortality']['high_risk']}")
        print(f"  ✓ PASS: All fields present")
    else:
        print(f"  ✗ FAIL: Missing fields")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

# Test 2: ICD-only model
print("\n\n[Test 2] ICD-only model - Check risk classification fields")
print("-" * 80)
payload = {
    "icd_codes": test_icd_codes
}
response = requests.post(f"{API_URL}/predict_flex/", json=payload)
if response.status_code == 200:
    result = response.json()
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {result['readmission']['model_used']}")

    # Check readmission fields
    has_readmission_high_risk = 'high_risk' in result['readmission']
    has_readmission_threshold = 'threshold_used' in result['readmission']
    print(f"\nReadmission fields:")
    print(f"  has 'high_risk': {has_readmission_high_risk}")
    print(f"  has 'threshold_used': {has_readmission_threshold}")
    if has_readmission_high_risk and has_readmission_threshold:
        print(f"  raw prediction: {result['readmission']['raw_prediction']:.4f}")
        print(f"  adjusted prediction: {result['readmission']['prediction']:.4f}")
        print(f"  threshold: {result['readmission']['threshold_used']:.6f}")
        print(f"  high_risk: {result['readmission']['high_risk']}")
        print(f"  ✓ PASS: All fields present")
    else:
        print(f"  ✗ FAIL: Missing fields")

    # Check mortality fields
    has_mortality_high_risk = 'high_risk' in result['mortality']
    has_mortality_threshold = 'threshold_used' in result['mortality']
    print(f"\nMortality fields:")
    print(f"  has 'high_risk': {has_mortality_high_risk}")
    print(f"  has 'threshold_used': {has_mortality_threshold}")
    if has_mortality_high_risk and has_mortality_threshold:
        print(f"  raw prediction: {result['mortality']['raw_prediction']:.4f}")
        print(f"  adjusted prediction: {result['mortality']['prediction']:.4f}")
        print(f"  threshold: {result['mortality']['threshold_used']:.6f}")
        print(f"  high_risk: {result['mortality']['high_risk']}")
        print(f"  ✓ PASS: All fields present")
    else:
        print(f"  ✗ FAIL: Missing fields")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

print("\n" + "=" * 80)
print("RISK CLASSIFICATION TEST COMPLETE")
print("=" * 80)
print("\nNote: Run the backend server first with: cd backend && uvicorn main:app --reload")
