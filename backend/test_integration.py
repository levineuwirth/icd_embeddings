"""
Integration test to verify model multiplexing works correctly.

Tests:
1. Full demographics → uses full_demographic model
2. Missing age → uses icd_only model
3. Missing gender → uses icd_only model
4. Missing pay1 → uses icd_only model
5. Missing zipinc_qrtl → uses icd_only model
6. No demographics (only ICD) → uses icd_only model
"""

import requests
import json

# Update this to match your backend URL
API_URL = "http://localhost:8000"

test_icd_codes = ["I10", "E11.9", "J44.0"]

print("=" * 80)
print("INTEGRATION TEST: Model Multiplexing")
print("=" * 80)

# Test 1: Full demographics
print("\n[Test 1] Full demographics provided")
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
    model_used = result['readmission']['model_used']
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {model_used}")
    print(f"✓ Expected: full_demographic")
    print(f"✓ PASS: {model_used == 'full_demographic'}")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

# Test 2: Missing age
print("\n[Test 2] Missing age (partial demographics)")
print("-" * 80)
payload = {
    "female": 1,
    "pay1": 1,
    "zipinc_qrtl": 3,
    "icd_codes": test_icd_codes
}
response = requests.post(f"{API_URL}/predict_flex/", json=payload)
if response.status_code == 200:
    result = response.json()
    model_used = result['readmission']['model_used']
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {model_used}")
    print(f"✓ Expected: icd_only")
    print(f"✓ PASS: {model_used == 'icd_only'}")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

# Test 3: Missing gender
print("\n[Test 3] Missing gender (partial demographics)")
print("-" * 80)
payload = {
    "age": 65,
    "pay1": 1,
    "zipinc_qrtl": 3,
    "icd_codes": test_icd_codes
}
response = requests.post(f"{API_URL}/predict_flex/", json=payload)
if response.status_code == 200:
    result = response.json()
    model_used = result['readmission']['model_used']
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {model_used}")
    print(f"✓ Expected: icd_only")
    print(f"✓ PASS: {model_used == 'icd_only'}")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

# Test 4: Missing pay1
print("\n[Test 4] Missing pay1 (partial demographics)")
print("-" * 80)
payload = {
    "age": 65,
    "female": 1,
    "zipinc_qrtl": 3,
    "icd_codes": test_icd_codes
}
response = requests.post(f"{API_URL}/predict_flex/", json=payload)
if response.status_code == 200:
    result = response.json()
    model_used = result['readmission']['model_used']
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {model_used}")
    print(f"✓ Expected: icd_only")
    print(f"✓ PASS: {model_used == 'icd_only'}")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

# Test 5: Missing zipinc_qrtl
print("\n[Test 5] Missing zipinc_qrtl (partial demographics)")
print("-" * 80)
payload = {
    "age": 65,
    "female": 1,
    "pay1": 1,
    "icd_codes": test_icd_codes
}
response = requests.post(f"{API_URL}/predict_flex/", json=payload)
if response.status_code == 200:
    result = response.json()
    model_used = result['readmission']['model_used']
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {model_used}")
    print(f"✓ Expected: icd_only")
    print(f"✓ PASS: {model_used == 'icd_only'}")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

# Test 6: Only ICD codes (no demographics)
print("\n[Test 6] Only ICD codes (no demographics)")
print("-" * 80)
payload = {
    "icd_codes": test_icd_codes
}
response = requests.post(f"{API_URL}/predict_flex/", json=payload)
if response.status_code == 200:
    result = response.json()
    model_used = result['readmission']['model_used']
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model used: {model_used}")
    print(f"✓ Expected: icd_only")
    print(f"✓ PASS: {model_used == 'icd_only'}")

    # Additional validation for ICD-only
    print(f"\nAdditional ICD-only validation:")
    print(f"  Readmission raw: {result['readmission']['raw_prediction']:.6f}")
    print(f"  Readmission adjusted: {result['readmission']['prediction']:.6f}")
    print(f"  Mortality raw: {result['mortality']['raw_prediction']:.6f}")
    print(f"  Mortality adjusted: {result['mortality']['prediction']:.6f}")
    print(f"  Readmission threshold: {result['readmission']['threshold_used']:.6f}")
    print(f"  Mortality threshold: {result['mortality']['threshold_used']:.6f}")
else:
    print(f"✗ FAILED: {response.status_code} - {response.text}")

print("\n" + "=" * 80)
print("INTEGRATION TEST COMPLETE")
print("=" * 80)
print("\nNote: Run the backend server first with: cd backend && uvicorn main:app --reload")
