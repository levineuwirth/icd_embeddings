"""
Test script for flexible ICD code parsing functionality.
"""

import json
import os
import sys

# Add parent directory to path to import from main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import parse_icd_codes_from_text, icd_codes

print("=" * 80)
print("ICD CODE PARSING & VALIDATION TESTS")
print("=" * 80)
print(f"Total ICD codes in database: {len(icd_codes)}\n")

# Test cases with different formats
test_cases = [
    {
        "name": "Comma-separated",
        "text": "I10, E11.9, J44.0, A00.0"
    },
    {
        "name": "Space-separated",
        "text": "I10 E11.9 J44.0 A00.0"
    },
    {
        "name": "Line-separated",
        "text": """I10
E11.9
J44.0
A00.0"""
    },
    {
        "name": "Mixed format",
        "text": """I10, E11.9
J44.0 A00.0"""
    },
    {
        "name": "With duplicates",
        "text": "I10, E11.9, I10, J44.0, E11.9"
    },
    {
        "name": "With invalid codes",
        "text": "I10, INVALID123, E11.9, BADCODE"
    },
    {
        "name": "Case insensitive",
        "text": "i10, e11.9, j44.0"
    },
    {
        "name": "More than 35 codes",
        "text": ", ".join([f"I{10+i}" for i in range(40)])
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['name']}")
    print("-" * 80)
    print(f"Input: {repr(test['text'][:80])}{' ...' if len(test['text']) > 80 else ''}")

    result = parse_icd_codes_from_text(test['text'])

    print(f"✓ Valid codes: {len(result['valid_codes'])}")
    if result['valid_codes']:
        print(f"  {', '.join(result['valid_codes'][:10])}")
        if len(result['valid_codes']) > 10:
            print(f"  ... and {len(result['valid_codes']) - 10} more")

    if result['invalid_codes']:
        print(f"⚠ Invalid codes: {len(result['invalid_codes'])}")
        for inv in result['invalid_codes'][:3]:
            suggestions = f" (suggestions: {', '.join(inv['suggestions'])})" if inv['suggestions'] else ""
            print(f"  - {inv['code']}{suggestions}")
        if len(result['invalid_codes']) > 3:
            print(f"  ... and {len(result['invalid_codes']) - 3} more")

    if result['warnings']:
        for warning in result['warnings']:
            print(f"⚠ {warning}")

    print()

print("=" * 80)
print("✓ All parsing tests completed!")
print("=" * 80)
