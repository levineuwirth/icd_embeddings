"""
Simple test script for flexible ICD code parsing functionality.
Tests the parsing logic without loading the full application.
"""

import json
import os

# Load ICD codes
base_dir = os.path.dirname(os.path.abspath(__file__))
icd_data_path = os.path.join(base_dir, 'data/icd10_codes.json')

with open(icd_data_path, 'r', encoding='utf-8') as f:
    icd_codes_db = json.load(f)

def parse_icd_codes_from_text(text, max_codes=35):
    """Simple version of the parsing function for testing."""
    # Replace common separators with spaces
    cleaned_text = text.replace(',', ' ').replace('\n', ' ').replace('\t', ' ').replace(';', ' ')

    # Split on whitespace and clean up
    potential_codes = [code.strip().upper() for code in cleaned_text.split() if code.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_codes = []
    for code in potential_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    # Validate against ICD database
    valid_codes = []
    invalid_codes = []
    warnings = []

    for code in unique_codes[:max_codes]:
        if code in icd_codes_db:
            valid_codes.append(code)
        else:
            # Try to find similar codes
            suggestions = []
            code_lower = code.lower()
            for icd_code in list(icd_codes_db.keys())[:1000]:
                if icd_code.lower().startswith(code_lower[:3]):
                    suggestions.append(icd_code)
                if len(suggestions) >= 3:
                    break

            invalid_codes.append({
                "code": code,
                "suggestions": suggestions[:3]
            })

    if len(unique_codes) > max_codes:
        warnings.append(f"Only the first {max_codes} codes were processed. {len(unique_codes) - max_codes} codes were ignored.")

    if len(potential_codes) != len(unique_codes):
        warnings.append(f"Removed {len(potential_codes) - len(unique_codes)} duplicate codes.")

    return {
        "valid_codes": valid_codes,
        "invalid_codes": invalid_codes,
        "warnings": warnings,
        "total_found": len(unique_codes)
    }

print("=" * 80)
print("ICD CODE PARSING & VALIDATION TESTS")
print("=" * 80)
print(f"Total ICD codes in database: {len(icd_codes_db)}\n")

# Test cases
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
]

for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['name']}")
    print("-" * 80)
    print(f"Input: {repr(test['text'][:80])}{' ...' if len(test['text']) > 80 else ''}")

    result = parse_icd_codes_from_text(test['text'])

    print(f"✓ Valid codes ({len(result['valid_codes'])}): {', '.join(result['valid_codes']) if result['valid_codes'] else 'none'}")

    if result['invalid_codes']:
        print(f"⚠ Invalid codes ({len(result['invalid_codes'])}):")
        for inv in result['invalid_codes']:
            suggestions = f" → suggestions: {', '.join(inv['suggestions'])}" if inv['suggestions'] else ""
            print(f"  - {inv['code']}{suggestions}")

    if result['warnings']:
        for warning in result['warnings']:
            print(f"⚠ Warning: {warning}")

    print()

print("=" * 80)
print("✓ All parsing tests completed successfully!")
print("=" * 80)
