"""
Standalone test script for ICD search functionality.
Tests the search_icd endpoint logic without requiring the full API.
"""

import json
import os


def search_icd(icd_codes_dict, query, limit=50):
    """
    Simulates the search_icd endpoint logic.

    Args:
        icd_codes_dict: Dictionary of ICD codes and descriptions
        query: Search query string
        limit: Maximum number of results

    Returns:
        Dictionary of matching ICD codes and descriptions
    """
    if not query or len(query.strip()) == 0:
        return {}

    q = query.strip().lower()
    results = {}

    # Categorize results by match type for better ordering
    exact_code_matches = {}
    code_starts_with = {}
    code_contains = {}
    desc_contains = {}

    for code, description in icd_codes_dict.items():
        code_lower = code.lower()
        desc_lower = description.lower()

        # Exact code match (highest priority)
        if code_lower == q:
            exact_code_matches[code] = description
        # Code starts with query (high priority)
        elif code_lower.startswith(q):
            code_starts_with[code] = description
        # Code contains query (medium priority)
        elif q in code_lower:
            code_contains[code] = description
        # Description contains query (lower priority)
        elif q in desc_lower:
            desc_contains[code] = description

    # Combine results in priority order
    results.update(exact_code_matches)
    results.update(code_starts_with)
    results.update(code_contains)
    results.update(desc_contains)

    # Limit results
    if len(results) > limit:
        results = dict(list(results.items())[:limit])

    return results


def main():
    # Load ICD codes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    icd_data_path = os.path.join(base_dir, 'data/icd10_codes.json')

    print("Loading ICD-10 codes...")
    with open(icd_data_path, 'r', encoding='utf-8') as f:
        icd_codes = json.load(f)

    print(f"✓ Loaded {len(icd_codes)} ICD-10 codes\n")

    # Test cases
    test_queries = [
        ("I10", 10),
        ("diabetes", 10),
        ("hypertension", 10),
        ("A00", 10),
        ("fracture", 10),
        ("covid", 10),
        ("J12", 10),
    ]

    print("=" * 80)
    print("ICD-10 SEARCH FUNCTIONALITY TESTS")
    print("=" * 80)

    for query, limit in test_queries:
        print(f"\n🔍 Search query: \"{query}\" (limit: {limit})")
        print("-" * 80)

        results = search_icd(icd_codes, query, limit)

        if not results:
            print("  No results found")
        else:
            print(f"  Found {len(results)} result(s):\n")
            for i, (code, desc) in enumerate(results.items(), 1):
                # Truncate long descriptions
                desc_display = desc if len(desc) <= 70 else desc[:67] + "..."
                print(f"  {i:2d}. {code:8s} - {desc_display}")

    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
