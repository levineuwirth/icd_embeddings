"""
Parse ICD-10-CM XML file and create a JSON file for efficient loading.
"""

import xml.etree.ElementTree as ET
import json
import os

def extract_codes(element, codes_dict):
    """
    Recursively extract all diagnosis codes and descriptions from XML.

    Args:
        element: XML element to process
        codes_dict: Dictionary to store code-description pairs
    """
    for diag in element.findall('.//diag'):
        name_elem = diag.find('name')
        desc_elem = diag.find('desc')

        if name_elem is not None and desc_elem is not None:
            code = name_elem.text
            description = desc_elem.text
            if code and description:
                codes_dict[code] = description

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(base_dir, 'Table and Index', 'icd10cm_tabular_2026.xml')
    output_file = os.path.join(base_dir, 'icd10_codes.json')

    print(f"Parsing {xml_file}...")

    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract all codes
    codes_dict = {}
    extract_codes(root, codes_dict)

    print(f"Found {len(codes_dict)} ICD-10 codes")

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(codes_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_file}")

    # Display sample
    print("\nSample codes:")
    for i, (code, desc) in enumerate(list(codes_dict.items())[:5]):
        print(f"  {code}: {desc}")

if __name__ == "__main__":
    main()
