# ICD-10 Code Data Setup

This directory contains the setup scripts and data for the ICD-10-CM (Clinical Modification) code database used for the ICD search functionality.

## Quick Start

To download and set up the ICD-10 data, run:

```bash
cd backend/data
./download_icd_data.sh
```

This script will:
1. Download the official CMS ICD-10-CM 2026 dataset (~22MB)
2. Extract the ZIP file
3. Parse the XML data and generate a JSON file with all codes
4. Display statistics about the loaded data

## Files

### Committed to Git
- `download_icd_data.sh` - Automated download and setup script
- `parse_icd10.py` - Python script to extract codes from XML and generate JSON
- `README.md` - This file

### Generated (excluded from Git)
- `icd-10.zip` - Official CMS ICD-10-CM 2026 dataset
- `icd10_codes.json` - Parsed JSON file containing 46,881 ICD-10 codes
- `Table and Index/` - Extracted XML files from the zip

## Data Source

The ICD-10 codes come from the **Centers for Medicare & Medicaid Services (CMS)** official ICD-10-CM dataset (2026 version).

**Download URL**: https://www.cms.gov/files/zip/2026-code-tables-tabular-and-index.zip

## Manual Setup

If you prefer to set up manually:

1. Download the ICD-10 data:
   ```bash
   wget https://www.cms.gov/files/zip/2026-code-tables-tabular-and-index.zip -O icd-10.zip
   ```

2. Extract the ZIP:
   ```bash
   unzip icd-10.zip
   ```

3. Parse and generate JSON:
   ```bash
   python parse_icd10.py
   ```

## How the Search Works

The search functionality in `backend/main.py` (`/search_icd/` endpoint):

1. **Loads codes at startup**: The JSON file is loaded into memory when the FastAPI app starts
2. **Search algorithm**: Results are prioritized by match type:
   - Exact code match (highest priority)
   - Code starts with query
   - Code contains query
   - Description contains query
3. **Configurable limit**: Returns up to 50 results by default (configurable via query parameter)

## Data Statistics

- **Total codes**: 46,881 ICD-10-CM codes
- **Format**: JSON dictionary `{code: description}`
- **File size**: ~5MB (JSON)
- **Coverage**: Complete ICD-10-CM 2026 code set

## Example Usage

Search by code:
```
GET /search_icd/?q=I10
→ Returns: {"I10": "Essential (primary) hypertension"}
```

Search by description:
```
GET /search_icd/?q=diabetes&limit=10
→ Returns: First 10 codes related to diabetes
```

## Updating to a New Version

When CMS releases a new ICD-10-CM version:

1. Update the `ICD_URL` in `download_icd_data.sh` to point to the new version
2. Run the download script again: `./download_icd_data.sh`
3. The script will prompt before overwriting existing data
