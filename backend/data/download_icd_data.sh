#!/bin/bash

# Script to download and process ICD-10-CM codes from CMS
# This downloads the official 2026 ICD-10-CM code tables from the Centers for Medicare & Medicaid Services

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ICD-10-CM Data Download and Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Download URL
ICD_URL="https://www.cms.gov/files/zip/2026-code-tables-tabular-and-index.zip"
ZIP_FILE="icd-10.zip"
JSON_FILE="icd10_codes.json"

# Check if JSON file already exists
if [ -f "$JSON_FILE" ]; then
    echo -e "${YELLOW}Warning: $JSON_FILE already exists.${NC}"
    read -p "Do you want to re-download and regenerate? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Skipping download. Using existing file.${NC}"
        exit 0
    fi
fi

# Download the ICD-10 data
echo -e "\n${BLUE}Step 1: Downloading ICD-10-CM data from CMS...${NC}"
if command -v wget &> /dev/null; then
    wget -O "$ZIP_FILE" "$ICD_URL" --progress=bar:force 2>&1
elif command -v curl &> /dev/null; then
    curl -L -o "$ZIP_FILE" "$ICD_URL" --progress-bar
else
    echo -e "${YELLOW}Error: Neither wget nor curl is installed.${NC}"
    echo "Please install wget or curl and try again."
    exit 1
fi

echo -e "${GREEN}✓ Download complete${NC}"

# Unzip the file
echo -e "\n${BLUE}Step 2: Extracting ZIP file...${NC}"
unzip -q -o "$ZIP_FILE"
echo -e "${GREEN}✓ Extraction complete${NC}"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python is not installed.${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Parse the XML and create JSON
echo -e "\n${BLUE}Step 3: Parsing XML and generating JSON...${NC}"
$PYTHON_CMD parse_icd10.py

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "ICD-10 codes are now ready to use."
echo -e "Total codes: $(grep -o '": "' "$JSON_FILE" | wc -l)"
echo -e ""
echo -e "Files created:"
echo -e "  - $ZIP_FILE (source file)"
echo -e "  - $JSON_FILE (parsed codes)"
echo -e "  - Table and Index/ (extracted XML files)"
