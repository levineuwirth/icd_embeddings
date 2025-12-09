# Use a specific, compatible Python version
# Multi-platform support (amd64, arm64 for Apple Silicon)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install required system packages for downloading and extracting ICD data
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching
COPY ./backend/requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY ./backend /app

# Copy the model files
COPY ./model /app/model

# Download and set up ICD-10 data
RUN cd /app/data && \
    echo "Downloading ICD-10-CM data from CMS..." && \
    wget -q https://www.cms.gov/files/zip/2026-code-tables-tabular-and-index.zip -O icd-10.zip && \
    echo "Extracting ICD-10 data..." && \
    unzip -q icd-10.zip && \
    echo "Parsing ICD-10 codes..." && \
    python parse_icd10.py && \
    echo "Cleaning up extracted XML files to reduce image size..." && \
    rm -rf "Table and Index" && \
    rm icd-10.zip && \
    echo "ICD-10 setup complete!"

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# The host must be 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
