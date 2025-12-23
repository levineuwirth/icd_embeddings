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
# Using shell form to allow environment variable expansion
# Uses gunicorn with uvicorn workers for production multi-process serving
# ${MODULE_NAME:-main} defaults to "main" if MODULE_NAME not set
# ${WORKERS:-1} defaults to 1 worker if WORKERS not set
# --timeout 300 allows ML predictions to complete without worker timeout
CMD gunicorn -k uvicorn.workers.UvicornWorker ${MODULE_NAME:-main}:app \
    --bind 0.0.0.0:8000 \
    --workers ${WORKERS:-1} \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
