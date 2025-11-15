# ICD Prediction Backend

This directory contains the backend for the ICD Prediction application. It is a FastAPI application that serves a machine learning model to predict 30-day hospital readmission risk.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To run the application, use the following command from within the `backend` directory:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

*   `GET /`: Root endpoint.
*   `POST /predict/`: Predicts the 30-day readmission risk.
*   `GET /search_icd/`: Searches for ICD-10 codes.

## Testing

To run the tests, use the following command from the root of the project:

```bash
pytest backend/
```
