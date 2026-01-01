---
title: ICD Prediction API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# ICD Prediction API (Backend)

This is the backend API for predicting 30-day hospital readmission and mortality risk based on patient data and ICD-10 diagnosis codes. It uses FastAPI with TensorFlow/Keras machine learning models.

## Repository Structure

This project is organized as a monorepo, with the frontend and backend code in separate directories:

*   `backend/`: Contains the FastAPI backend application, including the machine learning model and API endpoints. See the [backend/README.md](backend/README.md) for more details.
*   `src/`: Contains the React frontend application.
*   `model/`: Contains the machine learning model files.

## Deployment

This branch (`huggingface-backend`) is configured for deploying the backend API to Hugging Face Spaces using Docker.

### Hugging Face Spaces Deployment

This Space runs the FastAPI backend on port 7860. The frontend is deployed separately.

**API Base URL**: `https://your-space-name.hf.space`

### API Endpoints

- `GET /`: Welcome message
- `POST /predict/`: Predict readmission and mortality risk (requires full patient data)
- `POST /predict_flex/`: Flexible prediction (uses ICD-only model if demographics are incomplete)
- `GET /search_icd/?q=<query>&limit=<n>`: Search for ICD-10 codes
- `POST /parse_icd_codes/`: Parse and validate ICD codes from text
- `POST /upload_icd_file/`: Upload a file with ICD codes

### Example API Usage

```bash
# Search for ICD codes
curl "https://your-space-name.hf.space/search_icd/?q=diabetes&limit=10"

# Make a prediction
curl -X POST "https://your-space-name.hf.space/predict_flex/" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "female": 0,
    "pay1": 1,
    "zipinc_qrtl": 3,
    "icd_codes": ["E11.9", "I10", "J44.0"]
  }'
```

### CORS Configuration

The API is configured with CORS enabled (`allow_origins=["*"]`) to accept requests from any frontend origin.

## Development

### Backend

To run the backend development server, navigate to the `backend` directory and run:

```bash
uvicorn main:app --reload
```

### Frontend

To run the frontend development server, run the following command from the project root:

```bash
npm run dev
```
