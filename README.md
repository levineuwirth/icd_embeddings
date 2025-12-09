# ICD Prediction Application

This repository contains a full-stack application for predicting 30-day hospital readmission risk based on patient data. It includes a React frontend and a Python (FastAPI) backend.

## Repository Structure

This project is organized as a monorepo, with the frontend and backend code in separate directories:

*   `backend/`: Contains the FastAPI backend application, including the machine learning model and API endpoints. See the [backend/README.md](backend/README.md) for more details.
*   `src/`: Contains the React frontend application.
*   `model/`: Contains the machine learning model files.

## Deployment

This application is designed to be deployed as two separate services: a backend web service and a frontend static site.

TODO: provide more detailed information!

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
