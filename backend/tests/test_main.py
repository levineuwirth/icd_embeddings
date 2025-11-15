import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.main import app

client = TestClient(app)


def test_read_root():
    """
    Test the root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ICD Prediction API"}


def test_predict_valid_data():
    """
    Test the /predict endpoint with valid data.
    """
    response = client.post(
        "/predict/",
        json={"age": 50, "female": 1, "pay1": 1, "zipinc_qrtl": 2, "icd_codes": ["I10", "I11", "I12"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence_interval" in data
    assert "interpretation" in data
    assert 0 <= data["prediction"] <= 1
    assert len(data["confidence_interval"]) == 2
    assert 0 <= data["confidence_interval"][0] <= 1
    assert 0 <= data["confidence_interval"][1] <= 1


def test_predict_invalid_age():
    """
    Test the /predict endpoint with an invalid age.
    """
    response = client.post(
        "/predict/",
        json={"age": -1, "female": 1, "pay1": 1, "zipinc_qrtl": 2, "icd_codes": ["I10"]},
    )
    assert response.status_code == 422


def test_predict_invalid_female():
    """
    Test the /predict endpoint with an invalid 'female' value.
    """
    response = client.post(
        "/predict/",
        json={"age": 50, "female": 2, "pay1": 1, "zipinc_qrtl": 2, "icd_codes": ["I10"]},
    )
    assert response.status_code == 422


def test_predict_missing_icd_codes():
    """
    Test the /predict endpoint with missing ICD codes.
    """
    response = client.post(
        "/predict/",
        json={"age": 50, "female": 1, "pay1": 1, "zipinc_qrtl": 2, "icd_codes": []},
    )
    assert response.status_code == 422


def test_search_icd_found():
    """
    Test the /search_icd endpoint with a query that should return results.
    """
    response = client.get("/search_icd/?q=hypertension")
    assert response.status_code == 200
    assert response.json() == {"I10": "Essential (primary) hypertension"}


def test_search_icd_not_found():
    """
    Test the /search_icd endpoint with a query that should not return results.
    """
    response = client.get("/search_icd/?q=xyz")
    assert response.status_code == 200
    assert response.json() == {}
