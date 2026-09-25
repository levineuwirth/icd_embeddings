"""
Pin current API behavior against captured JSON fixtures.

Each file under ``tests/fixtures/`` records a request and the response that
the API produced at capture time. These tests replay every request and
assert the response is byte-for-byte identical, except that floating-point
fields are compared with a small absolute tolerance to tolerate
hardware/oneDNN jitter (TensorFlow warns about this on import).

Every prediction is deterministic: the randomised ``confidence_interval``
that needed a carve-out here was removed in September 2026.

To regenerate fixtures after an intentional API change:

    python -m backend.tests.capture_fixtures

then ``git diff`` the result and confirm every change is expected.
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import app  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FLOAT_TOLERANCE = 1e-6

client = TestClient(app)


def _fixture_files():
    return sorted(FIXTURES_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "fixture_path", _fixture_files(), ids=lambda p: p.stem
)
def test_response_matches_fixture(fixture_path):
    fixture = json.loads(fixture_path.read_text())
    request = fixture["request"]
    expected = fixture["response"]

    kwargs = {}
    if "json" in request:
        kwargs["json"] = request["json"]
    if "params" in request:
        kwargs["params"] = request["params"]

    response = client.request(request["method"], request["path"], **kwargs)

    assert response.status_code == expected["status"], (
        f"{fixture_path.name}: status {response.status_code} != "
        f"expected {expected['status']}\nbody: {response.text}"
    )
    actual_body = response.json()
    _assert_isomorphic(actual_body, expected["json"], path=fixture_path.stem)


def _assert_isomorphic(actual, expected, *, path):
    """Deep equality with a float tolerance (see module docstring)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), (
            f"{path}: expected dict, got {type(actual).__name__}"
        )
        assert actual.keys() == expected.keys(), (
            f"{path}: key set differs\n"
            f"  missing: {sorted(expected.keys() - actual.keys())}\n"
            f"  extra:   {sorted(actual.keys() - expected.keys())}"
        )
        for key, expected_value in expected.items():
            _assert_isomorphic(actual[key], expected_value, path=f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), (
            f"{path}: expected list, got {type(actual).__name__}"
        )
        assert len(actual) == len(expected), (
            f"{path}: length {len(actual)} != {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_isomorphic(a, e, path=f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=FLOAT_TOLERANCE), (
            f"{path}: {actual!r} != {expected!r} (tol {FLOAT_TOLERANCE})"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# -----------------------------------------------------------------------------
# Inline tests for endpoints that don't fixture cleanly.
# -----------------------------------------------------------------------------


def test_upload_txt_valid_codes():
    response = client.post(
        "/upload_icd_file/",
        files={"file": ("codes.txt", b"I10\nE11.9\nJ44.0", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "valid_codes",
        "invalid_codes",
        "warnings",
        "total_found",
    }
    assert "I10" in body["valid_codes"]


def test_upload_csv_valid_codes():
    response = client.post(
        "/upload_icd_file/",
        files={"file": ("codes.csv", b"I10,E11.9", "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "I10" in body["valid_codes"]


def test_upload_octet_stream_with_txt_extension_accepted():
    """The current endpoint falls back to extension when MIME isn't whitelisted.

    This pins the existing (intentional, per code comment) behavior; if the
    extension fallback is ever removed, this test will fail and force the
    isomorphism-break to be acknowledged.
    """
    response = client.post(
        "/upload_icd_file/",
        files={"file": ("codes.txt", b"I10", "application/octet-stream")},
    )
    assert response.status_code == 200


def test_upload_unknown_mime_unknown_extension_rejected():
    response = client.post(
        "/upload_icd_file/",
        files={"file": ("codes.bin", b"I10", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid file type. Please upload a TXT or CSV file."
    )


def test_upload_non_utf8_rejected():
    response = client.post(
        "/upload_icd_file/",
        files={"file": ("codes.txt", b"\xff\xfe\xfd", "text/plain")},
    )
    assert response.status_code == 400
    assert "encoding" in response.json()["detail"].lower()


# -----------------------------------------------------------------------------
# Regressions (September 2026).
# -----------------------------------------------------------------------------

FULL_BODY = {"age": 65, "female": 1, "pay1": 1, "zipinc_qrtl": 3,
             "icd_codes": ["E119", "I10", "J440"]}


def test_codes_never_reach_the_log(caplog):
    """The README and paper say inputs are not stored; the log used to
    record every request's codes at INFO, and unknown ones at WARNING."""
    import logging

    caplog.set_level(logging.DEBUG)
    codes = ["E119", "I10", "QQ999"]
    client.post("/predict_flex/", json={"icd_codes": codes})
    client.post("/predict/", json={**FULL_BODY, "icd_codes": codes})
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Prediction request" in logged, "request logging disappeared entirely"
    for code in codes:
        assert code not in logged, f"code {code} was logged"


def test_no_confidence_interval():
    """The interval was noise around the uncalibrated score; it is gone."""
    for path, body in (("/predict/", FULL_BODY),
                       ("/predict_flex/", {"icd_codes": FULL_BODY["icd_codes"]})):
        response = client.post(path, json=body)
        assert response.status_code == 200
        for outcome in ("readmission", "mortality"):
            assert "confidence_interval" not in response.json()[outcome]


def test_full_prediction_is_deterministic():
    first = client.post("/predict/", json=FULL_BODY).json()
    second = client.post("/predict/", json=FULL_BODY).json()
    assert first == second


def test_interpretation_follows_high_risk_flag():
    """A flagged mortality prediction is far below 0.2; the text used to
    call it low risk because it keyed on 0.2, not on the threshold."""
    from backend.main import _build_outcome_section

    flagged = _build_outcome_section(
        prediction=0.006, raw_prediction=0.61, high_risk=True,
        threshold=0.0039, outcome="mortality", model_used="full_demographic",
    )
    assert flagged["interpretation"].startswith("High risk of 30-day mortality")
    unflagged = _build_outcome_section(
        prediction=0.15, raw_prediction=0.49, high_risk=False,
        threshold=0.1224, outcome="readmission", model_used="full_demographic",
    )
    assert unflagged["interpretation"].startswith("Low risk of 30-day readmission")


@pytest.mark.parametrize("path", ["/predict/", "/predict_flex/"])
def test_minors_rejected(path):
    """The cohort was adults (18+); ages 0-17 used to be accepted."""
    response = client.post(path, json={**FULL_BODY, "age": 17})
    assert response.status_code == 422
    assert "18 or older" in response.text
    assert client.post(path, json={**FULL_BODY, "age": 18}).status_code == 200


@pytest.mark.parametrize("path", ["/predict/", "/predict_flex/"])
def test_all_unknown_codes_is_a_client_error(path):
    """The handlers' catch-all turned this deliberate 400 into a 500."""
    response = client.post(path, json={**FULL_BODY, "icd_codes": ["XYZ", "ABC123"]})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("No valid codes")
