"""
Pin current API behavior against captured JSON fixtures.

Each file under ``tests/fixtures/`` records a request and the response that
the API produced at capture time. These tests replay every request and
assert the response is byte-for-byte identical, with two documented
exceptions:

1. Floating-point fields are compared with a small absolute tolerance to
   tolerate hardware/oneDNN jitter (TensorFlow warns about this on
   import).
2. ``confidence_interval`` on full-demographic predictions is generated
   from unseeded ``np.random.normal`` (see ``calculate_prediction_ci`` in
   ``main.py``). For those responses we assert shape and bounds only.
   For ICD-only predictions the CI is deterministic
   (``[max(0, p-0.05), min(1, p+0.05)]``) and is pinned exactly.

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
    """Deep equality with two carve-outs (see module docstring)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), (
            f"{path}: expected dict, got {type(actual).__name__}"
        )
        assert actual.keys() == expected.keys(), (
            f"{path}: key set differs\n"
            f"  missing: {sorted(expected.keys() - actual.keys())}\n"
            f"  extra:   {sorted(actual.keys() - expected.keys())}"
        )
        # Carve-out: full-demographic CI is nondeterministic.
        skip_ci = expected.get("model_used") == "full_demographic"
        for key, expected_value in expected.items():
            if key == "confidence_interval" and skip_ci:
                _assert_ci_well_formed(actual[key], path=f"{path}.{key}")
                continue
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


def _assert_ci_well_formed(ci, *, path):
    assert isinstance(ci, list) and len(ci) == 2, (
        f"{path}: confidence_interval must be a 2-element list, got {ci!r}"
    )
    low, high = ci
    assert isinstance(low, (int, float)) and isinstance(high, (int, float)), (
        f"{path}: confidence_interval bounds must be numeric, got {ci!r}"
    )
    assert 0.0 <= low <= 1.0, f"{path}: lower bound {low} out of [0,1]"
    assert 0.0 <= high <= 1.0, f"{path}: upper bound {high} out of [0,1]"
    assert low <= high, f"{path}: lower {low} > upper {high}"


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
