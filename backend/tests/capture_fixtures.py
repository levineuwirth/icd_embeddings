"""
Record current API responses as JSON fixtures.

This is a one-shot, idempotent script. Run BEFORE refactoring to pin the
current behavior; the test suite then asserts the same responses survive
any future change.

Usage (from repo root):
    python -m backend.tests.capture_fixtures
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
client = TestClient(app)


def save(name, request, response):
    FIXTURES_DIR.mkdir(exist_ok=True)
    body_text = response.headers.get("content-type", "")
    if body_text.startswith("application/json"):
        body = response.json()
    else:
        body = response.text
    payload = {
        "request": request,
        "response": {"status": response.status_code, "json": body},
    }
    (FIXTURES_DIR / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def hit(name, method, path, *, json_body=None, params=None):
    request = {"method": method, "path": path}
    kwargs = {}
    if json_body is not None:
        request["json"] = json_body
        kwargs["json"] = json_body
    if params is not None:
        request["params"] = params
        kwargs["params"] = params
    response = client.request(method, path, **kwargs)
    save(name, request, response)
    print(f"  {response.status_code:>3} {method:6} {path:25} -> {name}")


def main():
    print(f"Recording fixtures to {FIXTURES_DIR}")

    # Root
    hit("root", "GET", "/")

    # /predict/ -- happy paths
    hit(
        "predict_full_valid",
        "POST",
        "/predict/",
        json_body={
            "age": 65,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 3,
            "icd_codes": ["E119", "I10", "J440"],
        },
    )
    # Same input but with periods in codes -- backend strips periods, so
    # predictions must be identical to the no-period case.
    hit(
        "predict_full_valid_with_periods",
        "POST",
        "/predict/",
        json_body={
            "age": 65,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 3,
            "icd_codes": ["E11.9", "I10", "J44.0"],
        },
    )
    # Age 95 -- silently capped to 90 by validator
    hit(
        "predict_full_age_capped",
        "POST",
        "/predict/",
        json_body={
            "age": 95,
            "female": 0,
            "pay1": 2,
            "zipinc_qrtl": 2,
            "icd_codes": ["I10"],
        },
    )

    # /predict/ -- validation errors
    hit(
        "predict_full_age_negative",
        "POST",
        "/predict/",
        json_body={
            "age": -1,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 1,
            "icd_codes": ["I10"],
        },
    )
    hit(
        "predict_full_age_too_high",
        "POST",
        "/predict/",
        json_body={
            "age": 130,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 1,
            "icd_codes": ["I10"],
        },
    )
    hit(
        "predict_full_female_invalid",
        "POST",
        "/predict/",
        json_body={
            "age": 50,
            "female": 2,
            "pay1": 1,
            "zipinc_qrtl": 1,
            "icd_codes": ["I10"],
        },
    )
    hit(
        "predict_full_pay1_invalid",
        "POST",
        "/predict/",
        json_body={
            "age": 50,
            "female": 1,
            "pay1": 7,
            "zipinc_qrtl": 1,
            "icd_codes": ["I10"],
        },
    )
    hit(
        "predict_full_qrtl_invalid",
        "POST",
        "/predict/",
        json_body={
            "age": 50,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 5,
            "icd_codes": ["I10"],
        },
    )
    hit(
        "predict_full_no_codes",
        "POST",
        "/predict/",
        json_body={
            "age": 50,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 1,
            "icd_codes": [],
        },
    )
    hit(
        "predict_full_too_many_codes",
        "POST",
        "/predict/",
        json_body={
            "age": 50,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 1,
            "icd_codes": ["I10"] * 41,
        },
    )
    # All codes missing from training -- 400
    hit(
        "predict_full_all_invalid",
        "POST",
        "/predict/",
        json_body={
            "age": 50,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 1,
            "icd_codes": ["XYZ", "ABC123"],
        },
    )

    # /predict_flex/ -- full demographic path (same model as /predict/)
    hit(
        "flex_full_demographic",
        "POST",
        "/predict_flex/",
        json_body={
            "age": 65,
            "female": 1,
            "pay1": 1,
            "zipinc_qrtl": 3,
            "icd_codes": ["E119", "I10", "J440"],
        },
    )
    # ICD-only path (no demographics provided)
    hit(
        "flex_icd_only",
        "POST",
        "/predict_flex/",
        json_body={"icd_codes": ["E119", "I10", "J440"]},
    )
    # Partial demographics -> falls back to ICD-only
    hit(
        "flex_partial_demographics",
        "POST",
        "/predict_flex/",
        json_body={"age": 65, "female": 1, "icd_codes": ["E119", "I10"]},
    )
    hit(
        "flex_all_invalid",
        "POST",
        "/predict_flex/",
        json_body={"icd_codes": ["XYZ", "ABC123"]},
    )
    hit(
        "flex_no_codes",
        "POST",
        "/predict_flex/",
        json_body={"icd_codes": []},
    )

    # /search_icd/
    hit("search_hypertension", "GET", "/search_icd/", params={"q": "hypertension"})
    hit("search_exact_i10", "GET", "/search_icd/", params={"q": "I10"})
    hit(
        "search_diabetes_limit",
        "GET",
        "/search_icd/",
        params={"q": "diabetes", "limit": 5},
    )
    hit("search_empty", "GET", "/search_icd/", params={"q": ""})
    hit("search_no_results", "GET", "/search_icd/", params={"q": "xyzzyqq"})

    # /parse_icd_codes/
    hit("parse_normal", "POST", "/parse_icd_codes/", json_body={"text": "I10, E11.9, J44.0"})
    hit(
        "parse_mixed",
        "POST",
        "/parse_icd_codes/",
        json_body={"text": "I10\nE11.9\nXYZ\nABC123"},
    )
    hit(
        "parse_duplicates",
        "POST",
        "/parse_icd_codes/",
        json_body={"text": "I10, I10, I10, E11.9"},
    )
    hit(
        "parse_too_many",
        "POST",
        "/parse_icd_codes/",
        json_body={"text": ", ".join([f"I{i:02d}" for i in range(40)])},
    )
    hit("parse_empty", "POST", "/parse_icd_codes/", json_body={"text": ""})
    hit("parse_missing_text", "POST", "/parse_icd_codes/", json_body={})
    hit(
        "parse_whitespace_only",
        "POST",
        "/parse_icd_codes/",
        json_body={"text": "   \n\t  "},
    )

    print(f"\nWrote {len(list(FIXTURES_DIR.glob('*.json')))} fixtures.")


if __name__ == "__main__":
    main()
