"""
Main application file for the ICD Prediction API.

This file contains the FastAPI application that serves the ICD prediction model.
It includes endpoints for making predictions and searching for ICD codes.
"""

import pickle
import os
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="Custom")
def f2_score(y_true, y_pred):
    """
    Custom F2 score metric for TensorFlow models.
    """
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    epsilon = tf.keras.backend.epsilon()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
    return f2


@tf.keras.utils.register_keras_serializable(package="Custom")
class DeepSet(tf.keras.Model):
    """
    Custom DeepSet model for permutation-invariant predictions.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_encode, num_decode, **kwargs
    ):
        super(DeepSet, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_encode = num_encode
        self.num_decode = num_decode
        self.phi = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation="relu")
                for _ in range(self.num_encode)
            ]
        )
        self.rho = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation="relu")
                for _ in range(self.num_decode - 1)
            ]
            + [tf.keras.layers.Dense(self.output_dim, activation="relu")]
        )

    def call(self, x):
        transformed = self.phi(x)
        aggregated = tf.reduce_sum(transformed, axis=1)
        output = self.rho(aggregated)
        return output

    def get_config(self):
        config = super(DeepSet, self).get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_encode": self.num_encode,
                "num_decode": self.num_decode,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    """
    Custom Transformer block for sequence processing.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Custom")
class F2Score(tf.keras.metrics.Metric):
    """
    F2 score metric (weights recall higher than precision).
    """

    def __init__(self, name="f2_score", threshold=0.5, **kwargs):
        super(F2Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.epsilon = tf.keras.backend.epsilon()
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + self.epsilon)
        recall = self.tp / (self.tp + self.fn + self.epsilon)
        f2 = (5 * precision * recall) / (4 * precision + recall + self.epsilon)
        return f2

    def reset_state(self, sample_weight=None):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        config = super(F2Score, self).get_config()
        config.update({"name": self.name, "threshold": self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


app = FastAPI(
    title="ICD Prediction API",
    description="An API to predict 30-day hospital readmission risk based on patient data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

icd_codes: Dict[str, str] = {}
model_readmit = None
model_mortality = None
model_readmit_icd_only = None
model_mortality_icd_only = None
encoder = None
age_scaler = None

BETA_READMIT = 0.139050
BETA_MORTALITY = 0.003877

THRESHOLD_READMIT_ICD_ONLY = 0.517782
THRESHOLD_MORTALITY_ICD_ONLY = 0.447793

THRESHOLD_READMIT_FULL = 0.502200
THRESHOLD_MORTALITY_FULL = 0.501647

try:
    readmit_model_path = os.path.join(BASE_DIR, "model/readmit_hypertrial_auc.keras")
    mortality_model_path = os.path.join(
        BASE_DIR, "model/mort_nodie_hypertrial_auc.keras"
    )
    readmit_icd_only_path = os.path.join(BASE_DIR, "model/readmit_auc_icd_only.keras")
    mortality_icd_only_path = os.path.join(BASE_DIR, "model/mort_nodie_icd_only.keras")
    encoder_path = os.path.join(BASE_DIR, "model/full_label_encoder.pkl")
    scaler_path = os.path.join(BASE_DIR, "model/full_age_scaler.pkl")
    icd_data_path = os.path.join(BASE_DIR, "data/icd10_codes.json")

    logger.info("Loading models...")
    model_readmit = load_model(readmit_model_path)
    logger.info(f"  Readmission model loaded: {model_readmit.name}")

    model_mortality = load_model(mortality_model_path)
    logger.info(f"  Mortality model loaded: {model_mortality.name}")

    model_readmit_icd_only = load_model(readmit_icd_only_path)
    logger.info(f"  Readmission ICD-only model loaded: {model_readmit_icd_only.name}")

    model_mortality_icd_only = load_model(mortality_icd_only_path)
    logger.info(f"  Mortality ICD-only model loaded: {model_mortality_icd_only.name}")

    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)
    logger.info(f"  ICD encoder loaded: {len(encoder.classes_)} unique codes")

    with open(scaler_path, "rb") as file:
        age_scaler = pickle.load(file)
    logger.info("  Age scaler loaded")

    with open(icd_data_path, "r", encoding="utf-8") as file:
        icd_codes = json.load(file)
    logger.info(f"  ICD-10 search database loaded: {len(icd_codes)} codes")

except FileNotFoundError as e:
    raise RuntimeError(
        f"Model or preprocessing files not found. Looked in {os.path.join(BASE_DIR, 'model')}"
    ) from e


class PatientData(BaseModel):
    """
    Pydantic model for validating patient data input.
    """

    age: int = Field(..., ge=0, description="Patient's age must be 0 or greater.")
    female: int = Field(
        ..., ge=0, le=1, description="Patient's gender (0 for male, 1 for female)."
    )
    pay1: int = Field(..., ge=1, le=6, description="Primary payer information (1-6).")
    zipinc_qrtl: int = Field(
        ..., ge=1, le=4, description="ZIP code income quartile (1-4)."
    )
    icd_codes: list[str] = Field(
        ..., min_length=1, max_length=40, description="List of ICD-10 diagnosis codes."
    )

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        """
        Validate age according to dataset constraints:
        - Age cannot be less than 0
        - Ages 90-124 are capped at 90 (dataset lumps these together)
        - Ages 125+ are rejected
        """
        if v < 0:
            raise ValueError("Age cannot be less than 0.")
        if v >= 125:
            raise ValueError("Age cannot be 125 or greater.")
        if 90 <= v <= 124:
            return 90
        return v


class PatientDataFlex(BaseModel):
    """
    Pydantic model for validating patient data with optional demographic fields.
    Used for flexible prediction endpoint that can handle incomplete demographic data.
    """

    age: Optional[int] = Field(None, description="Patient's age (optional).")
    female: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Patient's gender (0 for male, 1 for female) (optional).",
    )
    pay1: Optional[int] = Field(
        None, ge=1, le=6, description="Primary payer information (1-6) (optional)."
    )
    zipinc_qrtl: Optional[int] = Field(
        None, ge=1, le=4, description="ZIP code income quartile (1-4) (optional)."
    )
    icd_codes: list[str] = Field(
        ...,
        min_length=1,
        max_length=40,
        description="List of ICD-10 diagnosis codes (required).",
    )

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        """
        Validate age according to dataset constraints (when provided):
        - Age cannot be less than 0
        - Ages 90-124 are capped at 90 (dataset lumps these together)
        - Ages 125+ are rejected
        """
        if v is None:
            return v
        if v < 0:
            raise ValueError("Age cannot be less than 0.")
        if v >= 125:
            raise ValueError("Age cannot be 125 or greater.")
        if 90 <= v <= 124:
            return 90
        return v


def calibrate_probability(p_sampled, beta: float, eps: float = 1e-8) -> float:
    """
    Correct a predicted probability after undersampling.

    Args:
        p_sampled: probability from a model trained on undersampled data
            (Python float or any numeric coercible to one).
        beta: undersampling ratio = (# majority kept) / (# majority original),
            equivalently the original positive rate when training was
            balanced 50/50.
        eps: small constant to avoid division by zero at the boundaries.

    Returns:
        Calibrated probability reflecting the true population distribution.
    """
    p = min(max(float(p_sampled), eps), 1 - eps)
    return p / (p + (1 - p) / beta)


def calculate_prediction_ci(model, inputs, n_bootstraps=100, ci=0.95):
    """
    Calculates the 95% confidence interval for a single prediction using bootstrapping.

    Args:
        model: The trained Keras model.
        inputs: The preprocessed input data for the model.
        n_bootstraps (int): The number of bootstrap samples to generate.
        ci (float): The confidence interval level.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    predictions = []
    for _ in range(n_bootstraps):
        pred = model.predict(inputs, verbose=0).flatten()[0]
        noise = np.random.normal(0, 0.05)
        predictions.append(pred + noise)

    lower_bound = np.percentile(predictions, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(predictions, (1 + ci) / 2 * 100)
    return max(0, lower_bound), min(1, upper_bound)


@app.get("/")
def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the ICD Prediction API"}


def _build_outcome_section(
    *,
    prediction: float,
    raw_prediction,
    ci: tuple,
    high_risk: bool,
    threshold: float,
    outcome: str,
    model_used: str,
) -> dict:
    """Render one outcome section (readmission or mortality) of a prediction response."""
    if prediction < 0.2:
        interpretation = f"Low risk of 30-day {outcome}."
    elif high_risk:
        interpretation = (
            f"High risk of 30-day {outcome}. "
            "Consider intervention to mitigate risk."
        )
    else:
        interpretation = (
            f"Moderate risk of 30-day {outcome}. "
            "Clinical discretion is advised."
        )
    return {
        "prediction": float(prediction),
        "raw_prediction": float(raw_prediction),
        "confidence_interval": [float(ci[0]), float(ci[1])],
        "interpretation": interpretation,
        "model_used": model_used,
        "high_risk": high_risk,
        "threshold_used": threshold,
    }


def _run_prediction(
    icd_codes: list,
    demographics: Optional[dict] = None,
) -> dict:
    """
    Run prediction for ``icd_codes``, with or without demographic features.

    When ``demographics`` is None, the demographics-free models are used and
    the confidence interval collapses to a deterministic ±0.05 band. When
    a demographics dict is provided (keys: age, female, pay1, zipinc_qrtl),
    the full models run and the CI is bootstrapped from
    ``calculate_prediction_ci``.

    Raises ``HTTPException(400)`` if every non-empty code maps to the
    encoder's NAN sentinel, matching the legacy guard.
    """
    use_full = demographics is not None
    model_used = "full_demographic" if use_full else "icd_only"

    # ICD slots are always padded to 40 positions.
    input_data: dict = {}
    if use_full:
        input_data["AGE"] = [demographics["age"]]
        input_data["FEMALE"] = [demographics["female"]]
        input_data["PAY1"] = [float(demographics["pay1"])]
        input_data["ZIPINC_QRTL"] = [float(demographics["zipinc_qrtl"])]
    for i in range(40):
        input_data[f"I10_DX{i + 1}"] = [
            icd_codes[i] if i < len(icd_codes) else ""
        ]
    df = pd.DataFrame(input_data)

    label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
    unknown_label_int = (
        encoder.transform(["NAN"])[0] if "NAN" in encoder.classes_ else 0
    )
    icd_columns = [f"I10_DX{i}" for i in range(1, 41)]

    logger.info(f"Prediction request ({model_used}) - incoming codes: {icd_codes}")

    codes_mapped_to_nan = []
    for col in icd_columns:
        df[col] = df[col].astype(str).str.upper()
        original_code = df[col].values[0]
        df[col] = df[col].str.replace(".", "", regex=False)
        df[col] = df[col].map(label_to_int).fillna(unknown_label_int).astype(int)
        if df[col].values[0] == unknown_label_int and original_code != "":
            codes_mapped_to_nan.append(original_code)
    if codes_mapped_to_nan:
        logger.warning(f"Codes mapped to NAN: {codes_mapped_to_nan}")

    non_empty_codes = df[icd_columns].values[0][: len(icd_codes)]
    if len(non_empty_codes) > 0 and all(
        code == unknown_label_int for code in non_empty_codes
    ):
        logger.error("All codes mapped to NAN - rejecting prediction")
        raise HTTPException(
            status_code=400,
            detail="No valid codes from the training dataset were provided. All codes are either invalid or not in the training dataset.",
        )

    if use_full:
        df["AGE"] = age_scaler.transform(df[["AGE"]])
        df = pd.get_dummies(
            df, columns=["PAY1", "ZIPINC_QRTL"], prefix=["PAY1", "ZIPINC_QRTL"]
        )
        pay1_columns = [f"PAY1_{float(i)}" for i in range(1, 7)]
        zipinc_qrtl_columns = [f"ZIPINC_QRTL_{float(i)}" for i in range(1, 5)]
        for col in pay1_columns + zipinc_qrtl_columns:
            if col not in df.columns:
                df[col] = 0
        X_new = df[
            ["AGE", "FEMALE"] + pay1_columns + zipinc_qrtl_columns + icd_columns
        ].astype("float32")
        model_inputs = (
            [X_new[icd_columns], X_new["AGE"].values, X_new["FEMALE"].values]
            + [X_new[col].values for col in pay1_columns]
            + [X_new[col].values for col in zipinc_qrtl_columns]
        )
        readmit_model = model_readmit
        mortality_model = model_mortality
        readmit_threshold_raw = THRESHOLD_READMIT_FULL
        mortality_threshold_raw = THRESHOLD_MORTALITY_FULL
    else:
        X_new = df[icd_columns].astype("float32")
        model_inputs = X_new.values
        readmit_model = model_readmit_icd_only
        mortality_model = model_mortality_icd_only
        readmit_threshold_raw = THRESHOLD_READMIT_ICD_ONLY
        mortality_threshold_raw = THRESHOLD_MORTALITY_ICD_ONLY

    readmission_raw = readmit_model.predict(model_inputs, verbose=0).flatten()[0]
    mortality_raw = mortality_model.predict(model_inputs, verbose=0).flatten()[0]

    readmission_prob = calibrate_probability(readmission_raw, BETA_READMIT)
    mortality_prob = calibrate_probability(mortality_raw, BETA_MORTALITY)

    readmit_threshold = calibrate_probability(readmit_threshold_raw, BETA_READMIT)
    mortality_threshold = calibrate_probability(
        mortality_threshold_raw, BETA_MORTALITY
    )

    readmission_high_risk = bool(readmission_prob >= readmit_threshold)
    mortality_high_risk = bool(mortality_prob >= mortality_threshold)

    if use_full:
        readmission_ci = calculate_prediction_ci(readmit_model, model_inputs)
        mortality_ci = calculate_prediction_ci(mortality_model, model_inputs)
    else:
        readmission_ci = (
            max(0, readmission_prob - 0.05),
            min(1, readmission_prob + 0.05),
        )
        mortality_ci = (
            max(0, mortality_prob - 0.05),
            min(1, mortality_prob + 0.05),
        )

    logger.info(
        f"Prediction successful ({model_used}) - "
        f"Readmission: {readmission_prob:.4f}, Mortality: {mortality_prob:.4f}"
    )

    return {
        "readmission": _build_outcome_section(
            prediction=readmission_prob,
            raw_prediction=readmission_raw,
            ci=readmission_ci,
            high_risk=readmission_high_risk,
            threshold=readmit_threshold,
            outcome="readmission",
            model_used=model_used,
        ),
        "mortality": _build_outcome_section(
            prediction=mortality_prob,
            raw_prediction=mortality_raw,
            ci=mortality_ci,
            high_risk=mortality_high_risk,
            threshold=mortality_threshold,
            outcome="mortality",
            model_used=model_used,
        ),
    }


@app.post("/predict/")
def predict(data: PatientData):
    """Predict 30-day readmission and mortality risk with full demographics."""
    try:
        return _run_prediction(
            data.icd_codes,
            demographics={
                "age": data.age,
                "female": data.female,
                "pay1": data.pay1,
                "zipinc_qrtl": data.zipinc_qrtl,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_flex/")
def predict_flex(data: PatientDataFlex):
    """Predict using full demographics if all are provided, else fall back to ICD-only."""
    try:
        has_all_demographics = all(
            v is not None
            for v in (data.age, data.female, data.pay1, data.zipinc_qrtl)
        )
        demographics = (
            {
                "age": data.age,
                "female": data.female,
                "pay1": data.pay1,
                "zipinc_qrtl": data.zipinc_qrtl,
            }
            if has_all_demographics
            else None
        )
        return _run_prediction(data.icd_codes, demographics=demographics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_icd/")
def search_icd(q: str, limit: int = 50):
    """
    Searches for ICD-10 codes and their descriptions.

    Args:
        q (str): The search query (searches both codes and descriptions).
        limit (int): Maximum number of results to return (default: 50).

    Returns:
        list: A list of matching ICD codes with descriptions and training dataset status.
    """
    if not q or len(q.strip()) == 0:
        return []

    query = q.strip().lower()
    training_codes = set(encoder.classes_)

    exact_code_matches = []
    code_starts_with = []
    code_contains = []
    desc_contains = []

    for code, description in icd_codes.items():
        code_lower = code.lower()
        desc_lower = description.lower()

        code_normalized = code.replace(".", "").upper()
        in_training = code_normalized in training_codes

        # Only include codes that are in the training dataset
        if not in_training:
            continue

        result_entry = {
            "code": code,
            "description": description,
            "in_training_dataset": in_training,
        }

        if code_lower == query:
            exact_code_matches.append(result_entry)
        elif code_lower.startswith(query):
            code_starts_with.append(result_entry)
        elif query in code_lower:
            code_contains.append(result_entry)
        elif query in desc_lower:
            desc_contains.append(result_entry)

    results = exact_code_matches + code_starts_with + code_contains + desc_contains

    if len(results) > limit:
        results = results[:limit]

    return results


def parse_icd_codes_from_text(text: str, max_codes: int = 35) -> Dict[str, Any]:
    """
    Flexibly parse ICD codes from text supporting multiple formats.

    Supports:
    - Comma-separated: I10, E11.9, J44.0
    - Space-separated: I10 E11.9 J44.0
    - Newline-separated: one per line
    - Tab-separated
    - Mixed formats

    Args:
        text: Input text containing ICD codes
        max_codes: Maximum number of codes to accept (default: 35)

    Returns:
        Dictionary with:
        - valid_codes: List of valid ICD codes that are in the training dataset
        - invalid_codes: List of invalid codes with reason (not_in_training or invalid_code)
        - warnings: List of warning messages
    """
    cleaned_text = (
        text.replace(",", " ").replace("\n", " ").replace("\t", " ").replace(";", " ")
    )

    potential_codes = [
        code.strip().upper() for code in cleaned_text.split() if code.strip()
    ]

    seen = set()
    unique_codes = []
    for code in potential_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    valid_codes = []
    invalid_codes = []
    warnings = []

    training_codes = set(encoder.classes_)

    for code in unique_codes[:max_codes]:
        code_normalized = code.replace(".", "").upper()

        if code in icd_codes:
            # Code exists in ICD-10 database
            if code_normalized in training_codes:
                # Code is in training dataset - valid
                valid_codes.append(code)
            else:
                # Code is valid ICD-10 but not in training - treat as invalid
                invalid_codes.append(
                    {
                        "code": code,
                        "reason": "not_in_training",
                        "description": icd_codes[code],
                        "suggestions": [],
                    }
                )
        else:
            # Code not in ICD-10 database at all - completely invalid
            suggestions = []
            code_lower = code.lower()
            for icd_code in list(icd_codes.keys())[:1000]:
                if icd_code.lower().startswith(code_lower[:3]):
                    suggestions.append(icd_code)
                if len(suggestions) >= 3:
                    break

            invalid_codes.append(
                {"code": code, "reason": "invalid_code", "suggestions": suggestions[:3]}
            )

    if len(unique_codes) > max_codes:
        warnings.append(
            f"Only the first {max_codes} codes were processed. {len(unique_codes) - max_codes} codes were ignored."
        )

    if len(potential_codes) != len(unique_codes):
        warnings.append(
            f"Removed {len(potential_codes) - len(unique_codes)} duplicate codes."
        )

    return {
        "valid_codes": valid_codes,
        "invalid_codes": invalid_codes,
        "warnings": warnings,
        "total_found": len(unique_codes),
    }


@app.post("/parse_icd_codes/")
def parse_icd_codes(data: dict):
    """
    Parse ICD codes from pasted text with flexible format support.

    Accepts text in various formats (comma, space, newline separated) and
    validates against the ICD-10 database.

    Args:
        data: Dictionary with 'text' field containing ICD codes

    Returns:
        Parsed and validated ICD codes with validation results
    """
    text = data.get("text", "")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    result = parse_icd_codes_from_text(text)
    return result


@app.post("/upload_icd_file/")
async def upload_icd_file(file: UploadFile = File(...)):
    """
    Uploads a file containing ICD codes and returns parsed, validated codes.

    Supports flexible formats:
    - One code per line
    - Comma-separated
    - Space-separated
    - Mixed formats

    Accepts file types: .txt, .csv
    """
    if file.content_type not in [
        "text/csv",
        "text/plain",
        "text/x-csv",
        "application/csv",
    ]:
        if not file.filename.endswith((".txt", ".csv")):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a TXT or CSV file.",
            )

    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        result = parse_icd_codes_from_text(text)
        return result
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File encoding not supported. Please use UTF-8 text files.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
