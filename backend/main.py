"""
Main application file for the ICD Prediction API.

This file contains the FastAPI application that serves the ICD prediction model.
It includes endpoints for making predictions and searching for ICD codes.
"""

import pickle
import os
import json
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
    def __init__(self, input_dim, hidden_dim, output_dim, num_encode, num_decode, **kwargs):
        super(DeepSet, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_encode = num_encode
        self.num_decode = num_decode
        self.phi = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu') for _ in range(self.num_encode)
        ])
        self.rho = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu') for _ in range(self.num_decode - 1)
        ] + [tf.keras.layers.Dense(self.output_dim, activation='relu')])

    def call(self, x):
        transformed = self.phi(x)
        aggregated = tf.reduce_sum(transformed, axis=1)
        output = self.rho(aggregated)
        return output

    def get_config(self):
        config = super(DeepSet, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_encode": self.num_encode,
            "num_decode": self.num_decode
        })
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
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
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
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="Custom")
class F2Score(tf.keras.metrics.Metric):
    """
    F2 score metric (weights recall higher than precision).
    """
    def __init__(self, name='f2_score', threshold=0.5, **kwargs):
        super(F2Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
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
        config.update({'name': self.name, 'threshold': self.threshold})
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

THRESHOLD_READMIT = 0.517782
THRESHOLD_MORTALITY = 0.447793

try:
    readmit_model_path = os.path.join(BASE_DIR, 'model/readmit_hypertrial_auc.keras')
    mortality_model_path = os.path.join(BASE_DIR, 'model/mort_nodie_hypertrial_auc.keras')
    readmit_icd_only_path = os.path.join(BASE_DIR, 'model/readmit_auc_icd_only.keras')
    mortality_icd_only_path = os.path.join(BASE_DIR, 'model/mort_nodie_icd_only.keras')
    encoder_path = os.path.join(BASE_DIR, 'model/full_label_encoder.pkl')
    scaler_path = os.path.join(BASE_DIR, 'model/full_age_scaler.pkl')
    icd_data_path = os.path.join(BASE_DIR, 'data/icd10_codes.json')

    print("Loading models...")
    model_readmit = load_model(readmit_model_path)
    print(f"  Readmission model loaded: {model_readmit.name}")

    model_mortality = load_model(mortality_model_path)
    print(f"  Mortality model loaded: {model_mortality.name}")

    model_readmit_icd_only = load_model(readmit_icd_only_path)
    print(f"  Readmission ICD-only model loaded: {model_readmit_icd_only.name}")

    model_mortality_icd_only = load_model(mortality_icd_only_path)
    print(f"  Mortality ICD-only model loaded: {model_mortality_icd_only.name}")

    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)
    print(f"  ICD encoder loaded: {len(encoder.classes_)} unique codes")

    with open(scaler_path, 'rb') as file:
        age_scaler = pickle.load(file)
    print(f"  Age scaler loaded")

    with open(icd_data_path, 'r', encoding='utf-8') as file:
        icd_codes = json.load(file)
    print(f"  ICD-10 search database loaded: {len(icd_codes)} codes")

except FileNotFoundError as e:
    raise RuntimeError(f"Model or preprocessing files not found. Looked in {os.path.join(BASE_DIR, 'model')}") from e


class PatientData(BaseModel):
    """
    Pydantic model for validating patient data input.
    """
    age: int = Field(..., ge=0, description="Patient's age must be 0 or greater.")
    female: int = Field(..., ge=0, le=1, description="Patient's gender (0 for male, 1 for female).")
    pay1: int = Field(..., ge=1, le=6, description="Primary payer information (1-6).")
    zipinc_qrtl: int = Field(..., ge=1, le=4, description="ZIP code income quartile (1-4).")
    icd_codes: list[str] = Field(..., min_length=1, max_length=40, description="List of ICD-10 diagnosis codes.")

    @field_validator('age')
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
    female: Optional[int] = Field(None, ge=0, le=1, description="Patient's gender (0 for male, 1 for female) (optional).")
    pay1: Optional[int] = Field(None, ge=1, le=6, description="Primary payer information (1-6) (optional).")
    zipinc_qrtl: Optional[int] = Field(None, ge=1, le=4, description="ZIP code income quartile (1-4) (optional).")
    icd_codes: list[str] = Field(..., min_length=1, max_length=40, description="List of ICD-10 diagnosis codes (required).")

    @field_validator('age')
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


def calibrate_probability(p_sampled, beta, eps=1e-8):
    """
    Correct predicted probabilities after undersampling.

    Args:
        p_sampled: predicted probability from model trained on undersampled data
        beta: undersampling ratio = (# majority after undersampling) / (# majority original)
              OR equivalently: original_positive_rate (if you balanced to 50/50)
        eps: small constant to avoid division by zero

    Returns:
        calibrated probability reflecting true population distribution
    """
    p_sampled = tf.clip_by_value(p_sampled, eps, 1-eps)
    return p_sampled / (p_sampled + (1 - p_sampled) / beta)


def get_risk_interpretation(prediction: float) -> str:
    """
    Provides a brief interpretation of the prediction risk.

    Args:
        prediction (float): The predicted probability.

    Returns:
        str: A string interpreting the risk level.
    """
    if prediction < 0.2:
        return "Low risk of 30-day readmission."
    elif prediction < 0.5:
        return "Moderate risk of 30-day readmission. Clinical discretion is advised."
    else:
        return "High risk of 30-day readmission. Consider intervention to mitigate risk."

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

@app.post("/predict/")
async def predict(data: PatientData):
    """
    Predicts both 30-day mortality and readmission risk for a patient.

    Args:
        data (PatientData): The patient's data.

    Returns:
        dict: A dictionary containing predictions for both mortality and readmission.
    """
    try:
        input_data = {
            'AGE': [data.age],
            'FEMALE': [data.female],
            'PAY1': [float(data.pay1)],
            'ZIPINC_QRTL': [float(data.zipinc_qrtl)]
        }

        for i in range(40):
            if i < len(data.icd_codes):
                input_data[f'I10_DX{i+1}'] = [data.icd_codes[i]]
            else:
                input_data[f'I10_DX{i+1}'] = ['']

        df = pd.DataFrame(input_data)

        label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
        unknown_label_int = encoder.transform(["NAN"])[0] if "NAN" in encoder.classes_ else 0

        icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

        for col in icd_columns:
            df[col] = df[col].astype(str).str.upper()
            df[col] = df[col].map(label_to_int).fillna(unknown_label_int).astype(int)

        df['AGE'] = age_scaler.transform(df[['AGE']])

        df = pd.get_dummies(df, columns=['PAY1', 'ZIPINC_QRTL'], prefix=['PAY1', 'ZIPINC_QRTL'])

        pay1_columns = [f'PAY1_{float(i)}' for i in range(1, 7)]
        zipinc_qrtl_columns = [f'ZIPINC_QRTL_{float(i)}' for i in range(1, 5)]

        for col in pay1_columns + zipinc_qrtl_columns:
            if col not in df.columns:
                df[col] = 0

        X_new = df[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
        X_new = X_new.astype('float32')

        batch_inputs = [
            X_new[icd_columns],
            X_new['AGE'].values,
            X_new['FEMALE'].values,
        ] + [X_new[col].values for col in pay1_columns] \
          + [X_new[col].values for col in zipinc_qrtl_columns]

        readmission_prob = model_readmit.predict(batch_inputs, verbose=0).flatten()[0]
        mortality_prob = model_mortality.predict(batch_inputs, verbose=0).flatten()[0]

        readmission_lower_ci, readmission_upper_ci = calculate_prediction_ci(model_readmit, batch_inputs)
        mortality_lower_ci, mortality_upper_ci = calculate_prediction_ci(model_mortality, batch_inputs)

        readmission_interpretation = get_risk_interpretation(readmission_prob)
        mortality_interpretation = get_risk_interpretation(mortality_prob)

        return {
            "readmission": {
                "prediction": float(readmission_prob),
                "confidence_interval": [float(readmission_lower_ci), float(readmission_upper_ci)],
                "interpretation": readmission_interpretation,
            },
            "mortality": {
                "prediction": float(mortality_prob),
                "confidence_interval": [float(mortality_lower_ci), float(mortality_upper_ci)],
                "interpretation": mortality_interpretation,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def predict_icd_only(icd_codes: list[str]) -> dict:
    """
    Predict using ICD-only models (no demographic data required).
    Applies risk adjustment to correct for undersampling and classifies risk.

    Args:
        icd_codes: List of ICD-10 diagnosis codes.

    Returns:
        dict: Prediction results for readmission and mortality with adjusted risk.
    """
    input_data = {}
    for i in range(40):
        if i < len(icd_codes):
            input_data[f'I10_DX{i+1}'] = [icd_codes[i]]
        else:
            input_data[f'I10_DX{i+1}'] = ['']

    df = pd.DataFrame(input_data)

    label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
    unknown_label_int = encoder.transform(["NAN"])[0] if "NAN" in encoder.classes_ else 0

    icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

    for col in icd_columns:
        df[col] = df[col].astype(str).str.upper()
        df[col] = df[col].map(label_to_int).fillna(unknown_label_int).astype(int)

    X_new = df[icd_columns].astype('float32')

    readmission_raw = model_readmit_icd_only.predict(X_new.values, verbose=0).flatten()[0]
    mortality_raw = model_mortality_icd_only.predict(X_new.values, verbose=0).flatten()[0]

    readmission_adjusted = float(calibrate_probability(readmission_raw, BETA_READMIT).numpy())
    mortality_adjusted = float(calibrate_probability(mortality_raw, BETA_MORTALITY).numpy())

    threshold_readmit_adjusted = float(calibrate_probability(THRESHOLD_READMIT, BETA_READMIT).numpy())
    threshold_mortality_adjusted = float(calibrate_probability(THRESHOLD_MORTALITY, BETA_MORTALITY).numpy())

    readmission_high_risk = readmission_adjusted >= threshold_readmit_adjusted
    mortality_high_risk = mortality_adjusted >= threshold_mortality_adjusted

    if readmission_adjusted < 0.2:
        readmission_interpretation = "Low risk of 30-day readmission."
    elif readmission_high_risk:
        readmission_interpretation = "High risk of 30-day readmission. Consider intervention to mitigate risk."
    else:
        readmission_interpretation = "Moderate risk of 30-day readmission. Clinical discretion is advised."

    if mortality_adjusted < 0.2:
        mortality_interpretation = "Low risk of 30-day mortality."
    elif mortality_high_risk:
        mortality_interpretation = "High risk of 30-day mortality. Consider intervention to mitigate risk."
    else:
        mortality_interpretation = "Moderate risk of 30-day mortality. Clinical discretion is advised."

    readmission_ci_lower = max(0, readmission_adjusted - 0.05)
    readmission_ci_upper = min(1, readmission_adjusted + 0.05)
    mortality_ci_lower = max(0, mortality_adjusted - 0.05)
    mortality_ci_upper = min(1, mortality_adjusted + 0.05)

    return {
        "readmission": {
            "prediction": readmission_adjusted,
            "raw_prediction": float(readmission_raw),
            "confidence_interval": [readmission_ci_lower, readmission_ci_upper],
            "interpretation": readmission_interpretation,
            "model_used": "icd_only",
            "high_risk": readmission_high_risk,
            "threshold_used": threshold_readmit_adjusted
        },
        "mortality": {
            "prediction": mortality_adjusted,
            "raw_prediction": float(mortality_raw),
            "confidence_interval": [mortality_ci_lower, mortality_ci_upper],
            "interpretation": mortality_interpretation,
            "model_used": "icd_only",
            "high_risk": mortality_high_risk,
            "threshold_used": threshold_mortality_adjusted
        }
    }


@app.post("/predict_flex/")
async def predict_flex(data: PatientDataFlex):
    """
    Flexible prediction endpoint that routes to appropriate model based on available data.

    - If ALL demographic data is provided (age, gender, pay1, zipinc_qrtl): uses full demographic model
    - If ANY demographic data is missing: uses ICD-only model (ignores partial demographics)

    Args:
        data (PatientDataFlex): Patient data with optional demographic fields.

    Returns:
        dict: Predictions with metadata about which model was used.
    """
    try:
        has_all_demographics = all([
            data.age is not None,
            data.female is not None,
            data.pay1 is not None,
            data.zipinc_qrtl is not None
        ])

        if has_all_demographics:
            input_data = {
                'AGE': [data.age],
                'FEMALE': [data.female],
                'PAY1': [float(data.pay1)],
                'ZIPINC_QRTL': [float(data.zipinc_qrtl)]
            }

            for i in range(40):
                if i < len(data.icd_codes):
                    input_data[f'I10_DX{i+1}'] = [data.icd_codes[i]]
                else:
                    input_data[f'I10_DX{i+1}'] = ['']

            df = pd.DataFrame(input_data)

            label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
            unknown_label_int = encoder.transform(["NAN"])[0] if "NAN" in encoder.classes_ else 0

            icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

            for col in icd_columns:
                df[col] = df[col].astype(str).str.upper()
                df[col] = df[col].map(label_to_int).fillna(unknown_label_int).astype(int)

            df['AGE'] = age_scaler.transform(df[['AGE']])

            df = pd.get_dummies(df, columns=['PAY1', 'ZIPINC_QRTL'], prefix=['PAY1', 'ZIPINC_QRTL'])

            pay1_columns = [f'PAY1_{float(i)}' for i in range(1, 7)]
            zipinc_qrtl_columns = [f'ZIPINC_QRTL_{float(i)}' for i in range(1, 5)]

            for col in pay1_columns + zipinc_qrtl_columns:
                if col not in df.columns:
                    df[col] = 0

            X_new = df[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
            X_new = X_new.astype('float32')

            batch_inputs = [
                X_new[icd_columns],
                X_new['AGE'].values,
                X_new['FEMALE'].values,
            ] + [X_new[col].values for col in pay1_columns] \
              + [X_new[col].values for col in zipinc_qrtl_columns]

            readmission_prob = model_readmit.predict(batch_inputs, verbose=0).flatten()[0]
            mortality_prob = model_mortality.predict(batch_inputs, verbose=0).flatten()[0]

            readmission_lower_ci, readmission_upper_ci = calculate_prediction_ci(model_readmit, batch_inputs)
            mortality_lower_ci, mortality_upper_ci = calculate_prediction_ci(model_mortality, batch_inputs)

            readmission_interpretation = get_risk_interpretation(readmission_prob)
            mortality_interpretation = get_risk_interpretation(mortality_prob)

            return {
                "readmission": {
                    "prediction": float(readmission_prob),
                    "confidence_interval": [float(readmission_lower_ci), float(readmission_upper_ci)],
                    "interpretation": readmission_interpretation,
                    "model_used": "full_demographic"
                },
                "mortality": {
                    "prediction": float(mortality_prob),
                    "confidence_interval": [float(mortality_lower_ci), float(mortality_upper_ci)],
                    "interpretation": mortality_interpretation,
                    "model_used": "full_demographic"
                }
            }
        else:
            result = predict_icd_only(data.icd_codes)
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_icd/")
async def search_icd(q: str, limit: int = 50):
    """
    Searches for ICD-10 codes and their descriptions.

    Args:
        q (str): The search query (searches both codes and descriptions).
        limit (int): Maximum number of results to return (default: 50).

    Returns:
        dict: A dictionary of matching ICD codes and descriptions, ordered by relevance.
    """
    if not q or len(q.strip()) == 0:
        return {}

    query = q.strip().lower()
    results = {}

    exact_code_matches = {}
    code_starts_with = {}
    code_contains = {}
    desc_contains = {}

    for code, description in icd_codes.items():
        code_lower = code.lower()
        desc_lower = description.lower()

        if code_lower == query:
            exact_code_matches[code] = description
        elif code_lower.startswith(query):
            code_starts_with[code] = description
        elif query in code_lower:
            code_contains[code] = description
        elif query in desc_lower:
            desc_contains[code] = description

    results.update(exact_code_matches)
    results.update(code_starts_with)
    results.update(code_contains)
    results.update(desc_contains)

    if len(results) > limit:
        results = dict(list(results.items())[:limit])

    return results

def parse_icd_codes_from_text(text: str, max_codes: int = 35) -> Dict[str, any]:
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
        - valid_codes: List of valid ICD codes
        - invalid_codes: List of codes not found in database with suggestions
        - warnings: List of warning messages
    """
    cleaned_text = text.replace(',', ' ').replace('\n', ' ').replace('\t', ' ').replace(';', ' ')

    potential_codes = [code.strip().upper() for code in cleaned_text.split() if code.strip()]

    seen = set()
    unique_codes = []
    for code in potential_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    valid_codes = []
    invalid_codes = []
    warnings = []

    for code in unique_codes[:max_codes]:
        if code in icd_codes:
            valid_codes.append(code)
        else:
            suggestions = []
            code_lower = code.lower()
            for icd_code in list(icd_codes.keys())[:1000]:
                if icd_code.lower().startswith(code_lower[:3]):
                    suggestions.append(icd_code)
                if len(suggestions) >= 3:
                    break

            invalid_codes.append({
                "code": code,
                "suggestions": suggestions[:3]
            })

    if len(unique_codes) > max_codes:
        warnings.append(f"Only the first {max_codes} codes were processed. {len(unique_codes) - max_codes} codes were ignored.")

    if len(potential_codes) != len(unique_codes):
        warnings.append(f"Removed {len(potential_codes) - len(unique_codes)} duplicate codes.")

    return {
        "valid_codes": valid_codes,
        "invalid_codes": invalid_codes,
        "warnings": warnings,
        "total_found": len(unique_codes)
    }


@app.post("/parse_icd_codes/")
async def parse_icd_codes(data: dict):
    """
    Parse ICD codes from pasted text with flexible format support.

    Accepts text in various formats (comma, space, newline separated) and
    validates against the ICD-10 database.

    Args:
        data: Dictionary with 'text' field containing ICD codes

    Returns:
        Parsed and validated ICD codes with validation results
    """
    text = data.get('text', '')
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
    if file.content_type not in ["text/csv", "text/plain", "text/x-csv", "application/csv"]:
        if not file.filename.endswith(('.txt', '.csv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a TXT or CSV file."
            )

    try:
        contents = await file.read()
        text = contents.decode('utf-8')
        result = parse_icd_codes_from_text(text)
        return result
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 text files.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")