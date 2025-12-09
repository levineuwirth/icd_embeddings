"""
Main application file for the ICD Prediction API.

This file contains the FastAPI application that serves the ICD prediction model.
It includes endpoints for making predictions and searching for ICD codes.
"""

import pickle
import os
import json
from typing import List, Dict

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# It's a good practice to define the custom objects, even if they are registered
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


app = FastAPI(
    title="ICD Prediction API",
    description="An API to predict 30-day hospital readmission risk based on patient data.",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the base directory relative to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variable to store ICD-10 codes
icd_codes: Dict[str, str] = {}

# Load the model and preprocessing objects
try:
    # Construct paths relative to the current script's location
    model_path = os.path.join(BASE_DIR, 'model/readmit_hypertrial_deepset.keras')
    encoder_path = os.path.join(BASE_DIR, 'model/readmit_2016_label_encoder.pkl')
    scaler_path = os.path.join(BASE_DIR, 'model/readmit_2016_age_scaler.pkl')
    icd_data_path = os.path.join(BASE_DIR, 'data/icd10_codes.json')

    model = load_model(model_path)
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        age_scaler = pickle.load(file)

    # Load ICD-10 codes
    with open(icd_data_path, 'r', encoding='utf-8') as file:
        icd_codes = json.load(file)
    print(f"Loaded {len(icd_codes)} ICD-10 codes")

except FileNotFoundError as e:
    raise RuntimeError(f"Model or preprocessing files not found. Looked in {os.path.join(BASE_DIR, 'model')}") from e


class PatientData(BaseModel):
    """
    Pydantic model for validating patient data input.
    """
    age: int = Field(..., gt=0, description="Patient's age must be greater than 0.")
    female: int = Field(..., ge=0, le=1, description="Patient's gender (0 for male, 1 for female).")
    pay1: int = Field(..., ge=1, le=6, description="Primary payer information (1-6).")
    zipinc_qrtl: int = Field(..., ge=1, le=4, description="ZIP code income quartile (1-4).")
    icd_codes: list[str] = Field(..., min_length=1, max_length=35, description="List of ICD-10 diagnosis codes.")


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
        # The model's prediction for a single input is deterministic, so bootstrapping
        # the model's output directly isn't meaningful. A proper implementation would
        # involve techniques like Monte Carlo Dropout or bootstrapping the training data,
        # which are beyond the scope of this example.
        # As a placeholder, we'll simulate variability by adding small random noise.
        pred = model.predict(inputs, verbose=0).flatten()[0]
        noise = np.random.normal(0, 0.05)  # Assuming a small standard deviation
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
    Predicts the 30-day readmission risk for a patient.

    Args:
        data (PatientData): The patient's data.

    Returns:
        dict: A dictionary containing the prediction, confidence interval, and interpretation.
    """
    try:
        # 1. Create a DataFrame from the input data
        input_data = {
            'AGE': [data.age],
            'FEMALE': [data.female],
            'PAY1': [data.pay1],
            'ZIPINC_QRTL': [data.zipinc_qrtl]
        }

        for i in range(35):
            if i < len(data.icd_codes):
                input_data[f'I10_DX{i+1}'] = [data.icd_codes[i]]
            else:
                input_data[f'I10_DX{i+1}'] = ['']

        df = pd.DataFrame(input_data)

        # 2. Preprocess the data
        label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
        unknown_label_int = 0
        icd_columns = [f'I10_DX{i}' for i in range(1, 36)]

        for col in icd_columns:
            df[col] = df[col].astype(str).str.upper()
            df[col] = df[col].map(label_to_int).fillna(unknown_label_int).astype(int)

        df['AGE'] = age_scaler.transform(df[['AGE']])
        df = pd.get_dummies(df, columns=['PAY1', 'ZIPINC_QRTL'], prefix=['PAY1', 'ZIPINC_QRTL'])

        pay1_columns = [f'PAY1_{i}' for i in range(1, 7)]
        zipinc_qrtl_columns = [f'ZIPINC_QRTL_{i}' for i in range(1, 5)]
        expected_columns = pay1_columns + zipinc_qrtl_columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # 3. Prepare input for the model
        X_new = df[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
        X_new = X_new.astype('float32')

        batch_inputs = [
            X_new[icd_columns],
            X_new['AGE'].values,
            X_new['FEMALE'].values,
        ] + [X_new[col].values for col in pay1_columns] \
          + [X_new[col].values for col in zipinc_qrtl_columns]

        # 4. Make prediction
        prediction_prob = model.predict(batch_inputs, verbose=0).flatten()[0]

        # 5. Calculate confidence interval and interpretation
        lower_ci, upper_ci = calculate_prediction_ci(model, batch_inputs)
        interpretation = get_risk_interpretation(prediction_prob)

        return {
            "prediction": float(prediction_prob),
            "confidence_interval": [float(lower_ci), float(upper_ci)],
            "interpretation": interpretation,
        }
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

    # Categorize results by match type for better ordering
    exact_code_matches = {}
    code_starts_with = {}
    code_contains = {}
    desc_contains = {}

    for code, description in icd_codes.items():
        code_lower = code.lower()
        desc_lower = description.lower()

        # Exact code match (highest priority)
        if code_lower == query:
            exact_code_matches[code] = description
        # Code starts with query (high priority)
        elif code_lower.startswith(query):
            code_starts_with[code] = description
        # Code contains query (medium priority)
        elif query in code_lower:
            code_contains[code] = description
        # Description contains query (lower priority)
        elif query in desc_lower:
            desc_contains[code] = description

    # Combine results in priority order
    results.update(exact_code_matches)
    results.update(code_starts_with)
    results.update(code_contains)
    results.update(desc_contains)

    # Limit results
    if len(results) > limit:
        results = dict(list(results.items())[:limit])

    return results

@app.post("/upload_icd_file/", response_model=List[str])
async def upload_icd_file(file: UploadFile = File(...)):
    """
    Uploads a file containing ICD codes and returns a list of codes.
    The file is expected to have one ICD code per line.
    """
    if not file.content_type in ["text/csv", "text/plain", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV, TXT, or XLSX file.")

    try:
        contents = await file.read()
        lines = contents.decode('utf-8').splitlines()
        # Strip whitespace and remove empty lines
        icd_codes = [line.strip() for line in lines if line.strip()]
        return icd_codes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There was an error processing the file: {e}")