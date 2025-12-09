"""
Validation script to test the new models against expected predictions.
Based on evaluate_clean.py logic.
"""

import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
import sys

# Register custom Keras components
@tf.keras.utils.register_keras_serializable(package="Custom")
def f2_score(y_true, y_pred):
    """Custom F2 score function"""
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
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
    """DeepSet aggregation for permutation-invariant set modeling"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_encode, num_decode, **kwargs):
        super(DeepSet, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_encode = num_encode
        self.num_decode = num_decode
        self.phi = tf.keras.Sequential([
            Dense(self.hidden_dim, activation='relu') for _ in range(self.num_encode)
        ])
        self.rho = tf.keras.Sequential([
            Dense(self.hidden_dim, activation='relu') for _ in range(self.num_decode - 1)
        ] + [Dense(self.output_dim, activation='relu')])

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
    """Transformer encoder block with multi-head attention"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
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
    """F2 score metric"""
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

print("=" * 80)
print("MODEL VALIDATION SCRIPT")
print("=" * 80)

# Load models and preprocessing objects
print("\nLoading models and preprocessors...")
try:
    model_readmit = load_model('model/readmit_hypertrial_auc.keras')
    model_mortality = load_model('model/mort_nodie_hypertrial_auc.keras')

    with open('model/full_label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    with open('model/full_age_scaler.pkl', 'rb') as f:
        age_scaler = pickle.load(f)

    print(f"✓ Readmission model loaded: {model_readmit.name}")
    print(f"✓ Mortality model loaded: {model_mortality.name}")
    print(f"✓ Encoder loaded: {len(encoder.classes_)} codes")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    sys.exit(1)

# Load test datasets
print("\nLoading test datasets...")
test_rea30 = pd.read_csv('small_dataset_predictions_rea30.csv')
test_mor30 = pd.read_csv('small_dataset_predictions_mor30.csv')

print(f"✓ REA30 test data: {len(test_rea30)} samples")
print(f"✓ MOR30 test data: {len(test_mor30)} samples")

def prepare_data_for_prediction(df, encoder, age_scaler):
    """
    Prepare data for model prediction following evaluate_clean.py approach.
    """
    # Make a copy
    data = df.copy()

    # Standardize column names
    data.columns = data.columns.str.upper()

    # Define ICD columns
    icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

    # Encode ICD codes
    print("  Encoding ICD codes...")
    label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
    unknown_label_int = encoder.transform(["NAN"])[0]

    for col in icd_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.upper()
            data[col] = data[col].map(label_to_int).fillna(unknown_label_int).astype(int)
        else:
            data[col] = unknown_label_int

    # Normalize AGE
    print("  Normalizing AGE...")
    data['AGE'] = age_scaler.transform(data[['AGE']])

    # Handle missing value codes
    data['PAY1'] = data['PAY1'].replace([-8, -9], np.nan)
    data['ZIPINC_QRTL'] = data['ZIPINC_QRTL'].replace([-8, -9], np.nan)

    # One-hot encode
    print("  One-hot encoding...")
    data = pd.get_dummies(data, columns=['PAY1', 'ZIPINC_QRTL'],
                         prefix=['PAY1', 'ZIPINC_QRTL'])

    # Ensure all expected columns exist
    expected_pay1_cols = ['PAY1_1.0', 'PAY1_2.0', 'PAY1_3.0', 'PAY1_4.0', 'PAY1_5.0', 'PAY1_6.0']
    expected_zipinc_cols = ['ZIPINC_QRTL_1.0', 'ZIPINC_QRTL_2.0', 'ZIPINC_QRTL_3.0', 'ZIPINC_QRTL_4.0']

    for col in expected_pay1_cols + expected_zipinc_cols:
        if col not in data.columns:
            data[col] = 0

    pay1_columns = expected_pay1_cols
    zipinc_qrtl_columns = expected_zipinc_cols

    # Extract features
    X = data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
    X = X.dropna()

    return X, icd_columns, pay1_columns, zipinc_qrtl_columns

def prepare_model_inputs(X, icd_columns, pay1_columns, zipinc_qrtl_columns):
    """
    Prepare inputs in the format expected by the model.
    """
    return [
        X[icd_columns],
        X['AGE'],
        X['FEMALE'],
    ] + [X[c] for c in pay1_columns] \
      + [X[c] for c in zipinc_qrtl_columns]

# Test REA30 model
print("\n" + "=" * 80)
print("TESTING READMISSION MODEL (REA30)")
print("=" * 80)

X_rea, icd_cols, pay1_cols, zipinc_cols = prepare_data_for_prediction(test_rea30, encoder, age_scaler)
rea_inputs = prepare_model_inputs(X_rea, icd_cols, pay1_cols, zipinc_cols)

print("\nMaking predictions...")
rea_predictions = model_readmit.predict(rea_inputs, batch_size=32, verbose=0).squeeze()

print(f"\nSample predictions (first 5):")
for i in range(min(5, len(rea_predictions))):
    expected = test_rea30.iloc[i]['predicted_probability'] if 'predicted_probability' in test_rea30.columns else None
    pred = rea_predictions[i]
    print(f"  Row {i+1}: Predicted={pred:.6f}", end="")
    if expected is not None:
        print(f", Expected={expected:.6f}, Diff={abs(pred - expected):.6f}")
    else:
        print()

# Find the row with expected value 0.502200
target_value = 0.502200
if 'predicted_probability' in test_rea30.columns:
    matches = test_rea30[abs(test_rea30['predicted_probability'] - target_value) < 0.0001]
    if len(matches) > 0:
        idx = matches.index[0]
        print(f"\n✓ Found target value {target_value} at row {idx+1}")
        print(f"  Expected:  {test_rea30.iloc[idx]['predicted_probability']:.6f}")
        print(f"  Predicted: {rea_predictions[idx]:.6f}")
        print(f"  Difference: {abs(rea_predictions[idx] - test_rea30.iloc[idx]['predicted_probability']):.6f}")

# Test MOR30 model
print("\n" + "=" * 80)
print("TESTING MORTALITY MODEL (MOR30)")
print("=" * 80)

X_mor, icd_cols, pay1_cols, zipinc_cols = prepare_data_for_prediction(test_mor30, encoder, age_scaler)
mor_inputs = prepare_model_inputs(X_mor, icd_cols, pay1_cols, zipinc_cols)

print("\nMaking predictions...")
mor_predictions = model_mortality.predict(mor_inputs, batch_size=32, verbose=0).squeeze()

print(f"\nSample predictions (first 5):")
for i in range(min(5, len(mor_predictions))):
    expected = test_mor30.iloc[i]['predicted_probability'] if 'predicted_probability' in test_mor30.columns else None
    pred = mor_predictions[i]
    print(f"  Row {i+1}: Predicted={pred:.6f}", end="")
    if expected is not None:
        print(f", Expected={expected:.6f}, Diff={abs(pred - expected):.6f}")
    else:
        print()

# Find the row with expected value 0.501647
target_value = 0.501647
if 'predicted_probability' in test_mor30.columns:
    matches = test_mor30[abs(test_mor30['predicted_probability'] - target_value) < 0.0001]
    if len(matches) > 0:
        idx = matches.index[0]
        print(f"\n✓ Found target value {target_value} at row {idx+1}")
        print(f"  Expected:  {test_mor30.iloc[idx]['predicted_probability']:.6f}")
        print(f"  Predicted: {mor_predictions[idx]:.6f}")
        print(f"  Difference: {abs(mor_predictions[idx] - test_mor30.iloc[idx]['predicted_probability']):.6f}")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
