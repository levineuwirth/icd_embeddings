"""
Reference Integrated Gradients: the analysis pipeline's algorithm, for tests.

A port of src/feature_importance/IG.py in Rice-wxl/icd-10-embedding (commit
6492eaa) reduced to one record: interpolate the whole embedding table from the
NAN/padding row to its trained values by assigning into the embedding
variable, take the gradient of the mean logit at each of `steps` right
endpoints, and sum delta * mean-gradient over the embedding dimension. It
mutates the model it is given, so it loads a private copy.
"""

import numpy as np
import tensorflow as tf
from keras.models import load_model


def pipeline_ig(model_path, inputs, pad_id, steps=32, eps=1e-6):
    """Per-code-id attributions (an array over the whole vocabulary)."""
    model = load_model(model_path)
    emb_var = model.get_layer("icd_embedding").embeddings
    E_orig = emb_var.read_value()
    E0 = tf.repeat(tf.gather(E_orig, [pad_id]), repeats=int(E_orig.shape[0]), axis=0)
    deltaE = E_orig - E0
    grads_accum = tf.zeros_like(E_orig)
    try:
        for s in range(1, steps + 1):
            emb_var.assign(E0 + (s / steps) * deltaE)
            with tf.GradientTape() as tape:
                tape.watch(emb_var)
                out = tf.reshape(model(inputs, training=False), (-1,))
                out = tf.clip_by_value(out, eps, 1 - eps)
                scalar = tf.reduce_mean(tf.math.log(out) - tf.math.log1p(-out))
            grads_accum += tf.convert_to_tensor(tape.gradient(scalar, emb_var))
    finally:
        emb_var.assign(E_orig)
    return tf.reduce_sum(deltaE * grads_accum / steps, axis=1).numpy()
