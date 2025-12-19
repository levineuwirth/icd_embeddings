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


   p_sampled = tf.clip_by_value(p_sampled, eps, 1-eps, name=None)
   return p_sampled / (p_sampled + (1 - p_sampled) / beta)