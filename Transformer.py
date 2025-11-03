import tensorflow as tf
import numpy as np
from utils import split_sequences_TransMIT

def TransMIT(train_data, missing_matrix, TransMIT_parameters):
  '''Train
  
  Args:
    - data: original data with missing values
    - TransMIT_parameters: TransMIT model parameters
      - train_size:
      - s: 
      - alpha: 
      - batch_size: Batch size
      - num_layers:
      - d_model:
      - d_q: 
      - num_heads:
      - lr: Learning rate
      - epochs: Epochs
      
  Returns:
    - model: Trained model
  '''

  # System parameters
  train_size = TransMIT_parameters['train_size']
  batch_size = TransMIT_parameters['batch_size']
  lr = TransMIT_parameters['lr']
  epochs = TransMIT_parameters['epochs']
  alpha = TransMIT_parameters['alpha']
  s = TransMIT_parameters['s']
  d_model = TransMIT_parameters['d_model']
  d_q = TransMIT_parameters['d_q']
  num_layers = TransMIT_parameters['num_layers']
  num_heads = TransMIT_parameters['num_heads']
  #
  seq_length = s+1
  num_features = train_data.shape[1]

  #
  train_mask = train_data*missing_matrix
  train_x, train_y = split_sequences_TransMIT(train_data, s)
  train_x[:,-1,:] = train_mask[s:,:]  
  # train_train_size = int(round(train_size/8*6))
  # train_train_x = train_x[:train_train_size]
  # train_train_y = train_y[:train_train_size]
  # train_val_x = train_x[train_train_size:]
  # train_val_y = train_y[train_train_size:]  
  
  #shuffle the training dataset
  # indices = tf.range(start=0, limit=tf.shape(train_train_x)[0], dtype=tf.int32)
  # shuffled_indices = tf.random.shuffle(indices)
  # train_train_x = tf.gather(train_train_x,shuffled_indices)
  # train_train_y = tf.gather(train_train_y,shuffled_indices)
  indices = tf.range(start=0, limit=tf.shape(train_x)[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)
  train_x = tf.gather(train_x,shuffled_indices)
  train_y = tf.gather(train_y,shuffled_indices)
  
  # Define the input and output data
  inputs = tf.keras.layers.Input(shape=(seq_length, num_features))
  mask = tf.keras.layers.Lambda(lambda x: tf.cast(tf.math.not_equal(x, 0), tf.float32))(inputs)
  mask = mask[:,-1,:]
  Inputs = tf.keras.layers.Dense(d_model)(inputs)
  Inputs_t = tf.keras.layers.Permute((2, 1))(inputs)
  Inputs_t = tf.keras.layers.Dense(d_model)(Inputs_t)
  x = Inputs
  x_t = Inputs_t
  for i in range(num_layers):
      attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim= d_q, value_dim=d_q, dropout=0.1)(x, x, x, attention_mask=None)
      x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
      ffn_output = tf.keras.layers.Dense(d_model)(x)
      x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
      #x = tf.keras.layers.Dense(d_model)(x)
  for i in range(num_layers):
      attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_q, value_dim=d_q, dropout=0.1)(x_t, x_t, x_t, attention_mask=None)
      x_t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_t + attn_output)
      ffn_output = tf.keras.layers.Dense(d_model)(x_t)
      x_t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_t + ffn_output)

  x = tf.keras.layers.Dense(num_features)(x)
  x_t = tf.keras.layers.Dense(seq_length)(x_t)
  x_t = tf.keras.layers.Permute((2, 1))(x_t)
  outputs = tf.keras.layers.Concatenate(axis=-1)([x, x_t])
  outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
  outputs = tf.keras.layers.Dense(num_features)(outputs)
  outputs = tf.keras.layers.Concatenate(axis=-1)([outputs, mask])

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
  def MSE_TransMIT(y_true, y_pred):
    n = y_true.shape[-1]
    mask = y_pred[:,n:]
    total_elements = tf.cast(tf.size(mask), tf.float32)
    count_nonzeros = tf.math.count_nonzero(mask,dtype=tf.float32)
    count_zeros = total_elements-count_nonzeros
    reconstruction_loss = tf.math.reduce_mean(tf.math.square(tf.math.multiply(y_true, mask) - tf.math.multiply(y_pred[:,:n], mask)))*total_elements/count_nonzeros
    imputation_loss = tf.math.reduce_mean(tf.math.square(tf.math.multiply(y_true, 1-mask) - tf.math.multiply(y_pred[:,:n], 1-mask)))*total_elements/count_zeros
    loss = alpha*reconstruction_loss+(1-alpha)*imputation_loss
    return loss
      
  adam = tf.keras.optimizers.Adam(learning_rate=lr)
  model.compile(loss= MSE_TransMIT, optimizer=adam)
  es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
  # Train the model
  Epochs = epochs
  trainsize = int(round(train_x.shape[0] * 0.8)/Epochs)*Epochs
  history = model.fit(train_x[:trainsize], train_y[:trainsize], batch_size=batch_size, epochs=Epochs, validation_data=(train_x[trainsize:], train_y[trainsize:]),
                      validation_batch_size=Epochs,callbacks=[es],verbose=0)
  # history = model.fit(train_train_x, train_train_y, batch_size=batch_size, epochs=Epochs, validation_data=(train_val_x, train_val_y),
  #                     validation_batch_size=Epochs,callbacks=[es],verbose=0)  
  return model 
