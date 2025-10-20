def split_sequences_TransMIT(sequences, s):
    X, y = list(),list()
    for i in range(len(sequences)):
        end_ix = i + s + 1 
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1, :] 
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# def data_split(data_x, miss_data_x, data_m, train_size, s):
#   train = data_x[:train_size,:]
#   test = data_x[train_size-n_steps:,:]
#   mask_train = miss_data_x[:train_size,:]
#   mask_test = miss_data_x[train_size-n_steps:,:]
#   train_mask = train*mask_train
#   test_mask = test*mask_test
  
#   train_x, y = split_sequences(train, n_steps)
#   train_x[:,-1,:] = train_mask[n_steps:,:] 

def GPTImputer(data,MaskMatrix,train_size):
  n_steps = 16
  train = data[:train_size,:]
  test = data[train_size-n_steps:,:]
  mask_train = MaskMatrix[:train_size,:]
  mask_test = MaskMatrix[train_size-n_steps:,:]
  train_mask = train*mask_train
  test_mask = test*mask_test
  #sliding window
  train_x, y = split_sequences(train, n_steps)
  train_x[:,-1,:] = train_mask[n_steps:,:] #
  #shuffle the train
  indices = tf.range(start=0, limit=tf.shape(train_x)[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)
  train_x = tf.gather(train_x,shuffled_indices)
  y = tf.gather(y,shuffled_indices)
  #Define hyperparameters
  batch_size = 200
  alpha = 0.5 #0.5
  seq_length = n_steps+1 ##
  num_features = data.shape[1] #89
  num_layers = 4   ## 8
  d_model = 128
  num_heads = 4 ## 8
  dropout_rate = 0.1
  lr = 0.0005 #0.0001
  # Define the input and output data
  inputs = tf.keras.layers.Input(shape=(seq_length, num_features))
  Inputs = tf.keras.layers.Dense(d_model)(inputs)
  Inputs_t = Permute((2, 1))(inputs)
  Inputs_t = tf.keras.layers.Dense(d_model)(Inputs_t)
  #last_step = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(inputs)
  mask = tf.keras.layers.Lambda(lambda x: tf.cast(tf.math.not_equal(x, 0), tf.float32))(inputs)
  mask = mask[:,-1,:]

  x = Inputs
  x_t = Inputs_t
  # Create the transformer encoder layers
  for i in range(num_layers):
      attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=64,value_dim=64)(x, x, x, attention_mask=None)
      x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
      ffn_output = tf.keras.layers.Dense(d_model)(x)
      x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
  for i in range(num_layers):
      attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=64,value_dim=64)(x_t, x_t, x_t, attention_mask=None)
      x_t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_t + attn_output)
      ffn_output = tf.keras.layers.Dense(d_model)(x_t)
      x_t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_t + ffn_output)

  #x = tf.keras.layers.Dense(num_features)(x)
  x_t = tf.keras.layers.Dense(seq_length)(x_t)
  x_t = Permute((2, 1))(x_t)
  outputs = tf.keras.layers.Concatenate(axis=-1)([x, x_t])

  outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
  #outputs = tf.keras.Sequential([tf.keras.layers.Dense(256),tf.keras.layers.Dense(128),tf.keras.layers.Dense(num_features)])(outputs)
  outputs = tf.keras.layers.Dense(num_features)(outputs)
  #outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
  #outputs = tf.keras.layers.Dense(num_features)(outputs)
  outputs = tf.keras.layers.Concatenate(axis=-1)([outputs, mask])
  # Create the model
  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  def RMSE3D(y_true, y_pred):
    n = y_true.shape[-1]
    mask = y_pred[:,n:]
    total_elements = tf.cast(tf.size(mask), tf.float32)
    count_nonzeros = tf.math.count_nonzero(mask,dtype=tf.float32)
    count_zeros = total_elements-count_nonzeros
    reconstruction_loss = tf.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.multiply(y_true, mask) - tf.math.multiply(y_pred[:,:n], mask)))*total_elements/count_nonzeros)
    imputation_loss = tf.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.multiply(y_true, 1-mask) - tf.math.multiply(y_pred[:,:n], 1-mask)))*total_elements/count_zeros)
    loss = alpha*reconstruction_loss+(1-alpha)*imputation_loss
    return loss
  adam = tf.keras.optimizers.Adam(learning_rate=lr)
  model.compile(loss= RMSE3D, optimizer=adam)
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
  # Train the model
  Epochs = 200
  trainsize = int(round(train_x.shape[0] * 0.8)/Epochs)*Epochs
  history = model.fit(train_x[:trainsize], y[:trainsize], batch_size=batch_size, epochs=Epochs, validation_data=(train_x[trainsize:], y[trainsize:]),
                      validation_batch_size=Epochs,callbacks=[es],verbose=0)
