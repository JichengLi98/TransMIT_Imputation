import numpy as np
import tensorflow as tf

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


def train_test_split(data, missing_matrix, train_size,s):
  train = data[:train_size,:]
  test = data[train_size-s:,:]
  missing_train = missing_matrix[:train_size,:]
  missing_test = missing_matrix[train_size-n_steps:,:]
  train_mask = train*missing_train
  test_mask = test*missing_test
  train_x, train_y = split_sequences_TransMIT(train, n_steps)
  train_x[:,-1,:] = train_mask[n_steps:,:] 
  return train_x, train_y, test, test_mask

def online_imputation(model,test_data,data_m_test,s):
    data_m_test = tf.cast(data_m_test, tf.float32)
    test_mask = test_data*data_m_test
    test_copy = test_data.copy()
    X_hat = list()
    for i in range(test_data.shape[0]-s):
      #print(i)
      x = test_copy[i:i+s+1,:]
      x = x.reshape((1,x.shape[0],x.shape[1]))
      x[:,-1,:] = test_mask[i+s,:]
      x_hat = model.predict(x)[:,:test_data.shape[1]]
      X_hat.append(x_hat)
      #updating
      test_copy[i+s,:] = tf.math.multiply(test_copy[i+s,:],data_m_test[i+s,:]) + tf.math.multiply(x_hat,1-data_m_test[i+s,:])
    X_hat = np.array(X_hat) 
    X_hat = X_hat.reshape((X_hat.shape[0],X_hat.shape[2]))    

    count_zeros = np.count_nonzero(data_m_test[s:,:] == 0)
    rmse = tf.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.multiply(test_data[s:,:], 1-data_m_test[s:,:]) - tf.math.multiply(X_hat, 1-data_m_test[s:,:])))/count_zeros)
    mae = tf.math.reduce_sum(tf.math.abs(tf.math.multiply(test_data[s:,:], 1-data_m_test[s:,:]) - tf.math.multiply(X_hat, 1-data_m_test[s:,:])))/count_zeros
    rmse = np.array(rmse)
    mae = np.array(mae)

    return rmse, mae




