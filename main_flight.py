
# Necessary packages
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import data_loader
from TransMIT import TransMIT
from utils import split_sequences_TransMIT, train_test_split


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate

  TransMIT_parameters = {'train_size':args.train_size,
                         'batch_size':args.batch_size,
                         'lr':args.lr,
                         'epochs':args.epochs,
                         'alpha':args.alpha,
                         's':args.s,
                         'd_model':args.d_model,
                         'd_q':args.d_q,
                         'num_layers':args.num_layers,
                         'num_heads':args.num_heads}
  
  # Load data and introduce missingness
  data_x, data_m = data_loader(data_name, miss_rate)

  # data preprocessing
  train_size = int(round(data_x.shape[0] * 0.8))
  train_data = data_x[:train_size, :]
  test_data = data_x[train_size:, :]
  data_m_train = data_m[:train_size, :]
  data_m_test = data_m[train_size:, :]
  test_mask = test_data*data_m_test 

  # z-score normalizaiton
  scaler = StandardScaler()
  data_train = scaler.fit_transform(train_data)
  data_test = scaler.transform(test_data)

  # Train model
  model = TransMIT(data_train, data_m_train, TransMIT_parameters)
  
  # Evaluate the model performance on the test datast
  #online imputation
  def online_imputation(test_data,data_m_test,s):
    data_m_test = tf.cast(data_m_test, tf.float32)
    test_mask = test_data*data_m_test
    test_copy = test_data.copy()
    X_hat = list()
    for i in range(test_data.shape[0]-s):
      print(i)
      x = test_copy[i:i+s+1,:]
      x = x.reshape((1,x.shape[0],x.shape[1]))
      x[:,-1,:] = test_mask[i+s,:]
      x_hat = model.predict(x)[:,:test.shape[1]]
      X_hat.append(x_hat)
      #updating
      test_copy[i+s,:] = tf.math.multiply(test_copy[i+s,:],data_m_test[i+s,:]) + tf.math.multiply(x_hat,1-data_m_test[i+n_steps,:])
    X_hat = np.array(X_hat) #3d
    X_hat = X_hat.reshape((X_hat.shape[0],X_hat.shape[2]))    

  count_zeros = np.count_nonzero(mask_test[s:,:] == 0)
  rmse = tf.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.multiply(test_data[s:,:], 1-mask_test[s:,:]) - tf.math.multiply(X_hat, 1-mask_test[s:,:])))/count_zeros)
  mae = tf.math.reduce_sum(tf.math.abs(tf.math.multiply(test[s:,:], 1-mask_test[s:,:]) - tf.math.multiply(X_hat, 1-mask_test[s:,:])))/count_zeros
  rmse = np.array(rmse)
  mae = np.array(mae)

  return rmse, mae, X_hat





  
  def evaluate_test(test_data,data_m_test,s): 
    n_samples = int(test_data.shape[0]/s)
    n_features = test_data.shape[1]
    test_mask = test_data*data_m_test 
    a = test_mask[:n_samples*n_steps,:]
    a = a.reshape(n_samples,s,n_features)
    b = test_data[:n_samples*n_steps,:]
    b = b.reshape(n_samples,s,n_features)
    mask = data_m_test[:n_samples*s,:]
    mask = mask.reshape(n_samples,s,n_features) 
    count_zeros = np.sum(1-mask)
    preds = model.predict(a)[:,:,:n_features] 
    b = tf.cast(b, tf.float32)
    b = tf.math.multiply(b, 1-mask)
    preds = tf.math.multiply(preds, 1-mask)
    rmse = tf.sqrt(tf.math.reduce_sum(tf.math.square(b - preds))/count_zeros)
    mae = tf.math.reduce_sum(tf.math.abs(b - preds))/count_zeros
    mae = np.array(mae)
    rmse = np.array(rmse)

    return rmse,mae
  
  
  
  rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)
