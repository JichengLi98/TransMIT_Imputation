# Necessary packages
import argparse
import random
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import data_loader
from TransMIT import TransMIT
from utils import split_sequences_TransMIT, train_test_split, online_imputation

def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch_size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  # Load data and introduce missingness
  #data_x, data_m = data_loader(data_name, miss_rate)
  data_x = data_loader(data_name)
  data_m = np.random.rand(*data_x.shape) > miss_rate
  data_m = data_m.astype(float)
  
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

  TransMIT_parameters = {'train_size':train_size,
                         'batch_size':args.batch_size,
                         'lr':args.lr,
                         'epochs':args.epochs,
                         'alpha':args.alpha,
                         's':args.s,
                         'd_model':args.d_model,
                         'd_q':args.d_q,
                         'num_layers':args.num_layers,
                         'num_heads':args.num_heads}  
  # Train model
  model = TransMIT(data_train, data_m_train, TransMIT_parameters)
  
  # Evaluate the model performance on the test datast
  s = args.s
  rmse, mae = online_imputation(model,data_test,data_m_test,s)
  
  print()
  print('missing rate:' + str(miss_rate))
  print('RMSE Performance: ' + str(np.round(rmse, 3)) + ', MAE Performance: ' + str(np.round(mae, 3)))
  
  return rmse, mae

if __name__ == '__main__':  
  fix_seed = 2025
  # Fix Python built-in random seed
  random.seed(fix_seed)
  # Fix NumPy random seed
  np.random.seed(fix_seed)
  # Fix TensorFlow random seed
  tf.random.set_seed(fix_seed)
  
  # Enable GPU memory growth
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
      try:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          print("GPUs found and memory growth enabled")
      except RuntimeError as e:
          print(e)
    
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['FTD','Boiler'],
      default='FTD',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.1,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='batch_size',
      default=100,
      type=int)
  parser.add_argument(
      '--lr',
      help='lr',
      default=0.0005,
      type=float)
  parser.add_argument(
      '--epochs',
      help='epochs',
      default=200,
      type=int)
  parser.add_argument(
      '--alpha',
      help='alpha',
      default=0.5,
      type=float)
  parser.add_argument(
      '--s',
      help='s',
      default=16,
      type=int)
  parser.add_argument(
      '--d_model',
      help='d_model',
      default=128,
      type=int)
  parser.add_argument(
      '--d_q',
      help='d_q',
      default=128,
      type=int)
  parser.add_argument(
      '--num_layers',
      help='num_layers',
      default=4,
      type=int)
  parser.add_argument(
      '--num_heads',
      help='num_heads',
      default=4,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  if gpus:
      device_name = '/GPU:0'
  else:
      device_name = '/CPU:0'

  with tf.device(device_name):
      rmse, mae = main(args)  
  #rmse, mae = main(args)

