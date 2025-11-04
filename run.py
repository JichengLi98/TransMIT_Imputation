# Necessary packages
import argparse
import random
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import data_loader
from TransMIT import TransMIT
#from Transformer import Transformer
#from iTransformer import iTransformer
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
  s = args.s
  train_size = int(round(data_x.shape[0] * 0.8))
  train_data = data_x[:train_size, :]
  test_data = data_x[train_size-s:, :]  #
  data_m_train = data_m[:train_size, :]
  data_m_test = data_m[train_size-s:, :]
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
  rmse, mae = online_imputation(model,data_test,data_m_test,s)
  
  print()
  print('missing rate:' + str(miss_rate))
  print('RMSE Performance: ' + str(rmse) + ', MAE Performance: ' + str(mae))
  
  return rmse, mae

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument('--fix_seed', type=int, default=2025)
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
  
  fix_seed = args.fix_seed
  random.seed(fix_seed)
  np.random.seed(fix_seed)
  tf.random.set_seed(fix_seed)
  print(f"Running with seed = {fix_seed}")
 
  rmse, mae = main(args)

