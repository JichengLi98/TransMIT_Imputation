
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

  # z-score normalizaiton
  scaler = StandardScaler()
  data_train = scaler.fit_transform(train_data)
  data_test = scaler.transform(test_data)

  
  # Train model
  model = TransMIT(data_train, data_m_train, TransMIT_parameters)
  
  # Evaluate the model performance on the test datast
  
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
