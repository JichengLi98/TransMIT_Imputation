'''Data loader for flight test process, industrial 660MW boiler process datasets.
'''

# Necessary packages
import numpy as np

def data_loader(data_name):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  file_name = 'data/'+data_name+'.csv'
  #missing_file_name = f'missing/{data_name}_{int(miss_rate * 100)}.csv'
  data_x = np.loadtxt(file_name, delimiter=",", encoding='utf-8-sig')     # no header
  #data_m = np.loadtxt(missing_file_name, delimiter=",", encoding='utf-8-sig')

  #return data_x, data_m
  return data_x

  














