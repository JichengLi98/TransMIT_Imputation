'''Data loader for flight test process, industrial 660MW boiler process datasets.
'''

# Necessary packages
import numpy as np

def data_loader(data_name, miss_rate):
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
  missing_file_name = f'missing/{data_name}_{int(miss_rate * 100)}.csv'
  data_x = np.loadtxt(file_name, delimiter=",")     # no header
  data_m = np.loadtxt(missing_file_name, delimiter=",")

  # Parameters
  no, dim = data_x.shape

  # Introduce missing data
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m

  














