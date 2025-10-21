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
  #sliding window
  train_x, train_y = split_sequences_TransMIT(train, n_steps)
  train_x[:,-1,:] = train_mask[n_steps:,:] 
return train_x, train_y, test, test_mask







