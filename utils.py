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










