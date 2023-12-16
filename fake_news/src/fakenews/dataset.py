import os
import torch
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def tokenize(df, proj_dir, name):
    tokenizer = Tokenizer(num_words=5000) # default = 5000 words
    tokenizer.fit_on_texts(df['text'])

    tokenizer_folder = proj_dir + '/src/tokenizer'
    os.makedirs(tokenizer_folder, exist_ok=True)

    tokenizer_filename = f'{tokenizer_folder}/{name}.pickle'

    with open(tokenizer_filename, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    X = tokenizer.texts_to_sequences(df['text'])

    # Finding maximum sequence length
    max_l = 0
    for sequence in X:
        if len(sequence) > max_l:
            max_l = len(sequence)
    
    print("Maximum Sequence Length: ", max_l)
    
    X = pad_sequences(X, maxlen=max_l) # padding seq to maintain constant length

    X_train, X_test, y_train, y_test = train_test_split(X, list(df['label']), test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, tokenizer

def data_loaders(X_train, X_test, y_train, y_test):
    # Numpy to Tensors
    X_train_tensor = torch.Tensor(X_train)
    X_test_tensor = torch.Tensor(X_test)
    y_train_tensor = torch.Tensor(y_train)
    y_test_tensor = torch.Tensor(y_test)

    # TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader