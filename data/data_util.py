import pandas as pd
import os
from sklearn.datasets import load_svmlight_file
import numpy as np

def svm2pkl(source, save_path):
    X_train, y_train = load_svmlight_file(os.path.join(source, 'train'))
    X_valid, y_valid = load_svmlight_file(os.path.join(source, 'vali'))
    X_test, y_test = load_svmlight_file(os.path.join(source, 'test'))

    X_train = pd.DataFrame(X_train.todense())
    y_train = pd.Series(y_train)
    pd.concat([y_train, X_train], axis=1).T.reset_index(drop=True).T.to_pickle(os.path.join(save_path, 'train.pkl'))

    X_valid = pd.DataFrame(X_valid.todense())
    y_valid = pd.Series(y_valid)
    pd.concat([y_valid, X_valid], axis=1).T.reset_index(drop=True).T.to_pickle(os.path.join(save_path, 'valid.pkl'))

    X_test = pd.DataFrame(X_test.todense())
    y_test = pd.Series(y_test)
    pd.concat([y_test, X_test], axis=1).T.reset_index(drop=True).T.to_pickle(os.path.join(save_path, 'test.pkl'))

def csv2pkl(source, save_path):
    data = pd.read_csv(source, header=None)
    data.to_pickle(save_path)

def train_valid_test_sequential(train_size, valid_size, test_size, save_path='./yearpred/'):
    df_train = pd.DataFrame()
    df_train['0'] = range(train_size)
    df_train.to_csv(save_path + 'train_idx.csv')

    df_valid = pd.DataFrame()
    df_valid['0'] = range(train_size, train_size+valid_size)
    df_valid.to_csv(save_path + 'valid_idx.csv')

    df_test = pd.DataFrame()
    df_test['0'] = range(train_size+valid_size, train_size+valid_size+test_size)
    df_test.to_csv(save_path + 'test_idx.csv')

def train_valid_test_random(train_size, valid_size, test_size, save_path='./displacement_amplifier/'):
    full_range = np.arange(0, train_size+valid_size+test_size)

    train_idx = np.random.choice(full_range, size=train_size, replace=False, p=None)
    df_train = pd.DataFrame(train_idx, columns=['0'])
    df_train.to_csv(save_path + 'train_idx.csv')

    valid_test_idx = np.setdiff1d(full_range, train_idx, assume_unique=False)
    valid_idx = np.random.choice(a=valid_test_idx, size=valid_size, replace=False, p=None)
    df_valid = pd.DataFrame(valid_idx, columns=['0'])
    df_valid.to_csv(save_path + 'valid_idx.csv')

    test_idx = np.setdiff1d(valid_test_idx, valid_idx, assume_unique=False)
    df_test = pd.DataFrame(test_idx, columns=['0'])
    df_test.to_csv(save_path + 'test_idx.csv')

if __name__ == '__main__':
    # source = '/data/dataset/MSLR-WEB10K/Fold1'
    # save_path = './data/MSLR-WEB10K'
    # svm2pkl(source, save_path)

    # Generate appropriate data for year prediction problem
    # Original dataset can be downloaded from https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
    # source = './yearpred/YearPredictionMSD.txt'
    # save_path = './yearpred/YearPrediction.pkl'
    # csv2pkl(source, save_path)
    # train_valid_test_sequential(train_size=400000, valid_size=63715, test_size=51630)

    train_valid_test_random(train_size=int(590*0.8), valid_size=int(590*0.1), test_size=int(590*0.1))
