# -*- coding: utf-8 -*-

"""
LSTM for time series classification

Author: Zhou Ya'nan
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(datafile):
    """ Function to load csv data """
    all_data = np.loadtxt(datafile, delimiter=',')
    np.random.shuffle(all_data)
    all_data = all_data[0:, 1:]  # 去掉第一列PlotID
    segment = all_data[0:, 1:]
    label = all_data[0:, 0]

    segment = np.reshape(segment, [np.shape(segment)[0], -1, 2])
    # segment = np.reshape(segment, [np.shape(segment)[0], 2, -1])
    # segment = np.transpose(segment, (0, 2, 1))

    return segment, label


def sample_batch(x_train, y_train, batch_size):
    """ Function to sample a batch for batch training """
    shape0, _, _ = x_train.shape
    ind = np.random.choice(shape0, batch_size, replace=False)
    x_batch = x_train[ind]
    y_batch = y_train[ind]

    return x_batch, y_batch


def one_hot(labels):
    """ Function to one_hot encoding """
    relabels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

    return relabels


def argmax_label(labels):
    """ Function to retrieve the index for the max value"""
    index = np.argmax(labels, axis=1)
    index = np.reshape(index, -1, 1)

    return index


def standard_scaler(x):
    """ Function to """
    scaler = StandardScaler()
    ts_x = scaler.fit_transform(x)

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    ts_x = (x-mean)/std

    return ts_x
