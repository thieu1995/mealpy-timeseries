#!/usr/bin/env python
# Created by "Thieu" at 04:16, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# univariate mlp example
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.datasets.co2 as co2


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    ## https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def generate_data():
    ## Make dataset
    dataset = pd.DataFrame(co2.load().data)
    dataset = dataset.fillna(dataset.interpolate())
    scaler = MinMaxScaler()
    scaled_seq = scaler.fit_transform(dataset.values).flatten()

    # choose a number of time steps
    n_steps = 3
    # split into samples            60% - training
    x_train_point = int(len(scaled_seq) * 0.75)
    X_train, y_train = split_sequence(scaled_seq[:x_train_point], n_steps)
    X_test, y_test = split_sequence(scaled_seq[x_train_point:], n_steps)

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "n_steps": n_steps}
