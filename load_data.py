# -*- coding: utf-8 -*-

import csv
import numpy as np
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split

def load_data(Train=False):
    data = []

    # Read the training data
    f = open('data/spambase.data')
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        data.append(row)
    f.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    if Train:
        # returns X_train, X_test, y_train, y_test
        return train_test_split(X, y, test_size=0.3, random_state=RandomState())
    else:
        return X, y
