import os
import numpy as np

curpath = os.path.dirname(__file__)
x_train_int = np.load(curpath + '/x_train.npy')
x_train = x_train_int / 255.0
x_test_int = np.load(curpath + '/x_test.npy')
x_test = x_test_int / 255.0
y_train = np.load(curpath + '/y_train.npy')
y_test = np.load(curpath + '/y_test.npy')