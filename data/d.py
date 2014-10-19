import numpy as np
import os

x_train = np.load(os.path.join(os.path.dirname(__file__), 'x_train.npy'))
x_test = np.load(os.path.join(os.path.dirname(__file__), 'x_test.npy'))
y_train = np.load(os.path.join(os.path.dirname(__file__), 'y_train.npy'))
y_test = np.load(os.path.join(os.path.dirname(__file__), 'y_test.npy'))