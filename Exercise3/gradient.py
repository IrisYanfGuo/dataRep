# Load dataset, split into training and test sets, scale features and add intercept terms
# This step has been don for you because we
import numpy as np
from sklearn.datasets import load_boston
from matplotlib.legend_handler import HandlerTuple

# load boston housing price dataset
boston = load_boston()
x = boston.data
y = boston.target

# split into training and test sets, namely 80 percent of examples goes for the training, 20 percent goes for the test set
N_train = int(0.8 * x.shape[0])
x_train = x[:N_train,:]
y_train = y[:N_train]
x_test = x[N_train:,:]
y_test = y[N_train:]

# scale features by removing mean and dividing by the standard deviation
x_bar = np.mean(x_train,axis=0)
x_std = np.std(x_train,axis=0)
x_train_scaled = (x_train - x_bar)/x_std
x_test_scaled = (x_test - x_bar)/x_std

print('Number of training samples: ',x_train_scaled.shape[0])
print('Number of testing samples: ',x_test_scaled.shape[0])

# add intercept term
intercept_train = np.ones((N_train,1))
x_train_scaled = np.hstack((intercept_train,x_train_scaled))
intercept_test = np.ones((x.shape[0] - N_train,1))
x_test_scaled = np.hstack((intercept_test,x_test_scaled))


print('Training set shape: ',x_train_scaled.shape)
print('Testing set shape: ',x_test_scaled.shape)