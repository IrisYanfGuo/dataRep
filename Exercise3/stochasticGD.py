# Step 1: Load dataset, split into training and test sets, and scale features

import numpy as np
from sklearn.datasets import load_boston

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
x_train_scaled = (x_train - np.mean(x_train,axis=0))/np.std(x_train)     # YOUR CODE GOES HERE
x_test_scaled = (x_test - np.mean(x_test,axis=0))/np.std(x_test) # YOUR CODE GOES HERE

print(x_train_scaled.shape)
print(y_train.shape)
print(x_test_scaled.shape)
print(y_test.shape)


# Step 2: Add intercept terms and initialize parameters
# Note: If you run this step again, please run from step 1 because notebook keeps the value from the previous run
interc = np.ones((x_train_scaled.shape[0], 1))
x_train_scaled = np.hstack([interc, x_train_scaled])# YOUR CODE GOES HERE


interc = np.ones((x_test_scaled.shape[0], 1))
x_test_scaled = np.hstack((interc,x_test_scaled))# YOUR CODE GOES HERE

print(x_train_scaled.shape)
print(x_test_scaled.shape)

# init parameters using random values
mu,sigma = 0,0.5
theta = np.random.normal(mu,sigma,len(x_train_scaled[0]))# YOUR CODE GOES HERE
print(theta)


# Step 3: Implement the gradient and the cost function
# In this step, you have to calculate the gradient. You can use the provided formula but the best way is to vectorize
# that formula for efficiency
def compute_gradient(x,y,theta):
    return 1/len(x)*np.matmul(x.T,np.matmul(x,theta)-y)


def compute_cost(x,y,theta):
    return 1 / (2*len(x)) * np.matmul((np.matmul(x, theta) - y).T, np.matmul(x, theta) - y)

# Step 4: stochastic gradient descent
import matplotlib.pyplot as plt
import copy

# try different values for the learning rate
learning_rate = 0.01

# number of training iterations
num_samples = x_train_scaled.shape[0]
N_iterations = num_samples * 20 # loop over the training dataset 20 times

# prepare to plot
plt.subplot(111)

# calculate cost value and update theta
J = np.zeros(N_iterations)

# initialize new parameters using random distribution
theta_sgd = 0.5 * np.random.randn(x_train_scaled.shape[1])

for step in range(N_iterations):
    if step % num_samples == 0:
        index1 =np.arange(num_samples)
        np.random.shuffle(np.arange(num_samples))
        x_train_scaled = x_train_scaled[index1,:]
        y_train = y_train[index1]
        # shuffle the training data (must be done the same way for data and targets)
        # YOUR CODE GOES HERE

    # select the next sample to train
    index = step%len(x_train_scaled)
    x_step =  x_train_scaled[index,:]# YOUR CODE GOES HERE
    y_step =  y_train[index]# YOUR CODE GOES HERE
    x_step = x_step.reshape([1,-1])

    # calculate the cost on x_step and y_step
    J[step] =  compute_cost(x_step,y_step,theta_sgd)# YOUR CODE GOES HERE

    # update theta using a x_step and y_step
    theta_sgd =  theta_sgd - learning_rate*compute_gradient(x_step,y_step,theta_sgd)# YOUR CODE GOES HERE

# calculate the loss on the whole training set
J_train =  compute_cost(x_train_scaled,y_train,theta_sgd)# YOUR CODE GOES HERE
print('training cost: %f' %J_train)
# plot cost function
plt.plot(J)
plt.xlabel('Training step')
plt.ylabel('Cost')
plt.show()


# Step 5
# Predict the price of house
predict_price = 0 # YOUR CODE GOES HERE

# calculate the cost for the test set
test_cost = 0 # YOUR CODE GOES HERE
print('test cost: ',test_cost)

# plot the ground truth and the prediction
x_axis = np.linspace(1,len(y_test),len(y_test))
plt.plot(x_axis,y_test,'b',x_axis,predict_price,'r')
plt.legend(('Ground truth','Predicted'))
plt.show()