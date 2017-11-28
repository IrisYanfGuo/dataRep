# Step 1: Load dataset, split into training and test sets, and scale features
import numpy as np
from sklearn.datasets import load_boston

# load boston housing price dataset
boston = load_boston()
x = boston.data
y = boston.target


# scale features by removing mean and dividing by the standard deviation
x = (x - np.mean(x))/np.std(x)
print(x.shape)
interc = np.ones((x.shape[0], 1))
x = np.hstack((interc, x))
print(x.shape)
# split into training and test sets, namely 80 percent of examples goes for the training, 20 percent goes for the test set
N_train = int(0.8 * x.shape[0])
x_train_scaled = x[:N_train,:]
y_train = y[:N_train]
x_test_scaled = x[N_train:,:]
y_test = y[N_train:]


#print(x_test_scaled)

print(x_train_scaled.shape)
print(y_train.shape)
print(x_test_scaled.shape)
print(y_test.shape)



# init parameters using random values
mu,sigma = 0,0.5
theta = np.random.normal(mu,sigma,len(x_train_scaled[0]))# YOUR CODE GOES HERE
print(theta)



# Step 3: Implement the gradient and the cost function
# In this step, you have to calculate the gradient. You can use the provided formula but the best way is to vectorize
# that formula for efficiency
def compute_gradient(x,y,theta):
    return theta - 1/len(y)*np.matmul(x.T,(np.matmul(x,theta)-y))
    # YOUR CODE GOES HERE

def compute_cost(x,y,theta):
    return 1/(2*len(y))*np.matmul((np.matmul(x,theta)-y).T,(np.matmul(x,theta)-y))
    # YOUR CODE GOES HERE
