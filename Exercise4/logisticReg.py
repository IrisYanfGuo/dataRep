######## Exercise 4 - Part 1 - Logistic Regression ########
# In this exercise, we are going to classify the breast cancer wisconsin dataset.
# This dataset contains in total 569 examples, among them 212 examples are labelled as malignant (M or 0) and 357
# examples are marked as benign (B or 1). .Each example is a vector of 30 dimensions.
# We will train a binary logistic regression model using this dataset.

# Load, normalize, split and visualize your dataset. This step has been done for you.
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

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

print("Number of training examples: ",x_train.shape[0])
print("Number of testing examples: ",x_test.shape[0])

# visualize the dataset using histogram
labels = ['Benign','Malignant']
population = [np.sum(y),np.sum(y==0)]
y_pos = np.arange(len(labels))
barlist = plt.bar(y_pos, population, align='center',width=0.3)
plt.xticks(y_pos, labels)
plt.ylabel('Number of examples')
plt.title('Breast wisconsin dataset.')
barlist[1].set_color('r')
plt.show()

# Add intercept terms and initialize parameters
# This step has been done for you.
intercept_train = np.ones((N_train,1))
x_train_scaled = np.hstack((intercept_train,x_train_scaled))

intercept_test = np.ones((x.shape[0] - N_train,1))
x_test_scaled = np.hstack((intercept_test,x_test_scaled))

print(x_train_scaled.shape)
print(x_test_scaled.shape)

# Step 1: Implement the sigmoid, gradient and cost functions

# this function returns the probability of y=1
# x: data matrix
# theta: model's parameters
def sigmoid(x,theta):
    return 1/(1+np.exp(np.matmul(x,theta.T)))


# logarithmic loss
# x: data matrix (2D)
# y: label (1D)
# theta: model's parameters (1D)
def compute_cost(x,y,theta):
    result =1
    for i in range(len(x)):
        result = result*sigmoid(x[i],theta)**y[i]*[1-sigmoid(x[i],theta)]**(1-y[i])
    return -np.log(result)/len(x)


def compute_gradient(x,y,theta):
    return 1/len(x) * np.matmul(x.T,(sigmoid(x,theta)-y))

def approximate_gradient(x,y,theta,epsilon):
    n_features = x.shape[1]
    app_grad = np.zeros(n_features)
    for i in range(n_features):
        epsilon_one_hot = np.zeros(n_features)
        epsilon_one_hot[i] = epsilon
        theta_before = theta - epsilon_one_hot
        theta_after = theta + epsilon_one_hot
        app_grad[i] = (compute_cost(x,y,theta_after) - compute_cost(x,y,theta_before))/(2*epsilon)
    return app_grad

theta = 0.5 * np.random.randn(x_train_scaled.shape[1])
grad = compute_gradient(x_train_scaled,y_train,theta)
epsilon = 1e-4
app_grad = approximate_gradient(x_train_scaled,y_train,theta,epsilon)
print('Sum of gradient squared error: ',np.sum((grad - app_grad)**2))


