import sklearn
from sklearn.datasets import load_digits
import numpy as np
import time

# Step 1: Load the dataset
data = load_digits()
print('Classes:' , data.target_names.tolist())
X = data.data
y = data.target

# Step 2: Split into train and test sets
# Step 2: Split into train and test sets
num_samples = X.shape[0]
N_train = int(0.8 * X.shape[0])
X_tr =X[:N_train,:]### YOUR CODE HERE ###
y_tr = y[:N_train]### YOUR CODE HERE ###
X_te = X[N_train:,:]### YOUR CODE HERE ###
y_te = y[N_train:]### YOUR CODE HERE ###

# Step 3: Write function to calculate Euclidean distance
# between each row of two matrix
def euclidean(A, B):
    n, d = A.shape
    m, d1 = B.shape
    assert d == d1, 'Incompatible shape'
    distances = np.zeros((n,m))
    ### YOUR CODE HERE ###
    for i in range(n):
        for j in  range(m):
            distances[i,j]= np.sqrt(sum((A[i]-B[j])**2))
    return distances

def eval(predictions, targets):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            count +=1
    acc = count/len(predictions)### YOUR CODE HERE ###
    return acc

# Step 5: Perform kNN classification with explicit implementation of Euclidean distance
# Note: we use explicit implementation of kNN in this step
k = 5
st = time.time()
distances = euclidean(X_te, X_tr)
print('Finish function euclidean in %f seconds' %(time.time() - st))
n_test = X_te.shape[0]
n_classes = len(np.unique(y_tr))
st = time.time()
predictions = np.zeros(n_test, dtype=np.int8)
for i in range(n_test):
    # find indices of the closest neighbors
    nn_indices = distances[i].argsort()[:k]
    # increment the vote counts for the class of each closest neighbor
    votes = np.zeros(n_classes, dtype=np.int8)
    for j in range(k):
        votes[y_tr[nn_indices[j]]] +=1
        ### YOUR CODE HERE ###
    # take the class with highest vote as prediction for this sample
    predictions[i] = np.argmax(votes)
print('Finish function euclidean in %f seconds' %(time.time() - st))
print('Accuracy: %f %%' %(eval(predictions, y_te) * 100))


# Step 6: write function in vectorized form to calculate the Euclidean distance
def euclidean_vectorized(A, B):
    n, d = A.shape
    m, d1 = B.shape
    assert d == d1, 'Incompatible shape'
    distances = np.sqrt(((A-B)**2).sum())
    return distances

A = np.random.randn(3,5)
B = np.random.randn(4,5)
d1 = euclidean(A,B)
d2 = euclidean_vectorized(A,B)
assert np.allclose(d1,d2), 'Incorrect implementation of either of the two distance functions'