import numpy as np # import numpy for matrix operations

### this function load data from .dat file
def load_dat(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        dim = len(lines[0].strip().split())
        num_samples = len(lines)
        data = np.zeros((num_samples, dim))
        for i in range(num_samples):
            data[i, :] = np.array([float(x) for x in lines[i].strip().split()])
        return data

### load data
# call the load_dat function to load X and Y from the corresponding input files
X =  load_dat("ex1x.dat")### Your code here ###
Y =  load_dat("ex1y.dat")### Your code here ###
# get some statistics of the data
num_samples = X.shape[0] # get the first dimension of X (i.e. number of rows)
dim = X.shape[1] # get the second dimension of X (i.e. number of columns)
print('X (%d x %d)' %(num_samples, dim))
print('Y (%d)' %(num_samples))

### add intercept term to all samples in X
X = np.resize(X, (num_samples, dim + 1)) # resize X to add a new dimension
X[:,dim]= 1.0 # set all value in the new dimension of X to 1
Y = Y.reshape([-1,1])
print('X (%d x %d)' %(num_samples, dim + 1))
print('Y (%d x 1)' %(num_samples))


### main functions of multivariate linear regression
def pseudo_inverse(A):
    A_tran = np.transpose(A)
    temp = np.matmul(A_tran,A)
    return np.matmul(np.linalg.inv(temp),np.transpose(A))




# The pseudo inverse:
# Input: a matrix A
# Output: the pseudo_inverse of A
### Your code here ###


def sse(prediction, reference):
    s = 0
    for i in range(len(prediction)):
        s += (prediction[i]-reference[i])* (prediction[i]-reference[i])
    return s

# Calculate the sum of square error between the prediction and reference vectors
### Your code here ###


### estimate beta
# call the pseudo_inverse to estimate beta from X and Y
beta =  np.matmul(pseudo_inverse(X),Y)### Your code here
# print the estimated (learned) parameters
print(beta)


### evaluate the model
# calculate the predicted scores
prediction =  np.matmul(X,beta)### Your code here
# calculate the sum of square error
error = sse(prediction, Y)
print('Sum of square error: %f' %error)