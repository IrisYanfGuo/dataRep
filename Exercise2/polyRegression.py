from sklearn import datasets
import numpy as np

X, y = datasets.load_boston(return_X_y=True)
print(X.shape)
print(y.shape)

# create virtual features, including
#   second degree of the first variable
#   second degrees of the eighth variable
#   third and second degrees of the eleventh variable

### Your code here ###
a1 = X[:,0].reshape(-1,1)**2
a2 = X[:,7].reshape(-1,1)**2
a3 = X[:,10].reshape(-1,1)**3
a4 = X[:,10].reshape(-1,1)**2



# concatenate the virtual feature to the original features
### Your code here ###
X = np.hstack((X,a1,a2,a3,a4))
# add a dimension with all 1 to account for the intercept term
interc = np.ones((X.shape[0], 1))
X = np.hstack((interc, X))
print(X.shape)


train_ratio = 0.8
cutoff = int(X.shape[0] * train_ratio)
X_tr = X[:cutoff, :]
y_tr = y[:cutoff]
X_te = X[cutoff:,:]
y_te = y[cutoff:]
print('Train/Test: %d/%d' %(X_tr.shape[0], X_te.shape[0]))

def pseudo_inverse(A):
    # Calculate the pseudo_inverse of A
    pinv = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
    return pinv

lamb = 0.2
beta = np.matmul(pseudo_inverse(X_tr),y_tr)
print(beta)
print(len(beta))


def MSE(prediction,reference):
    s = 0
    for i in range(len(prediction)):
        s = s+(prediction[i]-reference[i])**2
    mse = s/len(prediction)### Your code here ###
    return mse

def MAE(prediction, reference):
    # Calculate the mean absolute error between the prediction and reference vectors
    s = 0
    for i in range(len(prediction)):
        s = s + abs(prediction[i] - reference[i])
    mae = s/len(prediction) ### Your code here ###
    return mae

pred = np.matmul(X_te,beta)### Your code here ###
print(pred)

mse = MSE(pred, y_te)
mae = MAE(pred, y_te)
print(mse)
print(mae)


def regularized_pseudo_inverse(A, theta):
    # Calculate the regularized pseudo_inverse of A
    pinv = np.matmul(np.linalg.inv(np.matmul(A.T,A)+theta*np.eye(A.shape[1])),A.T)
    return pinv

theta = 0.5
beta_regularized = np.matmul(regularized_pseudo_inverse(X_tr,theta),y_tr)
pred_2 =np.matmul(X_te,beta_regularized) ### Your code here ###
mse = MSE(pred_2, y_te)
mae = MAE(pred_2, y_te)
print(mse)
print(mae)

