# A simple linear regression program for approximating a linear equation using the normal equation


import numpy as np
import matplotlib.pyplot as plt

# Generate random data that follows the equation y = 2 + 7x
X_1 = np.random.rand(1000, 1)
Y_1 = 2 + 7*X_1+np.random.randn(1000, 1)


# add x0 = 1 to X vector to create y intercept
# use the normal equation [ theta = inv(X.T•X)•X.T•Y] to minimise error
def lin_reg(x,y):
    X_V = np.c_[np.ones((1000, 1)), x]
    args = np.linalg.inv(X_V.T.dot(X_V)).dot(X_V.T).dot(Y_1)
    return list(args.T)

# predictive function
def equate(args,xval):
    xval = np.matrix(xval)
    val_vec = np.tile(args,(np.shape(xval)[0],1))
    yvet = np.sum(np.array(xval)* np.array(val_vec),axis = 1)
    return yvet

# predict on value
xv = [1,0.8]
yv = equate(lin_reg(X_1,Y_1),xv)
plt.scatter(X_1,Y_1)
plt.xlabel("x_rand")
plt.ylabel("y_rand")
plt.title("Random Plot")
plt.plot(0.8, yv, marker="x", markersize=10, markeredgecolor="red")

X_V = np.c_[np.ones((1000, 1)), X_1]

ypredictions = equate(lin_reg(X_1,Y_1),X_V)
plt.plot(X_1,ypredictions)
plt.show()









