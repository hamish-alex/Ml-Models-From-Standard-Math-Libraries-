import numpy as np

# Stochastic gradient decent is dissimilar from batch gradient decent because it selects training data randomly
# and has a variable learning rate that changes but tends to decrease over time

import matplotlib.pyplot as plt
# simple linear regression using stochastic gradient decent

# Generate random data that follows the equation y = 2 + 7x +1y
X_1 = np.random.rand(1000, 1)
Y_1 = np.random.randn(1000, 1)
Z_1 = 2 +7*X_1+np.random.randn(1000, 1)+ 1*Y_1+np.random.randn(1000, 1)
# x with y intercept scalar
XY_V = np.c_[np.ones((1000, 1)), X_1,Y_1]


# learning rate specifies how much the approximator changes
lr0,lr1 = 10,100
n_epochs = 100
m = len(X_1)

# define a function for variable learning rate
def lrcompute(lr):
    return lr0/(lr1+lr)

# initialise random approximator with 3 components
appr = np.random.randn(3,1)

#iterate stochastic gradient decent

for epoch in range(n_epochs):
    for i in range(m):
        n_rand = np.random.randint(m)
        xi = XY_V[n_rand:n_rand+1]
        yi = Y_1[n_rand:n_rand+1]
        grad = 2/m * XY_V.T.dot(XY_V.dot(appr)-Z_1)
        damper = lrcompute(epoch*m*i)
        appr = appr - damper*grad


# picks a random iteration of the data and update the equation
# XY_V.dot(appr)-Z_1 calculates the error vector using the current appr
# 2/m * XY_V.T.dot() averages the error relative to each instance to compute the average steepness for each feature
# updates the approximator away from the error gradient scaled by the variable learning rate
# the learning rate changes each iteration but tends to decrease

print(appr)

# using sklearn
from sklearn.linear_model import SGDRegressor
# if the error drop by less than 0.001, the code stops running, else 100 iterations
# initial learning rate is 0.1
sgd_reg = SGDRegressor(max_iter = 10000, tol = 0.001, penalty=None,eta0 = 0.1)
sgd_reg.fit(XY_V,Z_1.ravel())

print(sgd_reg.intercept_)
print(sgd_reg.coef_)
