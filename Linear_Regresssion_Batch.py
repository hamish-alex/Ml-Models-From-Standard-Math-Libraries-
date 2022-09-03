import numpy as np
import matplotlib.pyplot as plt
# simple linear regression using batch gradient decent

# Generate random data that follows the equation y = 2 + 7x +1y
X_1 = np.random.rand(1000, 1)
Y_1 = np.random.randn(1000, 1)
Z_1 = 2 +7*X_1+np.random.randn(1000, 1)+ 1*Y_1+np.random.randn(1000, 1)
# x with y intercept scalar
XY_V = np.c_[np.ones((1000, 1)), X_1,Y_1]


# learning rate specifies how much the approximator changes based on the steepness of the error
lr = 0.2
n_iter = 1000
m = len(X_1)

# initialise random approximator with 3 components
appr = np.random.randn(3,1)

#iterate batch gradient decent
for iter in range(n_iter):
    grad = 2/m * XY_V.T.dot(XY_V.dot(appr)-Z_1)
    appr = appr - lr*grad

# XY_V.dot(appr)-Z_1 calculates the error vector using the current appr
# 2/m * XY_V.T.dot() averages the error relative to each instance to compute the average steepness for each feature
# updates the approximator away from the error gradient scaled by the learning rate

print(appr)