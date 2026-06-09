"""-------------------------------------------------------------------------
 File Name: TargetFunc.py
 Goal: Define the functions we want to use in our simulations
 Author: Francesca
 Author: Francesca Cannata
----------------------------------------------------------------------------"""
import numpy as np
import math

# Barron function
def barron_func(c, omega, x, phi):
    y = np.sum(c * np.sin(2 * math.pi * omega * x + phi), axis=1)
    return y


# Piecewise function
def piecewise_func(x, seed=0):
    np.random.seed(seed)
    x_points = np.array([0, np.random.rand(), 1])
    y_points = np.random.randn(len(x_points))
    pw_func = np.interp(x, x_points, y_points)  # It computes the 1D piecewise linear interpolant to a function with given discrete data points (x_points, y_points), evaluated at x
    return pw_func


# Define the hat functions considering the boundary conditions
def hat_function(x, J, M):
    return np.maximum(0, 1 - np.abs(M * x - J))
    # we need to define the boundary conditions
    #if J == 0:
        # if x <= 1/M:              # if x[i] is in the left side (i.e. between 0 and s_1 = 1/m), we want just a downward segment \
        #     return 1 - M*x
        # elif x >= 1 - 1/M:        # if x[i] is in the right side (i.e. between s_{m-1} and s_m = 1), we want just a upwards segment /
        #     return 1 + M*(x-1)
        # else:                        # if x[i] is between s_1 = 1/m and s_{m-1} then the hat function is zero
        #     return 0
        #return np.maximum(0,np.maximum(1 - M*x,1 + M*(x-1)))
    # elif 0 < J < M - 1:      # inner points (s_1, ..., s_{m-1})
    #     if S[J-1] < x <= S[J+1]:
    #         return 1 - np.abs(M*x - J)
    #     else:
    #         return 0
    # elif J == M - 1:  # last point
    #     if S[j - 1] < x <= 1:       # the end point of the domain is 1
    #         return 1 - np.abs(M * x - J)
    #     else:
    #         return 0
    #else:
        #return np.maximum(0, 1 - np.abs(M * x - J))