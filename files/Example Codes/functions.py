"""-------------------------------------------------------------------------
 File Name: TargetFunc.py
 Goal: Define the functions we want to use in our simulations
 Author: Francesca
 Author: Francesca Cannata
----------------------------------------------------------------------------"""
import numpy as np
import math
import torch.nn as nn
import torch


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


# 2D target function
def NN_func(x, seed=0, d=1, width=1):
    torch.manual_seed(seed)
    model = nn.Sequential()
    model.append(nn.Linear(d,width))
    model.append(nn.ReLU())
    model.append(nn.Linear(width, 1))

    # Parameter's initialization
    with torch.no_grad():
        nn.init.uniform_(model[0].bias, a=-1/2, b=1/2) # a = 0, b = 2
        nn.init.uniform_(model[0].weight, a=-1, b=1) # a = -2, b = -1
        nn.init.uniform_(model[2].bias, a=-1, b=1) # a = -1, b = 1
        nn.init.uniform_(model[2].weight, a=-1, b=1) # a = -1, b = 1

        # evaluate the Euclidean norm of the input weight
        input_weight_norm = model[0].weight.norm(dim=1, keepdim=True)
        print(f"input weight are {model[0].weight}")
        model[0].weight.copy_(model[0].weight / input_weight_norm)
        print(f'input normalized weight are {model[0].weight}')


    #with torch.no_grad():
        #p = torch.arange(1, width + 1) # Output weights: they have to sum up to 1 if we want the network in the variation space. Decay: 2^-p -> 1 for p -> \infty
        # 1st decay
        #decay = 1 / p**2

        # 2nd decay
        #decay = 2.0 ** (-p)

        # update the output weights
        #model[2].weight.copy_(decay.view(1, -1))

    model.eval()
    print(f'input weights: {model[0].weight}')
    print(f'input bias: {model[0].bias}')
    print(f'output weights: {model[2].weight}')
    print(f'output bias: {model[2].bias}')
    return model(torch.from_numpy(x).float()).detach().numpy()


# One dimensional hat function with boundary conditions
def hat_function(x, J, M):
    return np.maximum(0, 1 - np.abs(M * x - J))


# Two dimensional hat function
def twodim_hat_function(x, J, M):
    return np.prod(np.maximum(0, 1 - np.abs(M * x - J)), axis=1)