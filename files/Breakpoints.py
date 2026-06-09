"""----------------------------------------------------------------------
 File Name: Breakpoints.py
 Goal: Function approximation through a SNN with fixed neurons positions
 Author: Francesca Cannata
--------------------------------------------------------------------------"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import copy

# Read config from command line argument
parser = argparse.ArgumentParser(description='Training the network with different settings.')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0).')
#parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate (default: 0.01).')
#parser.add_argument('--stepsize', type=int, default=100000, help='Step size for the scheduler (default: 100k).')
#parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor for the scheduler (default: 0.1).')
#parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100).')
parser.add_argument('--units', type=int, default=10, help='Numbers of hidden neurons (default: 10).')

args = parser.parse_args() # Convert argument strings to objects and assign them as attributes of the namespace
print(f'This is the network\'s setting. \n Seed = {args.seed} \n Number of hidden neurons = {args.units}')


"""------------------------------------------
 Goal: Create a function hard to approximate
       (with a "big" Barron norm)
---------------------------------------------"""
# In order to have the same random generation each time (both in numpy and pytorch)
np.random.seed(args.seed)
#torch.manual_seed(args.seed)

# Parameters
N = 2**10
n = int(N/2)
alpha = 1
beta = 2.1                                              # beta > 2 for having a finite Barron norm
omega = np.arange(1, n+1)                   # low and high frequencies
phi = np.random.uniform(0, 2*math.pi, size=n)      # phase shift
w = 1+np.abs(omega)                                     # weight function
c = alpha / w**beta                                     # ansatz for the amplitude


# Evaluate the Barron norm (p = 1)
barron_norm = np.sum(w * np.abs(c))

# Evaluate the L2 norm
L2_norm = np.sum(np.square(c))

# Normalize the L2 norm -> \sum_{i = 1}^{n} |c_i| = 1 => \alpha_{norm} = 1 / np.sqrt(np.sum(np.square(1.0 / (w ** beta))))
alpha_norm = 1 / np.sqrt(np.sum(np.square(1.0 / (w ** beta))))
c_normalized = alpha_norm / w**beta
L2_norm_normalized = np.sum(np.square(c_normalized))

# Print the norms
print(f'The barron norm is {barron_norm}. \n The normalized L2 norm is {L2_norm_normalized}')


# Spatial domain
x = np.linspace(0,1,N).reshape(-1, 1)


# Numpy Array -> torch tensor
#omega_tensor =  torch.from_numpy(omega).view(1, -1)
#phi_tensor = torch.from_numpy(phi).float().view(1, -1)
#c_tensor = torch.from_numpy(c_normalized).float().view(1, -1)

# Create the target function that we want to approximate
y = np.sum(c * np.sin(2*math.pi*omega*x + phi), axis=1)


# Visualization
fig = plt.figure(1)
plt.plot(x, y, color='red', linewidth=2)
plt.title(f'Function with Barron norm {barron_norm}')
plt.xlabel('x')
plt.ylabel('Target function')
plt.grid(True)
plt.close()



"""----------------------------------------
 Goal: Piecewise linear interpolation
-------------------------------------------"""
# Number of intervals m << N
m = 100

# Initialization x-values of the breakpoints. We want to have m points between 0 (= s_0) and 1 (= s_m)
s = np.zeros(m)

# Define the values of s
for j in range(m):
    s[j] = j/m

print(f'The x coordinates of our breakpoints are {s}')

# Define the hat functions considering the boundary conditions
def hat_function(x, I, J, M, S):
    # we need to define the boundary conditions
    if J == 0:
        # if x <= 1/M:              # if x[i] is in the left side (i.e. between 0 and s_1 = 1/m), we want just a downward segment \
        #     return 1 - M*x
        # elif x >= 1 - 1/M:        # if x[i] is in the right side (i.e. between s_{m-1} and s_m = 1), we want just a upwards segment /
        #     return 1 + M*(x-1)
        # else:                        # if x[i] is between s_1 = 1/m and s_{m-1} then the hat function is zero
        #     return 0
        return np.maximum(0,np.maximum(1 - M*x,1 + M*(x-1)))
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
    else:
        return np.maximum(0, 1 - np.abs(M * x - J))




# Initialization matrix N x m with the values of the hat functions over x
x_flat = np.array(x).flatten()      # because x is a vector N x 1
phi = np.zeros((len(x_flat), m))

# Iterate the hat function on all i to generate columns of the matrix phi
#for i in range(len(x_flat)):
for j in range(m):
    phi[:, j] = hat_function(x_flat, 0, j, m, s)

phi_matrix = phi

# Solve the least square error and find the y-values of the breakpoints
ls_result = np.linalg.lstsq(phi_matrix, y)
t = ls_result[0]
error = ls_result[1]

# Target function approximation -> matrix product
y_pred = np.dot(phi_matrix, t)

# Final plot
plt.figure(2)
plt.plot(x, y, label='Target function', color='green', linewidth=2)
plt.plot(x, y_pred, label='Network Approximation', color='red')
#plt.plot(x, phi_matrix[:,0], label='Hat function', color='blue')
plt.title('Target Barron Function VS Network Approximation')
plt.xlabel('Input data')
plt.ylabel('Target function')
plt.legend(loc='best')
plt.grid(True)
plt.show()



# We want to know how the error evolves in terms of the intervals
# interval_num = np.linspace(1, 100, 100, dtype=int)
# error_mult = np.zeros(len(interval_num))
#
# for interval in interval_num:
#
#     # --- IL TUO CODICE PER COSTRUIRE LA MATRICE ---
#     s_mult = np.zeros(interval)
#     for j in range(interval):
#         s_mult[j] = j / interval
#
#     phi_mult = np.zeros((len(x_flat), interval))
#     for i in range(len(x_flat)):
#         for j in range(interval):
#             phi_mult[i, j] = hat_function(x_flat[i], i, j, interval, s_mult)
#
#     # --- CALCOLO DEI PESI ---
#     error_mult[interval] = np.linalg.lstsq(phi_mult, y)[1]
#
#
# # 3. PLOT FINALE DELL'ERRORE
# plt.figure(figsize=(8, 5))
# plt.plot(interval_num, error_mult, marker='o', linestyle='-', color='blue')
# plt.title("Andamento dell'errore di approssimazione")
# plt.xlabel("Numero di intervalli (m)")
# plt.ylabel("Mean Squared Error (MSE)")
# plt.grid(True)
# plt.show()



