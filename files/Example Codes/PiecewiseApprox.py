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
from functions import *

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


# Compute the norms
barron_norm = np.sum(w * np.abs(c)) # Barron norm (p = 1)
L2_norm = np.sum(np.square(c))      # L2 norm

# Normalize the L2 norm -> \sum_{i = 1}^{n} |c_i| = 1 => \alpha_{norm} = 1 / np.sqrt(np.sum(np.square(1.0 / (w ** beta))))
alpha_norm = 1 / np.sqrt(np.sum(np.square(1.0 / (w ** beta))))
c_normalized = alpha_norm / w**beta
L2_norm_normalized = np.sum(np.square(c_normalized))

# Print the norms
print(f'The barron norm is {barron_norm}. \n The normalized L2 norm is {L2_norm_normalized}')

# Define the spatial domain and call the barron_func
x = np.linspace(0,1,N).reshape(-1, 1)
y = barron_func(c, omega, x, phi)
#y = piecewise_func(x) #y = NN_func(x, width=4)


# Barron function visualization
plt.figure(1, figsize=(10, 8))
plt.plot(x, y, color='C0', linewidth=3)
#plt.title(f'Function with Barron norm {barron_norm}')
plt.xlabel('x', fontsize=13)
plt.ylabel('Target function $f_{targ}(x)$', fontsize=13)
plt.grid(True)
plt.show()
# plt.close()



"""----------------------------------------
 Goal: Piecewise linear interpolation
-------------------------------------------"""
# Number of intervals m << N
m = np.arange(1, 100)

# Because x is a vector N x 1
x_flat = np.array(x).flatten()

# Initialization error list
error_list = []
error_th = []

# Iterate the hat function on all i to generate columns of the matrix phi
for k in m:
  s = np.zeros(k+1)                   # Initialization x-values of the breakpoints for each m value
  phi = np.zeros((len(x_flat), k+1))  # Initialization matrix N x k with the values of the hat functions over x

  # Define the values of s for each m value and generate the matrix's columns
  for j in range(k+1):
      s[j] = j/k
      phi[:, j] = hat_function(x_flat, j, k)

  phi_matrix = phi

  # Solve the least square error and find the y-values of the breakpoints
  ls_result = np.linalg.lstsq(phi_matrix, y)
  t = ls_result[0]
  error = np.sqrt(ls_result[1])
  error_list.append(float(error[0]))
  error_th.append(float(1/k**(1.4174))) # 1/k**2

  # Target function approximation -> matrix product
  y_pred = np.dot(phi_matrix, t)

  # # Save s and t
  # dir_name = f'Breakpoints_k_{k}'
  # os.makedirs(dir_name, exist_ok=True)
  # np.save(os.path.join(dir_name, 's.npy'), s)
  # np.save(os.path.join(dir_name, 't.npy'), t)


  # Final plot
  if k == 99:
      plt.figure(2, figsize=(10, 8))
      plt.plot(x, y, label='Target function', color='C0', linewidth=3)
      plt.plot(x, y_pred, linestyle='--', label='Piecewise linear approximation', color='C1', linewidth=2.5)
      #plt.plot(x, phi_matrix[:,-1], label='Hat function', color='blue')
      #plt.title(f'Target Function VS Piecewise Approximation (intervals = {k})')
      plt.xlabel('x', fontsize=13)
      #plt.ylabel('Target function and approximation')
      plt.legend(loc='best', fontsize=12)
      plt.grid(True, alpha=0.5)
      plt.show()


c_err = error_list[-1] / error_th[-1]
error_th = c_err * np.array(error_th)
print(error_th)

# Fit lineare su log-log (escludi i primi valori dove il transitorio è grande)
log_k = np.log(m[10:])        # escludi i primi 10 punti
log_e = np.log(error_list[10:])

coeffs = np.polyfit(log_k, log_e, 1)
slope = coeffs[0]
print(f'Estimated exponent: {slope:.4f}')  # dovrebbe essere circa -1.25

# We want to know how the theoretical and the experimental error evolves in terms of the intervals

plt.figure(3, figsize=(10, 8))
plt.loglog(m, error_list, color='C1', label='Experimental error', linewidth=2.3)
plt.loglog(m, error_th, color='purple', label='Theoretical error $\propto k^{-1.41}$', linewidth=2.3)
#plt.title('Errors vs Intervals')
plt.xlabel('Intervals $k$', fontsize=13)
plt.ylabel('Least-squared error', fontsize=13)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()

