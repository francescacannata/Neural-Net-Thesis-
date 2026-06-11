"""----------------------------------------------------------------------
 File Name: "2DApprox".py
 Goal: Approximation in 2 dimensions
 Author: Francesca Cannata
--------------------------------------------------------------------------"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import pandas as pd
import math
import copy
from functions import *
from nn_approx import *

# Read config from command line argument
parser = argparse.ArgumentParser(description='Training the network with different settings.')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0).')
#parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate (default: 0.01).')
#parser.add_argument('--stepsize', type=int, default=100000, help='Step size for the scheduler (default: 100k).')
#parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor for the scheduler (default: 0.1).')
#parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100).')
parser.add_argument('--units', type=int, default=10, help='Numbers of hidden neurons (default: 10).')
parser.add_argument('--size', type=int, default=2**7, help='Size of input data (default: 2**7).')
parser.add_argument('--bp', type=int, default=10, help='SNumber of breakpoints (default: 10).')

args = parser.parse_args() # Convert argument strings to objects and assign them as attributes of the namespace
#print(f'This is the network\'s setting. \n Seed = {args.seed} \n Number of hidden neurons = {args.units}')


"""------------------------------------------
 Goal: Create a function hard to approximate
       (with a "big" Barron norm)
---------------------------------------------"""
# In order to have the same random generation each time (both in numpy and pytorch)
np.random.seed(args.seed)
#torch.manual_seed(args.seed)

# Parameters
N = args.size #2**8
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

# Define the spatial domain and call the target function NN_func
x1 = np.linspace(-1,1,N).reshape(-1, 1)
x2 = np.linspace(-1,1,N).reshape(-1, 1)
X1, X2 = np.meshgrid(x1, x2)
X = np.array([X1.ravel(), X2.ravel()]).T        # N**2 x 2
y = NN_func(X, width=4, d=2)
print('target function ready')



# Target function visualization
dir_name = f'results_N_{args.size}_bp_{args.bp}'
os.makedirs(dir_name, exist_ok=True)

ax = plt.figure(1).add_subplot(111, projection='3d')
#ax.contour(X1, X2, y.reshape((N,N)))
ax.plot_surface(X1, X2, y.reshape((N,N)))
plt.title(f'Function with Barron norm {barron_norm}')
plt.xlabel('x')
plt.ylabel('Target function')
plt.grid(True)

plt.savefig(os.path.join(dir_name, f'Target Function.png'))
#plt.clf()
#plt.close()
plt.show()


"""----------------------------------------
 Goal: Piecewise linear interpolation
-------------------------------------------"""
# Number of intervals m << N
m = np.arange(1, args.bp)

# Initialization error list
error_list = []
error_th = []

# Iterate the hat function on all i to generate columns of the matrix phi
for k in m:
    # s = np.zeros(k+1)                         # Initialization x-values of the breakpoints for each m value
    phi = np.zeros((N**2, 2*(k+1), 2*(k+1)))  # Initialization matrix N x k with the values of the hat functions over x

    # Define the values of s for each m value and generate the matrix's columns
    for j1 in range(-k, k):
      for j2 in range(-k, k):
          phi[:, j1+k, j2+k] = twodim_hat_function(X, np.array([j1,j2]).reshape((1,2)), k) # j_i + k because negative indices are at the end of the array. We are doing a translation from [-k, k] to [0, 2k]
          if j1 == 0 and j2 == 0 and k==8:
            print(f'phi = {phi[:, j1, j2].reshape((N, N))}')

    phi_matrix = phi.reshape((N**2, (2*(k+1))**2))

    # Solve the least square error and find the y-values of the breakpoints
    print(phi_matrix.shape)
    ls_result = np.linalg.lstsq(phi_matrix, y)
    t = ls_result[0]
    #error = ls_result[1]
    #print(f'The error is {error}')
    #error_list.append(float(error[0]))
    #error_th.append(float(1/k**2.5))

    # Target function approximation with piecewise -> matrix product
    y_pred_PW = np.dot(phi_matrix, t)
    error = np.sum((y - y_pred_PW)**2)
    error_list.append(float(error))
    error_th.append(float(1 / k ** 2.5))

    # Final plot
    if k == 8:
      ax = plt.figure(2).add_subplot(111, projection='3d')
      # ax.contour(X1, X2, y.reshape((N,N)))
      ax.plot_wireframe(X1, X2, y.reshape((N, N)), color='orange')
      ax.plot_wireframe(X1, X2, y_pred_PW.reshape((N, N)), color='green')
      #ax.plot_surface(X1, X2, phi[:,1,1].reshape((N, N)))
      #ax.plot_surface(X1, X2, phi[:,2,2].reshape((N, N)))
      #plt.title(f'Hat functions (intervals = {k})')
      plt.title(f'Target Function VS Piecewise Approximation (interval = {k})')
      plt.xlabel('Input data')
      plt.ylabel('Target function')
      plt.legend(loc='best')
      plt.grid(True)
      plt.savefig(os.path.join(dir_name, f'TargetApproxPW.png'))
      #plt.clf()
      #plt.close()
      plt.show()

# Theoretical piecewise error
c_err = error_list[-1] / error_th[-1]
error_th = c_err * np.array(error_th)
print(error_th)

# We want to know how the theoretical and the experimental error evolves in terms of the intervals (Piecewise Approx)
plt.figure(2)
plt.figure(figsize=(8, 5))
plt.loglog(m, error_list, color='blue', label='Experimental Error')
plt.loglog(m, error_th, color='red', label='Theoretical Error')
plt.title('Errors vs Intervals')
plt.xlabel('Intervals')
plt.ylabel('Error')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(dir_name, f'ErrorVsIntervals.png'))
#plt.clf()
#plt.close()
plt.show()


# Target function approximation with Neural network
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
total_neurons = []
min_loss_list = []
min_loss_th_list = []


for neurons in range(1, args.units+1):
    y_pred_NN, min_loss = approx(X_tensor, y_tensor, neurons, 0.01, 1000, 0.3, 5000)
    total_neurons.append(neurons)
    min_loss_list.append(float(min_loss))
    min_loss_th_list.append(float(1/neurons))

    # Neural Network approximation plot
    if neurons == 10:
        ax = plt.figure(3).add_subplot(111, projection='3d')
        ax.plot_wireframe(X1, X2, y.reshape((N, N)), color='orange')
        ax.plot_wireframe(X1, X2, y_pred_NN.reshape((N, N)), color='green')
        plt.title(f'Target Function VS Neural network approximation (Hidden Neurons = {neurons})')
        plt.xlabel('Input data')
        plt.ylabel('Target function')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(dir_name, f'TargetApproxNN.png'))
        #plt.clf()
        #plt.close()
        plt.show()

# We want to know how the theoretical and the experimental error evolves in terms of the number of hidden neurons (NN Approx)
plt.figure(3)
plt.figure(figsize=(8, 5))
plt.loglog(total_neurons, min_loss_list, color='blue', label='Experimental NN Error')
plt.loglog(total_neurons, min_loss_th_list, color='red', label='Theoretical NN Error')
plt.title('Errors vs Hidden Neurons')
plt.xlabel('Hidden Neurons')
plt.ylabel('Error')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(dir_name, f'ErrorVsNeurons.png'))
#plt.clf()
#plt.close()
plt.show()



