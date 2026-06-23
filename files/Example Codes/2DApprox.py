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
parser.add_argument('--stepsize', type=int, default=100, help='Step size for the scheduler (default: 100k).')
#parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor for the scheduler (default: 0.1).')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (default: 100).')
parser.add_argument('--units', type=int, default=20, help='Numbers of hidden neurons (default: 10).')
parser.add_argument('--size', type=int, default=2**7, help='Size of input data (default: 2**7).')
parser.add_argument('--bp', type=int, default=20, help='Number of breakpoints (default: 10).')

args = parser.parse_args() # Convert argument strings to objects and assign them as attributes of the namespace
#print(f'This is the network\'s setting. \n Seed = {args.seed} \n Number of hidden neurons = {args.units}')


"""------------------------------------------
 Goal: Create a function hard to approximate
       (with a "big" Barron norm)
---------------------------------------------"""
# In order to have the same random generation each time (both in numpy and pytorch)
np.random.seed(args.seed)

# Parameters
N = args.size #2**7 default

# Define the spatial domain and call the target function NN_func
x1 = np.linspace(-1,1,N).reshape(-1, 1)
x2 = np.linspace(-1,1,N).reshape(-1, 1)
X1, X2 = np.meshgrid(x1, x2)
X = np.array([X1.ravel(), X2.ravel()]).T        # dimension: N**2 x 2
y = NN_func(X, width=1, d=2) # change the width from 1 to 4
print('Target function ready')


# Target function visualization
dir_name = f'resultsW1exp_StepSize_{args.stepsize}_Epochs_{args.epochs}_HN_{args.units}_N_{args.size}_bp_{args.bp}_seed_{args.seed}'
os.makedirs(dir_name, exist_ok=True)

ax = plt.figure(1, figsize=(10, 8)).add_subplot(111, projection='3d')
#ax.contour(X1, X2, y.reshape((N,N)))
ax.plot_surface(X1, X2, y.reshape((N,N)))
ax.set_xlabel('$x_1$', fontsize=16, labelpad=10)
ax.set_ylabel('$x_2$', fontsize=16, labelpad=10)
ax.set_zlabel('$f_{targ}(x_1, x_2)$', fontsize=16, labelpad=10)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax.tick_params(axis='both', labelsize=12)
plt.grid(True, alpha=0.5)
ax.view_init(elev=15, azim=50)
plt.savefig(f'Target2DFunction_W2_exp.pdf', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()
#plt.show()


"""----------------------------------------
 Goal: Piecewise linear approximation
-------------------------------------------"""
# Number of breakpoints m << N
m = np.arange(1, args.bp)

# Initialization error lists
error_list = []
#error_th = []

# Iterate on the number of breakpoints
for k in m:
    phi = np.zeros((N**2, 2*(k+1), 2*(k+1)))    # Initialization matrix N x k with the values of the hat functions over x

    # Iterate the hat function on both dimensions the matrix phi
    for j1 in range(-k, k+1):
      for j2 in range(-k, k+1):
          phi[:, j1+k, j2+k] = twodim_hat_function(X, np.array([j1,j2]).reshape((1,2)), k) # j_i + k because negative indices are at the end of the array. We are doing a translation from [-k, k] to [0, 2k]
          if j1 == 0 and j2 == 0 and k==8:
            print(f'phi = {phi[:, j1, j2].reshape((N, N))}')

    phi_matrix = phi.reshape((N**2, (2*(k+1))**2))
    print(phi_matrix.shape)

    # Solve the least square error and find the y-values of the breakpoints. Target function approximation with piecewise -> matrix product
    ls_result = np.linalg.lstsq(phi_matrix, y)
    t = ls_result[0]
    y_pred_PW = np.dot(phi_matrix, t)
    error = np.sum((y - y_pred_PW)**2) / N**2 # we are dividing for N**2 in order to have the same scale of the error obtained with the neural net approx
    error_list.append(float(error))
    #error_th.append(float(1 / k ** 2.5))

    # Final plot for a fixed number of breakpoints
    if k == 8:
      ax = plt.figure(2, figsize=(10, 8)).add_subplot(111, projection='3d')
      # ax.contour(X1, X2, y.reshape((N,N)))
      ax.plot_wireframe(X1, X2, y.reshape((N, N)), color='C0')
      ax.plot_wireframe(X1, X2, y_pred_PW.reshape((N, N)), color='C1')
      #ax.plot_surface(X1, X2, phi[:,1,1].reshape((N, N)))
      #ax.plot_surface(X1, X2, phi[:,2,2].reshape((N, N)))
      #plt.title(f'Hat functions (intervals = {k})')
      #plt.title(f'Target Function VS Piecewise Approximation (interval = {k})')
      #plt.xlabel('Input data', fontsize=13)
      #plt.ylabel('Target function', fontsize=13)
      ax.set_xticks([-1, -0.5, 0, 0.5, 1])
      ax.set_yticks([-1, -0.5, 0, 0.5, 1])
      ax.set_zticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
      #plt.legend(loc='best')
      plt.grid(True)
      #plt.savefig(os.path.join(dir_name, f'TargetApproxPW.png'), dpi=300)
      plt.clf()
      plt.close()
      #plt.show()

# Theoretical piecewise error
# Fine del loop
log_k = np.log(m[2:])
log_e = np.log(error_list[2:])
coeffs = np.polyfit(log_k, log_e, 1)
slope = coeffs[0]
print(f'Estimated exponent: {slope:.4f}')

# Costruisci la curva teorica con l'esponente trovato
error_th = [1 / k ** -slope for k in m]

# Calibratioin
c_err = error_list[-1] / error_th[-1]
error_th = c_err * np.array(error_th)

pd.DataFrame({'k': m, 'error': error_list, 'error_th': error_th}).to_csv(f'PwErrors_W1_exp.csv', index=False)

print(error_th)
print(error_list)

# We want to know how the theoretical and the experimental error evolves in terms of the intervals (Piecewise Approx)
plt.figure(2, figsize=(10, 8))
plt.loglog(m, error_list, color='C1', label='Experimental error', linewidth=2.3)
plt.loglog(m, error_th, color='purple', label='Theoretical error $\propto k^{-1.19}$', linewidth=2.3)
plt.xlabel('Intervals $k$', fontsize=16)
plt.ylabel('Least-squared error', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(loc='best', fontsize=13)
plt.grid(True, which='major', linewidth=0.8)
plt.grid(True, which='minor', linewidth=0.3, linestyle=':')
#plt.savefig(f'LSEVsIntervals_pol.pdf', bbox_inches='tight', dpi=300)
#plt.savefig(os.path.join(dir_name, f'LSEVsIntervals_pol.pdf'), bbox_inches='tight', dpi=300)
plt.clf()
plt.close()
#plt.show()



"""----------------------------------------
 Goal: Neural Network approximation
-------------------------------------------"""
# Target function approximation with Neural network
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
total_neurons = []
min_loss_list = []
min_loss_th_list = []

# Iterate on the number of hidden neurons. arg.units is the maximum number of hidden neurons of the approximating network
for neurons in range(1, args.units+1):
    y_pred_NN, min_loss = approx(X_tensor, y_tensor, neurons, 0.01, args.stepsize, 0.3, args.epochs, args.seed)
    total_neurons.append(neurons)
    min_loss_list.append(float(min_loss))
    min_loss_th_list.append(float(1/neurons))

    # Neural Network approximation plot for a fixed number of hidden neurons
    if neurons == 1:
        ax = plt.figure(3, figsize=(10, 8)).add_subplot(111, projection='3d')
        ax.plot_wireframe(X1, X2, y.reshape((N, N)), color='C0')
        ax.plot_wireframe(X1, X2, y_pred_NN.reshape((N, N)), color='C2')
        #plt.title(f'Target Function VS Neural network approximation (Hidden Neurons = {neurons})')
        # plt.xlabel('Input data', fontsize=13)
        # plt.ylabel('Target function', fontsize=13)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        plt.legend(loc='best')
        plt.grid(True)
        #plt.savefig(os.path.join(dir_name, f'TargetApproxNN.png'), dpi=300)
        plt.clf()
        plt.close()
        #plt.show()

# Save the min_loss_list
pd.DataFrame(min_loss_list, columns=['min_loss']).to_csv(os.path.join(dir_name, f'min_loss_seed_{args.seed}.csv'), index=False)

# We want to know how the theoretical and the experimental error evolves in terms of the number of hidden neurons (NN Approx)
plt.figure(3, figsize=(10, 8))
plt.loglog(total_neurons, min_loss_list, color='limegreen', label='Experimental error', linewidth=2)
plt.loglog(total_neurons, min_loss_th_list, color='purple', label='Theoretical error', linewidth=2)
plt.xlabel('Hidden neurons', fontsize=16)
plt.ylabel('MSE loss', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(loc='best', fontsize=14)
plt.grid(True, which='major', linewidth=0.5)
plt.grid(True, which='minor', linewidth=0.3, linestyle=':')
#plt.title('Errors vs Hidden Neurons')
#plt.savefig(os.path.join(dir_name, f'ErrorVsNeurons.png'), dpi=300)
plt.clf()
plt.close()
#plt.show()


