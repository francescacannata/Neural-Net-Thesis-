import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Read config from command line argument
parser = argparse.ArgumentParser(description='Training the network with different settings.')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0).')
parser.add_argument('--stepsize', type=int, default=1000, help='Step size for the scheduler (default: 100k).')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs (default: 100).')
parser.add_argument('--units', type=int, default=10, help='Numbers of hidden neurons (default: 10).')
parser.add_argument('--size', type=int, default=2**7, help='Size of input data (default: 2**7).')
parser.add_argument('--bp', type=int, default=10, help='Number of breakpoints (default: 10).')
args = parser.parse_args()

# initialization of the matrix with the min_loss for each seed
loss_array_seeds = []

# folder's path
path = '/Users/francesca/Desktop/2DresultPlots'
dir_name = f'MinLoss_StepSize_{args.stepsize}_Epochs_{args.epochs}_HN_{args.units}_N_{args.size}_bp_{args.bp}'
os.makedirs(dir_name, exist_ok=True)

# Load the min_loss_seed_{args.seed} file for each seed
for seed in args.seeds:
    dir_name_dyn = f'results_StepSize_{args.stepsize}_Epochs_{args.epochs}_HN_{args.units}_N_{args.size}_bp_{args.bp}_seed_{seed}'
    df = pd.read_csv(os.path.join(path, dir_name_dyn, f'min_loss_seed_{seed}.csv'))
    loss_array_seeds.append(df['min_loss'].values)

# Loss matrix (row  = neurons, columns = seed)
loss_matrix = np.array(loss_array_seeds).T

# Take the minimum of each row
min_loss_seed = np.min(loss_matrix, axis=1)

# Print the results
print(loss_matrix.shape)
print(min_loss_seed)

# Final plot
total_neurons = np.arange(1, args.units + 1)
min_loss_th_list = 1 / total_neurons

plt.figure(3)
plt.figure(figsize=(10, 8))
plt.loglog(total_neurons, min_loss_seed, color='C2', label='Experimental NN Error', linewidth=2)
plt.loglog(total_neurons, min_loss_th_list, color='purple', label='Theoretical NN Error', linewidth=2)
plt.xlabel('Hidden Neurons', fontsize=13)
plt.ylabel('Error', fontsize=13)
plt.tick_params(axis='both', labelsize=10)
plt.legend(loc='best', fontsize=13)
plt.grid(True, which='major', linewidth=0.8)
plt.grid(True, which='minor', linewidth=0.3, linestyle=':')
#plt.title('Errors vs Hidden Neurons')
plt.savefig(os.path.join(dir_name, f'ErrorSeedVsNeurons.png'), dpi=300)
#plt.clf()
#plt.close()
plt.show()