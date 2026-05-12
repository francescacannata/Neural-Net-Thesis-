"""---------------------------
 File Name: NN2HiddenLayers.py
 Purpose: Implement a SNN to approximate f(x) = cos(x)
 Author: Francesca Cannata
-----------------------------""""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# 1. Input Data and Target Function
x = np.linspace(-1, 1, 100).reshape(-1, 1) # we're creating 1000 points between -1 and 1, organized in one column
y = x^2 # target function

# Convert Numpy arrays to PyTorch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()



# 2. Network Architecture and Model Definition
# Define the number of neurons for each layer
n_in = 1
n_Hid1 = 2
n_Hid2 = 3
n_out = 1

# Define a class which contains our training model.
class NeuralNet(nn.Module): # our class "NeuralNet" inherits all methods and properties from the superclass nn.Module in PyTorch
    def __init__(self):
        super().__init__() # "super()" allows us to have access to methods from the superclass nn.Module

        # Architecture of our network: two hidden layers with 2 and 3 neurons respectively
        self.InHid1 = nn.Linear(n_in, n_Hid1)   # input -> 1st Hidden: each input is multiplied by a weight and summed with a bias. They will become the 1st-hidden neurons
        self.Hid1Hid2 = nn.Linear(n_Hid1, n_Hid2) # 1st Hidden -> 2nd Hidden: each 1st-hidden output is multiplied by a weight and summed with a bias. They will become the 2nd-hidden neurons
        self.Hid2Out = nn.Linear(n_Hid2, n_out)  # 2nd Hidden -> output: each 2nd-hidden output is multiplied by a weight and summed with a bias. They will become the final output of the net
        self.activation = nn.ReLU() # examples of other non-linear activation functions: nn.Tanh(), nn.Sigmoid(), nn.ELU() etc.

    def forward(self, x):
        x = self.activation(self.InHid1(x))
        x = self.activation(self.Hid1Hid2(x))
        x = self.Hid2Out(x)
        return x

model = NeuralNet()


# 3. Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 4. Training Loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if epoch==0 or (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label='Actual $x$', color='blue', linewidth=2)
        plt.plot(x, predictions.detach().numpy(), label='Network Approximation', color='red')
        plt.title('Shallow Network Approximating $x$')
        plt.legend()
        plt.grid(True)
        plt.show()


# 5. Visualization
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).numpy()

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Actual $x^3 + x^2$', color='blue', linewidth=2)
plt.plot(x, predicted, label='Network Approximation', color='red')
plt.title('Shallow Network Approximating $x^3 + x^2$')
plt.legend()
plt.grid(True)
plt.show()
