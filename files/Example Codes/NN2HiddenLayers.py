"""---------------------------
 File Name: NN2HiddenLayers.py
 Purpose: Implement a NN with 2 hidden layers to approximate f(x) = x^2
 Author: Francesca Cannata
-----------------------------"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# ---------- Input Data and Target Function ----------
x = np.linspace(-1, 1, 100).reshape(-1, 1) # we're creating 1000 points between -1 and 1, organized in one column
y = x**2  # target function

# Convert Numpy arrays to PyTorch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()



# ---------- Network Architecture and Model Definition ----------
# Define the number of neurons for each layer
n_in = 1
n_Hid1 = 2
n_Hid2 = 100
n_out = 1

# Set the learning rate
learning_rate = 0.01

# Define a class which contains our training model.
class NeuralNet(nn.Module): # our class "NeuralNet" inherits all methods and properties from the superclass nn.Module in PyTorch
    def __init__(self):
        super().__init__() # initialization of nn.Module

        # Architecture of our network: two hidden layers with 2 and 3 neurons respectively and ReLU activation.
        # The function "nn.Linear(in_features, out_features)" applies an affine transformation to input data (x * W^T + b). It creates
        # the weight matrix of dimension (in_features, out_features) and the bias tensor of length in_features; it fills them with
        # random values. The bias argument is optional: by default is True, but one can disable it
        self.InHid1 = nn.Linear(n_in, n_Hid1)       # it creates the input-weight matrix with shape (n_in, n_Hid1) and the input-bias vector with size n_ind
        self.Hid1Hid2 = nn.Linear(n_Hid1, n_Hid2)   # it creates the hidden-weight matrix with shape (n_Hid1, n_Hid2) and the hidden-bias vector with size n_Hid1
        self.Hid2Out = nn.Linear(n_Hid2, n_out)     # it creates the output-weight matrix with shape (n_Hid1, n_Hid2) and the output-bias vector with size n_Hid2
        self.activation = nn.ELU() # examples of non-linear activation functions: nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.ELU() etc.

    # We need a function for establishing the order in which our data go through the layers
    # and for turning the input to the network (x) to its output
    def forward(self, x):
        x = self.activation(self.InHid1(x)) # input -> 1st Hidden: 1. With "self.InHid1(x)" each input x is multiplied by the input-weight and added to
                                            #                         the input-bias.
                                            #                      2. With "self.activation()", the net is applying the activation function to the result of
                                            #                         the previous step. They will get the new input x for the 1st-hidden layer.
        x = self.activation(self.Hid1Hid2(x)) # 1st Hidden -> 2nd Hidden
        x = self.Hid2Out(x) # 2nd Hidden -> output
        return x

# Define the model
model = NeuralNet()

# Create the loss function (we are taking the Mean Square Error MSE).
error = nn.MSELoss()

# Create the optimizer: it changes the "model.parameters()" (== the parameters of the network)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# ---------- Training Loop ----------
epochs = 30000
for epoch in range(epochs):
    # Reset gradients in order to not accumulate them in the .grad attribute during next epochs
    optimizer.zero_grad()

    # Forward pass: compute predicted output y_pred passing the input data x_tensor to our model
    y_pred = model(x_tensor)

    # Compare the prediction y_pred to the target y_tensor evaluating the loss. We want it goes near to zero as much as possible
    loss = error(y_pred, y_tensor)
    # print(f"For epoch ", epoch, " the loss is: ", loss.item())    # "loss.item()" gives us just the number inside the tensor.
                                                                    # Otherwise, if we print "loss", we will see the number and its derivation


    # Backward pass: determine how much each parameter contribute to the error. To this end, we compute the
    # derivatives of the error function with respect to every model parameter. It starts from the back to the start.
    # It stores the result for each parameter in their ".grad()" attribute
    loss.backward()

    # Gradient descent: update the parameters in the direction stored in .grad() attribute
    optimizer.step()

    # Let us create a plot which compare y_pred and y_tensor for the first epoch and then every 100 epochs
    # if epoch == 0 or (epoch + 1) % 100 == 0:     # epoch + 1 is needed because the counting starts from 0
    #     # Print the epoch and the corresponding loss
    #     print(f"The current epoch is {(epoch+1)}/{epochs}. The loss is: ", loss.item())
    #
    #     # Plot y_tensor and y_pred at that epoch. For this, it is important that Pytorch doesn't track y_pred anymore so, to this end, we will use the function ".detach()"
    #     plt.plot(x, y, label='Target function $y(x)$', color='blue', linewidth=2) # target
    #     plt.plot(x, y_pred.detach().numpy(), label='Output of the network', color='red', linewidth=2) # output of the network
    #
    #     # Plot attributes
    #     plt.xlabel('Input values $x$')
    #     plt.ylabel('$x^2$')
    #     plt.title('Target VS Prediction')
    #     plt.legend(loc='best')
    #
    #     plt.show()


# ---------- Visualization ----------
# model.eval() # not useful for fully connected

# Compute the prediction corresponding to the last epoch, without Autograd tracking
with torch.no_grad(): # all commands in here, will be without attribute "grad_fn"
    y_pred_final = model(x_tensor).numpy() # we are converting the tensor to a numpy array just for making the plot

# Print the final epoch with the corresponding loss
print(f"The final epoch is {(epoch+1)}/{epochs} and the loss is {(loss.item())}")

# Final plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Target function $x^2$', color='blue', linewidth=2)
plt.plot(x, y_pred_final, label='Network Approximation', color='red')

# Plot attributes
plt.xlabel('Input data')
plt.ylabel('$x^2$')
plt.title('Target $x^2$ VS Network Approximation')
plt.legend(loc='best')
plt.grid(True)

plt.show()
