"""---------------------------
 File Name: SNN4cos.py
 Purpose: Implement a SNN to approximate f(x) = cos(x)
 Author: Francesca Cannata
-----------------------------"""


import matplotlib.pyplot as plt
import torch
import torch.nn as nn       # module with pre-made tools for NN (i.e. loss and activation functions)

# Define the number of neurons for the input, hidden and output layers
n_in = 1
n_hidden = 150
n_out = 1

# Define the batch size (= number of tests which the network has to do before updating its parameters) and the learning rate
batch = 5
learning_rate = 0.01


# Define the input tensor with number of rows equal to batch and number of columns equal to n_in, from the normal distribution
x = torch.linspace(-10, 10, 100).view(-1, 1)  # .view() is equivalent to .reshape() in Numpy. The second argument (1) refers to the number of column,
                                                       # while the first argument (-1) tells Pytorch to calculate how many rows are necessarily for organizing
                                                       # our data in one column

# Define the target tensor y = cos(x)
y = torch.cos(x)


# Create a sequential model (our NN is a feedforward one). "Sequential" allows us to stack layers in a
# defined order, passing data through each layer, from start to finish.
model = nn.Sequential(
    nn.Linear(n_in, n_hidden),      # input layer: each input is multiplied by a weight and summed with a bias. They will become the hidden neurons
    nn.ReLU(),                      # activation layer: if the previous computation has generated negative numbers, they will set to 0; otherwise, they remain the same
    nn.Linear(n_hidden, n_out))     # output layer: each hidden output is multiplied by a weight and summed with a bias. They will become the output

# Create the loss function (we are taking the Mean Square Error MSE).
error = nn.MSELoss()

# Create the optimizer: it changes the "model.parameters()" (== the parameters of the network)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Implement the gradient descent algorithm
epochs = 30000
for epoch in range(epochs):
    # Reset gradients in order to not accumulate them in the .grad attribute during next epoch
    optimizer.zero_grad()

    # Forward pass: compute predicted output y_pred passing the input data x to our model
    y_pred = model(x)

    # Compare the prediction y_pred to the target y evaluating the loss. We want it goes near to zero as much as possible
    loss = error(y_pred, y)
    # print(f"For epoch ", epoch, " the loss is: ", loss.item()) # "loss.item()" gives us just the number inside the tensor.
                                                               # Otherwise, if we print "loss", we will see the number and its derivation


    # Backward pass: determine how much each parameter contributed to the error. To this end, we compute the
    # derivatives of the error function with respect to every model parameter. It starts from the back to the start.
    # It stores the result for each parameter in their ".grad()" attribute
    loss.backward()

    # Gradient descent: update the parameters in the direction stored in .grad() attribute
    optimizer.step()


    # Let us create a plot which compare y_pred and y for the first epoch and then every 100 epochs
    # if epoch == 0 or (epoch + 1) % 100 == 0:     # epoch + 1 is needed because the counting starts from 0
    #     # Print the epoch and the corresponding loss
    #     print(f"The current epoch is {(epoch+1)}/{epochs}. The loss is: ", loss.item())
    #
    #     # Plot y and y_pred at that epoch. For this, it is important that Pytorch doesn't track y_pred anymore so, to this end, we will use the function ".detach()"
    #     plt.plot(x, y, label='Target function $y(x)$', color='blue', linewidth=2) # target
    #     plt.plot(x, y_pred.detach().numpy(), label='Output of the network', color='red', linewidth=2) # output of the network
    #
    #     # Plot attributes
    #     plt.xlabel('Input values $x$')
    #     plt.ylabel('$cos(x)$')
    #     plt.title('Target VS Prediction')
    #     plt.legend(loc='best')
    #
    #     plt.show()



# Let us create a plot which compare y_pred and y for the final epoch
# Compute the prediction corresponding to the last epoch, without Autograd tracking
with torch.no_grad(): # all commands in here, will be without attribute "grad_fn"
    y_pred_final = model(x)

# Print the final epoch with the corresponding loss
print(f"The final epoch is {(epoch+1)}/{epochs} and the loss is {(loss.item())}")

# Final plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Target function $y(x)$', color='blue', linewidth=2) # target
plt.plot(x, y_pred_final, label='Output of the network', color='red', linewidth=2) # output of the network

# Plot attributes
plt.xlabel('Input values $x$')
plt.ylabel('$cos(x)$')
plt.title('Target VS Final prediction')
plt.legend(loc='best')

plt.show()



