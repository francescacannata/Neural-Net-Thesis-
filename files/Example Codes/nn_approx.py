# Neural Approximation model
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
import copy
import os


def approx(x, y, units, learning_rate, stepsize, gamma, epochs, seed):
    torch.manual_seed(seed)
    n_in = x.shape[1]
    n_out = 1


    # Define a class which contains our training model.
    class NeuralNet(nn.Module): # our class "NeuralNet" inherits all methods and properties from the superclass nn.Module in PyTorch
        def __init__(self):
            super().__init__() # initialization of nn.Module

            # Architecture of our network: two hidden layers with 2 and 3 neurons respectively and ReLU activation.
            # The function "nn.Linear(in_features, out_features)" applies an affine transformation to input data (x * W^T + b). It creates
            # the weight matrix of dimension (in_features, out_features) and the bias tensor of length in_features; it fills them with
            # random values. The bias argument is optional: by default is True, but one can disable it
            self.InHid1 = nn.Linear(n_in, units)       # it creates the input-weight matrix with shape (n_in, n_Hid1) and the input-bias vector with size n_in
            self.Hid1Out = nn.Linear(units, n_out)     # it creates the output-weight matrix with shape (n_Hid1, n_out) and the output-bias vector with size n_Hid1
            self.activation = nn.ReLU()                 # examples of non-linear activation functions: nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.ELU() etc.

        # We need a function for establishing the order in which our data go through the layers
        # and for turning the input to the network (x) to its output
        def forward(self, _x):
            _x = self.activation(self.InHid1(_x)) # input -> 1st Hidden: 1. With "self.InHid1(x)" each input x is multiplied by the input-weight and added to
                                                #                         the input-bias.
                                                #                      2. With "self.activation()", the net is applying the activation function to the result of
                                                #                         the previous step. They will get the new input x for the 1st-hidden layer.
            _x = self.Hid1Out(_x) # 2st Hidden -> output
            return _x

    # Define the model
    model = NeuralNet()

    # Create the loss function (we are taking the Mean Square Error MSE).
    error = nn.MSELoss()

    # Create the optimizer: it changes the "model.parameters()" (== the parameters of the network)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Change the learning rate dynamically with the Scheduler algorithm: we want to adjust it during the training
    # Step Decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma= gamma) # if we are not using it, uncomment also the lower bound for the learning rate in the for loop

    # ReduceLROnPlateau
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=500, min_lr=1e-5)


    # Parameter's initialization
    with torch.no_grad():
        # Input Biases
        nn.init.uniform_(model.InHid1.bias, a=0, b=0)

        # Input Weights
        init_weights = nn.init.uniform_(model.InHid1.weight, a=-1, b=1)
        model.InHid1.weight.copy_(init_weights)


    # ---------- Training Loop ----------
    min_loss = float('inf') # initialization of the minimum loss at infinity
    min_epoch = 0           # initialization of the epoch corresponding to the minimum loss
    best_model = {}         # initialization of the dictionary which will contain the best model

    loss_history = np.zeros(epochs) # initialization of the array which will contain all errors
    learning_rate_history = np.zeros(epochs) # initialization of the array which will contain all learning rates
    for epoch in range(epochs):

        # Reset gradients in order to not accumulate them in the .grad attribute during next epochs
        optimizer.zero_grad()

        # Forward pass: compute predicted output y_pred passing the input data x_tensor to our model
        y_pred = model(x)

        # Compare the prediction y_pred to the target y_tensor evaluating the loss. We want it goes near to zero as much as possible
        loss = error(y_pred, y.view(-1,1))
        # print(f"For epoch ", epoch, " the loss is: ", loss.item())    # "loss.item()" gives us just the number inside the tensor.
                                                                        # Otherwise, if we print "loss", we will see the number and its derivation


        # Backward pass: determine how much each parameter contribute to the error. To this end, we compute the
        # derivatives of the error function with respect to every model parameter. It starts from the back to the start.
        # It stores the result for each parameter in their ".grad()" attribute
        loss.backward()

        # Gradient descent: update the parameters in the direction stored in .grad() attribute
        optimizer.step()

        # Update the learning rate when reaching the "step_size" and when it is grater than or equal to 10^-5 (lower bound)
        if scheduler.get_last_lr()[0] > 10**(-5):   # scheduler.get_last_lr()[0] get the most recent learning rates computed by the scheduler.
            scheduler.step()

        # Update the learning rate with Plateau scheduler
        #comscheduler.step()

        # Keep track of the minimum loss, its epoch and the corresponding model
        aux_loss = loss.item() # auxiliary array with the epoch and the corresponding loss
        if loss.item() < min_loss:
            min_loss = loss.item()
            min_epoch = epoch

            best_model = copy.deepcopy(model.state_dict())      # in order to have a copy (and not a reference) we need to use "copy.deepcopy()"

        # Let us create a plot which compare y_pred and y_tensor for the first epoch and then every 100 epochs
        if epoch == 0 or (epoch + 1) % 100 == 0:     # epoch + 1 is needed because the counting starts from 0
            # Print the epoch and the corresponding loss
            print(f"The current epoch is {(epoch+1)}/{epochs}. The loss is: ", loss.item())

        # add the loss and the learning_rate histories to the array
        loss_history[epoch] = loss.item()
        learning_rate_history[epoch] = scheduler.get_last_lr()[0]


    # To be sure the net uses the best model found, we load it with "model.load_state_dict()" and save it
    model.load_state_dict(best_model)
    print(f"The best model is {best_model}")

    # We want to have the same size for bias and weights using .flatten()
    bias_InHid1 = best_model["InHid1.bias"].flatten().numpy()
    bias_Hid1Out = best_model["Hid1Out.bias"].flatten().numpy()
    weight_InHid1 = best_model["InHid1.weight"].flatten().numpy()
    weight_Hid1Out = best_model["Hid1Out.weight"].flatten().numpy()

    best_model_dataframe = pd.DataFrame({'Bias_Intput_Layer': pd.Series(bias_InHid1),
                                'Weights_Intput_Layer': pd.Series(weight_InHid1),
                                'Bias_Output_Layer': pd.Series(bias_Hid1Out),
                                'Weights_Output_Layer': pd.Series(weight_Hid1Out)})

    dir_name = f'results_HN_{units}_Epochs_{epochs}_InitLR_{learning_rate}_StepSize_{stepsize}'
    os.makedirs(dir_name, exist_ok=True)
    best_model_dataframe.to_csv(os.path.join(dir_name, f'BestModel.csv'), index=False)




    # ---------- Visualization ----------
    # Compute the prediction corresponding to the last epoch, without Autograd tracking
    with torch.no_grad(): # all commands in here, will be without attribute "grad_fn"
        y_pred_final = model(x).numpy() # we are converting the tensor to a numpy array just for making the plot

    # Print the final epoch with the corresponding loss
    print(f"The final epoch is {(epoch+1)}/{epochs} and the loss is {(loss.item())}")

    # Print the minimum loss and its epoch
    print(f"The minimum loss is {(min_loss)} and its epoch is {(min_epoch+1)}")

    return y_pred_final, min_loss