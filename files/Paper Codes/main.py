import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data import Data

def run():
    # setup data
    data = Data(input_dimension = 1, training_samples = 1000, validation_samples = 100, test_samples = 100)

    # setup Model
    width = 160 # =N such that N = 4/3M
    model = torch.nn.Sequential(
        torch.nn.Linear(data.input_dimension, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    )
    # torch.nn.init.zeros_(model[0].bias)
    # torch.nn.init.zeros_(model[2].bias)
    # print(model[0].weight, model[-1].weight)
    # print(model)
    # initialize training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()


    # load data
    data.generate()

    # --- error tracking setup (ADD) ---
    train_hist = []
    val_hist = []
    x_val = data.validation_data["input"]
    y_val = data.validation_data["target"]


    # training loop
    epochs = 50000
    target = data.training_data["target"]
    print(target)

    for epoch in range(epochs):
        # model.train()
        prediction = model(data.training_data["input"])
        loss = loss_function(prediction.squeeze(1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # --- record train/val losses (ADD) ---
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_function(val_pred.squeeze(1), y_val)

        train_hist.append(loss.item())
        val_hist.append(val_loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"epoch: {epoch+1}, loss: {loss.item()}")

    # -------------------------
    # Shapes + trainable flag for all nn.Parameters
    # -------------------------
    print(f"{'name':35} {'shape':20} {'dtype':12} trainable")
    print("-" * 85)
    for name, p in model.named_parameters():
        shape_str = str(tuple(p.shape))  # make it a string first
        print(f"{name:35} {shape_str:20} {str(p.dtype):12} {p.requires_grad}")

    print("total learned params:", sum(p.numel() for p in model.parameters()))  # total learned params

    print(f"layer zero weights: {model[0].weight}")
    print(f"layer zero biases: {model[0].bias}")
    print(f"layer one weights: {model[-1].weight}")
    print(f"layer one biases: {model[-1].bias}")
    # plot results
    prediction = model(data.test_data["input"]).detach().numpy()
    target = data.test_data["target"].numpy()
    input_data = data.test_data["input"].numpy()
    i_sort = np.argsort(input_data, axis=0)
    target = target[i_sort]
    prediction = prediction[i_sort, 0]
    input_data = input_data[i_sort, 0]

    training_input = data.training_data["input"].numpy()
    i_sort = np.argsort(training_input, axis=0)
    training_input = training_input[i_sort, 0]
    training_target = data.training_data["target"].numpy()[i_sort]

    print(input_data[:10])

    # --- plot error decay (ADD) ---
    plt.figure()
    plt.plot(range(epochs), train_hist, label="ReLU Train MSE")
    plt.plot(range(epochs), val_hist, label="ReLU Val MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Error Decay for 2 Layers each {width} with ReLU")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.plot(input_data, prediction, label="ReLU prediction")
    plt.plot(input_data, target, label="target")
    plt.plot(training_input, training_target, label="training")
    plt.legend()
    plt.show()




    print("Hello from modulationnetwork!")


if __name__ == "__main__":
    run()
