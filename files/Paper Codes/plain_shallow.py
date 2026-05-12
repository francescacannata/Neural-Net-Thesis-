"""Self-contained shallow network approximating x^2."""

# from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class ShallowNetwork(nn.Module):
    r"""A shallow network that approximates :math:`x^2`."""

    def __init__(self, units: int = 120) -> None:
        super().__init__()
        self.units = units

        self.a = nn.Parameter(torch.randn(units) * 0.01)
        self.w = nn.Parameter(torch.randn(units))
        self.b = nn.Parameter(torch.zeros(units))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=self.a.dtype, device=self.a.device)

        if x.dim() == 1:
            x = x.unsqueeze(1)

        x = x.to(self.a.dtype)

        w = self.w.unsqueeze(0)
        b = self.b.unsqueeze(0)

        lin =  x * w + b

        output = (F.relu(lin) * self.a.unsqueeze(0)).sum(dim=1) + self.c
        return output.squeeze(-1)


def generate_quadratic_samples(num_samples: int, input_dimension: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (input, target) pairs sampled from a standard normal input."""

    inputs = torch.randn(num_samples, input_dimension)
    targets = inputs.pow(2).sum(dim=1)
    return inputs, targets


def train_and_plot(
    epochs: int = 10000,
    lr: float = 1e-3,
    training_samples: int = 1000,
    validation_samples: int = 200,
    test_samples: int = 200,
    units: int = 120,
) -> None:
    """Train the shallow network and plot predictions vs. targets."""

    model = ShallowNetwork(units=units)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    train_input, train_target = generate_quadratic_samples(training_samples)
    val_input, val_target = generate_quadratic_samples(validation_samples)
    test_input, test_target = generate_quadratic_samples(test_samples)

    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(epochs):
        prediction = model(train_input)
        loss = loss_function(prediction, train_target)
        train_loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss_value = loss_function(model(val_input), val_target).item()

        train_losses.append(train_loss_value)
        val_losses.append(val_loss_value)

        if (epoch + 1) % 100 == 0:
            print(
                f"epoch: {epoch + 1}, "
                f"training loss: {train_loss_value}, "
                f"validation loss: {val_loss_value}"
            )

    with torch.no_grad():
        final_test_loss = loss_function(model(test_input), test_target).item()
        print(f"final test loss: {final_test_loss:.6f}")

    with torch.no_grad():
        predicted_test = model(test_input).cpu().numpy()
        target_test = test_target.cpu().numpy()
        input_test = test_input.cpu().numpy()

        predicted_train = model(train_input).cpu().numpy()
        target_train = train_target.cpu().numpy()
        input_train = train_input.cpu().numpy()

    sort_test = np.argsort(input_test[:, 0])
    sort_train = np.argsort(input_train[:, 0])

    plt.figure()
    plt.plot(input_test[sort_test, 0], predicted_test[sort_test],label="test prediction")
    plt.plot(input_test[sort_test, 0], target_test[sort_test], label="test target")
    plt.scatter(input_train[sort_train, 0], target_train[sort_train], s=10, alpha=0.4, label="training samples")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Shallow Network Approximating x^2")
    plt.tight_layout()
    plt.show()


    # plt.figure()
    # epochs_axis = np.arange(epochs)
    # plt.plot(epochs_axis, train_losses, label="training MSE")
    # plt.plot(epochs_axis, val_losses, label="validation MSE")
    # plt.xlabel("Epoch")
    # plt.ylabel("Mean Squared Error")
    # plt.title("Training and Validation Errors")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    plt.figure()
    epochs_axis = np.arange(epochs)
    eps = 1e-8
    plt.semilogy(epochs_axis, np.asarray(train_losses) + eps, label="training MSE")
    plt.semilogy(epochs_axis, np.asarray(val_losses) + eps, label="validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title("Training and Validation Errors")
    plt.legend()
    plt.tight_layout()
    plt.show()

train_and_plot()