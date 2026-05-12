#%%
import argparse
import time
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import torch
from torch import nn
from torch.nn import functional as F

#%% md
# # Network Architecture
#%%
class ShallowModulationNetwork(nn.Module):
    r"""
    A 1D *shallow* neural network (one hidden layer) designed to approximate a real valued function`.

    -- What the model computes (mathematical form) -----------------------------------------------
    For an input scalar x (or a batch of scalars), the model outputs

        y(x) = sum_{k=1}^units a_k * g_k(x) + c,

    where each hidden unit k uses either
      • "modulated ReLU" (if modulation=True):
            g_k(x) = ReLU(η * w_k * x + b_k) * exp( -0.5 * (η * w_k * x + b_k - t)^2 )
                                       * exp( -0.5 * (x - y_k)^2 )
        This is a ReLU gate multiplied by two Gaussians:
          (i) one centered at the *linear response* (≈ t),
          (ii) one centered at the *input* (≈ y_k).

      • plain ReLU (if modulation=False):
            g_k(x) = ReLU(w_k * x + b_k)

    The model is a weighted sum of these g_k(x) with output bias c.

    -- Why "modulation"? --------------------------------------------------------------------------
    The Gaussian factors "focus" each unit on specific regions:
      • exp( -0.5 * (x - y_k)^2 ) emphasizes inputs near y_k,
      • exp( -0.5 * (η w_k x + b_k - t)^2 ) emphasizes when the *linear response* ≈ t.
    This can help fit smooth shapes like x^2 with fewer units.

    -- Parameters (learnable) ---------------------------------------------------------------------
      a : (units,)    output weights for each hidden unit
      w : (units,)    slope of the linear response inside each unit
      b : (units,)    bias of the linear response
      y : (units,)    preferred input center for the Gaussian over x
      c : (1,)        global output bias

    -- Buffers (fixed hyperparameters, not optimized) ---------------------------------------------
      eta : scalar     scales the linear response inside ReLU and the first Gaussian
      t   : scalar     target "center" for the linear response in the first Gaussian
      (Buffers move with the model to GPU/CPU but are not updated by the optimizer.)

    -- Expected input / output shapes -------------------------------------------------------------
      Input  x : shape (N,) or (N,1) or a Python scalar/float/int
      Output y : shape (N,)   (or scalar if input is scalar)

    -- Suggested usage ----------------------------------------------------------------------------
      model = ShallowModulationNetwork(units=120, modulation=True, eta=0.5, t=1.0)
      x = torch.linspace(-2, 2, 201)          # (201,)
      y_hat = model(x)                         # (201,)

    """

    def __init__(self, units: int = 300, seed:int = 42, modulation: bool = True, eta: float = 0.5, t: float = 1.0, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        """
        Args:
            units: Number of hidden units (basis functions). Larger = more expressive.
            modulation: If True, use ReLU * Gaussian * Gaussian. If False, plain ReLU.
            eta: Fixed scale for the linear part inside each unit (buffer, not learned).
            t:   Fixed "target" center for the linear-response Gaussian (buffer, not learned).
        """
        super().__init__()
        self.units = units
        self.modulation = modulation

        # Register fixed scalars as BUFFERS (not trainable, but move with .to(device))
        self.register_buffer("eta", torch.tensor(float(eta), device=device))
        self.register_buffer("t", torch.tensor(float(t), device=device))

        # Seeding for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # --- Learnable parameters --------------------
        # Small init for a to avoid huge outputs at the start.
        self.a = nn.Parameter(torch.randn(units, device=device) * 0.01)
        self.w = nn.Parameter(torch.randn(2, units, device=device))
        self.b = nn.Parameter(torch.randn(units, device=device))
        if self.modulation:
            self.y = nn.Parameter(torch.randn(2, units, device=device))
        self.c = nn.Parameter(torch.randn(1, device=device))

    def forward(self, x: torch.Tensor | float) -> torch.Tensor:
        """
        Forward pass.

        Steps:
          1) Ensure x is a tensor with dtype/device matching the parameters.
          2) Ensure x has shape (N, 1) for batched inputs (or be a scalar).
          3) Broadcast parameters to shape (1, units) so x * w is (N, units).
          4) Compute g_k(x) either with modulation or plain ReLU.
          5) Weighted sum over units with coefficients a_k and add bias c.

        Args:
            x: Input tensor of shape (N,), (N,1), or a Python scalar/float/int.

        Returns:
            Tensor of shape (N,) for batched input, or a scalar tensor if input is scalar.
        """

        # 1) Convert scalars to tensors and move dtype/device to match parameters
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=self.a.dtype, device=self.a.device)

        # 2) If x is a 1D batch (N,), make it (N,1) so broadcasting with (1, units) gives (N, units)
        #    If x is a scalar tensor (shape ()), we keep it as-is; broadcasting still works.
        if x.dim() == 1:
            x = x.unsqueeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)

        # 3) Reshape parameter vectors to (1, units) for clean broadcasting with (N,1)
        #    (If x is scalar, broadcasting will yield shape (1, units).)
        W = self.w.unsqueeze(0)
        b = self.b.unsqueeze(0)


        # 4) Compute unit activations g_k(x)
        if self.modulation:
            y = self.y.unsqueeze(0)
            # Linear response per unit: shape (N, units) or (1, units) for scalar input
            lin = self.eta * x @ W + b

            # ReLU gate
            relu_term = F.relu(lin)

            # Gaussian centered on the linear response ≈ t
            gaussian_center = torch.exp(-0.5 * (lin - self.t) ** 2)

            # # Gaussian centered on the input ≈ y_k (broadcasts x with y)
            # # If x is (N,1), (x - y) -> (N, units); if x is scalar, -> (1, units)
            # gaussian_input = torch.exp(-0.5 * ((x - y)**2).sum(dim=1))

            # Gaussian centered on the inputs at y_k ∈ R^d, one center per unit
            # Broadcast to [N, units, d] and reduce over the feature dimension
            # ensure y has shape [units, d] = [300, 2]
            y_centers = self.y.t()  # from [2, 300] -> [300, 2]

            # broadcast to [N, units, d] and reduce over features
            diff = x.unsqueeze(1) - y_centers.unsqueeze(0)  # [10000, 300, 2]
            gaussian_input = torch.exp(-0.5 * (diff ** 2).sum(dim=-1))  # [10000, 300]  # [N, units, d]


            # Final per-unit contribution
            contribution = relu_term * gaussian_center * gaussian_input

            # Weighted sum across units (dim=1), then add output bias c
            output = (contribution * self.a.unsqueeze(0)).sum(dim=-1) + self.c
        else:
            # Plain one-hidden-layer ReLU network: sum_k a_k * ReLU(w_k x + b_k) + c
            lin = x @ W + b
            output = (F.relu(lin) * self.a).sum(dim=-1) + self.c

        return output[-1]

#%% md
# # Data Generating Distribution
#%%

def generate_samples(
    num_samples: int,
    input_dimension: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device = torch.device("cuda"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic grid on [-1, 1]^d with target:
        f(x) = exp(-||x||_2^2) * sin(sum(x))

    Requirement:
        num_samples must be a perfect d-th power: num_samples = n**input_dimension.

    Returns:
        X: [N, d]  inputs
        Z: [N, 1]  targets
        n: int  # points per dimension, with N = n**d
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if input_dimension < 1:
        raise ValueError("input_dimension must be >= 1.")

    d = input_dimension
    # Enforce N = n^d
    n = int(round(num_samples ** (1.0 / d)))
    if n ** d != num_samples:
        floor_n = int(num_samples ** (1.0 / d))
        # adjust for potential floating-point quirks
        while (floor_n + 1) ** d <= num_samples:
            floor_n += 1
        while floor_n ** d > num_samples and floor_n > 0:
            floor_n -= 1
        ceil_n = floor_n if floor_n ** d == num_samples else floor_n + 1
        lower = floor_n ** d
        upper = ceil_n ** d
        raise ValueError(
            f"num_samples must equal n^{d}. Got {num_samples}. "
            f"Nearest valid choices: {lower} (= {floor_n}^{d}) or {upper} (= {ceil_n}^{d})."
        )

    # Build axes and Cartesian grid
    axes = [
        torch.linspace(-1.0, 1.0, steps=n, dtype=dtype, device=device)
        for _ in range(d)
    ]

    if d == 1:
        X = axes[0].unsqueeze(1)  # [n,1]
    else:
        mesh = torch.meshgrid(*axes, indexing="ij")
        X = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # [n^d, d]

    # Target: f(x) = exp(-sum x_i^2) * sin(sum x_i)
    r2 = (X ** 2).sum(dim=1)           # [N]
    s  = X.sum(dim=1)                  # [N]
    Z = torch.exp(-r2) * torch.sin(s)  # [N]
    Z = Z.unsqueeze(1)                 # [N,1]

    return X, Z, n

#
# def generate_samples(num_samples: int, input_dimension: int = 1, seed:int = 42) -> tuple[torch.Tensor, torch.Tensor]:
#     """Generate (input, target) pairs sampled from a standard normal input."""
#
#     # Seeding for reproducibility
#     # torch.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed)
#     inputs = torch.rand(num_samples, input_dimension)
#     X, Y = inputs[:, 0], inputs[:, 1]
#     targets = torch.exp(-(X ** 2 + Y ** 2)) * torch.sin(16 * (X ** 2 + Y ** 2))
#
#     # targets = torch.exp(-inputs.pow(2))*torch.sin(3*inputs)
#     return inputs, targets
#%% md
# # Sobolev Loss Function
#%%
def sobolev_h1_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    create_graph: bool = False,
    sobolev: bool = False,
) -> torch.Tensor:
    """Compute the H1 Sobolev loss combining function and gradient errors."""
    device = next(model.parameters()).device
    inputs.to(device)
    targets.to(device)
    inputs_for_grad = inputs.detach().clone().detach().requires_grad_(True)
    predictions = model(inputs_for_grad).unsqueeze(1)

    # print(f"Inputs for grad shape:{inputs_for_grad.shape}")
    # print(f"Prediction shape:{predictions.shape}")
    # print(f"Targets shape:{targets.shape}")
    l2_term = F.mse_loss(predictions, targets)

    grad_outputs = torch.ones_like(predictions)
    gradients = torch.autograd.grad(
        predictions,
        inputs_for_grad,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    # Derivative of the target
    X = inputs_for_grad # [N,d]
    r2 = (X ** 2).sum(dim=1, keepdim=True)  # [N,1]
    s = X.sum(dim=1, keepdim=True)  # [N,1]

    # Gradient dZ/dX: [N, d]
    grad = torch.exp(-r2) * (torch.cos(s) - 2.0 * torch.sin(s) * X)

    dfdx = grad[:, 0]

    dfdy = grad[:, 1]

    final_loss = l2_term + F.mse_loss(gradients[:, 0], dfdx) + F.mse_loss(gradients[:, 1], dfdy)

    return final_loss

#%% md
# # Trainig and Plotting (the plotting part could be moved to a separate section)
#%%
def train_and_plot(
    epochs: int = 10000,
    lr: float = 1e-3,
    training_samples: int = 10000,
    validation_samples: int = 2500,
    test_samples: int = 2500,
    units: int = 300,
    seed:int = 42,
    modulation : bool = True) -> tuple[list[float]]:
    # ndarray[tuple[Any, ...], dtype[_ScalarT]]:

    device = torch.device("cuda")

    """Train the shallow modulation network and plot predictions vs. targets."""
    model_type = "Modulation" if modulation else "Plain"

    if not modulation:
        #1d       units = int(4/3*units)
        #2d
        units = int(3 / 2 * units)

    print({f"Neurons={units}"})
    model = ShallowModulationNetwork(units=units, modulation=modulation, seed=seed)
    model.to(device)
    print(f"Model device: {torch.cuda.is_available()}")
    print(f"Model device: {torch.cuda.get_device_name(model.a.device)}")
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # we want to minimize the loss
        factor=0.9,  # lr *= 0.9 on plateau
        patience=50,  # epochs with no improvement before reducing
        threshold=0.0,  # improvement must be strictly better than best
        threshold_mode="rel",  # compare relative improvement (default)
        cooldown=100,  # wait this many epochs after a reduction
        min_lr=1e-8  # optional floor
    )

    train_input, train_target, n_train = generate_samples(num_samples=training_samples, input_dimension=2)
    val_input, val_target, n_val = generate_samples(num_samples=validation_samples, input_dimension=2)
    test_input, test_target, n_test = generate_samples(num_samples=test_samples, input_dimension=2)

    train_losses: list[float] = []
    val_losses: list[float] = []

    # timing before the training loop
    start_time = time.perf_counter()
    last_mark = start_time
    for epoch in range(epochs):
        # Forward pass
        prediction = model(train_input)

        # Compute loss
        loss = sobolev_h1_loss(model, train_input, train_target, create_graph=True, sobolev=True)

        # Store the loss value
        train_loss_value = loss.item()

        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        # # Step the scheduler
        scheduler.step(loss.item())

        # log current LR
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss_value = sobolev_h1_loss(model, val_input, val_target, create_graph=False, sobolev=True).item()


        train_losses.append(train_loss_value)
        val_losses.append(val_loss_value)

        # Check the shape of the loss
        # print(np.asarray(train_losses).shape)

        if epoch==0 or  (epoch + 1) % 100 == 0:
            now = time.perf_counter()
            elapsed_since_last = now - last_mark
            elapsed_total = now - start_time
            last_mark = now
            mins, secs = divmod(elapsed_since_last, 60)
            tmins, tsecs = divmod(elapsed_total, 60)
            print(
                f"epoch: {epoch + 1} ",
                f"time: {int(mins)}m{secs:05.2f}s since, {int(tmins)}m{tsecs:05.2f}s total",
                f"lr={current_lr:.0e}",
                f"training loss: {train_loss_value}",
                f"validation loss: {val_loss_value}"
                )

    total_sec = time.perf_counter() - start_time
    mins, secs = divmod(total_sec, 60)
    hrs, mins = divmod(mins, 60)
    print(f"Total training time: {int(hrs)}h {int(mins)}m {secs:05.2f}s  (~{total_sec:.2f}s)")
    # -------------------------
    # Shapes + trainable flag for all nn.Parameters
    # -------------------------
    print(f"{'name':35} {'shape':20} {'dtype':12} trainable")
    print("-" * 85)
    for name, p in model.named_parameters():
        shape_str = str(tuple(p.shape))  # make it a string first
        print(f"{name:35} {shape_str:20} {str(p.dtype):12} {p.requires_grad}")

    print(f"total learned {model_type} params:", sum(p.numel() for p in model.parameters()))  # total learned params

    ##################################################################
    # Saving state_dict
    ##################################################################
    ckpt_dir = Path("checkpoints")  # or Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # creates it if missing (no error if it exists)
    ckpt = {
        "model_state": model.state_dict(),
        # "optimizer_state": optimizer.state_dict(),  # optional
        # "epoch": epoch,                              # optional
        # "config": {"Modulation":modulation, "width": units, "lr": lr},   # optional
    }
    torch.save(ckpt, ckpt_dir/f"{model_type}_net_Width{units}_Seed{seed}_Units{units}.pt")


    with torch.no_grad():
        predicted_test = model(test_input)
        predicted_test = predicted_test.detach().cpu().numpy()
        target_test = test_target.detach().cpu().numpy()
        input_test = test_input.detach().cpu().numpy()


        predicted_train = model(train_input)
        predicted_train = predicted_train.detach().cpu().numpy()
        target_train = train_target.detach().cpu().numpy()
        input_train = train_input.detach().cpu().numpy()

    sort_test = np.argsort(input_test[:, 0])
    sort_train = np.argsort(input_train[:, 0])

    plots_dir = Path("plots")  # or Path(args.ckpt_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)  # creates it if missing (no error if it exists)

    # Sort and reshape into grid
    order = np.lexsort((input_test[:, 1], input_test[:, 0]))
    input_test_sorted = input_test[order]
    predicted_test_sorted = predicted_test[order]
    xs = np.unique(input_test_sorted[:, 0])
    ys = np.unique(input_test_sorted[:, 1])
    predicted_test_grid = predicted_test_sorted.reshape(n_test, n_test)
    test_target_grid = target_test.reshape(n_test, n_test)
    # # Use shared contour levels so both panels are comparable
    # zmin = min(predicted_test_grid.min(), test_target_grid.min())
    # zmax = max(predicted_test_grid.max(), test_target_grid.max())
    level_values = 20

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # LEFT: prediction
    ax = axes[0]
    cs = ax.contour(xs, ys, predicted_test_grid, levels=level_values)  # contour lines (not filled)
    # ax.clabel(cs, inline=True, fontsize=8)  # optional
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{model_type} Network (Prediction)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.6)

    # RIGHT: ground truth
    ax = axes[1]
    cs_true = ax.contour(xs, ys, test_target_grid, levels=level_values)
    # ax.clabel(cs_true, inline=True, fontsize=8)  # optional
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Target Function (Ground Truth)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.6)

    plt.savefig(f"plots/{model_type}_Network_Approximation_Seed_{seed}_Units{units}_side_by_side.pdf")
    # plt.figure()
    # cs = plt.contour(xs, ys, predicted_test_grid, levels=20)  # contour lines (not filled)
    # # plt.clabel(cs, inline=True, fontsize=8)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title(f"{model_type} Network vs Target Function")
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.6)
    # plt.savefig(f"plots/{model_type}_Network_Approximation_Seed_{seed}_Units{units}.pdf")



    # Prepare data for derivative  (DO NOT use torch.no_grad() here)
    test_input_for_grad = test_input.clone().detach().requires_grad_(True)

    # The prediction of the model f_θ(x) on test data
    z_hat = model(test_input_for_grad)

    # The derivative of the model w.r.t the test data
    grad_z_hat = torch.autograd.grad(
        z_hat, test_input_for_grad, grad_outputs=torch.ones_like(z_hat),
        create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True
    )[0]

    # print(grad_z_hat.shape)
    # print(grad_z_hat[:, 0])
    dzdx_hat = grad_z_hat[:, 0]
    
    dzdy_hat = grad_z_hat[:, 1]


    print(test_input_for_grad.shape)
    # 4) Plotting
    x_np = test_input_for_grad.detach().cpu().numpy()
    print(x_np.shape)
    # z_hat_np = z_hat.detach().cpu().numpy().ravel()
    dzdx_np = dzdx_hat.detach().cpu().numpy()
    dzdy_np = dzdy_hat.detach().cpu().numpy()

    r2 = (x_np ** 2).sum(axis=1, keepdims=True)  # [N,1]
    s = x_np.sum(axis=1, keepdims=True)  # [N,1]
    # Gradient dZ/dX: [N, d]
    grad = np.exp(-r2) * (np.cos(s) - 2.0 * np.sin(s) * x_np)

    dzdx_true_test = grad[:, 0]
    dzdy_true_test = grad[:, 1]


    # Plot dx derivative approximation vs target derivative
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    # Left: from scattered samples
    cs0 = axes[0].tricontour(x_np[:, 0], x_np[:, 1], dzdx_np, levels=level_values)
    axes[0].set_title(f"{model_type} Network dx Derivative Prediction")

    # Right: analytic grid
    cs1 = axes[1].tricontour(x_np[:, 0], x_np[:, 1], dzdx_true_test, levels=level_values)
    # axes[1].clabel(cs1, inline=True, fontsize=8)
    axes[1].set_title(f"True dx Derivative")

    # one shared colorbar
    fig.colorbar(cs1, ax=axes, location="right")
    fig.suptitle(f"{model_type} Network vs Derivative of Target Function")
    plt.savefig(f"plots/{model_type}_dx_Network_Approximation_Sobolev_Seed_{seed}_Units{units}.pdf")


    # Plot dy derivative approximation vs target derivative
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    # Left: from scattered samples
    cs0 = axes[0].tricontour(x_np[:, 0], x_np[:, 1], dzdy_np, levels=level_values)
    axes[0].set_title(f"{model_type} Network dy Derivative Prediction")

    # Right: analytic grid
    cs1 = axes[1].tricontour(x_np[:, 0], x_np[:, 1], dzdy_true_test, levels=level_values)
    # axes[1].clabel(cs1, inline=True, fontsize=8)
    axes[1].set_title(f"True dy Derivative")

    # one shared colorbar
    fig.colorbar(cs1, ax=axes, location="right")
    fig.suptitle(f"{model_type} Network vs Derivative of Target Function")
    plt.savefig(f"plots/{model_type}_dy_Network_Approximation_Sobolev_Seed_{seed}_Units{units}.pdf")



    plt.figure()
    epochs_axis = np.arange(epochs)
    # eps = 1e-8
    plt.semilogy(epochs_axis, np.asarray(train_losses), label=f"{model_type} training MSE")
    plt.semilogy(epochs_axis, np.asarray(val_losses), label=f"{model_type} validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title(f"{model_type} Training and Validation Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model_type}_Training_vs_Validation_Seed_{seed}_Units{units}.pdf")


    losses = np.asarray(np.asarray(train_losses), dtype=float)  # ensure 1-D float array
    n = losses.shape[0]
    df = pd.DataFrame({
        "seed": seed,
        "epoch": np.arange(1, n + 1, dtype=int),
        "loss": losses
    })
    # df.to_csv(f"loss/{model_type}_losses_seed_{seed}_Units{units}.txt", sep="\t", index=False, float_format="%.9g")

    loss_dir = Path("loss")  # or Path(args.ckpt_dir)
    loss_dir.mkdir(parents=True, exist_ok=True)

    return df.to_csv(f"loss/{model_type}_losses_seed_{seed}_units{units}.txt", sep="\t", index=False, float_format="%.9g")
#%% md
# # Modulation Network
#%%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations for different network architectures.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for the simulation.')
    parser.add_argument('--units', type=int, default=100, help='How many neurons for the hidden layer.')
    parser.add_argument("--modulation",type=bool, default=True, help="Use modulation architecture if set.")

    args = parser.parse_args()
    print(f'Plain is activated and the network running with seed {args.seed} and {args.units} hidden neurons')
    train_and_plot(modulation =  args.modulation, seed = args.seed, units = args.units, training_samples=10000, epochs=100000)


#
# #%% md
# # # Modulation loss across seeds
# #%%
# m_ls = np.array(all_loss_his)
# print(m_ls.shape)
#
#
# rows = (
#     {"seed": s, "epoch": e, "loss": v}
#     for s, lst in enumerate(m_ls)          # seeds 0..9
#     for e, v in enumerate(lst, start=1)      # epochs 1..len(lst)
# )
# df_modulation = pd.DataFrame(rows, columns=["seed", "epoch", "loss"])
# # print(df_modulation)
# # df_modulation.to_csv("modulation_losses_long.txt", sep="\t", index=False)
#
# g = sns.relplot(x="epoch", y="loss", hue="seed",kind="line", data=df_modulation)
# g.set_axis_labels("Epoch", "Loss")
# g.figure.suptitle("Modulation Training Loss per Seed", y=1.02)
# g.set(ylim=(1e-4, None), yscale="log")
# plt.show()
#
# # sns.set_theme(style="whitegrid")
#
# # Median line + percentile band (10–90%)
# ax = sns.lineplot(
#     data=df_modulation, x="epoch", y="loss",
#     estimator=np.median,         # central tendency
#     errorbar=("pi", 90),         # percentile interval width (80% => 10–90)
#     n_boot=0,                    # no bootstrap; use sample percentiles
#     linewidth=1.5,               # thick middle line
# )
# ax.set_yscale("log")         # focus near zero
# ax.set_ylim(1e-4, None)
# ax.set(title="Modulation Loss across seeds", xlabel="Epoch", ylabel="Loss")
# plt.show()
#
# #%% md
# # #  Seed-wise Visualization: Target vs Prediction and Loss Evolution
# #%%
# # the following plots are a reproduction of the training Target vs Prediction plots and the training loss
# for i, s in enumerate(seeds):
#     print(f"Run {i+1} with seed {s}")
#     plt.plot(all_data[s], all_pred[s] ,label="test prediction")
#     plt.plot(all_data[s], all_target[s] ,label="target", ls='--', dashes=(10, 5))
#     plt.legend()
#     plt.show()
# #%% md
# # # Plain Network
# #%%
# seeds = [k for k in range(10)]
# # seeds = [0]
# print(seeds)
# all_loss_his_plain, all_data_plain, all_target_plain, all_pred_plain = [], [], [], []
# for i, s in enumerate(seeds):
#      print(f"Run {i+1} with seed {s}")
#      loss_his_plain, data_plain, target_plain, pred_plain = train_and_plot(modulation=False, seed=s, epochs=5000)
#      all_loss_his_plain.append(loss_his_plain)
#      all_data_plain.append(data_plain)
#      all_target_plain.append(target_plain)
#      all_pred_plain.append(pred_plain)
# #%% md
# # # Plain loss across seeds
# #%%
# p_ls = np.array(all_loss_his_plain)
# print(p_ls.shape)
#
#
# rows = (
#     {"seed": s, "epoch": e, "loss": v}
#     for s, lst in enumerate(p_ls)          # seeds 0..9
#     for e, v in enumerate(lst, start=1)      # epochs 1..len(lst)
# )
# df_plain = pd.DataFrame(rows, columns=["seed", "epoch", "loss"])
# # print(df_plain)
# # df_plain.to_csv("losses_long.txt", sep="\t", index=False)
#
# g = sns.relplot(x="epoch", y="loss", hue="seed",kind="line", data=df_plain)
# g.set_axis_labels("Epoch", "Loss")
# g.figure.suptitle("Plain Training Loss per Seed", y=1.02)
# g.set(yscale="log")
# plt.show()
#
# # sns.set_theme(style="whitegrid")
#
# # Median line + percentile band (10–90%)
# ax = sns.lineplot(
#     data=df_plain, x="epoch", y="loss",
#     estimator=np.median,         # central tendency
#     errorbar=("pi", 90),         # percentile interval width (80% => 10–90)
#     n_boot=0,                    # no bootstrap; use sample percentiles
#     linewidth=1.5,               # thick middle line
# )
# ax.set_yscale("log")         # focus near zero
# # ax.set_ylim(1e-7, None)
# ax.set(title="Plain loss across seeds", xlabel="Epoch", ylabel="Loss")
# plt.show()
#
# #%% md
# # #  Seed-wise Visualization: Target vs Prediction on Test Data
# #%%
#
# for i, s in enumerate(seeds):
#     print(f"Run {i+1} with seed {s}")
#     plt.plot(all_data_plain[s], all_pred_plain[s] ,label="test prediction")
#     plt.plot(all_data_plain[s], all_target_plain[s] ,label="target", ls='--', dashes=(10, 5))
#     plt.legend()
#     plt.show()
#
#
# #%% md
# # # Load and Test Models on New Data for Evaluation
# #%%
#
# def load_models_from_checkpoints(checkpoint_paths, model_class, **model_kwargs):
#     """
#     Load model instances from checkpoint files.
#
#     Args:
#         checkpoint_paths: List of Path objects pointing to .pt files
#         model_class: The model class to instantiate (ShallowNetwork or ShallowModulationNetwork)
#         **model_kwargs: Keyword arguments to pass to model constructor (e.g., units=300)
#
#     Returns:
#         List of loaded model instances
#     """
#     models = []
#     for path in checkpoint_paths:
#         # Create a new model instance
#         model = model_class(**model_kwargs)
#
#         # Load the saved state dictionary
#         state_dict = torch.load(path, map_location='cpu')
#         model.load_state_dict(state_dict["model_state"])
#
#         # Set to evaluation mode
#         model.eval()
#
#         models.append(model)
#
#     return models
#
# def stack_preds(models, x_sorted):
#     """
#     Stack predictions from multiple models.
#
#     Args:
#         models: List of PyTorch model instances (not paths!)
#         x_sorted: Input tensor of shape (N,) or (N, 1)
#
#     Returns:
#         Array of shape (num_models, N) with predictions
#     """
#     x_tensor = torch.tensor(x_sorted, dtype=torch.float32)
#     if x_tensor.dim() == 1:
#         x_tensor = x_tensor.unsqueeze(1)
#
#     predictions = []
#     with torch.no_grad():
#         for model in models:
#             # Model is already in eval mode from loading
#             # But we can ensure it here too
#             model.eval()
#
#             # Get predictions from the model
#             y_pred = model(x_tensor)
#             predictions.append(y_pred.cpu().numpy())
#
#     return np.array(predictions)
#
# # Example usage - load your checkpoints
# checkpoint_dir = Path("checkpoints")
#
# # Load modulation networks
# mod_checkpoint_paths = sorted(checkpoint_dir.glob("Modulation_net_Width300_Seed*.pt"))
# mod_models = load_models_from_checkpoints(
#     mod_checkpoint_paths,
#     ShallowModulationNetwork,
#     units=300
# )
#
# for mod_model in mod_models:
#     mod_model.eval()
#
#
# # Load plain networks
# plain_checkpoint_paths = sorted(checkpoint_dir.glob("Plain_net_Width400_Seed*.pt"))
# plain_models = load_models_from_checkpoints(
#     plain_checkpoint_paths,
#     ShallowModulationNetwork,
#     units=400,
#     modulation=False
# )
#
# for plain_model in plain_models:
#     plain_model.eval()
#
#
#
# # Now create your pair dictionary with actual model objects
# pair = {
#     "mod": mod_models,
#     "plain": plain_models
# }
#
# # Fix the random seed for reproducibility
# torch.manual_seed(99)
#
# x = -2 + 4 * torch.rand(100, 1)
#
#
# # sort by the 1D coordinate
# x_sorted, idx = torch.sort(x[:, 0])          # x_sorted: (100,)
# x_sorted = x_sorted.unsqueeze(1)              # (100, 1)
#
#
# true_target = torch.exp(-x_sorted.pow(2))*torch.sin(3*x_sorted)
# true_target = true_target.squeeze(0).cpu().numpy()
#
# # # mod_pred_test, plain_pred_test= [], []
# # with torch.no_grad():
# #     for type in pair:
# #         if type in pair and pair["mod"]:
# #             for i, mod_model in enumerate(mod_models):
# #                 mod_pred_test = mod_model(x_sorted)
# #                 x_np = x_sorted.squeeze(1).cpu().numpy()
# #                 y_modulation_np = mod_pred_test.squeeze(0).cpu().numpy()
# #                 plt.plot(x_np, y_modulation_np, label=f"modulation network Seed={i}")
# #                 plt.legend()
# #                 plt.show()
# #         if type in pair and pair["plain"]:
# #             for i, plain_model in enumerate(plain_models):
# #                 plain_pred_test = plain_model(x_sorted)# shape (1000, 1)
# #                 x_np = x_sorted.squeeze(1).cpu().numpy()
# #                 y_plain_np = plain_pred_test.squeeze(0).cpu().numpy()
# #                 plt.plot(x_np, y_plain_np, label=f"plain network Seed={i}")
# #                 plt.legend()
# #                 plt.show()
#
#
# # assume x_sorted already built; do this once, not in the loop
# x_np = x_sorted.squeeze(1).cpu().numpy()
#
# with torch.no_grad():
#     # pair index i from mod with index i from plain (i and i+10 overall)
#     n = max(len(mod_models), len(plain_models))
#     for i in range(n):
#         plt.figure(figsize=(9, 5))
#
#         # plot modulation i (if exists)
#         if i < len(mod_models):
#             y = mod_models[i](x_sorted).cpu().numpy()
#             plt.plot(x_np, y, label=f"Modulation Seed: {i}")
#
#         # plot plain i (if exists)
#         if i < len(plain_models):
#             z = plain_models[i](x_sorted).cpu().numpy()
#             plt.plot(x_np, z, label=f"Plain Seed: {i}", dashes=(20,10))
#
#         plt.plot(x_np, true_target, label="target", ls='-.', dashes=(10,5))
#         plt.title(f"Prediction on new data Plain vs Modulation Seed:{i}")
#         plt.xlabel("x")
#         plt.ylabel("prediction")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#
# #%% md
# # # Seed-wise losses: Modulation Network vs Plain Network
# #%%
# for i, s in enumerate(seeds):
#     print(f"Run {i+1} with seed {s}")
#     plt.semilogy(all_loss_his[s], label=f"modulation network Seed={s}")
#     plt.semilogy(all_loss_his_plain[s], label=f"plain network Seed={s}")
#     plt.legend()
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training Errors")
#     plt.tight_layout()
#     plt.show()
# #%% md
# # # Plot Modulation Network vs Plain Network losses
# #%%
#
# # Example: list of lists; each inner list is the loss history for a seed
# # losses = [
# #     [0.9, 0.7, 0.5],     # seed 0
# #     [0.95, 0.72, 0.52],  # seed 1
# #     ...
# # ]
#
# def save_losses_list_of_lists(losses, path_txt, start_epoch=1):
#     rows = []
#     for seed, loss_list in enumerate(losses):
#         for e, loss in enumerate(loss_list, start=start_epoch):
#             rows.append({"seed": seed, "epoch": e, "loss": float(loss)})
#     df = pd.DataFrame(rows, columns=["seed", "epoch", "loss"])
#     # space-separated text with header
#     df.to_csv(path_txt, sep=" ", index=False, header=True, float_format="%.9g")
#     return df
#
# df_modulation = save_losses_list_of_lists(all_loss_his, "modulation_losses.txt")
# df_plain = save_losses_list_of_lists(all_loss_his_plain, "plain_losses.txt")
#
#
# # --- point these to your files ---
# path_mod = Path("modulation_losses.txt")
# path_plain = Path("plain_losses.txt")
#
# def load_three_col_txt(path, module: str):
#     """
#     Loads a 3-column text file into a DataFrame with columns: seed, epoch, loss,
#     and adds a fixed 'module' column (e.g., 'modulation', 'plain').
#     Works for whitespace-, tab-, comma-, or semicolon-separated values.
#     Ignores lines starting with '#'.
#     """
#     df = pd.read_csv(
#         path,
#         sep=r"[\s,;]+",
#         engine="python",
#         header=0,
#         names=["seed", "epoch", "loss"],
#         comment="#",
#         usecols=[0, 1, 2],
#         skipinitialspace=True,
#         on_bad_lines="skip",
#     )
#     df["module"] = module  # <- fixed label for this file
#     # Optional strict dtypes:
#     # df = df.astype({"seed": "int64", "epoch": "int64", "loss": "float64", "module": "string"})
#     return df
#
# df_m = load_three_col_txt(path_mod, module="modulation")
# df_p = load_three_col_txt(path_plain, module="plain")
#
# df = pd.concat([df_m, df_p], ignore_index=True).sort_values(["seed", "epoch"]).reset_index(drop=True)
#
# # df = pd.merge(
# #     df_p, df_m,
# #     on=["seed", "epoch"],
# #     how="inner", # or "outer" if keys may be missing on one side
# #     suffixes=("_plain", "_modulation") # becomes loss_plain, loss_modulation
# # )
#
# # If you later want them together:
# df_all = pd.concat([df_m, df_p], ignore_index=True)
#
#
# print("\nMerged DataFrame:")
# print(df_all)
#
# #%%
#
# sns.set_theme(style="darkgrid")
#
#
# # Plot the responses for different events and regions
# ax = sns.lineplot(x="epoch", y="loss",
#              style='module', data=df_all)
# ax.set(yscale="log")
#
# plt.show()
# #%%
# # change k to the desired jumps on the epochs
# k = 1000
# reduced_df = df_all.iloc[::k]
# print(reduced_df)
# #%%
#
# sns.set_theme(style="darkgrid")
#
#
# # Plot the responses for different events and regions
# ax = sns.lineplot(x="epoch", y="loss", estimator='mean',
#              hue="module", data=reduced_df)
# ax.set(yscale="log")
# plt.savefig('loss_modulation_plain.pdf',  dpi=600)
# plt.show()
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run simulations for different network architectures.')
#     parser.add_argument('--seed', type=int, default=0, help='Random seed for the simulation.')
#     args = parser.parse_args()
#     print(f'Running with seed {args.seed}')
#
#     run(seed=args.seed)