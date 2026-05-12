import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path


path_dir = "loss"
loss_dir = Path(path_dir)
print(f"Loading from: {loss_dir}")

# pick your pattern; e.g. "losses_seed*.txt" or "*.txt"
files = sorted(loss_dir.glob("Plain_losses_seed_0_units*.txt"))

if not files:
    raise FileNotFoundError(f"No files matched in {loss_dir}")

# read and concat with a single header: seed, lr, epoch, loss
dfs = [
    pd.read_csv(
        f, sep="\t", header=0,
        dtype={"seed": "int64", "epoch": "int64", "loss": "float64"}
    )
    for f in files
]


df_all = pd.concat(dfs, ignore_index=True)

# (optional) save combined table back to loss/
out_path = loss_dir / "plain_combined_losses.txt"
df_all.to_csv(out_path, sep="\t", index=False, float_format="%.9g")
print(f"Saved combined file to: {out_path}")

print(df_all.head(), df_all.shape)




# sns.set_theme(style="whitegrid")

# Median line + percentile band (10–90%)
ax = sns.lineplot(
    data=df_all, x="epoch", y="loss",
    estimator=np.median,         # central tendency
    errorbar=("pi", 90),         # percentile interval width (80% => 10–90)
    n_boot=0,                    # no bootstrap; use sample percentiles
    linewidth=1.5,               # thick middle line
)
ax.set_yscale("log")         # focus near zero
# ax.set_ylim(1e-4, None)
ax.set(title="Plain Loss across seeds", xlabel="Epoch", ylabel="Loss")
plt.savefig(loss_dir/"plain_median_loss_across_seeds.pdf")
