import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path


# --- point these to your files ---
path_mod = Path("loss/modulation_combined_losses.txt")
path_plain = Path("loss/plain_combined_losses.txt")

def load_three_col_txt(path, module: str):
    """
    Loads a 3-column text file into a DataFrame with columns: seed, epoch, loss,
    and adds a fixed 'module' column (e.g., 'modulation', 'plain').
    Works for whitespace-, tab-, comma-, or semicolon-separated values.
    Ignores lines starting with '#'.
    """
    df = pd.read_csv(
        path,
        sep=r"[\s,;]+",
        engine="python",
        header=0,
        names=["seed", "epoch", "loss"],
        comment="#",
        usecols=[0, 1, 2],
        skipinitialspace=True,
        on_bad_lines="skip",
    )
    df["module"] = module  # <- fixed label for this file
    # Optional strict dtypes:
    # df = df.astype({"seed": "int64", "epoch": "int64", "loss": "float64", "module": "string"})
    return df

k = 1

# Load
df_m = load_three_col_txt(path_mod, module="Modulation")
df_p = load_three_col_txt(path_plain, module="Plain")

# (Optional but recommended) sort so cummin follows time
df_m = df_m.sort_values(["seed", "epoch"])
df_p = df_p.sort_values(["seed", "epoch"])

# Downsample per seed (keeps order); or keep your iloc if you prefer
reduced_df_m = df_m.groupby("seed", group_keys=False).apply(lambda g: g.iloc[::k]).copy()
reduced_df_p = df_p.groupby("seed", group_keys=False).apply(lambda g: g.iloc[::k]).copy()

# Running min per seed — assign back as a column (overwrite or new)
reduced_df_m["loss"] = reduced_df_m.groupby("seed")["loss"].cummin()
reduced_df_p["loss"] = reduced_df_p.groupby("seed")["loss"].cummin()

print(reduced_df_m)
print(reduced_df_p)
# Equivalent alternative using transform (also returns a Series aligned to the DF):
# reduced_df_m["loss"] = reduced_df_m.groupby("seed")["loss"].transform("cummin")
# reduced_df_p["loss"] = reduced_df_p.groupby("seed")["loss"].transform("cummin")

# If you want to combine for plotting:
df_all = pd.concat(
    [reduced_df_m.assign(module="Modulation Network"),
     reduced_df_p.assign(module="Plain Network")],
    ignore_index=True,
)

# k = 1000
# df_m = load_three_col_txt(path_mod, module="modulation")
# df_p = load_three_col_txt(path_plain, module="plain")
#
# reduced_df_m = df_m.iloc[::k]
# reduced_df_p = df_p.iloc[::k]
#
# cummin_df_m = reduced_df_m.groupby("seed")["loss"].cummin()
# cummin_df_p = reduced_df_p.groupby("seed")["loss"].cummin()
# df = pd.concat([reduced_df_m, reduced_df_p], ignore_index=True).sort_values(["seed", "epoch"]).reset_index(drop=True)

# df = pd.merge(
#     df_p, df_m,
#     on=["seed", "epoch"],
#     how="inner", # or "outer" if keys may be missing on one side
#     suffixes=("_plain", "_modulation") # becomes loss_plain, loss_modulation
# )

# If you later want them together:
# df_all = pd.concat([cummin_df_m, cummin_df_p], ignore_index=True)


print("\nMerged DataFrame:")
print(df_all)

#%%

# sns.set_theme(style="darkgrid")


# Plot the responses for different events and regions
# ax = sns.lineplot(x="epoch", y="loss",
#              style='module', data=df_all)
# ax.set(yscale="log")
# plt.savefig("loss/modulation_plain_full_loss.pdf")

#%%
# change k to the desired jumps on the epochs
# k = 1000
# reduced_df = df_all.iloc[::k]
# print(reduced_df)
#%%

# sns.set_theme(style="darkgrid")
#
#
# # Plot the responses for different events and regions
# ax = sns.lineplot(x="epoch", y="loss", estimator='mean',
#              hue="module", data=reduced_df)
# ax.set(yscale="log")
# plt.savefig('loss/loss_modulation_plain.pdf')
# # plt.show()



# new_df = df_all.copy()
# new_df["loss"] = new_df.groupby("seed")["loss"].cummin()


sns.set_theme(style="darkgrid")


# Plot the responses for different events and regions
axx = sns.lineplot(x="epoch", y="loss", estimator='mean',
             hue="module", data=df_all)
axx.set(yscale="log")
axx.legend(title=None)
plt.savefig('loss/reduced_cummin_loss_modulation_plain.pdf')
