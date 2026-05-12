import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------------------
# 1. Load all files and parse metadata
# ----------------------------------------

# Adjust pattern if needed, or use "losses/*.txt" if in subfolder
paths = glob.glob("loss/*.txt")

# Filenames like:
#   Modulation_losses_seed_0_units1000.txt
#   Plain_losses_seed_0_units1500.txt
pattern = re.compile(
    r"(Modulation|Plain)_losses_seed_(\d+)_units(\d+)\.txt",
    re.IGNORECASE,
)

rows = []

for path in paths:
    fname = os.path.basename(path)
    m = pattern.match(fname)
    if not m:
        print("Skipping file that doesn't match pattern:", fname)
        continue

    arch_str, seed_str, units_str = m.groups()
    network = "Modulation" if arch_str.lower().startswith("modulation") else "Plain"
    seed = int(seed_str)
    units = int(units_str)

    # Each file has header: "epoch  loss"
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=0,
        engine="python",
    )

    # Make sure epoch and loss are numeric
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["epoch", "loss"])

    df["network"] = network
    df["units"] = units
    df["seed"] = seed

    rows.append(df)

if not rows:
    raise RuntimeError("No files matched the pattern. Check your filenames/pattern.")

all_df = pd.concat(rows, ignore_index=True)
print("Loaded shape:", all_df.shape)
print(all_df.head())

# ----------------------------------------
# 2. Select the pairs: modulation N, plain 3/2 N
# ----------------------------------------

mod_target_units = [1000, 10000]  # modulation widths of interest

dfs_pairs = []  # we will collect only the matching pairs

for N_mod in mod_target_units:
    N_plain = int(1.5 * N_mod)  # 3/2 * N

    df_mod = all_df[(all_df["network"] == "Modulation") & (all_df["units"] == N_mod)]
    df_plain = all_df[(all_df["network"] == "Plain") & (all_df["units"] == N_plain)]

    if df_mod.empty:
        print(f"Warning: no Modulation data found for units={N_mod}")
    if df_plain.empty:
        print(f"Warning: no Plain data found for units={N_plain}")

    dfs_pairs.append(df_mod)
    dfs_pairs.append(df_plain)

pair_df = pd.concat(dfs_pairs, ignore_index=True)

# ----------------------------------------
# 3. Optional: downsample (e.g. every 100 epochs)
# ----------------------------------------

pair_df_ds = (
    pair_df
    .sort_values(["network", "units", "seed", "epoch"])
    .groupby(["network", "units", "seed"], as_index=False)
    .apply(lambda g: g.iloc[::100])   # keep every 100th point
    .reset_index(drop=True)
)

df = pair_df_ds   # use downsampled data
# df = pair_df    # uncomment this instead if you want full resolution

# ----------------------------------------
# 4. Plot: single panel, modulation vs plain, paired by N and 3/2 N
# ----------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

# linestyle by architecture
linestyles = {"Modulation": "-", "Plain": "--"}

# color per modulation N (so pair shares color)
colors = {
    1000: "tab:blue",
    10000: "tab:orange",
}

for N_mod in mod_target_units:
    N_plain = int(1.5 * N_mod)

    for network, units in [("Modulation", N_mod), ("Plain", N_plain)]:
        sub = df[(df["network"] == network) & (df["units"] == units)]
        if sub.empty:
            continue

        color = colors[N_mod]
        ls = linestyles[network]

        label = (
            f"Modulation, {N_mod} neurons"
            if network == "Modulation"
            else f"Plain, {N_plain} neurons"
        )

        ax.plot(
            sub["epoch"], sub["loss"],
            linestyle=ls,
            color=color,
            label=label,
        )

ax.set_yscale("log")
ax.set_ylabel("Loss")
# ax.set_title("Training loss vs epochs\nModulation N vs Plain 1.5×N neurons")

# ---- X-axis ticks in multiples of 10k ----

max_epoch = df["epoch"].max()
tick_step = 10000  # 10k
ticks = list(range(0, int(max_epoch) + tick_step, tick_step))
ax.set_xticks(ticks)

def k_formatter(x, pos):
    if x == 0:
        return "0"
    return f"{int(x/1000)}k"

ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_formatter))
ax.set_xlabel("Epochs")

ax.legend(title="Architecture and width", ncol=1)
plt.tight_layout()
plt.savefig('loss/expressivity_modulation_vs_plain.pdf')
plt.show()