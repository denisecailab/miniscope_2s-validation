# %% definition and imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from plotly.express.colors import qualitative
from scipy.stats import ttest_ind

IN_BEHAV = "./intermediate/frame_label/behav.feat"
IN_V4_BEHAV = "./intermediate/frame_label/behav_v4.feat"
PARAM_SUB_ANM = [
    "mc23",
    "mc26",
    "mc27",
    "mc28",
    "mc29",
    "m20",
    "m21",
    "m22",
    "m23",
    "m24",
    "m25",
    "m26",
    "m27",
    "m29",
    "m30",
]  # only control animals from szplace are included
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
PARAM_CMAP = {
    "Dual-channel\nMiniscope": qualitative.Plotly[5],
    "Single-channel\nMiniscope": qualitative.Plotly[8],
}
FIG_PATH = "./figs/behav_comparison"

os.makedirs(FIG_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)


def calculate_speed(df, scale=1, fps=30):
    if "frame" in df.columns:
        df = df.sort_values("frame")
        try:
            spd = np.abs(np.gradient(df["linpos"])) * fps * scale
        except ValueError:
            spd = np.nan
        return pd.DataFrame({"frame": df["frame"], "speed": spd}).set_index("frame")
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        spd = np.abs(df["linpos"].diff()) / df["timestamp"].diff() * scale
        return pd.DataFrame({"timestamp": df["timestamp"], "speed": spd}).set_index(
            "timestamp"
        )


def count_ss(ss):
    ss_vals = ss.unique()
    ss_vals.sort()
    ss_map = {s: i for i, s in enumerate(ss_vals)}
    return ss.map(ss_map)


# %% load data and calculate speeds and trials
behav = pd.read_feather(IN_BEHAV)
behav_v4 = pd.read_feather(IN_V4_BEHAV)
behav_v4["linpos"] = behav_v4["linpos"].abs()
# take only the last training day
behav_v4["ss_ct"] = behav_v4.groupby("animal", group_keys=False)["session"].apply(
    count_ss
)
behav_v4 = behav_v4[behav_v4["ss_ct"] == 4].copy()
scale = 103 / (behav["linpos"].max() - behav["linpos"].min())  # LT length 103cm
scale_v4 = 103 / (
    behav_v4["linpos"].max() - behav_v4["linpos"].min()
)  # MultiCon LT length 103cm
spd = (
    behav.groupby(["animal", "session", "trial"], group_keys=True)
    .apply(calculate_speed, scale=scale)
    .reset_index()
)
spd_v4 = (
    behav_v4.groupby(["animal", "session", "trial"], group_keys=True)
    .apply(calculate_speed, scale=scale_v4)
    .reset_index()
)
spd["group"] = "2s"
spd_v4["group"] = "v4"
spd = pd.concat([spd, spd_v4], ignore_index=True)
spd_agg = spd.groupby(["group", "animal"])["speed"].quantile(0.95).reset_index()
ntrials = behav.groupby(["animal", "session"])["trial"].max().reset_index()
spd_prt = spd_agg[spd_agg["group"] == "2s"]["speed"]
print("speed: {} +/- {}".format(spd_prt.mean(), spd_prt.sem()))
trial_prt = ntrials.groupby("animal")["trial"].mean()
print("trial: {} +/- {}".format(trial_prt.mean(), trial_prt.sem()))

# %% plot speeds
lmap = {"2s": "Dual-channel\nMiniscope", "v4": "Single-channel\nMiniscope"}
spd_agg_plt = spd_agg[spd_agg["animal"].isin(PARAM_SUB_ANM)].copy()
spd_agg_plt["group"] = spd_agg_plt["group"].map(lmap)
fig, ax = plt.subplots(figsize=(2.8, 2))
ax = sns.barplot(
    spd_agg_plt,
    x="group",
    y="speed",
    hue="group",
    palette=PARAM_CMAP,
    errorbar="se",
    err_kws={"linewidth": 3},
    capsize=0.2,
    saturation=0.9,
    alpha=0.75,
    width=0.5,
    legend=False,
)
ax = sns.swarmplot(
    spd_agg_plt,
    x="group",
    y="speed",
    hue="group",
    palette=PARAM_CMAP,
    linewidth=1.2,
    warn_thresh=0.8,
    edgecolor="gray",
    alpha=0.9,
    legend=False,
)
ax.set_xlabel("")
ax.set_ylabel("Running Speed (cm/s)", style="italic")
fig.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "comparison.svg"), bbox_inches="tight")

# %% stats info
spd_agg_sub = spd_agg[spd_agg["animal"].isin(PARAM_SUB_ANM)].copy()
print("speed t-test")
print(
    ttest_ind(
        spd_agg_sub[spd_agg_sub["group"] == "2s"]["speed"],
        spd_agg_sub[spd_agg_sub["group"] == "v4"]["speed"],
    )
)
print("speeds")
print(spd_agg_sub.groupby("group")["speed"].agg(["mean", "sem"]).reset_index())
print("trials")
print(ntrials["trial"].agg(["mean", "sem"]))


# %% plot example
def plot_exp(data, color=None, **kwargs):
    ax = sns.lineplot(data=data, **kwargs)
    ax.get_xaxis().set_minor_locator(AutoMinorLocator())
    ax.grid(which="minor", color="w", linewidth=0.5)
    return ax


sns.set_theme(context="paper", style="darkgrid")
behav = pd.read_feather(IN_BEHAV)
behav_v4 = pd.read_feather(IN_V4_BEHAV)
behav_v4["linpos"] = behav_v4["linpos"].abs()
scale = 103 / (behav["linpos"].max() - behav["linpos"].min())  # LT length 103cm
scale_v4 = 103 / (
    behav_v4["linpos"].max() - behav_v4["linpos"].min()
)  # MultiCon LT length 103cm
behav = behav.set_index(["animal", "session"]).loc[("m22", "rec5")].copy().reset_index()
behav_v4 = (
    behav_v4.set_index(["animal", "session"])
    .loc[("mc23", "2023_10_20")]
    .copy()
    .reset_index()
)
behav["location"] = behav["linpos"] * scale
behav_v4["location"] = behav_v4["linpos"] * scale_v4
behav["time"] = behav["frame"] / 30
behav_v4["time"] = behav_v4["timestamp"] - behav_v4["timestamp"].min()
behav["group"] = "Dual-channel\nMiniscope"
behav_v4["group"] = "Single-channel\nMiniscope"
behav_all = pd.concat([behav, behav_v4], ignore_index=True)
behav_all = behav_all[behav_all["time"].between(450, 650)].copy()
behav_all["time"] = behav_all["time"] - behav_all["time"].min()
g = sns.FacetGrid(
    behav_all,
    row="animal",
    legend_out=False,
    height=1.8,
    aspect=2.5,
)
g.map_dataframe(
    plot_exp, x="time", y="location", lw=1.5, hue="group", palette=PARAM_CMAP
)
fig = g.figure
ax = fig.axes[0]
g.add_legend(
    loc="lower center",
    bbox_to_anchor=(0.1, 1.2, 0.8, 0.15),
    mode="expand",
    ncol=2,
    bbox_transform=ax.transAxes,
    title="",
)
g.set_xlabels("Time (s)", style="italic")
g.set_ylabels("Position (cm)", style="italic")
g.set_titles(row_template="Animal: {row_name}")
fig.savefig(os.path.join(FIG_PATH, "example.svg"), dpi=500, bbox_inches="tight")
# plt.close(fig)
