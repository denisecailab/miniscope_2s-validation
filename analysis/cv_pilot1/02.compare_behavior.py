#%% definition and imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.express.colors import qualitative
from scipy.stats import ttest_ind

IN_BEHAV = "./intermediate/frame_label/behav.feat"
IN_PFD_BEHAV = "./data-pfd/behav.feather"
PARAM_SUB_ANM = [
    "ts44-3",
    "ts45-4",
    "ts46-1",
    "ts30-0",
    "ts31-0",
    "ts32-1",
    "ts32-3",
    "m09",
    "m10",
    "m11",
    "m12",
    "m15",
    "m16",
]  # only control animals from pfd are included
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
FIG_PATH = "./figs/behav_comparison"

os.makedirs(FIG_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)


def calculate_speed(df, h=6, scale=1, fps=30):
    df = df.sort_values("frame")
    spd = np.abs(np.gradient(df["linpos"], h)) * fps * scale
    return pd.DataFrame({"frame": df["frame"], "speed": spd}).set_index("frame")


#%% load data and calculate speeds and trials
behav = pd.read_feather(IN_BEHAV)
behav_pfd = pd.read_feather(IN_PFD_BEHAV).rename(
    columns={"X": "linpos", "fmCam1": "frame"}
)
scale = 103 / (behav["linpos"].max() - behav["linpos"].min())  # LT length 103cm
scale_pfd = 200 / (
    behav_pfd["linpos"].max() - behav_pfd["linpos"].min()
)  # pfd LT length 200cm
spd = (
    behav[behav["linpos_sign"].notnull()]
    .groupby(["animal", "session", "trial"], group_keys=True)
    .apply(calculate_speed, scale=scale)
    .reset_index()
)
spd_pfd = (
    behav_pfd[
        (behav_pfd["state"].isin(["run_left", "run_right"]))
        & (behav_pfd["linpos"].notnull())
    ]
    .groupby(["animal", "session", "trial"], group_keys=True)
    .apply(calculate_speed, scale=scale_pfd)
    .reset_index()
)
spd["group"] = "2s"
spd_pfd["group"] = "pfd"
spd = pd.concat([spd, spd_pfd], ignore_index=True)
spd_agg = spd.groupby(["group", "animal"])["speed"].quantile(0.75).reset_index()
ntrials = behav.groupby(["animal", "session"])["trial"].max().reset_index()

#%% plot speeds
lmap = {"2s": "Dual-channel\nMiniscope", "pfd": "Single-channel\nMiniscope"}
spd_agg_plt = spd_agg[spd_agg["animal"].isin(PARAM_SUB_ANM)].copy()
spd_agg_plt["group"] = spd_agg_plt["group"].map(lmap)
fig, ax = plt.subplots(figsize=(2.6, 2))
ax = sns.barplot(
    spd_agg_plt,
    x="group",
    y="speed",
    color=qualitative.D3[0],
    errorbar="se",
    errwidth=3,
    capsize=0.2,
    saturation=0.4,
    width=0.5,
)
ax = sns.swarmplot(
    spd_agg_plt,
    x="group",
    y="speed",
    color=qualitative.D3[0],
    linewidth=1,
    warn_thresh=0.8,
    alpha=0.8,
)
ax.set_xlabel("")
ax.set_ylabel("Running Speed (cm/s)", style="italic")
fig.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "comparison.svg"), bbox_inches="tight")

#%% stats info
spd_agg_sub = spd_agg[spd_agg["animal"].isin(PARAM_SUB_ANM)].copy()
print("speed t-test")
print(
    ttest_ind(
        spd_agg_sub[spd_agg_sub["group"] == "2s"]["speed"],
        spd_agg_sub[spd_agg_sub["group"] == "pfd"]["speed"],
    )
)
print("speeds")
print(spd_agg_sub.groupby("group")["speed"].agg(["mean", "sem"]).reset_index())
print("trials")
print(ntrials["trial"].agg(["mean", "sem"]))
