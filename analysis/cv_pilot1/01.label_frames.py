#%% import and def
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from routine.behavior import (
    agg_behav,
    code_direction,
    determine_trial,
    extract_location,
    label_ts,
    linearize_pos,
)
from routine.utilities import df_set_metadata, norm, unique_seg

IN_DPATH = "./data"
IN_SSMAP = "./log/sessions.csv"
PARAM_SMOOTH = 15
PARAM_DIFF = {(0, 3): 0.1, (3, 97): 0.02, (97, 100): 0.1}
FIG_PATH = "./figs/frame_label/"
OUT_FM_LABEL = "./intermediate/frame_label"

os.makedirs(OUT_FM_LABEL, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% load and align data
ssmap = pd.read_csv(IN_SSMAP)
ssmap = ssmap[ssmap["analyze"]]
behav_ls = []
for _, row in tqdm(list(ssmap.iterrows())):
    # load data
    anm, ss, dpath = row["animal"], row["name"], row["data"]
    behav = pd.read_csv(os.path.join(IN_DPATH, dpath, "linear_track.csv")).astype(
        {"timestamp": float}
    )
    ms_ts = pd.read_csv(
        os.path.join(IN_DPATH, dpath, "miniscope_side", "timeStamps.csv")
    ).rename(columns={"Time Stamp (ms)": "timestamp", "Frame Number": "frame"})
    # align_ts and extract locations
    behav = label_ts(behav, ms_ts)
    loc = behav.apply(extract_location, axis="columns").astype(float).interpolate()
    behav = (
        determine_trial(behav.join(loc))
        .groupby("ms_frame")
        .apply(agg_behav)
        .reset_index()
        .astype({"ms_frame": int})
        .interpolate()
        .ffill()
        .bfill()
        .astype({"trial": int})
    )
    behav = df_set_metadata(
        behav[["ms_frame", "x", "y", "trial"]], {"animal": anm, "session": ss}
    ).rename(columns={"ms_frame": "frame"})
    behav_ls.append(behav)
behav = pd.concat(behav_ls, ignore_index=True)
behav["linpos"] = norm(linearize_pos(behav[["x", "y"]].values)) * 99 + 1
behav["linpos_sign"] = behav.groupby(["animal", "session"])["linpos"].transform(
    code_direction, smooth=PARAM_SMOOTH, diff_thres=PARAM_DIFF
)
behav.to_feather(os.path.join(OUT_FM_LABEL, "behav.feat"))
fm_lab = behav.set_index(["animal", "session", "frame"]).to_xarray()
fm_lab.to_netcdf(os.path.join(OUT_FM_LABEL, "fm_label.nc"))

#%% plot result
def plot_behav(data, fm, pos, pos_sgn, trial, **kwargs):
    data = data.copy()
    ax = plt.gca()
    # plot trial
    tidx = np.diff(data[trial], prepend=0) > 0
    for f in data[fm][tidx]:
        ax.axvline(f, linestyle=":", color="black", linewidth=0.5)
    # plot position
    seg_cmap = kwargs.get("seg_cmap", {0: "gray", 1: "blue", -1: "orange"})
    seg_lmap = kwargs.get(
        "seg_lmap", {0: "idle", 1: "running right", -1: "running left"}
    )
    data["seg_sign"] = np.sign(data[pos_sgn]).fillna(0)
    data["pos_seg"] = unique_seg(data["seg_sign"])
    for _, seg in data.groupby("pos_seg"):
        s = seg["seg_sign"].unique().item()
        sns.lineplot(
            seg, x=fm, y=pos, color=seg_cmap[s], label=seg_lmap[s], ax=ax, linewidth=0.5
        )


behav = pd.read_feather(os.path.join(OUT_FM_LABEL, "behav.feat"))
for anm, anm_df in behav.groupby("animal"):
    g = sns.FacetGrid(anm_df, col="session", col_wrap=3, aspect=2, legend_out=True)
    g.map_dataframe(
        plot_behav, fm="frame", pos="linpos", pos_sgn="linpos_sign", trial="trial"
    )
    g.add_legend()
    fig = g.figure
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_PATH, "{}.svg".format(anm)), dpi=500, bbox_inches="tight"
    )
    plt.close(fig)
