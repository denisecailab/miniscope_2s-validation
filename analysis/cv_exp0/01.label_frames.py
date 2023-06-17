# %% import and def
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from routine.behavior import (
    agg_behav,
    code_direction,
    determine_trial,
    extract_location,
    label_ts,
    linearize_pos,
)
from routine.plotting import plot_behav
from routine.utilities import df_set_metadata, norm
from tqdm.auto import tqdm

IN_DPATH = "./data"
IN_SSMAP = "./log/sessions.csv"
PARAM_SMOOTH = 15
PARAM_DIFF = {(0, 3): 0.1, (3, 97): 0.02, (97, 100): 0.1}
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
FIG_PATH = "./figs/frame_label/"
OUT_FM_LABEL = "./intermediate/frame_label"

os.makedirs(OUT_FM_LABEL, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)

# %% load and align data
ssmap = pd.read_csv(IN_SSMAP)
ssmap = ssmap[ssmap["analyze"]]
behav_ls = []
for _, row in tqdm(list(ssmap.iterrows())):
    # load data
    anm, ss, dpath = row["animal"], row["session"], row["data"]
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
        behav.join(loc)
        .groupby("ms_frame")
        .apply(agg_behav)
        .reset_index()
        .astype({"ms_frame": int})
        .interpolate()
        .ffill()
        .bfill()
    )
    behav["linpos"] = norm(linearize_pos(behav[["x", "y"]].values)) * 99 + 1
    behav["linpos_sign"] = code_direction(
        behav["linpos"], smooth=PARAM_SMOOTH, diff_thres=PARAM_DIFF
    )
    behav = determine_trial(behav).astype({"trial": int})
    behav = df_set_metadata(
        behav[["ms_frame", "x", "y", "trial", "linpos", "linpos_sign"]].copy(),
        {"animal": anm, "session": ss},
    ).rename(columns={"ms_frame": "frame"})
    behav_ls.append(behav)
behav = pd.concat(behav_ls, ignore_index=True)
behav.to_feather(os.path.join(OUT_FM_LABEL, "behav.feat"))
fm_lab = behav.set_index(["animal", "session", "frame"]).to_xarray()
fm_lab.to_netcdf(os.path.join(OUT_FM_LABEL, "fm_label.nc"))

# %% plot result
behav = pd.read_feather(os.path.join(OUT_FM_LABEL, "behav.feat"))
for anm, anm_df in behav.groupby("animal"):
    g = sns.FacetGrid(anm_df, col="session", col_wrap=3, aspect=2, legend_out=True)
    g.map_dataframe(plot_behav)
    g.add_legend()
    fig = g.figure
    fig.savefig(
        os.path.join(FIG_PATH, "{}.svg".format(anm)), dpi=500, bbox_inches="tight"
    )
    plt.close(fig)

# %% plot specific animals
# behav = pd.read_feather(os.path.join(OUT_FM_LABEL, "behav.feat"))
# behav = behav[
#     (behav["session"] == "rec6") & (behav["animal"].isin(["m09", "m12"]))
# ].copy()
# behav["time"] = behav["frame"] / 30
# behav = behav[behav["time"] <= 600].copy()
# g = sns.FacetGrid(behav, row="animal", legend_out=False, height=1.8, aspect=2.5)
# g.map_dataframe(plot_behav, fm="time", lw=1.2, lw_dash=0.8)
# fig = g.figure
# g.add_legend(
#     loc="lower center",
#     bbox_to_anchor=(0.1, 0.96, 0.8, 0.1),
#     mode="expand",
#     ncol=3,
#     bbox_transform=fig.transFigure,
# )
# g.set_xlabels("Time (s)", style="italic")
# g.set_ylabels("Linearized Position", style="italic")
# g.set_titles(row_template="Animal: {row_name}")
# fig.savefig(os.path.join(FIG_PATH, "example.svg"), dpi=500, bbox_inches="tight")
# plt.close(fig)
