#%% import and def
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from routine.behavior import (
    code_direction,
    determine_trial,
    linearize_pos,
    map_ts,
    merge_ts,
)
from routine.utilities import df_set_metadata, norm

IN_DPATH = "./data"
IN_EZTRACK_RES = "./intermediate/behav/eztrack"
IN_GREEN_PS = "./intermediate/processed/green"
IN_SSMAP = "./log/sessions.csv"
PARAM_SMOOTH = 15
PARAM_DIFF = 0.1
FIG_PATH = "./figs/frame_label/"
OUT_FM_LABEL = "./intermediate/frame_label"

os.makedirs(OUT_FM_LABEL, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% load and align data
ssmap = pd.read_csv(IN_SSMAP)
ssmap = ssmap[ssmap["analyze"]]
behav_ls = []
for _, row in tqdm(list(ssmap.iterrows())):
    anm, ss, dpath = row["animal"], row["name"], row["data"]
    eztrack_df = pd.read_csv(os.path.join(IN_EZTRACK_RES, "{}-{}.csv".format(anm, ss)))
    behav = eztrack_df[["Frame", "X", "Y"]].rename(columns=lambda c: c.lower())
    ms_ts = pd.read_csv(
        os.path.join(IN_DPATH, dpath, "miniscope_side", "timeStamps.csv")
    )
    behav_ts = pd.read_csv(os.path.join(IN_DPATH, dpath, "behavcam", "timeStamps.csv"))
    ts_map = map_ts(merge_ts(ms_ts, behav_ts))
    behav = (
        ts_map.merge(behav, how="left", left_on="fmCam1", right_on="frame")
        .drop(columns=["frame", "fmCam1"])
        .rename(columns={"fmCam0": "frame"})
    )
    behav = df_set_metadata(behav, {"animal": anm, "session": ss})
    behav_ls.append(behav)
behav = pd.concat(behav_ls, ignore_index=True)
behav["linpos"] = norm(linearize_pos(behav[["x", "y"]].values)) * 99 + 1
behav["linpos_sign"] = behav.groupby(["animal", "session"])["linpos"].transform(
    code_direction, smooth=PARAM_SMOOTH, diff_thres=PARAM_DIFF
)
behav["trial"] = behav.groupby(["animal", "session"])["linpos"].transform(
    determine_trial, smooth=PARAM_SMOOTH
)
fm_lab = behav.set_index(["animal", "session", "frame"]).to_xarray()
fm_lab.to_netcdf(os.path.join(OUT_FM_LABEL, "fm_label.nc"))

#%% plot result
def plot_trial(fm, trial, **kwargs):
    ax = plt.gca()
    tidx = np.diff(trial, prepend=0) > 0
    for f in fm[tidx]:
        ax.axvline(f, linestyle=":", color="black", linewidth=1)


for anm, anm_df in behav.groupby("animal"):
    g = sns.FacetGrid(anm_df, col="session", col_wrap=5)
    g.map(plot_trial, "frame", "trial")
    g.map_dataframe(sns.lineplot, x="frame", y="linpos")
    fig = g.figure
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_PATH, "{}.svg".format(anm)), dpi=500, bbox_inches="tight"
    )
    plt.close(fig)
