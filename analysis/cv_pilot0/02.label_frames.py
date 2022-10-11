#%% import and def
import os

import pandas as pd
from tqdm.auto import tqdm

from routine.behavior import code_direction, linearize_pos, map_ts, merge_ts
from routine.utilities import df_set_metadata, norm

IN_DPATH = "./data"
IN_EZTRACK_RES = "./intermediate/behav/eztrack"
IN_GREEN_PS = "./intermediate/processed/green"
IN_SSMAP = "./log/sessions.csv"
PARAM_SMOOTH = 15
PARAM_DIFF = 0.1
OUT_FM_LABEL = "./intermediate/frame_label"

os.makedirs(OUT_FM_LABEL, exist_ok=True)

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
fm_lab = behav.set_index(["animal", "session", "frame"]).to_xarray()
fm_lab.to_netcdf(os.path.join(OUT_FM_LABEL, "fm_label.nc"))

#%%
import holoviews as hv

hv.notebook_extension("bokeh")
hv.Curve(behav["linpos"]) + hv.Curve(behav["linpos_sign"])
