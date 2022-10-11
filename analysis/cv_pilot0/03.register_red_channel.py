#%% imports and definition
import os

import numpy as np
import pandas as pd
import xarray as xr
from minian.cross_registration import (
    calculate_centroid_distance,
    calculate_centroids,
    calculate_mapping,
    fill_mapping,
    group_by_session,
    resolve_mapping,
)
from minian.motion_correction import apply_transform, estimate_motion
from minian.utilities import open_minian, open_minian_mf

IN_DPATH = "./intermediate/processed/red"
IN_SS_CSV = "./log/sessions.csv"
IN_DPAT = r".*\.nc$"
OUT_PATH = "./intermediate/cross_reg/red"
os.makedirs(OUT_PATH, exist_ok=True)


def set_window(wnd):
    return wnd == wnd.min()


#%% align videos
meta_df = pd.read_csv(IN_SS_CSV)
meta_df = meta_df[meta_df["analyze"]]
temps = []
As = []
for anm, anmdf in meta_df.groupby("animal"):
    anm_ds = [
        xr.open_dataset(os.path.join(IN_DPATH, "{}-{}.nc".format(anm, ss)), chunks=-1)
        for ss in anmdf["name"]
    ]
    temps.append(xr.concat([ds["max_proj"] for ds in anm_ds], "session"))
    As.append(xr.concat([ds["A"] for ds in anm_ds], "session"))
temps = xr.concat(temps, "animal")
As = xr.concat(As, "animal")

#%%
shifts = estimate_motion(temps, dim="session").compute().rename("shifts")
temps_sh = apply_transform(temps, shifts, fill=np.nan).compute().rename("temps_shifted")
shiftds = xr.merge([temps, shifts, temps_sh])
window = shiftds["temps_shifted"].isnull().sum("session")
window, _ = xr.broadcast(window, shiftds["temps_shifted"])
window = xr.apply_ufunc(
    set_window,
    window,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
)
A_shifted = apply_transform(As, shiftds["shifts"])

#%%
param_dist = 10
cents = calculate_centroids(A_shifted, window)
dist = calculate_centroid_distance(cents, index_dim=["animal"])
dist_ft = dist[dist["variable", "distance"] < param_dist].copy()
dist_ft = group_by_session(dist_ft)
mappings = calculate_mapping(dist_ft)
mappings_meta = resolve_mapping(mappings)
mappings_meta_fill = fill_mapping(mappings_meta, cents)

#%%
shiftds.to_netcdf(os.path.join(OUT_PATH, "shiftds.nc"))
cents.to_pickle(os.path.join(OUT_PATH, "cents.pkl"))
dist.to_pickle(os.path.join(OUT_PATH, "dist.pkl"))
mappings.to_pickle(os.path.join(OUT_PATH, "mappings.pkl"))
mappings_meta.to_pickle(os.path.join(OUT_PATH, "mappings_meta.pkl"))
mappings_meta_fill.to_pickle(os.path.join(OUT_PATH, "mappings_meta_fill.pkl"))
