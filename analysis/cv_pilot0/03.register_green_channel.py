#%% imports and definition
import os
import warnings

import dask.array as darr
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

IN_DPATH = "./intermediate/processed/green"
IN_SS_CSV = "./log/sessions.csv"
IN_DPAT = r".*\.nc$"
OUT_PATH = "./intermediate/cross_reg/green"
os.makedirs(OUT_PATH, exist_ok=True)


def set_window(wnd):
    return wnd == wnd.min()


#%% align videos
meta_df = pd.read_csv(IN_SS_CSV)
meta_df = meta_df[meta_df["analyze"]]
temps = []
As = []
for anm, anmdf in meta_df.groupby("animal"):
    A_ls = []
    temp_ls = []
    for ss in anmdf["name"]:
        try:
            cur_ds = xr.open_dataset(
                os.path.join(IN_DPATH, "{}-{}.nc".format(anm, ss)), chunks=-1
            )
        except FileNotFoundError:
            warnings.warn("Missing data for {} {}".format(anm, ss))
            continue
        A_ls.append(cur_ds["A"])
        temp_ls.append(cur_ds["max_proj"])
    temps.append(xr.concat(temp_ls, "session"))
    As.append(xr.concat(A_ls, "session"))
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
param_dist = 5
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
