#%% imports and definition
import itertools as itt
import os

import holoviews as hv
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from bokeh.palettes import Category20
from minian.cross_registration import (
    calculate_centroid_distance,
    calculate_centroids,
    calculate_mapping,
    fill_mapping,
    group_by_session,
    resolve_mapping,
)
from minian.motion_correction import apply_transform, estimate_motion
from natsort import natsorted
from tqdm.auto import tqdm

from routine.plotting import plotA_contour

hv.notebook_extension("bokeh")

IN_DPATH = "./intermediate/processed/red"
IN_SS_CSV = "./log/sessions.csv"
IN_DPAT = r".*\.nc$"
PARAM_DIST = 5
OUT_PATH = "./intermediate/cross_reg/red"
FIG_PATH = "./figs/cross_reg/red"
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


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
shifts = estimate_motion(temps, dim="session").compute().rename("shifts")
temps_sh = apply_transform(temps, shifts, fill=np.nan).compute().rename("temps_shifted")
window = temps_sh.isnull().sum("session")
window, _ = xr.broadcast(window, temps_sh)
window = xr.apply_ufunc(
    set_window,
    window,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
).rename("window")
A_shifted = apply_transform(As, shifts).rename("A_sh")
shiftds = xr.merge([A_shifted, temps, shifts, temps_sh, window])
shiftds.to_netcdf(os.path.join(OUT_PATH, "shiftds.nc"))

#%% calculate mappings
shift_ds = xr.open_dataset(os.path.join(OUT_PATH, "shiftds.nc"))
A_shifted = shift_ds["A_sh"]
window = shift_ds["window"]
cents = calculate_centroids(A_shifted, window)
dist = calculate_centroid_distance(cents, index_dim=["animal"])
dist_ft = dist[dist["variable", "distance"] < PARAM_DIST].copy()
dist_ft = group_by_session(dist_ft)
mappings = calculate_mapping(dist_ft)
mappings_meta = resolve_mapping(mappings)
mappings_meta_fill = fill_mapping(mappings_meta, cents)

#%% save results
cents.to_pickle(os.path.join(OUT_PATH, "cents.pkl"))
dist.to_pickle(os.path.join(OUT_PATH, "dist.pkl"))
mappings.to_pickle(os.path.join(OUT_PATH, "mappings.pkl"))
mappings_meta.to_pickle(os.path.join(OUT_PATH, "mappings_meta.pkl"))
mappings_meta_fill.to_pickle(os.path.join(OUT_PATH, "mappings_meta_fill.pkl"))

#%% plot registration
fig_path = os.path.join(FIG_PATH, "cells")
os.makedirs(fig_path, exist_ok=True)
shiftds = xr.open_dataset(os.path.join(OUT_PATH, "shiftds.nc"))
A_sh = shiftds["A_sh"]
projs = shiftds["temps_shifted"]
mappings_meta = pd.read_pickle(os.path.join(OUT_PATH, "mappings_meta.pkl"))
im_opts = {"cmap": "gray", "frame_width": 400, "frame_height": 400}
for anm, anm_df in tqdm(list(mappings_meta.groupby(("meta", "animal")))):
    plt_dict = dict()
    cmap = itt.cycle(Category20[20])
    anm_df = anm_df.dropna(axis="columns", how="all").copy()
    anm_df["variable", "color"] = [next(cmap) for _ in range(len(anm_df))]
    for ss in tqdm(natsorted(anm_df["session"]), leave=False):
        curA = A_sh.sel(animal=anm, session=ss).dropna("unit_id").compute()
        cur_proj = projs.sel(animal=anm, session=ss)
        uids_all = np.array(curA.coords["unit_id"])
        idx_ma = anm_df["session", ss].dropna()
        idx_nm = np.array(list(set(uids_all) - set(np.array(idx_ma))))
        im_ma = plotA_contour(
            curA.sel(unit_id=np.array(idx_ma)),
            cur_proj,
            cmap=anm_df.loc[idx_ma.index]
            .astype({("session", ss): int})
            .set_index(("session", ss))[("variable", "color")]
            .to_dict(),
            im_opts=im_opts,
        )
        im_nm = plotA_contour(curA.sel(unit_id=idx_nm), cur_proj, im_opts=im_opts)
        plt_dict[(ss, "match")] = im_ma
        plt_dict[(ss, "mismatch")] = im_nm
    cur_plt = hv.NdLayout(plt_dict, ["session", "ma"]).cols(4)
    hv.save(cur_plt, os.path.join(fig_path, "{}.html".format(anm)))

#%% plot summary
mappings_meta_fill = pd.read_pickle(os.path.join(OUT_PATH, "mappings_meta_fill.pkl"))
mappings_meta_fill.columns = mappings_meta_fill.columns.droplevel(0)
mappings_meta_fill["grp_len"] = mappings_meta_fill["group"].map(len)
mappings_meta_fill = (
    mappings_meta_fill.groupby(["animal", "grp_len"])["group"].count().reset_index()
)
fig = px.bar(mappings_meta_fill, x="grp_len", y="group", facet_col="animal")
fig.write_html(os.path.join(FIG_PATH, "summary.html"))
