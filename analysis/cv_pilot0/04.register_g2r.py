#%% imports and definitions
import os
import warnings

import holoviews as hv
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from minian.cross_registration import (
    calculate_centroid_distance,
    calculate_mapping,
    group_by_session,
)
from minian.visualization import centroid
from tqdm.auto import tqdm

from routine.alignment import apply_affine, est_affine
from routine.plotting import plot_overlap
from routine.utilities import df_set_metadata

IN_GREEN_PATH = "./intermediate/processed/green"
IN_RED_PATH = "./intermediate/processed/red"
IN_SS_FILE = "./log/sessions.csv"
IN_RED_MAP = "./intermediate/cross_reg/red/mappings_meta.pkl"
IN_GREEN_MAP = "./intermediate/cross_reg/green/mappings_meta.pkl"
PARAM_DIST_THRES = 10
OUT_PATH = "./intermediate/register_g2r"
FIG_PATH = "./figs/register_g2r/"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% load data and align
im_opts = {"xaxis": None, "yaxis": None}
fig_path = os.path.join(FIG_PATH, "alignment")
os.makedirs(fig_path, exist_ok=True)
ss_df = pd.read_csv(IN_SS_FILE)
ss_df = ss_df[ss_df["analyze"]]
map_ls = []
for anm, anm_df in ss_df.groupby("animal"):
    plt_dict = dict()
    for _, row in tqdm(anm_df.iterrows(), leave=False):
        ss = row["name"]
        try:
            green_ds = xr.open_dataset(
                os.path.join(IN_GREEN_PATH, "{}-{}.nc".format(anm, ss))
            ).squeeze()
            red_ds = xr.open_dataset(
                os.path.join(IN_RED_PATH, "{}-{}.nc".format(anm, ss))
            ).squeeze()
        except FileNotFoundError:
            warnings.warn("Missing data for {} {}".format(anm, ss))
            continue
        # alignment
        trans, _ = est_affine(green_ds["max_proj"], red_ds["max_proj"])
        A_red = red_ds["A"]
        A_green = green_ds["A"]
        A_green_trans = xr.apply_ufunc(
            apply_affine,
            green_ds["A"],
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            kwargs={"tx": trans},
        ).compute()
        # registration
        cent_red = centroid(A_red)
        cent_green = centroid(A_green_trans)
        cent_red["session"] = "red"
        cent_green["session"] = "green"
        cents = pd.concat([cent_red, cent_green], ignore_index=True)
        dist = calculate_centroid_distance(cents, index_dim=[])
        dist = dist[dist["variable", "distance"] < PARAM_DIST_THRES]
        dist = group_by_session(dist)
        mapping = calculate_mapping(dist)
        mapping.columns = mapping.columns.droplevel(0)
        mapping = (
            df_set_metadata(mapping, {"animal": anm, "session": ss})
            .rename(columns={"green": "uid_green", "red": "uid_red"})
            .drop(columns="group")
        )
        map_ls.append(mapping)
        # plotting
        plt_dict[(ss, "0.before_align")] = hv.RGB(
            plot_overlap(A_green.max("unit_id"), A_red.max("unit_id"))
        ).opts(**im_opts)
        plt_dict[(ss, "1.after_align")] = hv.RGB(
            plot_overlap(A_green_trans.max("unit_id"), A_red.max("unit_id"))
        ).opts(**im_opts)
        plt_dict[(ss, "2.registered")] = hv.RGB(
            plot_overlap(
                A_green_trans.sel(unit_id=mapping["uid_green"].values).max("unit_id"),
                A_red.sel(unit_id=mapping["uid_red"].values).max("unit_id"),
            )
        ).opts(**im_opts)
    ovlp_plot = hv.NdLayout(plt_dict, ["session", "reg"]).cols(6)
    hv.save(ovlp_plot, os.path.join(fig_path, "{}.html".format(anm)))
mapping = pd.concat(map_ls, ignore_index=True)
mapping.to_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"), index=False)

#%% compute green mapping based on red
map_red = pd.read_pickle(IN_RED_MAP).drop(columns=[("session", "rec4")])
map_green = pd.read_pickle(IN_GREEN_MAP).drop(columns=[("session", "rec4")])
map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv")).set_index(
    ["animal", "session", "uid_red"]
)["uid_green"]
# map_red = map_red[map_red["session"].notnull().all(axis="columns")]
map_green_reg = []
for _, row in map_red.iterrows():
    anm = row["meta", "animal"]
    row_ss = row["session"]
    row_new = row.copy()
    idxs = [(anm, s, u) for s, u in zip(row_ss.index, row_ss.values)]
    row_new.loc["session"] = map_g2r.reindex(idxs).fillna(-1).values
    row_new.loc[row.isnull()] = np.nan
    map_green_reg.append(row_new.to_frame().T)
map_green_reg = pd.concat(map_green_reg, ignore_index=True)
map_green_reg.to_pickle(os.path.join(OUT_PATH, "green_mapping_reg.pkl"))

#%% plot results
map_green_reg["meta", "method"] = "red/registered"
map_green["meta", "method"] = "green/raw"
map_green_reg["variable", "stable"] = (
    map_green_reg["session"].notnull().all(axis="columns")
)
map_green_reg["variable", "nactive"] = (map_green_reg["session"] >= 0).sum(
    axis="columns"
)
map_green_reg["variable", "pactive"] = map_green_reg[
    "variable", "nactive"
] / map_green_reg["session"].notnull().sum(axis="columns")
map_green["variable", "stable"] = True
map_green["variable", "nactive"] = map_green["session"].notnull().sum(axis="columns")
map_green["variable", "pactive"] = map_green["variable", "nactive"] / len(
    map_green["session"].columns
)
map_master = pd.concat([map_green_reg, map_green], ignore_index=True)
map_master.columns = map_master.columns.droplevel(0)
fig_nactive = px.histogram(
    map_master[map_master["stable"]],
    x="nactive",
    facet_col="animal",
    facet_row="method",
    range_x=[-0.5, 10.5],
    range_y=[0, 0.6],
    histnorm="probability",
    labels={"nactive": "Number of<br>Active Sessions"},
)
fig_pactive = px.histogram(
    map_master,
    x="pactive",
    facet_col="animal",
    facet_row="method",
    range_x=[-0.1, 1.1],
    range_y=[0, 0.6],
    histnorm="probability",
    labels={"pactive": "Probability of<br>Activation"},
)
fig_nactive.write_html(os.path.join(FIG_PATH, "nactive.html"))
fig_pactive.write_html(os.path.join(FIG_PATH, "pactive.html"))
