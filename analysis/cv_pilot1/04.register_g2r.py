#%% imports and definitions
import os
import warnings

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
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

from plotly.express.colors import qualitative

IN_GREEN_PATH = "./intermediate/processed/green"
IN_RED_PATH = "./intermediate/processed/red"
IN_SS_FILE = "./log/sessions.csv"
IN_RED_MAP = "./intermediate/cross_reg/red/mappings_meta_fill.pkl"
IN_GREEN_MAP = "./intermediate/cross_reg/green/mappings_meta_fill.pkl"
PARAM_DIST_THRES = 8
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
    "svg.fonttype": "none",
}
OUT_PATH = "./intermediate/register_g2r"
FIG_PATH = "./figs/register_g2r/"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)

#%% load data and align
red_path = os.path.join(OUT_PATH, "Ared")
gn_path = os.path.join(OUT_PATH, "Agn")
gn_trans_path = os.path.join(OUT_PATH, "Agn_trans")
os.makedirs(red_path, exist_ok=True)
os.makedirs(gn_path, exist_ok=True)
os.makedirs(gn_trans_path, exist_ok=True)
ss_df = pd.read_csv(IN_SS_FILE)
ss_df = ss_df[ss_df["analyze"]]
map_ls = []
for anm, anm_df in tqdm(list(ss_df.groupby("animal"))):
    plt_algn_dict = dict()
    for _, row in tqdm(list(anm_df.iterrows()), leave=False):
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
        A_red = red_ds["A"].compute()
        A_green = green_ds["A"].compute()
        trans, _ = est_affine(A_green.max("unit_id"), A_red.max("unit_id"))
        A_green_trans = xr.apply_ufunc(
            apply_affine,
            green_ds["A"],
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            kwargs={"tx": trans},
        ).compute()
        A_red.to_netcdf(os.path.join(red_path, "{}-{}.nc".format(anm, ss)))
        A_green.to_netcdf(os.path.join(gn_path, "{}-{}.nc".format(anm, ss)))
        A_green_trans.to_netcdf(os.path.join(gn_trans_path, "{}-{}.nc".format(anm, ss)))
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
mapping = pd.concat(map_ls, ignore_index=True)
mapping.to_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"), index=False)

#%% compute green mapping based on red
map_red = pd.read_pickle(IN_RED_MAP)
map_green = pd.read_pickle(IN_GREEN_MAP)
map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv")).set_index(
    ["animal", "session", "uid_red"]
)["uid_green"]
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

#%% plot cells
im_opts = {"xaxis": None, "yaxis": None}
red_path = os.path.join(OUT_PATH, "Ared")
gn_path = os.path.join(OUT_PATH, "Agn")
gn_trans_path = os.path.join(OUT_PATH, "Agn_trans")
fig_algn_path = os.path.join(FIG_PATH, "alignment")
fig_cells_path = os.path.join(FIG_PATH, "cells")
os.makedirs(fig_algn_path, exist_ok=True)
os.makedirs(fig_cells_path, exist_ok=True)
map_red = pd.read_pickle(IN_RED_MAP)
map_green = pd.read_pickle(IN_GREEN_MAP)
map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"))
map_green_reg = pd.read_pickle(os.path.join(OUT_PATH, "green_mapping_reg.pkl"))
map_red = map_red[map_red["session"].notnull().sum(axis="columns") > 1].copy()
map_green = map_green[map_green["session"].notnull().sum(axis="columns") > 1].copy()
map_green_reg = map_green_reg[
    map_green_reg["session"].notnull().sum(axis="columns") > 1
].copy()
cells_im = []
for anm, anm_df in tqdm(list(map_g2r.groupby("animal"))):
    plt_algn_dict = dict()
    plt_cells_dict = dict()
    for ss, mapping in tqdm(list(anm_df.groupby("session")), leave=False):
        dsname = "{}-{}.nc".format(anm, ss)
        A_red = xr.open_dataarray(os.path.join(red_path, dsname)).compute()
        A_green = xr.open_dataarray(os.path.join(gn_path, dsname)).compute()
        A_green_trans = xr.open_dataarray(os.path.join(gn_trans_path, dsname)).compute()
        plt_algn_dict[(ss, "0.before_align")] = hv.RGB(
            plot_overlap(A_green.max("unit_id"), A_red.max("unit_id"))
        ).opts(**im_opts)
        plt_algn_dict[(ss, "1.after_align")] = hv.RGB(
            plot_overlap(A_green_trans.max("unit_id"), A_red.max("unit_id"))
        ).opts(**im_opts)
        plt_algn_dict[(ss, "2.registered")] = hv.RGB(
            plot_overlap(
                A_green_trans.sel(unit_id=mapping["uid_green"].values).max("unit_id"),
                A_red.sel(unit_id=mapping["uid_red"].values).max("unit_id"),
            )
        ).opts(**im_opts)
        idx_red = map_red[map_red["meta", "animal"] == anm]["session", ss].dropna()
        idx_green = map_green_reg[map_green_reg["meta", "animal"] == anm][
            "session", ss
        ].dropna()
        idx_green = idx_green[idx_green > 0]
        if len(idx_green) > 0:
            ag = A_green_trans.sel(unit_id=np.array(idx_green)).max("unit_id")
        else:
            h, w = np.array(A_green_trans.coords["height"]), np.array(
                A_green_trans.coords["width"]
            )
            ag = xr.DataArray(
                np.zeros((len(h), len(w))),
                dims=["height", "width"],
                coords={"height": h, "width": w},
            )
        c_ovly, c_gn, c_red = plot_overlap(
            ag,
            A_red.sel(unit_id=np.array(idx_red)).max("unit_id"),
            return_raw=True,
        )
        plt_cells_dict[(ss, "0.red")] = hv.RGB(c_red).opts(**im_opts)
        plt_cells_dict[(ss, "1.green")] = hv.RGB(c_gn).opts(**im_opts)
        plt_cells_dict[(ss, "2.overlay")] = hv.RGB(c_ovly).opts(**im_opts)
        cells_im.append(
            pd.DataFrame(
                {
                    "animal": anm,
                    "session": ss,
                    "kind": ["red", "green", "ovly"],
                    "im": [c_red, c_gn, c_ovly],
                }
            )
        )
    ovlp_plot = hv.NdLayout(plt_algn_dict, ["session", "reg"]).cols(6)
    cells_plot = hv.NdLayout(plt_cells_dict, ["session", "kind"]).cols(6)
    hv.save(ovlp_plot, os.path.join(fig_algn_path, "{}.html".format(anm)))
    hv.save(cells_plot, os.path.join(fig_cells_path, "{}.html".format(anm)))
cells_im = pd.concat(cells_im, ignore_index=True)
cells_im.to_pickle(os.path.join(OUT_PATH, "cells_im.pkl"))

#%% generate cells im figure
def plot_cells(x, gain=1.8, **kwargs):
    ax = plt.gca()
    im = x.values[0]
    im[:, :, :3] = (im[:, :, :3] * gain).clip(0, 1)
    ax.imshow(im)
    ax.set_axis_off()


fig_cells_path = os.path.join(FIG_PATH, "cells")
os.makedirs(fig_cells_path, exist_ok=True)
cells_im = pd.read_pickle(os.path.join(OUT_PATH, "cells_im.pkl"))
cells_im = cells_im[cells_im["session"] == "rec3"].copy()
cells_im["kind"] = cells_im["kind"].map(
    {"red": "tdTomato", "green": "GCaMP", "ovly": "Overlay"}
)
for anm, anm_df in cells_im.groupby("animal"):
    g = sns.FacetGrid(anm_df, col="kind", margin_titles=True, height=2)
    g.map(plot_cells, "im")
    g.set_titles(col_template="{col_name}")
    fig = g.fig
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_cells_path, "{}.svg".format(anm)), dpi=500, bbox_inches="tight"
    )
    plt.close(fig)

#%% plot results
def find_active(df):
    df["variable", "nactive"] = (df["session"] >= 0).sum(axis="columns")
    df["variable", "npresent"] = df["session"].notnull().sum(axis="columns")
    nsess = df["variable", "npresent"].max()
    df["variable", "nsess"] = nsess
    return df


map_red = pd.read_pickle(IN_RED_MAP)
map_green = pd.read_pickle(IN_GREEN_MAP)
map_green_reg = pd.read_pickle(os.path.join(OUT_PATH, "green_mapping_reg.pkl"))
map_red["meta", "method"] = "red/raw"
map_green["meta", "method"] = "green/raw"
map_green_reg["meta", "method"] = "red/registered"
map_red = map_red.groupby(("meta", "animal")).apply(find_active)
map_green = map_green.groupby(("meta", "animal")).apply(find_active)
map_green_reg = map_green_reg.groupby(("meta", "animal")).apply(find_active)
map_red["variable", "stable"] = True
map_green["variable", "stable"] = True
map_green_reg["variable", "stable"] = (
    map_green_reg["variable", "npresent"] == map_green_reg["variable", "nsess"]
)
map_red["variable", "pactive"] = (
    map_red["variable", "nactive"] / map_red["variable", "nsess"]
)
map_green["variable", "pactive"] = (
    map_green["variable", "nactive"] / map_green["variable", "nsess"]
)
map_green_reg["variable", "pactive"] = (
    map_green_reg["variable", "nactive"] / map_green_reg["variable", "npresent"]
)
map_master = pd.concat([map_green_reg, map_green, map_red], ignore_index=True)
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

#%% plot summary
def find_active(df):
    df["variable", "nactive"] = (df["session"] >= 0).sum(axis="columns")
    df["variable", "npresent"] = df["session"].notnull().sum(axis="columns")
    nsess = df["variable", "npresent"].max()
    df["variable", "nsess"] = nsess
    return df


def agg_pactive(df):
    df_agg = df.groupby("pactive").size().reset_index(name="counts")
    df_agg["density"] = df_agg["counts"] / df_agg["counts"].sum()
    return df_agg.set_index("pactive")


def agg_per_ss(df):
    ret_df = []
    for ss in df["session"]:
        cur_df = df[df[("session", ss)].notnull()].copy()
        cur_df.columns = cur_df.columns.droplevel(0)
        cur_agg = cur_df.groupby("nactive").size().reset_index(name="counts")
        cur_agg["density"] = cur_agg["counts"] / cur_agg["counts"].sum()
        cur_agg["session"] = ss
        ret_df.append(cur_agg)
    return pd.concat(ret_df, ignore_index=True).set_index("nactive")


def hist_wrap(data, x, **kwargs):
    data = data[data[x].notnull()].copy()
    if len(data) > 0:
        ax = plt.gca()
        sns.histplot(data=data, x=x, ax=ax, **kwargs)


def bar_wrap(data, x, **kwargs):
    data = data[data[x].notnull()].copy()
    if len(data) > 0:
        ax = plt.gca()
        data[x] = data[x].astype(int)
        sns.barplot(data=data, x=x, ax=ax, **kwargs)


def swarm_wrap(data, x, **kwargs):
    data = data[data[x].notnull()].copy()
    if len(data) > 0:
        ax = plt.gca()
        sns.swarmplot(data=data, x=x, ax=ax, **kwargs)


map_red = pd.read_pickle(IN_RED_MAP)
map_green = pd.read_pickle(IN_GREEN_MAP)
map_green_reg = pd.read_pickle(os.path.join(OUT_PATH, "green_mapping_reg.pkl"))
map_red = map_red.groupby(("meta", "animal")).apply(find_active)
map_green = map_green.groupby(("meta", "animal")).apply(find_active)
map_green_reg = map_green_reg.groupby(("meta", "animal")).apply(find_active)
red_agg = (
    map_red.groupby(("meta", "animal"))
    .apply(agg_per_ss)
    .reset_index()
    .rename(columns={("meta", "animal"): "animal"})
)
green_agg = (
    map_green.groupby(("meta", "animal"))
    .apply(agg_per_ss)
    .reset_index()
    .rename(columns={("meta", "animal"): "animal"})
)
map_green_reg["variable", "pactive"] = (
    map_green_reg["variable", "nactive"] / map_green_reg["variable", "npresent"]
)
map_green_reg.columns = map_green_reg.columns.droplevel(0)
green_reg_agg = map_green_reg.groupby("animal").apply(agg_pactive).reset_index()
red_agg["method"] = "tdTomato"
green_agg["method"] = "GCaMP"
green_reg_agg["method"] = "GCaMP cells registered\nwith tdTomato"
cmap = {
    "tdTomato": qualitative.Plotly[1],
    "GCaMP": qualitative.Plotly[2],
    "GCaMP cells registered\nwith tdTomato": qualitative.Plotly[4],
}
agg_df = pd.concat([red_agg, green_agg], ignore_index=True)
agg_df = agg_df[agg_df["animal"] == "m16"]
g = sns.FacetGrid(
    agg_df,
    col="method",
    margin_titles=True,
    sharex="row",
    sharey=True,
    height=2.2,
    aspect=0.8,
)
g.set_xlabels(clear_inner=False)
g.map_dataframe(
    bar_wrap,
    x="nactive",
    y="density",
    errorbar="se",
    saturation=0.8,
    errwidth=1.5,
    capsize=0.3,
    hue="method",
    palette=cmap,
)
g.map_dataframe(
    swarm_wrap,
    x="nactive",
    y="density",
    hue="method",
    palette=cmap,
    edgecolor="gray",
    linewidth=1,
    size=3,
)
g.set_titles(col_template="{col_name}")
g.set_xlabels("# of sessions active", style="italic")
g.set_ylabels("Proportion of cells", style="italic")
g.fig.savefig(os.path.join(FIG_PATH, "summary.svg"), dpi=500, bbox_inches="tight")
plt.close(g.fig)
