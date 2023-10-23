# %% imports and definitions
import os
import warnings

import dask.array as darr
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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from plotly.express.colors import qualitative
from routine.alignment import apply_affine, est_affine
from routine.plotting import plot_overlap
from routine.utilities import df_set_metadata, norm
from scipy.stats import zscore
from tqdm.auto import tqdm

IN_GREEN_PATH = "./intermediate/processed/green"
IN_RED_PATH = "./intermediate/processed/red"
IN_SS_FILE = "./log/sessions.csv"
IN_RED_MAP = "./intermediate/cross_reg/red/mappings_meta_fill.pkl"
IN_GREEN_MAP = "./intermediate/cross_reg/green/mappings_meta_fill.pkl"
PARAM_DIST_THRES = 10
PARAM_SUB_SS = None
PARAM_SUB_ANM = ["m20", "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m29"]
PARAM_EXP_ANM = ["m20"]
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
OUT_PATH = "./intermediate/register_g2r"
FIG_PATH = "./figs/register_g2r/"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)

# %% load data and align
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
        ss = row["session"]
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
            A_green,
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

# %% compute green mapping based on red
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

# %% plot cells
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
if PARAM_SUB_SS is not None:
    map_g2r = map_g2r.loc[map_g2r["session"].isin(PARAM_SUB_SS)].copy()
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


# %% generate cells im figure
def plot_cells(x, gain=1.8, **kwargs):
    ax = plt.gca()
    im = x.values[0]
    im[:, :, :3] = (im[:, :, :3] * gain).clip(0, 1)
    ax.imshow(im)
    ax.set_axis_off()


ss_dict = {
    "rec0": "Day 1",
    "rec1": "Day 3",
    "rec2": "Day 5",
    "rec3": "Day 7",
    "rec4": "Day 9",
    "rec5": "Day 11",
    "rec6": "Day 13",
}
fig_cells_path = os.path.join(FIG_PATH, "cells")
os.makedirs(fig_cells_path, exist_ok=True)
cells_im = pd.read_pickle(os.path.join(OUT_PATH, "cells_im.pkl"))
cells_im["session"] = cells_im["session"].map(ss_dict)
cells_im["kind"] = cells_im["kind"].map(
    {"red": "tdTomato", "green": "GCaMP", "ovly": "Overlay"}
)
for anm, anm_df in cells_im.groupby("animal"):
    g = sns.FacetGrid(
        anm_df,
        row="kind",
        col="session",
        col_order=list(ss_dict.values()),
        margin_titles=True,
        height=1.8,
    )
    g.map(plot_cells, "im")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    fig = g.fig
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.004, hspace=0.04)
    fig.savefig(
        os.path.join(fig_cells_path, "{}.svg".format(anm)), dpi=500, bbox_inches="tight"
    )
    plt.close(fig)


# %% plot results
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


# %% plot summary
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


def subset_sessions(df, ss):
    if ss is not None:
        cols = list(filter(lambda c: c[0] != "session" or c[1] in ss, df.columns))
        return df[cols].copy()
    else:
        return df


map_red = subset_sessions(pd.read_pickle(IN_RED_MAP), PARAM_SUB_SS)
map_green = subset_sessions(pd.read_pickle(IN_GREEN_MAP), PARAM_SUB_SS)
map_green_reg = subset_sessions(
    pd.read_pickle(os.path.join(OUT_PATH, "green_mapping_reg.pkl")), PARAM_SUB_SS
)
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
reg_agg = (
    map_green_reg[map_green_reg["variable", "npresent"] == 7]
    .groupby(("meta", "animal"))
    .apply(agg_per_ss)
    .reset_index()
    .rename(columns={("meta", "animal"): "animal"})
)
map_green_reg["variable", "pactive"] = (
    map_green_reg["variable", "nactive"] / map_green_reg["variable", "npresent"]
)
map_green_reg.columns = map_green_reg.columns.droplevel(0)
map_green_reg = map_green_reg[
    (map_green_reg["npresent"] > 1) & (map_green_reg["nactive"] > 0)
].copy()
map_green_reg.to_csv(os.path.join(OUT_PATH, "green_reg_pactive.csv"), index=False)
green_reg_agg = map_green_reg.groupby("animal").apply(agg_pactive).reset_index()
red_agg["method"] = "tdTomato"
green_agg["method"] = "GCaMP"
reg_agg["method"] = "GCaMP cells\nregistered with tdTomato"
green_reg_agg["method"] = "GCaMP cells pactive"
cmap = {
    "tdTomato": qualitative.Plotly[1],
    "GCaMP": qualitative.Plotly[2],
    "GCaMP cells\nregistered with tdTomato": qualitative.Plotly[4],
    "GCaMP cells pactive": qualitative.Plotly[4],
}
agg_df = pd.concat([red_agg, green_agg, reg_agg, green_reg_agg], ignore_index=True)
sum_agg_df = pd.concat([red_agg, green_agg, reg_agg], ignore_index=True)
if PARAM_SUB_ANM is not None:
    sum_agg_df = sum_agg_df[sum_agg_df["animal"].isin(PARAM_SUB_ANM)]
sum_agg_df = (
    sum_agg_df.groupby(["method", "animal", "nactive"])["density"].mean().reset_index()
)
df_dict = {"summary": agg_df, "summary_agg": sum_agg_df}
for plt_type, cur_data in df_dict.items():
    if plt_type == "summary":
        plt_args = {"row": "method", "col": "animal", "sharex": False, "aspect": 0.8}
    elif plt_type == "summary_agg":
        plt_args = {
            "col": "method",
            "sharex": False,
            "aspect": 1,
            "col_order": list(cmap.keys())[:3],
        }
    g = sns.FacetGrid(cur_data, margin_titles=True, sharey=True, height=3, **plt_args)
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
        linewidth=0.8,
        size=2,
        warn_thresh=0.8,
    )
    if plt_type == "summary":
        g.map_dataframe(
            hist_wrap,
            x="pactive",
            weights="density",
            stat="probability",
            bins=9,
            binrange=(0, 1),
            hue="method",
            palette=cmap,
            alpha=0.9,
        )
        g.set_titles(row_template="{row_name}", col_template="Animal: {col_name}")
        for ax in g.axes[0, :]:
            ax.set_xlabel("# of sessions active", style="italic")
        for ax in g.axes[1, :]:
            ax.set_xlabel("# of sessions active", style="italic")
        for ax in g.axes[2, :]:
            ax.set_xlabel("Probability of active", style="italic")
            if ax.texts:
                for tx in ax.texts:
                    x, y = tx.get_unitless_position()
                    tx.set(horizontalalignment="center", x=x + 0.08)
    elif plt_type == "summary_agg":
        g.set_titles(col_template="{col_name}")
        g.set_xlabels("# of sessions active", style="italic")
    g.set_ylabels("Per-session probability", style="italic")
    g.fig.savefig(
        os.path.join(FIG_PATH, "{}.svg".format(plt_type)), dpi=500, bbox_inches="tight"
    )
    plt.close(g.fig)

# %% plot example traces
nsmp = 5
Awnd = 15
Twnd = 14000
sub_idx = [286, 509, 176, 530, 227]
brt_offset = 0.1
trace_offset = 5
lw = 1
cmap = {"red": qualitative.Plotly[1], "green": qualitative.Plotly[2]}
map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"))
map_g2r = map_g2r[(map_g2r["animal"].isin(PARAM_EXP_ANM)) & (map_g2r["distance"] < 3)]
# map_smp = map_g2r.sample(nsmp, replace=False).reset_index()
map_smp = map_g2r.loc[sub_idx].reset_index(drop=True)
fig, axs = plt.subplots(
    len(map_smp), 2, figsize=(6.4, 3.5), gridspec_kw={"width_ratios": [1, 8]}
)
for ir, row in map_smp.iterrows():
    anm, ss, uid_red, uid_gn = (
        row["animal"],
        row["session"],
        row["uid_red"],
        row["uid_green"],
    )
    Ared = (
        xr.open_dataarray(os.path.join(OUT_PATH, "Ared", "{}-{}.nc".format(anm, ss)))
        .sel(unit_id=uid_red)
        .compute()
    )
    Tred = (
        xr.open_dataset(os.path.join(IN_RED_PATH, "{}-{}.nc".format(anm, ss)))["C_init"]
        .sel(unit_id=uid_red)
        .squeeze()
        .compute()
    )
    Agn = (
        xr.open_dataarray(
            os.path.join(OUT_PATH, "Agn_trans", "{}-{}.nc".format(anm, ss))
        )
        .sel(unit_id=uid_gn)
        .compute()
    )
    Tgn = (
        xr.open_dataset(os.path.join(IN_GREEN_PATH, "{}-{}.nc".format(anm, ss)))["YrA"]
        .sel(unit_id=uid_gn)
        .squeeze()
        .compute()
    )
    ct = centroid(Ared.expand_dims("unit_id"))
    ct2 = centroid(Agn.expand_dims("unit_id"))
    h, w = ct["height"].values.item(), ct["width"].values.item()
    hs = slice(h - Awnd, h + Awnd)
    ws = slice(w - Awnd, w + Awnd)
    Ared, Agn = Ared.sel(height=hs, width=ws), Agn.sel(height=hs, width=ws)
    Tred, Tgn = Tred.isel(frame=slice(5, Twnd)), Tgn.isel(frame=slice(5, Twnd))
    im = plot_overlap(norm(np.array(Agn)), norm(np.array(Ared)), brt_offset=0.1)
    ax_A, ax_tr = axs[ir, 0], axs[ir, 1]
    ax_A.imshow(im)
    ax_tr.plot(
        zscore(Tred),
        label="tdTomato Channel" if ir == 0 else "",
        color=cmap["red"],
        lw=lw,
    )
    ax_tr.plot(
        zscore(Tgn) + trace_offset,
        label="GCaMP Channel" if ir == 0 else "",
        color=cmap["green"],
        lw=lw,
    )
    ax_A.set_axis_off()
    ax_tr.set_axis_off()
    ax_tr.set_xlim(0, Twnd)
fig.legend(
    title=None,
    loc="lower center",
    bbox_to_anchor=(0.2, 1, 0.6, 0.001),
    bbox_transform=fig.transFigure,
    mode="expand",
    ncol=2,
)
fig.tight_layout()
bar = AnchoredSizeBar(
    ax_tr.transData,
    600,
    "10 sec",
    loc="upper right",
    bbox_to_anchor=(0.97, 0),
    bbox_transform=ax_tr.transAxes,
    frameon=False,
    pad=0.1,
    sep=4,
    size_vertical=0.5,
)
ax_tr.add_artist(bar)
plt.subplots_adjust(top=1, hspace=0.08)
fig.savefig(os.path.join(FIG_PATH, "traces.svg"), bbox_inches="tight")
