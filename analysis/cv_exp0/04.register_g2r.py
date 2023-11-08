# %% imports and definitions
import itertools as itt
import os
import warnings

import dask.array as darr
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from bokeh.palettes import Category20
from minian.cross_registration import (
    calculate_centroid_distance,
    calculate_mapping,
    group_by_session,
)
from minian.visualization import centroid
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from natsort import natsorted
from plotly.express.colors import qualitative
from routine.alignment import apply_affine, est_affine
from routine.g2r_mapping import calculate_mapping_linear_sum
from routine.plotting import plot_overlap, plotA_contour
from routine.utilities import df_set_metadata, norm
from scipy.ndimage import median_filter
from scipy.spatial.distance import correlation, cosine
from scipy.stats import ttest_ind, zscore
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm.auto import tqdm

IN_GREEN_PATH = "./intermediate/processed/green"
IN_RED_PATH = "./intermediate/processed/red"
IN_SS_FILE = "./log/sessions.csv"
IN_RED_MAP = "./intermediate/cross_reg/red/mappings_meta_fill.pkl"
IN_GREEN_MAP = "./intermediate/cross_reg/green/mappings_meta_fill.pkl"
IN_WV_GFP = "./data/wavelength/fpbase_spectra_EGFP.csv"
IN_WV_EMM_RED = "./data/wavelength/et600-50m.txt"
IN_WV_EMM_GN = "./data/wavelength/et525-50m.txt"
PARAM_DIST_THRES = 15
PARAM_SUB_SS = None
PARAM_SUB_ANM = ["m20", "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m29"]
PARAM_EXP_ANM = ["m20"]
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
REG_PATH = "./intermediate/cross_reg/red"
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
map_lsm_ls = []
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
        mapping_lsm = calculate_mapping_linear_sum(
            cent_red, cent_green, thres=PARAM_DIST_THRES
        )
        mapping_lsm = df_set_metadata(
            mapping_lsm, {"animal": anm, "session": ss}
        ).rename(columns={"idxA": "uid_red", "idxB": "uid_green"})
        map_lsm_ls.append(mapping_lsm)
mapping = pd.concat(map_ls, ignore_index=True)
mapping.to_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"), index=False)
mapping_lsm = pd.concat(map_lsm_ls, ignore_index=True)
mapping_lsm.to_csv(os.path.join(OUT_PATH, "g2r_mapping_lsm.csv"), index=False)


# %% compute green mapping based on red
def reg_map(reg_df, map_df):
    map_reg = []
    for _, row in reg_df.iterrows():
        anm = row["meta", "animal"]
        row_ss = row["session"]
        row_new = row.copy()
        idxs = [(anm, s, u) for s, u in zip(row_ss.index, row_ss.values)]
        row_new.loc["session"] = map_df.reindex(idxs).fillna(-1).values
        row_new.loc[row.isnull()] = np.nan
        map_reg.append(row_new.to_frame().T)
    return pd.concat(map_reg, ignore_index=True)


map_red = pd.read_pickle(IN_RED_MAP)
map_green = pd.read_pickle(IN_GREEN_MAP)
map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv")).set_index(
    ["animal", "session", "uid_red"]
)["uid_green"]
map_r2g = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv")).set_index(
    ["animal", "session", "uid_green"]
)["uid_red"]
map_green_reg = reg_map(map_red, map_g2r)
map_red_reg = reg_map(map_green, map_r2g)
map_green_reg.to_pickle(os.path.join(OUT_PATH, "green_mapping_reg.pkl"))
map_red_reg.to_pickle(os.path.join(OUT_PATH, "red_mapping_reg.pkl"))

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


def plot_animal(anm_df, col_order):
    g = sns.FacetGrid(
        anm_df,
        row="kind",
        col="session",
        col_order=col_order,
        margin_titles=True,
        height=1.8,
    )
    g.map(plot_cells, "im")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    fig = g.fig
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.004, hspace=0.04)
    return fig


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
exp_anm = "m22"
exp_sess = ["Day 1", "Day 5", "Day 9", "Day 13"]
for anm, anm_df in cells_im.groupby("animal"):
    fig = plot_animal(anm_df, col_order=list(ss_dict.values()))
    fig.savefig(
        os.path.join(fig_cells_path, "{}.svg".format(anm)), dpi=500, bbox_inches="tight"
    )
    plt.close(fig)
    if anm == exp_anm:
        fig = plot_animal(anm_df[anm_df["session"].isin(exp_sess)], col_order=exp_sess)
        fig.savefig(
            os.path.join(fig_cells_path, "{}_example.svg".format(anm)),
            dpi=500,
            bbox_inches="tight",
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
    map_green_reg[
        (map_green_reg["variable", "npresent"] == 7)
        & (map_green_reg["session"] >= 0).any(axis="columns")
    ]
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
    g = sns.FacetGrid(cur_data, margin_titles=True, sharey=True, height=3.2, **plt_args)
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
        g.set(ylim=(0, 0.65))
        for mthd, grp_data in cur_data.groupby("method"):
            lm = ols("density ~ C(nactive)", data=grp_data).fit()
            anova = sm.stats.anova_lm(lm, typ=3)
            print(mthd)
            print(anova)
            if anova.loc["C(nactive)", "PR(>F)"] < 0.05:
                tk = pairwise_tukeyhsd(grp_data["density"], grp_data["nactive"])
                tk_df = pd.DataFrame(
                    tk.summary().data[1:], columns=tk.summary().data[0]
                )
                rej_count = pd.DataFrame(
                    {
                        "nactive": grp_data["nactive"].unique(),
                        "rej_count": [
                            tk_df.loc[
                                (tk_df[["group1", "group2"]] == nact).any(
                                    axis="columns"
                                ),
                                "reject",
                            ].sum()
                            for nact in grp_data["nactive"].unique()
                        ],
                    }
                )
                rej_count["rej_all"] = (
                    rej_count["rej_count"] == grp_data["nactive"].nunique() - 1
                )
                cur_ax = g.axes_dict[mthd]
                max_dat = grp_data.groupby("nactive")["density"].max()
                for nact in rej_count.loc[rej_count["rej_all"], "nactive"]:
                    cur_ax.annotate(
                        "*",
                        (str(nact), 0.85),
                        xytext=(0, 0.2),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset fontsize",
                        fontsize=20,
                        horizontalalignment="center",
                    )
                    cur_ax.relim()
                    cur_ax.autoscale_view(tight=True)
    g.set_ylabels("Per-session probability", style="italic")
    g.fig.savefig(
        os.path.join(FIG_PATH, "{}.svg".format(plt_type)), dpi=500, bbox_inches="tight"
    )
    plt.close(g.fig)

# %% plot example traces
nsmp = 5
Awnd = 15
Twnd = 14000
sub_idx = [1, 9, 10, 15, 38]
brt_offset = 0.1
trace_offset = 5
lw = 1
cmap = {"red": qualitative.Plotly[1], "green": qualitative.Plotly[2]}
map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"))
map_g2r = (
    map_g2r[(map_g2r["animal"].isin(PARAM_EXP_ANM)) & (map_g2r["distance"] < 3)]
    .sort_values("distance")
    .reset_index(drop=True)
)
map_smp = map_g2r.loc[sub_idx].reset_index(drop=True)
fig, axs = plt.subplots(
    len(map_smp), 2, figsize=(7, 3.8), gridspec_kw={"width_ratios": [1, 8]}
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


# %% compute trace correlations
def med_baseline(a: np.ndarray, wnd: int) -> np.ndarray:
    base = median_filter(a, size=wnd)
    a -= base
    return a


def compute_pair_corr(row, med_wnd=500):
    anm, ss, uid_red, uid_gn = (
        row["animal"],
        row["session"],
        row["uid_red"],
        row["uid_green"],
    )
    Tred = (
        xr.open_dataset(os.path.join(IN_RED_PATH, "{}-{}.nc".format(anm, ss)))["C_init"]
        .sel(unit_id=uid_red)
        .squeeze()
        .compute()
    )
    Tgn = (
        xr.open_dataset(os.path.join(IN_GREEN_PATH, "{}-{}.nc".format(anm, ss)))["YrA"]
        .sel(unit_id=uid_gn)
        .squeeze()
        .compute()
    )
    common_idx = sorted(
        list(
            set(Tred.coords["frame"].values.tolist()).intersection(
                set(Tgn.coords["frame"].values.tolist())
            )
        )
    )
    Tred = Tred.sel(frame=common_idx)
    Tgn = Tgn.sel(frame=common_idx)
    if med_wnd is not None:
        Tred = med_baseline(Tred, med_wnd) + Tred.mean()
        Tgn = med_baseline(Tgn, med_wnd) + Tgn.mean()
    sh = int(np.random.randint(-Tred.sizes["frame"], Tred.sizes["frame"]))
    Tred_sh = np.roll(Tred, sh)
    reg = LinearRegression().fit(np.array(Tgn).reshape((-1, 1)), np.array(Tred))
    coef = reg.coef_[0]
    reg_sh = LinearRegression().fit(np.array(Tgn).reshape((-1, 1)), np.array(Tred_sh))
    coef_sh = reg_sh.coef_[0]
    return pd.Series(
        {
            "corr_org": 1 - correlation(Tred, Tgn),
            "corr_sh": 1 - correlation(Tred_sh, Tgn),
            "cos_org": 1 - cosine(Tred, Tgn),
            "cos_sh": 1 - cosine(Tred_sh, Tgn),
            "coef_org": coef,
            "coef_sh": coef_sh,
            "sh": sh,
        }
    )


map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"))
map_g2r_corr = pd.concat(
    [map_g2r, map_g2r.apply(compute_pair_corr, axis="columns")], axis="columns"
)
map_g2r_corr.to_csv(os.path.join(OUT_PATH, "g2r_mapping_corr.csv"), index=False)


# %% trace example
def exp_trace(row, med_wnd=None):
    anm, ss, uid_red, uid_gn = (
        row["animal"],
        row["session"],
        row["uid_red"],
        row["uid_green"],
    )
    Tred = (
        xr.open_dataset(os.path.join(IN_RED_PATH, "{}-{}.nc".format(anm, ss)))["C_init"]
        .sel(unit_id=uid_red)
        .squeeze()
        .compute()
    )
    Tgn = (
        xr.open_dataset(os.path.join(IN_GREEN_PATH, "{}-{}.nc".format(anm, ss)))["YrA"]
        .sel(unit_id=uid_gn)
        .squeeze()
        .compute()
    )
    Tgn_C = (
        xr.open_dataset(os.path.join(IN_GREEN_PATH, "{}-{}.nc".format(anm, ss)))["C"]
        .sel(unit_id=uid_gn)
        .squeeze()
        .compute()
    )
    if med_wnd is not None:
        Tred = med_baseline(Tred, med_wnd)
        Tgn = med_baseline(Tgn, med_wnd)
    return Tred, Tgn, Tgn_C


map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv"))
exp_row = map_g2r.sort_values("cos_org", ascending=False).iloc[0]
Tred, Tgn, Tgn_C = exp_trace(exp_row)
opts_cv = {"frame_width": 900}
(
    hv.Curve(Tred).opts(**opts_cv)
    + hv.Curve(Tgn).opts(**opts_cv)
    + hv.Curve(Tgn_C).opts(**opts_cv)
).cols(1)


# %% plot trace correlation distribution
# theoretical calculation
dat_gfp = pd.read_csv(IN_WV_GFP)
dat_emm_gn = pd.read_csv(
    IN_WV_EMM_GN,
    sep="\t",
    names=["wavelength", "trans_gn"],
    dtype={"wavelength": int, "trans_gn": float},
)
dat_emm_red = pd.read_csv(
    IN_WV_EMM_RED,
    sep="\t",
    names=["wavelength", "trans_red"],
    dtype={"wavelength": int, "trans_red": float},
)
dat = (
    dat_gfp.merge(dat_emm_red, on="wavelength", how="inner")
    .merge(dat_emm_gn, on="wavelength", how="inner")
    .fillna(0)
)
dat = dat[dat["wavelength"].between(300, 700)]
dat_gn = dat.loc[dat["trans_gn"] > 5e-6]
dat_red = dat.loc[dat["trans_red"] > 5e-6]
gfp = dat["EGFP em"]
emm_red = dat["trans_red"]
emm_gn = dat["trans_gn"]
prop = (gfp * emm_red).sum() / (gfp * emm_gn).sum()
fig, ax = plt.subplots(figsize=(4.7, 2.7))
sns.lineplot(x=dat["wavelength"], y=np.zeros(len(dat)), color="grey", ax=ax, lw=1)
sns.lineplot(
    dat_red,
    x="wavelength",
    y="trans_red",
    color=qualitative.Plotly[1],
    label="tdTomato Channel\nEmission Filter",
    ax=ax,
)
sns.lineplot(
    dat_gn,
    x="wavelength",
    y="trans_gn",
    color=qualitative.Plotly[2],
    label="GCaMP Channel\nEmission Filter",
    ax=ax,
)
sns.lineplot(
    dat, x="wavelength", y="EGFP em", color="black", label="GCaMP Emission", ax=ax
)
ax.fill_between(
    x=dat_red["wavelength"],
    y1=np.zeros_like(dat_red["wavelength"]),
    y2=dat_red[["EGFP em", "trans_red"]].min(axis="columns"),
    color=qualitative.Plotly[1],
    alpha=0.4,
)
ax.fill_between(
    x=dat_gn["wavelength"],
    y1=np.zeros_like(dat_gn["wavelength"]),
    y2=dat_gn[["EGFP em", "trans_gn"]].min(axis="columns"),
    color=qualitative.Plotly[2],
    alpha=0.4,
)
ax.set_xlabel("Wavelength (nm)", style="italic")
ax.set_ylabel("Relative\nTransmission/Power (A.U.)", style="italic")
ax.set_xlim(260, 720)
fig.savefig(os.path.join(FIG_PATH, "crosstalk_wavelength.svg"), bbox_inches="tight")
# distribution
map_g2r_corr = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping_corr.csv"))
g2r_df = map_g2r_corr.melt(
    id_vars=["animal", "session", "uid_green", "uid_red"],
    value_vars=["coef_org", "coef_sh"],
    var_name="corr_type",
    value_name="corr",
)
g2r_df["corr_type"] = g2r_df["corr_type"].map(
    {"coef_org": "Observed", "coef_sh": "Shuffled"}
)
fig, ax = plt.subplots(figsize=(4.7, 2.7))
xlim = (-0.15, 0.25)
ax.axvline(prop, color="dimgrey", dashes=(3, 2))
ax.annotate(
    "Exptected\nCrosstalk\nRatio",
    xy=(prop, 0.7),
    xycoords=("data", "axes fraction"),
    xytext=(0.5, 0),
    textcoords="offset fontsize",
    color="dimgrey",
)
sns.histplot(
    g2r_df,
    x="corr",
    hue="corr_type",
    stat="proportion",
    bins=50,
    binrange=xlim,
    kde=True,
    kde_kws={"clip": xlim},
    palette={"Observed": qualitative.Plotly[5], "Shuffled": qualitative.Plotly[8]},
    alpha=0.4,
    line_kws={"lw": 2},
    ax=ax,
)
ax.get_legend().set_title("")
sns.move_legend(ax, "upper left")
ax.set_xlabel("Crosstalk Ratio", style="italic")
ax.set_ylabel("Proportion", style="italic")
fig.savefig(os.path.join(FIG_PATH, "crosstalk_distribution.svg"), bbox_inches="tight")

# %% compute overlap over time
map_red = pd.read_pickle(IN_RED_MAP).set_index(("meta", "animal"))
map_green = pd.read_pickle(IN_GREEN_MAP).set_index(("meta", "animal"))
map_green_reg = pd.read_pickle(os.path.join(OUT_PATH, "green_mapping_reg.pkl"))
map_red_reg = pd.read_pickle(os.path.join(OUT_PATH, "red_mapping_reg.pkl"))
map_green_reg = map_green_reg[
    (map_green_reg["session"].notnull().sum(axis="columns") == 7)
    & (map_green_reg["session"] >= 0).any(axis="columns")
].set_index(("meta", "animal"))
map_red_reg = map_red_reg[
    (map_red_reg["session"].notnull().sum(axis="columns") == 7)
    & (map_red_reg["session"] >= 0).any(axis="columns")
].set_index(("meta", "animal"))
map_g2r = pd.read_csv(os.path.join(OUT_PATH, "g2r_mapping.csv")).set_index(
    ["animal", "session"]
)
if PARAM_SUB_SS is not None:
    map_g2r = map_g2r.loc[map_g2r["session"].isin(PARAM_SUB_SS)].copy()
all_anms = PARAM_SUB_ANM
all_ss = map_red["session"].columns
ovlp_df = []
for anm, ss in itt.product(all_anms, all_ss):
    nred = len(map_red.loc[anm, ("session", ss)].dropna())
    ngreen = len(map_green.loc[anm, ("session", ss)].dropna())
    nreg = len(map_g2r.loc[anm, ss])
    cur_green_reg = map_green_reg.loc[anm, ("session", ss)].dropna()
    cur_red_reg = map_red_reg.loc[anm, ("session", ss)].dropna()
    ngreen_reg = len(cur_green_reg)
    nred_reg = len(cur_red_reg)
    prop_red = nreg / nred
    prop_green = nreg / ngreen
    prop_green_reg = (cur_green_reg >= 0).sum() / ngreen_reg
    prop_red_reg = (cur_red_reg >= 0).sum() / nred_reg
    ovlp_df.append(
        pd.Series(
            {
                "animal": anm,
                "session": ss,
                "nred": nred,
                "ngreen": ngreen,
                "nreg": nreg,
                "ngreen_reg": ngreen_reg,
                "nred_reg": nred_reg,
                "prop_red": prop_red,
                "prop_green": prop_green,
                "prop_green_reg": prop_green_reg,
                "prop_red_reg": prop_red_reg,
            }
        )
    )
ovlp_df = pd.concat(ovlp_df, axis="columns").T

# %% plot overlap over time
cmap = {
    "tdTomato": qualitative.Plotly[1],
    "GCaMP": qualitative.Plotly[2],
    "Stable tdTomato": qualitative.Plotly[4],
    "Stable GCaMP": qualitative.Plotly[7],
}
lmap = {
    "overlap_ncell": {
        "nred": "tdTomato",
        "ngreen": "GCaMP",
        "ngreen_reg": "Stable tdTomato",
        "nred_reg": "Stable GCaMP",
    },
    "overlap_prop": {
        "prop_red": "tdTomato",
        "prop_green": "GCaMP",
        "prop_green_reg": "Stable tdTomato",
        "prop_red_reg": "Stable GCaMP",
    },
}
ylab = {
    "overlap_ncell": "Number of cells",
    "overlap_prop": "Proportion of registered cells",
}
ss_dict = {
    "rec0": "Day 1",
    "rec1": "Day 3",
    "rec2": "Day 5",
    "rec3": "Day 7",
    "rec4": "Day 9",
    "rec5": "Day 11",
    "rec6": "Day 13",
}
for plt_type, cur_lmap in lmap.items():
    ovlp_df_long = ovlp_df.melt(
        id_vars=["animal", "session"],
        value_vars=list(cur_lmap),
        var_name="denom",
        value_name="value",
    )
    ovlp_df_long["day"] = ovlp_df_long["session"].map(ss_dict)
    ovlp_df_long["Denominator"] = ovlp_df_long["denom"].map(cur_lmap)
    fig, ax = plt.subplots(figsize=(9, 3))
    sns.barplot(
        ovlp_df_long,
        x="day",
        y="value",
        errorbar="se",
        saturation=0.8,
        errwidth=1.5,
        capsize=0.1,
        hue="Denominator",
        palette=cmap,
        ax=ax,
    )
    sns.swarmplot(
        ovlp_df_long,
        x="day",
        y="value",
        hue="Denominator",
        palette=cmap,
        edgecolor="gray",
        linewidth=0.8,
        size=2.5,
        warn_thresh=0.8,
        ax=ax,
        dodge=True,
        legend=False,
    )
    ax.get_legend().set_title(None)
    sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
    sns.despine(fig)
    ax.set_xlabel("")
    ax.set_ylabel(ylab[plt_type], style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, "{}.svg".format(plt_type)), bbox_inches="tight")

# %% plot missing cells
map_red_reg = pd.read_pickle(os.path.join(OUT_PATH, "red_mapping_reg.pkl"))
map_missing = map_red_reg[
    (map_red_reg["session"].notnull().sum(axis="columns") == 7)
    & (map_red_reg["session"] >= 0).any(axis="columns")
    & (map_red_reg["session"] == -1).any(axis="columns")
].set_index(("meta", "animal"))
fig_path = os.path.join(FIG_PATH, "missing_cells")
os.makedirs(fig_path, exist_ok=True)
shiftds = xr.open_dataset(os.path.join(REG_PATH, "shiftds.nc"))
A_sh = shiftds["A_sh"]
projs = shiftds["temps_shifted"]
im_opts = {"cmap": "gray", "frame_width": 400, "frame_height": 400}
for anm, anm_df in tqdm(list(map_missing.groupby(("meta", "animal")))):
    plt_dict = dict()
    cmap = itt.cycle(Category20[20])
    anm_df = anm_df.dropna(axis="columns", how="all").copy()
    anm_df["variable", "color"] = [next(cmap) for _ in range(len(anm_df))]
    for ss in tqdm(natsorted(anm_df["session"]), leave=False):
        curA = A_sh.sel(animal=anm, session=ss).dropna("unit_id").compute()
        cur_proj = projs.sel(animal=anm, session=ss)
        uids_all = np.array(curA.coords["unit_id"])
        idx_ma = anm_df["session", ss].dropna()
        idx_ma = idx_ma[idx_ma >= 0]
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
