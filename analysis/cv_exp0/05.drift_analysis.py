# %% imports and definitions
import itertools as itt
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from plotly.express.colors import qualitative
from routine.place_cell import (
    aggregate_fr,
    classify_cell,
    compute_metrics,
    kde_est,
    shuffleS,
)
from routine.plotting import scatter_agg
from routine.utilities import df_set_metadata, norm, thres_gmm
from scipy.stats import ttest_ind, zscore
from sklearn.metrics import pairwise_distances
from statsmodels.formula.api import ols
from tqdm.auto import tqdm, trange

IN_SS_CSV = "./log/sessions.csv"
IN_PS_PATH = "./intermediate/processed/green"
IN_FM_LABEL = "./intermediate/frame_label/fm_label.nc"
IN_RAW_MAP = "./intermediate/cross_reg/green/mappings_meta_fill.pkl"
IN_RED_MAP = "./intermediate/cross_reg/red/mappings_meta_fill.pkl"
IN_REG_MAP = "./intermediate/register_g2r/green_mapping_reg.pkl"
PARAM_CORR_ZSCORE = True
PARAM_BW = 5
PARAM_BW_OCCP = 5
PARAM_NSHUF = 500
PARAM_STB_QTHRES = 0.95
PARAM_SI_QTHRES = 0.95
PARAM_SMP_SPACE = np.linspace(-100, 100, 200)
PARAM_MIN_NCELL = 0
PARAM_SUB_ANM = None
PARAM_SUB_TDIST = (0, 14)
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
PARAM_AGG_ANM = True
OUT_PATH = "./intermediate/drift"
FIG_PATH = "./figs/drift"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)

# %% compute fr
ss_csv = pd.read_csv(IN_SS_CSV)
ss_csv = ss_csv[ss_csv["analyze"]]
fm_label = xr.open_dataset(IN_FM_LABEL)
fr_ls = []
metric_ls = []
for _, row in tqdm(list(ss_csv.iterrows())):
    anm, ss = row["animal"], row["session"]
    try:
        ps_ds = xr.open_dataset(
            os.path.join(IN_PS_PATH, "{}-{}.nc".format(anm, ss))
        ).assign_coords(linpos=fm_label["linpos_sign"], trial=fm_label["trial"])
    except FileNotFoundError:
        warnings.warn("missing data for {}-{}".format(anm, ss))
        continue
    # Sbin = xr.apply_ufunc(
    #     thres_gmm,
    #     ps_ds["S"].dropna("unit_id", how="all"),
    #     input_core_dims=[["frame"]],
    #     output_core_dims=[["frame"]],
    #     kwargs={"pos_thres": 0.5},
    #     vectorize=True,
    # ).compute()
    S_df = ps_ds["S"].dropna("unit_id", how="all").to_dataframe().dropna().reset_index()
    pos_df = S_df[["animal", "session", "frame", "linpos", "trial"]].drop_duplicates()
    occp = (
        pos_df.groupby(["animal", "session", "trial"])
        .apply(
            kde_est,
            var_name="linpos",
            bandwidth=PARAM_BW_OCCP,
            smp_space=PARAM_SMP_SPACE,
        )
        .rename(columns={"linpos": "occp"})
        .reset_index()
    )
    fr_df = aggregate_fr(S_df, occp, PARAM_BW, PARAM_SMP_SPACE)
    metric_df = compute_metrics(fr_df)
    metric_df["ishuf"] = -1
    met_shuf_ls = []
    for ishuf in trange(PARAM_NSHUF, leave=False):
        S_df_shuf = shuffleS(S_df)
        fr_shuf = aggregate_fr(S_df_shuf, occp, PARAM_BW, PARAM_SMP_SPACE)
        met_df = compute_metrics(fr_shuf)
        met_df["ishuf"] = ishuf
        met_shuf_ls.append(met_df)
    fr_ls.append(fr_df)
    metric_ls.append(metric_df)
    metric_ls.extend(met_shuf_ls)
fr_df = pd.concat(fr_ls, ignore_index=True)
metric_df = pd.concat(metric_ls, ignore_index=True)
fr_df.to_feather(os.path.join(OUT_PATH, "fr.feat"))
metric_df.to_feather(os.path.join(OUT_PATH, "metric.feat"))
metric_df_agg = (
    metric_df.groupby(["animal", "session", "unit_id"])
    .apply(classify_cell, stb_thres=PARAM_STB_QTHRES, si_thres=PARAM_SI_QTHRES)
    .reset_index()
)
metric_df_agg.to_feather(os.path.join(OUT_PATH, "metric_agg.feat"))


# %% compute pv corr and ovlp
def zscore_active(a, thres=0):
    az = zscore(a)
    if np.isnan(az).any():
        return az
    else:
        return np.where(az > thres, az, 0)


ss_csv = pd.read_csv(IN_SS_CSV, parse_dates=["date"])
ss_csv = (
    ss_csv[ss_csv["analyze"]]
    .sort_values(["animal", "session"])
    .set_index(["animal", "session"])
)
map_red = pd.read_pickle(IN_RED_MAP)
map_green = pd.read_pickle(IN_RAW_MAP)
map_reg = pd.read_pickle(IN_REG_MAP)
map_red = map_red[
    map_red["session"].notnull().all(axis="columns")
    & (map_red["session"] >= 0).any(axis="columns")
].copy()
map_reg = map_reg[
    map_reg["session"].notnull().all(axis="columns")
    & (map_red["session"] >= 0).any(axis="columns")
].copy()
mapping_dict = {"red/raw": map_red, "green/raw": map_green, "red/registered": map_reg}
metric_df = pd.read_feather(os.path.join(OUT_PATH, "metric_agg.feat"))
metric_df["act"] = metric_df["peak"].notnull()
metric_df = metric_df.set_index(["animal", "session", "unit_id"])
fr_df = pd.read_feather(os.path.join(OUT_PATH, "fr.feat"))
fr = (
    fr_df.groupby(["animal", "session", "unit_id", "smp_space"])["fr_norm"]
    .mean()
    .to_xarray()
)
if PARAM_CORR_ZSCORE:
    fr = xr.apply_ufunc(
        zscore_active,
        fr,
        input_core_dims=[["smp_space"]],
        output_core_dims=[["smp_space"]],
        vectorize=True,
    )
pv_corr_ls = []
ovlp_ls = []
for mmethod, mmap in mapping_dict.items():
    for ssA, ssB in tqdm(
        list(itt.combinations(mmap["session"].columns, 2)), leave=False
    ):
        mmap_sub = mmap[[("meta", "animal"), ("session", ssA), ("session", ssB)]]
        mmap_sub = mmap_sub[mmap_sub["session"].notnull().any(axis="columns")]
        mmap_sub.columns = mmap_sub.columns.droplevel(0)
        for anm, mp in mmap_sub.groupby("animal"):
            # compute tdist
            try:
                tdist = np.abs(
                    (ss_csv.loc[anm, ssA]["date"] - ss_csv.loc[anm, ssB]["date"]).days
                )
            except KeyError:
                warnings.warn(
                    "cannot find session pair {} and {} for animal {}".format(
                        ssA, ssB, anm
                    )
                )
            if mmethod == "red_registered":
                mp = mp[mp[[ssA, ssB]].notnull().all(axis="columns")]
            # compute overlap
            me_all = (
                pd.DataFrame(
                    {
                        "animal": mp["animal"],
                        "sessionA": ssA,
                        "unit_idA": mp[ssA],
                        "sessionB": ssB,
                        "unit_idB": mp[ssB],
                    }
                )
                .merge(
                    metric_df[["stb", "si", "peak", "sig", "act"]],
                    left_on=["animal", "sessionA", "unit_idA"],
                    right_on=["animal", "session", "unit_id"],
                    how="left",
                )
                .merge(
                    metric_df[["stb", "si", "peak", "sig", "act"]],
                    left_on=["animal", "sessionB", "unit_idB"],
                    right_on=["animal", "session", "unit_id"],
                    suffixes=("A", "B"),
                    how="left",
                )
            )
            me_dict = {
                "all_cells": me_all,
                "place_cells": me_all[
                    (me_all[["sigA", "sigB"]].fillna(0).any(axis="columns"))
                    & (me_all[["actA", "actB"]].all(axis="columns"))
                ],
                "non-place_cells": me_all[
                    (~me_all[["sigA", "sigB"]].fillna(0)).all(axis="columns")
                    | (~me_all[["actA", "actB"]].any(axis="columns"))
                ],
            }
            for subset_plc, cur_me in me_dict.items():
                novlp = (
                    (cur_me[["unit_idA", "unit_idB"]] >= 0).all(axis="columns").sum()
                )
                nAll = (cur_me[["unit_idA", "unit_idB"]] >= 0).any(axis="columns").sum()
                nA = (cur_me["unit_idA"] >= 0).sum()
                nB = (cur_me["unit_idB"] >= 0).sum()
                actA = novlp / nA if nA > 0 else np.nan
                actB = novlp / nB if nB > 0 else np.nan
                act = np.mean([actA, actB])
                ovlp = novlp / nAll
                ovlp_ls.append(
                    pd.DataFrame(
                        [
                            {
                                "map_method": mmethod,
                                "animal": anm,
                                "ssA": ssA,
                                "ssB": ssB,
                                "tdist": tdist,
                                "inclusion": subset_plc,
                                "actA": actA,
                                "actB": actB,
                                "actMean": act,
                                "ovlp": ovlp,
                            }
                        ]
                    )
                )
            if mmethod == "red/raw":
                continue
            mp_dict = {
                "shared": mp[(mp[[ssA, ssB]] >= 0).all(axis="columns")],
                "zero_padded": mp.fillna(-1),
            }
            for mp_method, cur_mp in mp_dict.items():
                if len(cur_mp) == 0:
                    continue
                # compute pv corr
                me = (
                    pd.DataFrame(
                        {
                            "animal": cur_mp["animal"],
                            "sessionA": ssA,
                            "unit_idA": cur_mp[ssA],
                            "sessionB": ssB,
                            "unit_idB": cur_mp[ssB],
                        }
                    )
                    .merge(
                        metric_df[["stb", "si", "peak", "sig", "act"]],
                        left_on=["animal", "sessionA", "unit_idA"],
                        right_on=["animal", "session", "unit_id"],
                        how="left",
                    )
                    .merge(
                        metric_df[["stb", "si", "peak", "sig", "act"]],
                        left_on=["animal", "sessionB", "unit_idB"],
                        right_on=["animal", "session", "unit_id"],
                        suffixes=("A", "B"),
                        how="left",
                    )
                )
                me_dict = {
                    "all_cells": me,
                    "place_cells": me[
                        (me[["sigA", "sigB"]].fillna(0).any(axis="columns"))
                        & (me[["actA", "actB"]].all(axis="columns"))
                    ],
                    "non-place_cells": me[
                        (~me[["sigA", "sigB"]].fillna(0)).all(axis="columns")
                        | (~me[["actA", "actB"]].any(axis="columns"))
                    ],
                }
                for subset_plc, cur_me in me_dict.items():
                    frA = fr.reindex(
                        animal=cur_me["animal"].unique(),
                        session=cur_me["sessionA"].unique(),
                        unit_id=cur_me["unit_idA"],
                        fill_value=0,
                    ).squeeze()
                    frB = fr.reindex(
                        animal=cur_me["animal"].unique(),
                        session=cur_me["sessionB"].unique(),
                        unit_id=cur_me["unit_idB"],
                        fill_value=0,
                    ).squeeze()
                    idx = np.logical_and(
                        np.array(frA.notnull().all("smp_space")),
                        np.array(frB.notnull().all("smp_space")),
                    )
                    if subset_plc == "place_cells":
                        assert len(idx) == idx.sum()
                    if idx.sum() > PARAM_MIN_NCELL:
                        frA = frA.isel(unit_id=idx)
                        frB = frB.isel(unit_id=idx)
                        corr = (
                            xr.DataArray(
                                1
                                - pairwise_distances(
                                    frA.transpose("smp_space", "unit_id"),
                                    frB.transpose("smp_space", "unit_id"),
                                    metric="correlation",
                                ),
                                dims=["smp_spaceA", "smp_spaceB"],
                                coords={
                                    "smp_spaceA": np.array(frA.coords["smp_space"]),
                                    "smp_spaceB": np.array(frB.coords["smp_space"]),
                                },
                                name="corr",
                            )
                            .to_series()
                            .reset_index()
                        )
                        corr["diag"] = corr["smp_spaceA"] == corr["smp_spaceB"]
                        corr = df_set_metadata(
                            corr,
                            {
                                "map_method": mmethod,
                                "cell_map": mp_method,
                                "inclusion": subset_plc,
                                "animal": anm,
                                "ssA": ssA,
                                "ssB": ssB,
                                "tdist": tdist,
                            },
                        )
                        pv_corr_ls.append(corr)
pv_corr = pd.concat(pv_corr_ls, ignore_index=True)
ovlp = pd.concat(ovlp_ls, ignore_index=True)
pv_corr.to_feather(os.path.join(OUT_PATH, "pv_corr.feat"))
ovlp.to_csv(os.path.join(OUT_PATH, "ovlp.csv"), index=False)
pv_corr_agg = (
    pv_corr[pv_corr["diag"]]
    .groupby(["map_method", "cell_map", "inclusion", "animal", "ssA", "ssB", "tdist"])[
        "corr"
    ]
    .mean()
    .reset_index()
)
pv_corr_agg.to_csv(os.path.join(OUT_PATH, "pv_corr_agg.csv"), index=False)
pv_corr_mat = (
    pv_corr.groupby(
        ["smp_spaceA", "smp_spaceB", "map_method", "cell_map", "inclusion", "tdist"]
    )["corr"]
    .median()
    .reset_index()
)
pv_corr_mat.to_feather(os.path.join(OUT_PATH, "pv_corr_mat.feat"))


# %% plot pv corr matrices
def corr_mat_to_array(df):
    return np.array(df.set_index(["smp_spaceA", "smp_spaceB"])["corr"].to_xarray())


def plot_mat(x, **kwargs):
    ax = plt.gca()
    ax.imshow(x.values[0], cmap="plasma", aspect="auto", interpolation="none")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)


pv_corr_mat = (
    pd.read_feather(os.path.join(OUT_PATH, "pv_corr_mat.feat"))
    .groupby(["map_method", "cell_map", "inclusion", "tdist"])
    .apply(corr_mat_to_array)
    .rename("mat")
    .reset_index()
    .sort_values(["inclusion", "map_method", "cell_map"])
)
pv_corr_mat["cat"] = (
    pv_corr_mat["inclusion"]
    + "\n"
    + pv_corr_mat["map_method"]
    + "\n"
    + pv_corr_mat["cell_map"]
)
g = sns.FacetGrid(
    pv_corr_mat,
    row="cat",
    col="tdist",
    margin_titles=True,
    sharex=True,
    sharey=True,
    height=2,
    aspect=1,
)
g.map(plot_mat, "mat")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
fig = g.fig
fig.savefig(os.path.join(FIG_PATH, "corr_mat.svg"))
plt.close(fig)

# %% plot pv corr
show_sig = False
cmap = {
    # "green/raw-shared": qualitative.Plotly[2],
    "green/raw-zero_padded": qualitative.Plotly[2],
    # "red/registered-shared": qualitative.Plotly[4],
    "red/registered-zero_padded": qualitative.Plotly[4],
}
smap = {
    # "green/raw-shared": (3, 1),
    "green/raw-zero_padded": "",
    # "red/registered-shared": (3, 1),
    "red/registered-zero_padded": "",
}
lmap = {
    # "green/raw-shared": "Always active GCaMP cells",
    "green/raw-zero_padded": "All GCaMP cells",
    # "red/registered-shared": "Active GCaMP cells\nregistered with tdTomato",
    "red/registered-zero_padded": "GCaMP cells\nregistered with tdTomato",
}
pv_corr = pd.read_csv(os.path.join(OUT_PATH, "pv_corr_agg.csv"))
if PARAM_SUB_ANM is not None:
    pv_corr = pv_corr[pv_corr["animal"].isin(PARAM_SUB_ANM)].copy()
if PARAM_SUB_TDIST is not None:
    pv_corr = pv_corr[pv_corr["tdist"].between(*PARAM_SUB_TDIST)].copy()
pv_corr["cat"] = pv_corr["map_method"] + "-" + pv_corr["cell_map"]
pv_corr["cat"] = pv_corr["cat"].map(lmap)
if PARAM_AGG_ANM:
    grp_dims = list(set(pv_corr.columns) - set(["ssA", "ssB", "corr"]))
    pv_corr = pv_corr.groupby(grp_dims).agg({"corr": "median"}).reset_index()
corr_dict = {"master": pv_corr}
for by, cur_corr in corr_dict.items():
    for inclusion, corr_sub in cur_corr.groupby("inclusion"):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax = sns.lineplot(
            corr_sub,
            x="tdist",
            y="corr",
            hue="cat",
            style="cat",
            palette={lmap[k]: v for k, v in cmap.items()},
            dashes={lmap[k]: v for k, v in smap.items()},
            errorbar="se",
            ax=ax,
            zorder=5,
        )
        ax = sns.swarmplot(
            corr_sub,
            x="tdist",
            y="corr",
            hue="cat",
            palette={lmap[k]: v for k, v in cmap.items()},
            edgecolor="gray",
            dodge=False,
            ax=ax,
            legend=False,
            native_scale=True,
            size=3.5,
            linewidth=0.8,
            warn_thresh=0.8,
        )
        ax.set_xlabel("Days apart", style="italic")
        ax.set_ylabel("PV correlation", style="italic")
        ax.set_ylim(0, 1.05)
        if show_sig:
            y_pos = ax.get_ylim()[1]
            ax.set_ylim(top=y_pos * 1.1)
            for t in corr_sub["tdist"].unique():
                ttA, ttB = (
                    corr_sub[
                        (
                            corr_sub["map_method"]
                            == "Active GCaMP cells registered\nwith tdTomato (zero-padded)"
                        )
                        & (corr_sub["tdist"] == t)
                    ]["corr"],
                    corr_sub[
                        (corr_sub["map_method"] == "All GCaMP cells\n(zero-padded)")
                        & (corr_sub["tdist"] == t)
                    ]["corr"],
                )
                stat, pval = ttest_ind(ttA, ttB)
                if pval < 0.05:
                    ttext = "*"
                else:
                    ttext = "ns"
                ax.text(x=t, y=y_pos * 1.03, s=ttext)
        plt.legend(
            title=None,
            loc="lower center",
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            mode="expand",
            ncol=2,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(FIG_PATH, "pv_corr-{}.svg".format(inclusion)),
            dpi=500,
            bbox_inches="tight",
        )
        plt.close(fig)

# %% plot overlap
show_sig = False
cmap = {
    # "red/raw": qualitative.Plotly[1],
    "green/raw": qualitative.Plotly[2],
    "red/registered": qualitative.Plotly[4],
}
lmap = {
    # "red/raw": "tdTomato cells",
    "green/raw": "All GCaMP cells",
    "red/registered": "GCaMP cells\nregistered with tdTomato",
}
ovlp = pd.read_csv(os.path.join(OUT_PATH, "ovlp.csv"))
if PARAM_SUB_ANM is not None:
    ovlp = ovlp[ovlp["animal"].isin(PARAM_SUB_ANM)].copy()
if PARAM_SUB_TDIST is not None:
    ovlp = ovlp[ovlp["tdist"].between(*PARAM_SUB_TDIST)].copy()
ovlp["color"] = ovlp["map_method"].map(cmap)
ovlp["map_method"] = ovlp["map_method"].map(lmap)
ovlp = ovlp.dropna()
if PARAM_AGG_ANM:
    grp_dims = list(
        set(ovlp.columns) - set(["ssA", "ssB", "actMean", "ovlp", "actA", "actB"])
    )
    ovlp = (
        ovlp.groupby(grp_dims)
        .agg(
            {"actMean": "median", "ovlp": "median", "actA": "median", "actB": "median"}
        )
        .reset_index()
    )
for inclusion, cur_ovlp in ovlp.groupby("inclusion"):
    for metric in ["actMean", "ovlp"]:
        fig = scatter_agg(
            cur_ovlp,
            x="tdist",
            y=metric,
            facet_row=None,
            facet_col="animal",
            col_wrap=3,
            legend_dim="map_method",
            marker={"color": "color"},
        )
        fig.update_xaxes(title="Days apart")
        fig.update_yaxes(range=(0, 1), title="Overlap")
        fig.write_html(
            os.path.join(FIG_PATH, "overlap-{}-{}.html".format(inclusion, metric))
        )
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax = sns.swarmplot(
            cur_ovlp,
            x="tdist",
            y=metric,
            hue="map_method",
            palette={lmap[k]: v for k, v in cmap.items()},
            edgecolor="gray",
            dodge=False,
            ax=ax,
            legend=False,
            native_scale=True,
            size=3.5,
            linewidth=0.8,
            warn_thresh=0.8,
        )
        ax = sns.lineplot(
            cur_ovlp,
            x="tdist",
            y=metric,
            hue="map_method",
            palette={lmap[k]: v for k, v in cmap.items()},
            errorbar="se",
            ax=ax,
            zorder=5,
        )
        ax.set_xlabel("Days apart", style="italic")
        ax.set_ylabel("Reactivation probability", style="italic")
        ax.set_ylim(0.2, 1.05)
        if show_sig:
            y_pos = ax.get_ylim()[1]
            ax.set_ylim(top=y_pos * 1.2)
            for t in cur_ovlp["tdist"].unique():
                ttA, ttB = (
                    cur_ovlp[
                        (cur_ovlp["map_method"] == "All GCaMP cells")
                        & (cur_ovlp["tdist"] == t)
                    ][metric],
                    cur_ovlp[
                        (
                            cur_ovlp["map_method"]
                            == "GCaMP cells\nregistered with tdTomato"
                        )
                        & (cur_ovlp["tdist"] == t)
                    ][metric],
                )
                stat, pval = ttest_ind(ttA, ttB)
                if pval < 0.05:
                    ttext = "*"
                else:
                    ttext = "ns"
                ax.text(x=t, y=y_pos * 1.03, s=ttext)
        plt.legend(
            title=None,
            loc="lower center",
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            mode="expand",
            ncol=2,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(FIG_PATH, "{}-{}.svg".format(metric, inclusion)),
            dpi=500,
            bbox_inches="tight",
        )
        plt.close(fig)

# %% run stats on pv corr
from patsy.contrasts import ContrastMatrix


def _name_levels(prefix, levels):
    return ["[%s%s]" % (prefix, level) for level in levels]


class Simple(object):
    def _simple_contrast(self, levels):
        nlevels = len(levels)
        contr = -1.0 / nlevels * np.ones((nlevels, nlevels - 1))
        contr[1:][np.diag_indices(nlevels - 1)] = (nlevels - 1.0) / nlevels
        return contr

    def code_with_intercept(self, levels):
        contrast = np.column_stack(
            (np.ones(len(levels)), self._simple_contrast(levels))
        )
        return ContrastMatrix(contrast, _name_levels("Simp.", levels))

    def code_without_intercept(self, levels):
        contrast = self._simple_contrast(levels)
        return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))


df = pd.read_csv(os.path.join(OUT_PATH, "pv_corr_agg.csv"))
df = df[df["inclusion"] == "place_cells"].copy()
if PARAM_SUB_ANM is not None:
    df = df[df["animal"].isin(PARAM_SUB_ANM)].copy()
df["cat"] = df["map_method"] + "-" + df["cell_map"]
if PARAM_AGG_ANM:
    grp_dims = list(set(df.columns) - set(["ssA", "ssB", "corr"]))
    df = df.groupby(grp_dims).agg({"corr": "median"}).reset_index()
    cov_type = "nonrobust"
else:
    cov_type = "HC1"
lm = ols("corr ~ C(cat, Simple)*tdist", data=df).fit(cov_type=cov_type)
anova = sm.stats.anova_lm(lm, typ=3)
df_alt = df[df["cat"].isin(["green/raw-zero_padded", "red/registered-zero_padded"])]
lm_alt = ols("corr ~ C(cat, Simple)*tdist", data=df_alt).fit(cov_type=cov_type)
anova_alt = sm.stats.anova_lm(lm_alt, typ=3)
lm_dict = dict()
anova_dict = dict()
for ct, mdf in df.groupby("cat"):
    cur_lm = ols("corr ~ tdist", data=mdf).fit(cov_type=cov_type)
    lm_dict[ct] = cur_lm
    anova_dict[ct] = sm.stats.anova_lm(cur_lm, typ=3)

# %% run stats on overlap
df = pd.read_csv(os.path.join(OUT_PATH, "ovlp.csv"))
df = df[
    (df["map_method"].isin(["green/raw", "red/registered"]))
    & (df["inclusion"] == "place_cells")
].copy()
if PARAM_SUB_ANM is not None:
    df = df[df["animal"].isin(PARAM_SUB_ANM)].copy()
if PARAM_AGG_ANM:
    grp_dims = list(
        set(df.columns) - set(["ssA", "ssB", "actMean", "ovlp", "actA", "actB"])
    )
    df = (
        df.groupby(grp_dims)
        .agg(
            {"actMean": "median", "ovlp": "median", "actA": "median", "actB": "median"}
        )
        .reset_index()
    )
    cov_type = "nonrobust"
else:
    cov_type = "HC1"
lm = ols("actMean ~ C(map_method)*tdist", data=df).fit(cov_type=cov_type)
anova = sm.stats.anova_lm(lm, typ=3)
lm_dict = dict()
anova_dict = dict()
for mmethod, mdf in df.groupby("map_method"):
    cur_lm = ols("actMean ~ tdist", data=mdf).fit(cov_type=cov_type)
    lm_dict[mmethod] = cur_lm
    anova_dict[mmethod] = sm.stats.anova_lm(cur_lm, typ=3)


# %% plot cells
def plot_fr(x, **kwargs):
    ax = plt.gca()
    ax.imshow(x.values[0], cmap="plasma", aspect="auto", interpolation="none")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)


map_gn = pd.read_pickle(IN_RAW_MAP)
map_rd = pd.read_pickle(IN_REG_MAP)
if PARAM_SUB_ANM is not None:
    map_gn = map_gn[map_gn["meta", "animal"].isin(PARAM_SUB_ANM)].copy()
    map_rd = map_rd[map_rd["meta", "animal"].isin(PARAM_SUB_ANM)].copy()
mapping_dict = {
    "green": map_gn,
    "red": map_rd,
}
lmap = {
    "green": "All GCaMP cells",
    "red": "GCaMP cells\nregistered with tdTomato",
}
fr_df = pd.read_feather(os.path.join(OUT_PATH, "fr.feat"))
metric_df = pd.read_feather(os.path.join(OUT_PATH, "metric_agg.feat"))
metric_df = metric_df.set_index(["animal", "session", "unit_id"])
fr_df = (
    fr_df.groupby(["animal", "session", "unit_id", "smp_space"])["fr_norm"]
    .mean()
    .reset_index()
    .set_index(["animal", "session", "unit_id"])
)
sess_sub = ["rec{}".format(i) for i in range(7)]
day_dict = {ss: "Day {}".format(2 * i + 1) for i, ss in enumerate(sess_sub)}
fr_ma_ls = []
for mmethod, mmap in mapping_dict.items():
    mmap_sub = mmap[[("session", s) for s in sess_sub] + [("meta", "animal")]]
    mmap_sub = mmap_sub[(mmap_sub.notnull()).sum(axis="columns") >= 2]
    mmap_sub.columns = mmap_sub.columns.droplevel(0)
    for srt_ss in [sess_sub[0], sess_sub[-1]]:
        cur_mmap = mmap_sub[mmap_sub[srt_ss] >= 0].copy()
        cur_met = metric_df.loc[
            [(a, srt_ss, u) for a, u in zip(cur_mmap["animal"], cur_mmap[srt_ss])]
        ]
        cur_mmap["sig"] = cur_met["sig"].values
        cur_mmap["peak"] = cur_met["peak"].values
        cur_mmap = cur_mmap[cur_mmap["sig"]].sort_values("peak").reset_index(drop=True)
        for ss in sess_sub:
            cur_fr = []
            for muid, row in cur_mmap.iterrows():
                uid = row[ss]
                if np.isnan(uid) or uid == -1:
                    f = np.zeros_like(PARAM_SMP_SPACE)
                else:
                    f = np.array(fr_df.loc[row["animal"], ss, row[ss]]["fr_norm"])
                    if f.sum() > 0:
                        f = norm(f)
                cur_fr.append(f)
            cur_fr = np.stack(cur_fr, axis=0)
            fr_ma_ls.append(
                pd.DataFrame(
                    {
                        "mmethod": mmethod,
                        "sortby": srt_ss,
                        "session": ss,
                        "fr_mat": [cur_fr],
                    }
                )
            )
fr_ma = pd.concat(fr_ma_ls, ignore_index=True)
fr_ma["day"] = fr_ma["session"].map(day_dict)
fr_ma["mmethod"] = fr_ma["mmethod"].map(lmap)
fr_ma["row"] = fr_ma["mmethod"] + " x " + fr_ma["sortby"]
g = sns.FacetGrid(
    fr_ma,
    row="row",
    col="day",
    margin_titles=True,
    sharey="row",
    sharex=True,
    height=2,
    aspect=0.58,
)
g.map(plot_fr, "fr_mat")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
for ax in g.axes[:, -1]:
    if ax.texts:
        for tx in ax.texts:
            x, y = tx.get_unitless_position()
            tx.set(
                horizontalalignment="center",
                x=x + 0.25,
                text=tx.get_text().partition(" x ")[0],
            )
for crd in [(0, 0), (1, -1), (2, 0), (3, -1)]:
    ax = g.axes[crd]
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(3)
        spine.set_linestyle(":")
        spine.set_color("black")
        spine.set_position(("outward", 1.6))
fig = g.fig
fig.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.05)
fig.savefig(os.path.join(FIG_PATH, "drifting_cells.svg"), dpi=500, bbox_inches="tight")
