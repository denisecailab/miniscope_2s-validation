#%% imports and definitions
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
from sklearn.metrics import pairwise_distances
from statsmodels.formula.api import ols
from tqdm.auto import tqdm, trange

IN_SS_CSV = "./log/sessions.csv"
IN_PS_PATH = "./intermediate/processed/green"
IN_FM_LABEL = "./intermediate/frame_label/fm_label.nc"
IN_RAW_MAP = "./intermediate/cross_reg/green/mappings_meta_fill.pkl"
IN_RED_MAP = "./intermediate/cross_reg/red/mappings_meta_fill.pkl"
IN_REG_MAP = "./intermediate/register_g2r/green_mapping_reg.pkl"
PARAM_BW = 5
PARAM_BW_OCCP = 5
PARAM_NSHUF = 500
PARAM_STB_QTHRES = 0.95
PARAM_SI_QTHRES = 0.95
PARAM_SMP_SPACE = np.linspace(-100, 100, 200)
PARAM_MIN_NCELL = 3
PARAM_SUB_ANM = ["m12", "m15", "m16"]
PARAM_SUB_TDIST = (0, 12)
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
OUT_PATH = "./intermediate/drift"
FIG_PATH = "./figs/drift"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)

#%% compute fr
ss_csv = pd.read_csv(IN_SS_CSV)
ss_csv = ss_csv[ss_csv["analyze"]]
fm_label = xr.open_dataset(IN_FM_LABEL)
fr_ls = []
metric_ls = []
for _, row in tqdm(list(ss_csv.iterrows())):
    anm, ss = row["animal"], row["name"]
    try:
        ps_ds = xr.open_dataset(
            os.path.join(IN_PS_PATH, "{}-{}.nc".format(anm, ss))
        ).assign_coords(linpos=fm_label["linpos_sign"], trial=fm_label["trial"])
    except FileNotFoundError:
        warnings.warn("missing data for {}-{}".format(anm, ss))
        continue
    Sbin = xr.apply_ufunc(
        thres_gmm,
        ps_ds["S"].dropna("unit_id", how="all"),
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        kwargs={"pos_thres": 0.5},
        vectorize=True,
    ).compute()
    S_df = Sbin.to_dataframe().dropna().reset_index()
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

#%% compute pv corr and ovlp
ss_csv = (
    pd.read_csv(IN_SS_CSV, parse_dates=["date"])
    .sort_values(["animal", "name"])
    .set_index(["animal", "name"])
)
map_red = pd.read_pickle(IN_RED_MAP)
map_green = pd.read_pickle(IN_RAW_MAP)
map_reg = pd.read_pickle(IN_REG_MAP)
map_red = map_red[map_red["session"].notnull().all(axis="columns")].copy()
map_reg = map_reg[map_reg["session"].notnull().all(axis="columns")].copy()
mapping_dict = {"red/raw": map_red, "green/raw": map_green, "red/registered": map_reg}
metric_df = pd.read_feather(os.path.join(OUT_PATH, "metric_agg.feat"))
metric_df = metric_df.set_index(["animal", "session", "unit_id"])
fr_df = pd.read_feather(os.path.join(OUT_PATH, "fr.feat"))
fr = (
    fr_df.groupby(["animal", "session", "unit_id", "smp_space"])["fr_norm"]
    .mean()
    .to_xarray()
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
            tdist = np.abs(
                (ss_csv.loc[anm, ssA]["date"] - ss_csv.loc[anm, ssB]["date"]).days
            )
            # compute overlap
            if mmethod == "red_registered":
                mp = mp[mp[[ssA, ssB]].notnull().all(axis="columns")]
            novlp = (mp[[ssA, ssB]] >= 0).all(axis="columns").sum()
            nAll = (mp[[ssA, ssB]] >= 0).any(axis="columns").sum()
            nA = (mp[ssA] >= 0).sum()
            nB = (mp[ssB] >= 0).sum()
            actA = novlp / nA
            actB = novlp / nB
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
                me = pd.DataFrame(
                    {
                        "animal": cur_mp["animal"],
                        "sessionA": ssA,
                        "unit_idA": cur_mp[ssA],
                        "sessionB": ssB,
                        "unit_idB": cur_mp[ssB],
                    }
                )
                for subset_plc in ["all_cells", "place_cells"]:
                    if subset_plc == "place_cells":
                        me_plc = me.merge(
                            metric_df[["stb", "si", "peak", "sig"]],
                            left_on=["animal", "sessionA", "unit_idA"],
                            right_on=["animal", "session", "unit_id"],
                            how="left",
                        ).merge(
                            metric_df[["stb", "si", "peak", "sig"]],
                            left_on=["animal", "sessionB", "unit_idB"],
                            right_on=["animal", "session", "unit_id"],
                            suffixes=("A", "B"),
                            how="left",
                        )
                        me_plc = me_plc[me_plc[["sigA", "sigB"]].any(axis="columns")]
                    else:
                        me_plc = me
                    if len(me_plc) > PARAM_MIN_NCELL:
                        frA = (
                            fr.reindex(
                                animal=me_plc["animal"].unique(),
                                session=me_plc["sessionA"].unique(),
                                unit_id=me_plc["unit_idA"],
                            )
                            .fillna(0)
                            .squeeze()
                        )
                        frB = (
                            fr.reindex(
                                animal=me_plc["animal"].unique(),
                                session=me_plc["sessionB"].unique(),
                                unit_id=me_plc["unit_idB"],
                            )
                            .fillna(0)
                            .squeeze()
                        )
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
pv_corr["diag"] = pv_corr["smp_spaceA"] == pv_corr["smp_spaceB"]
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

#%% plot pv corr matrices
def corr_mat_to_array(df):
    return np.array(df.set_index(["smp_spaceA", "smp_spaceB"])["corr"].to_xarray())


def plot_mat(x, **kwargs):
    ax = plt.gca()
    ax.imshow(x.values[0], cmap="plasma", aspect="auto", interpolation="none")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines[:].set_visible(False)


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

#%% plot pv corr
cmap = {
    "green/raw-shared": qualitative.Plotly[2],
    "green/raw-zero_padded": qualitative.D3[2],
    "red/registered-shared": qualitative.Plotly[4],
    "red/registered-zero_padded": qualitative.D3[1],
}
smap = {
    "green/raw-shared": "",
    "green/raw-zero_padded": (3, 1),
    "red/registered-shared": "",
    "red/registered-zero_padded": (3, 1),
}
lmap = {
    "green/raw-shared": "All active GCaMP cells",
    "green/raw-zero_padded": "All GCaMP cells\n(zero-padded)",
    "red/registered-shared": "Active GCaMP cells\nregistered with tdTomato",
    "red/registered-zero_padded": "Active GCaMP cells registered\nwith tdTomato (zero-padded)",
}
pv_corr = pd.read_csv(os.path.join(OUT_PATH, "pv_corr_agg.csv"))
pv_corr = pv_corr[
    (pv_corr["animal"].isin(PARAM_SUB_ANM))
    & (pv_corr["tdist"].between(*PARAM_SUB_TDIST))
].copy()
pv_corr["cat"] = pv_corr["map_method"] + "-" + pv_corr["cell_map"]
pv_corr["cat"] = pv_corr["cat"].map(lmap)
corr_dict = {"master": pv_corr}
for by, cur_corr in corr_dict.items():
    for inclusion, corr_sub in cur_corr.groupby("inclusion"):
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
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
        # ax = sns.swarmplot(
        #     cur_corr,
        #     x="tdist",
        #     y="corr",
        #     hue="cat",
        #     palette={lmap[k]: v for k, v in cmap.items()},
        #     edgecolor="gray",
        #     dodge=True,
        #     ax=ax,
        #     legend=False,
        #     native_scale=True,
        #     size=3,
        #     linewidth=1,
        #     warn_thresh=0.8,
        # )
        ax.set_xlabel("Days apart", style="italic")
        ax.set_ylabel("PV correlation", style="italic")
        plt.legend(
            title=None,
            loc="lower center",
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            mode="expand",
            ncol=2,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_PATH, "pv_corr-{}.svg".format(inclusion)), dpi=500)
        plt.close(fig)

#%% plot overlap
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
ovlp = ovlp[
    (ovlp["animal"].isin(PARAM_SUB_ANM)) & (ovlp["tdist"].between(*PARAM_SUB_TDIST))
].copy()
ovlp["color"] = ovlp["map_method"].map(cmap).dropna()
ovlp["map_method"] = ovlp["map_method"].map(lmap).dropna()
for metric in ["actMean", "ovlp"]:
    fig = scatter_agg(
        ovlp,
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
    fig.write_html(os.path.join(FIG_PATH, "overlap-{}.html".format(metric)))
    fig, ax = plt.subplots(figsize=(5.4, 3))
    ax = sns.swarmplot(
        ovlp,
        x="tdist",
        y=metric,
        hue="map_method",
        palette={lmap[k]: v for k, v in cmap.items()},
        edgecolor="gray",
        dodge=False,
        ax=ax,
        legend=False,
        native_scale=True,
        size=3,
        linewidth=1,
        warn_thresh=0.8,
    )
    ax = sns.lineplot(
        ovlp,
        x="tdist",
        y=metric,
        hue="map_method",
        palette={lmap[k]: v for k, v in cmap.items()},
        errorbar="se",
        ax=ax,
        zorder=5,
    )
    ax.set_xlabel("Days apart", style="italic")
    ax.set_ylabel("Reactivation", style="italic")
    plt.legend(
        title=None,
        loc="lower center",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, "overlap-{}.svg".format(metric)), dpi=500)
    plt.close(fig)

#%% run stats on pv corr
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
df = df[(df["animal"].isin(PARAM_SUB_ANM)) & (df["inclusion"] == "place_cells")].copy()
df["cat"] = df["map_method"] + "-" + df["cell_map"]
lm = ols("corr ~ C(cat, Simple)*tdist", data=df).fit(cov_type="HC1")
anova = sm.stats.anova_lm(lm, typ=3)
df_alt = df[df["cat"] != "green/raw-zero_padded"]
lm_alt = ols("corr ~ C(cat, Simple)*tdist", data=df_alt).fit(cov_type="HC1")
anova_alt = sm.stats.anova_lm(lm_alt, typ=3)

#%% run stats on overlap
df = pd.read_csv(os.path.join(OUT_PATH, "ovlp.csv"))
df = df[(df["animal"].isin(PARAM_SUB_ANM)) & (df["map_method"] != "red/raw")].copy()
lm = ols("actMean ~ C(map_method)*tdist", data=df).fit(cov_type="HC1")
anova = sm.stats.anova_lm(lm, typ=3)
lm_dict = dict()
for mmethod, mdf in df.groupby("map_method"):
    lm_dict[mmethod] = ols("actMean ~ tdist", data=mdf).fit(cov_type="HC1")

#%% plot cells
def plot_fr(x, **kwargs):
    ax = plt.gca()
    ax.imshow(x.values[0], cmap="plasma", aspect="auto", interpolation="none")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines[:].set_visible(False)


map_gn = pd.read_pickle(IN_RAW_MAP)
map_rd = pd.read_pickle(IN_REG_MAP)
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
sess_sub = ["rec3", "rec4", "rec5", "rec6", "rec7", "rec8", "rec9", "rec10", "rec11"]
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
    ax.spines[:].set_visible(True)
    ax.spines[:].set_linewidth(3)
    ax.spines[:].set_linestyle(":")
    ax.spines[:].set_color("black")
    ax.spines[:].set_position(("outward", 1.6))
fig = g.fig
fig.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.05)
fig.savefig(os.path.join(FIG_PATH, "drifting_cells.svg"), dpi=500, bbox_inches="tight")
