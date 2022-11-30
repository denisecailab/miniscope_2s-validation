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
from statsmodels.formula.api import ols
from tqdm.auto import tqdm

from routine.place_cell import compute_si, compute_stb, find_peak_field, kde_est
from routine.plotting import scatter_agg
from routine.utilities import corr_mat, nan_corr, norm, thres_gmm

IN_SS_CSV = "./log/sessions.csv"
IN_PS_PATH = "./intermediate/processed/green"
IN_FM_LABEL = "./intermediate/frame_label/fm_label.nc"
IN_RAW_MAP = "./intermediate/cross_reg/green/mappings_meta_fill.pkl"
IN_RED_MAP = "./intermediate/cross_reg/red/mappings_meta_fill.pkl"
IN_REG_MAP = "./intermediate/register_g2r/green_mapping_reg.pkl"
PARAM_BW = 5
PARAM_BW_OCCP = 5
PARAM_SMP_SPACE = np.linspace(-100, 100, 200)
PARAM_SUB_ANM = ["m12", "m15", "m16"]
PARAM_STB_THRES = 0
PARAM_SI_THRES = 0
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
    fr = (
        S_df.groupby(["animal", "session", "trial", "unit_id"])
        .apply(
            kde_est,
            var_name="linpos",
            bandwidth=PARAM_BW,
            smp_space=PARAM_SMP_SPACE,
            weight_name="S",
        )
        .rename(columns={"linpos": "fr"})
        .reset_index()
    )
    fr_df = fr.merge(occp, how="left", on=["animal", "session", "trial", "smp_space"])[
        ["animal", "session", "trial", "unit_id", "smp_space", "fr", "occp"]
    ]
    fr_df = fr_df[fr_df["occp"].notnull()]
    fr_df["fr"] = fr_df["fr"].fillna(0)
    fr_df["fr_norm"] = np.nan_to_num(fr_df["fr"] / fr_df["occp"], posinf=0)
    fr_ls.append(fr_df)
fr_df = pd.concat(fr_ls, ignore_index=True)
fr_df.to_feather(os.path.join(OUT_PATH, "fr.feat"))
stb_df = (
    fr_df.groupby(["animal", "session", "unit_id"])
    .apply(compute_stb)
    .rename("stb")
    .reset_index()
)
fr_agg = (
    fr_df.groupby(["animal", "session", "unit_id", "smp_space"])
    .agg({"fr_norm": "mean", "occp": "mean", "fr": "mean"})
    .reset_index()
)
pos_df = (
    fr_agg.groupby(["animal", "session", "unit_id"])
    .apply(find_peak_field)
    .rename("peak")
    .reset_index()
)
si_df = (
    fr_agg.groupby(["animal", "session", "unit_id"])
    .apply(compute_si)
    .rename("si")
    .reset_index()
)
metric_df = stb_df.merge(
    pos_df, on=["animal", "session", "unit_id"], validate="one_to_one"
).merge(si_df, on=["animal", "session", "unit_id"], validate="one_to_one")
metric_df.to_feather(os.path.join(OUT_PATH, "metric.feat"))

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
fr_df = pd.read_feather(os.path.join(OUT_PATH, "fr.feat"))
metric_df = pd.read_feather(os.path.join(OUT_PATH, "metric.feat"))
metric_df["valid"] = np.logical_and(
    metric_df["stb"] > PARAM_STB_THRES, metric_df["si"] > PARAM_SI_THRES
)
metric_df = metric_df.set_index(["animal", "session", "unit_id"])
fr_df = (
    fr_df.groupby(["animal", "session", "unit_id", "smp_space"])["fr_norm"]
    .mean()
    .reset_index()
    .set_index(["animal", "session", "unit_id"])
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
            tdist = np.abs(
                (ss_csv.loc[anm, ssA]["date"] - ss_csv.loc[anm, ssB]["date"]).days
            )
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
            mp = mp[(mp[[ssA, ssB]] >= 0).all(axis="columns")]
            if mmethod == "red/raw" or len(mp) == 0:
                continue
            meA = metric_df.loc[anm, ssA, mp[ssA].values].reset_index()
            meB = metric_df.loc[anm, ssB, mp[ssB].values].reset_index()
            me = meA.join(meB, lsuffix="A", rsuffix="B")
            me = me[me[["validA", "validB"]].any(axis="columns")]
            if len(me) > 0:
                uidA, uidB = np.array(me["unit_idA"]), np.array(me["unit_idB"])
                frA = (
                    fr_df.loc[anm, ssA, uidA]
                    .reset_index()
                    .set_index(["unit_id", "smp_space"])["fr_norm"]
                    .to_xarray()
                )
                frB = (
                    fr_df.loc[anm, ssB, uidB]
                    .reset_index()
                    .set_index(["unit_id", "smp_space"])["fr_norm"]
                    .to_xarray()
                )
                r_all = nan_corr(np.array(frA), np.array(frB))
                r_cell = corr_mat(np.array(frA), np.array(frB), agg_axis=1)
                pv_corr_ls.append(
                    pd.DataFrame(
                        {
                            "map_method": mmethod,
                            "animal": anm,
                            "ssA": ssA,
                            "ssB": ssB,
                            "uidA": np.concatenate([["mat"], uidA]),
                            "uidB": np.concatenate([["mat"], uidB]),
                            "tdist": tdist,
                            "corr": np.concatenate([[r_all], r_cell]),
                        }
                    )
                )
pv_corr = pd.concat(pv_corr_ls, ignore_index=True)
ovlp = pd.concat(ovlp_ls, ignore_index=True)
pv_corr.to_csv(os.path.join(OUT_PATH, "pv_corr.csv"), index=False)
ovlp.to_csv(os.path.join(OUT_PATH, "ovlp.csv"), index=False)

#%% plot pv corr
cmap = {
    "green/raw": qualitative.Plotly[2],
    "red/registered": qualitative.Plotly[4],
}
lmap = {
    "green/raw": "All GCaMP cells",
    "red/registered": "GCaMP cells\nregistered with tdTomato",
}
pv_corr = pd.read_csv(os.path.join(OUT_PATH, "pv_corr.csv"))
pv_corr = pv_corr[pv_corr["animal"].isin(PARAM_SUB_ANM)].copy()
pv_corr_med = (
    pv_corr[pv_corr["uidA"] != "mat"]
    .groupby(["map_method", "animal", "ssA", "ssB", "tdist"])["corr"]
    .median()
    .reset_index()
)
pv_corr["color"] = pv_corr["map_method"].map(cmap)
pv_corr["map_method"] = pv_corr["map_method"].map(lmap)
pv_corr_med["color"] = pv_corr_med["map_method"].map(cmap)
pv_corr_med["map_method"] = pv_corr_med["map_method"].map(lmap)
corr_dict = {
    "by_cell": pv_corr[pv_corr["uidA"] != "mat"].copy(),
    "by_session": pv_corr[pv_corr["uidA"] == "mat"].copy(),
    "by_cell_med": pv_corr_med,
}
for by, cur_corr in corr_dict.items():
    fig = scatter_agg(
        cur_corr,
        x="tdist",
        y="corr",
        facet_row=None,
        facet_col="animal",
        col_wrap=3,
        legend_dim="map_method",
        marker={"color": "color"},
    )
    fig.update_xaxes(title="Days apart")
    fig.update_yaxes(range=(0, 1), title="PV correlation")
    fig.write_html(os.path.join(FIG_PATH, "pv_corr-{}.html".format(by)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.lineplot(
        cur_corr,
        x="tdist",
        y="corr",
        hue="map_method",
        palette={lmap[k]: v for k, v in cmap.items()},
        errorbar="se",
        ax=ax,
        zorder=5,
    )
    ax = sns.swarmplot(
        cur_corr,
        x="tdist",
        y="corr",
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
    fig.savefig(os.path.join(FIG_PATH, "pv_corr-{}.svg".format(by)), dpi=500)
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
ovlp = ovlp[ovlp["animal"].isin(PARAM_SUB_ANM)].copy()
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
    fig, ax = plt.subplots(figsize=(6, 4))
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
df = pd.read_csv(os.path.join(OUT_PATH, "pv_corr.csv"))
df = df[(df["animal"].isin(PARAM_SUB_ANM)) & (df["uidA"] == "mat")].copy()
lm = ols("corr ~ C(map_method)*tdist", data=df).fit(cov_type="HC1")
anova = sm.stats.anova_lm(lm, typ=3)

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
metric_df = pd.read_feather(os.path.join(OUT_PATH, "metric.feat"))
metric_df["valid"] = np.logical_and(
    metric_df["stb"] > PARAM_STB_THRES, metric_df["si"] > PARAM_SI_THRES
)
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
        cur_mmap["valid"] = cur_met["valid"].values
        cur_mmap["peak"] = cur_met["peak"].values
        cur_mmap = (
            cur_mmap[cur_mmap["valid"]].sort_values("peak").reset_index(drop=True)
        )
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
    aspect=0.5,
)
g.map(plot_fr, "fr_mat")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
for ax in g.axes[:, -1]:
    if ax.texts:
        for tx in ax.texts:
            x, y = tx.get_unitless_position()
            tx.set(
                horizontalalignment="center",
                x=x + 0.3,
                text=tx.get_text().partition(" x ")[0],
            )
for crd in [(0, 0), (1, -1), (2, 0), (3, -1)]:
    ax = g.axes[crd]
    ax.spines[:].set_visible(True)
    ax.spines[:].set_linewidth(3)
    ax.spines[:].set_linestyle(":")
    ax.spines[:].set_color("dimgray")
    ax.spines[:].set_position(("outward", 1.6))
fig = g.fig
fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.08)
fig.savefig(os.path.join(FIG_PATH, "drifting_cells.svg"), dpi=500, bbox_inches="tight")
