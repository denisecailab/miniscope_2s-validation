#%% imports and definition
import itertools as itt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from plotly.express.colors import qualitative
from routine.utilities import nan_corr
from scipy.stats import ttest_ind
from tqdm.auto import tqdm

IN_PACTIVE = "./intermediate/register_g2r/green_reg_pactive.csv"
IN_PLC_METRIC = "./intermediate/drift/metric.feat"
IN_PLC_FR = "./intermediate/drift/fr.feat"
IN_PS_PATH = "./intermediate/processed/green"
IN_SS_CSV = "./log/sessions.csv"
PARAM_SUB_ANM = ["m12", "m15", "m16"]
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
PARAM_CMAP = {"hub": qualitative.Dark24[0], "drift": qualitative.Dark24[1]}
OUT_PATH = "./intermediate/hub_cells"
FIG_PATH = "./figs/hub_cells"

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)
plt.rcParams.update(**PARAM_PLT_RC)

#%% compute act quantile
meta_df = pd.read_csv(IN_SS_CSV)
meta_df = meta_df[meta_df["analyze"]]
Sqt_df = []
for _, row in tqdm(list(meta_df.iterrows())):
    anm, ss = row["animal"], row["name"]
    curds = xr.open_dataset(os.path.join(IN_PS_PATH, "{}-{}.nc".format(anm, ss)))
    S_mean = (
        curds["S"]
        .mean("frame")
        .rename("Smean")
        .squeeze()
        .to_dataframe()
        .reset_index()
        .sort_values("Smean")
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "Sqt"})
    )
    S_mean["Sqt"] = S_mean["Sqt"] / len(S_mean)
    Sqt_df.append(S_mean)
Sqt_df = pd.concat(Sqt_df, ignore_index=True)
Sqt_df.to_csv(os.path.join(OUT_PATH, "Sqt.csv"), index=False)

#%% load and classify cells
def classify_cells(p):
    if p <= 5 / 9:
        return "drift"
    elif p >= 8 / 9:
        return "hub"
    else:
        return "unclassified"


pactive = pd.read_csv(IN_PACTIVE)
pactive = pactive[pactive["npresent"] == 9].copy()
metric = pd.read_feather(IN_PLC_METRIC)
Sqt_df = pd.read_csv(os.path.join(OUT_PATH, "Sqt.csv"))
fr = (
    pd.read_feather(IN_PLC_FR)
    .groupby(["animal", "session", "unit_id", "smp_space"])["fr_norm"]
    .mean()
    .reset_index()
    .sort_values(["animal", "session", "unit_id", "smp_space"])
    .set_index(["animal", "session", "unit_id"])
)
pactive = pactive[(pactive["nactive"] > 1)]
pactive["cell_type"] = pactive["pactive"].map(classify_cells)
pactive = pactive[pactive["cell_type"] != "unclassified"].copy()
sess = list(filter(lambda c: c.startswith("rec"), pactive.columns))
cell_df = []
for idx, row in tqdm(list(pactive.iterrows())):
    anm = row["animal"]
    cur_ss = row[sess]
    cur_ss = cur_ss[cur_ss >= 0].to_dict()
    cell_df.append(
        pd.DataFrame(
            {
                "animal": anm,
                "master_uid": row.name,
                "session": list(cur_ss.keys()),
                "unit_id": list(cur_ss.values()),
                "cell_type": row["cell_type"],
            }
        )
    )
    corr_ls = []
    for (ssA, uidA), (ssB, uidB) in itt.combinations(cur_ss.items(), 2):
        frA = np.array(fr.loc[(anm, ssA, uidA), "fr_norm"])
        frB = np.array(fr.loc[(anm, ssB, uidB), "fr_norm"])
        corr_ls.append(nan_corr(frA, frB))
    pactive.loc[idx, "corr_med"] = np.nanmedian(corr_ls)
cell_df = (
    pd.concat(cell_df, ignore_index=True)
    .merge(
        metric, on=["animal", "session", "unit_id"], how="left", validate="one_to_one"
    )
    .merge(
        Sqt_df, on=["animal", "session", "unit_id"], how="left", validate="one_to_one"
    )
)
cell_df.to_csv(os.path.join(OUT_PATH, "cell_df.csv"), index=False)
pactive.to_csv(os.path.join(OUT_PATH, "pactive.csv"), index=False)

#%% plot peak
cell_df = pd.read_csv(os.path.join(OUT_PATH, "cell_df.csv"))
cell_df = cell_df[cell_df["animal"].isin(PARAM_SUB_ANM)].copy()
fig, ax = plt.subplots(figsize=(4, 2))
ax = sns.ecdfplot(cell_df, x="peak", hue="cell_type", ax=ax, palette=PARAM_CMAP)
ax.get_legend().set_title("")
ax.set_xlabel("Place Field Location", style="italic")
ax.set_ylabel("Proportion", style="italic")
fig.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "peak_cdf.svg"), bbox_inches="tight")

#%% plot metrics
def plot_metrics(df, x="cell_type", y="si", ax=None):
    ax = sns.pointplot(
        df,
        x=x,
        y=y,
        hue=x,
        ax=ax,
        errorbar="se",
        capsize=0.04,
        errwidth=1.2,
        scale=0.35,
        palette=PARAM_CMAP,
    )
    plt.setp(ax.lines, zorder=5)
    plt.setp(ax.collections, zorder=5)
    ax = sns.swarmplot(
        df,
        x=x,
        y=y,
        hue=x,
        ax=ax,
        alpha=0.2,
        size=3,
        linewidth=1,
        edgecolor="gray",
        warn_thresh=0.8,
        legend=False,
        palette=PARAM_CMAP,
    )
    return ax


cell_df = pd.read_csv(os.path.join(OUT_PATH, "cell_df.csv"))
pactive = pd.read_csv(os.path.join(OUT_PATH, "pactive.csv"))
cell_df = cell_df[cell_df["animal"].isin(PARAM_SUB_ANM)].copy()
pactive = pactive[pactive["animal"].isin(PARAM_SUB_ANM)].copy()
cell_df_agg = (
    cell_df.groupby(["animal", "master_uid", "cell_type"])
    .agg({"stb": "median", "si": "median", "Sqt": "median"})
    .reset_index()
)
fig, axs = plt.subplot_mosaic([["Sqt", "corr_med"], ["si", "stb"]], figsize=(4, 4))
dfs = {"Sqt": cell_df_agg, "corr_med": pactive, "si": cell_df_agg, "stb": cell_df_agg}
ylabs = {
    "Sqt": "Activity",
    "corr_med": "Cross-day Stability",
    "si": "Spatial Information",
    "stb": "Within-day Stability",
}
for met, ax in axs.items():
    dat = dfs[met]
    ax = plot_metrics(dat, y=met, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(ylabs[met], style="italic")
    ax.get_legend().remove()
    ts, pval = ttest_ind(
        dat.loc[dat["cell_type"] == "drift", met].dropna(),
        dat.loc[dat["cell_type"] == "hub", met].dropna(),
    )
    print("metric: {}, ts: {}, pv: {}".format(met, ts, pval))
fig.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "metrics.svg"), bbox_inches="tight")
