#%% imports and definition
import itertools as itt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from routine.utilities import nan_corr
from scipy.stats import ttest_ind
from tqdm.auto import tqdm

IN_PACTIVE = "./intermediate/register_g2r/green_reg_pactive.csv"
IN_PLC_METRIC = "./intermediate/drift/metric.feat"
IN_PLC_FR = "./intermediate/drift/fr.feat"
IN_PS_PATH = "./intermediate/processed/green"
IN_SS_CSV = "./log/sessions.csv"
PARAM_SUB_ANM = ["m12", "m15", "m16"]
OUT_PATH = "./intermediate/hub_cells"
FIG_PATH = "./figs/hub_cells"

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

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

#%% load
def classify_cells(p):
    if p <= 4 / 9:
        return "drift"
    elif p >= 8 / 9:
        return "hub"
    else:
        return "unclassified"


pactive = pd.read_csv(IN_PACTIVE)
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
fig, ax = plt.subplots()
sns.ecdfplot(cell_df, x="peak", hue="cell_type", ax=ax)
ax.set_xlabel("Peak Location")
fig.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "peak_cdf.svg"), bbox_inches="tight")

#%% plot metrics
def plot_metrics(df, x="cell_type", y="si"):
    fig, ax = plt.subplots()
    sns.swarmplot(
        df,
        x=x,
        y=y,
        hue=x,
        ax=ax,
        alpha=0.3,
        warn_thresh=0.8,
    )
    sns.pointplot(
        df,
        x=x,
        y=y,
        hue=x,
        ax=ax,
        errorbar="se",
        capsize=0.03,
        scale=0.5,
    )
    fig.tight_layout()
    return fig


cell_df = pd.read_csv(os.path.join(OUT_PATH, "cell_df.csv"))
pactive = pd.read_csv(os.path.join(OUT_PATH, "pactive.csv"))
cell_df = cell_df[cell_df["animal"].isin(PARAM_SUB_ANM)].copy()
pactive = pactive[pactive["animal"].isin(PARAM_SUB_ANM)].copy()
cell_df_agg = (
    cell_df.groupby(["animal", "master_uid", "cell_type"])
    .agg({"stb": "median", "si": "median", "Sqt": "median"})
    .reset_index()
)
for met in ["si", "stb", "Sqt"]:
    fig = plot_metrics(cell_df_agg, y=met)
    ts, pval = ttest_ind(
        cell_df_agg.loc[cell_df_agg["cell_type"] == "drift", met].dropna(),
        cell_df_agg.loc[cell_df_agg["cell_type"] == "hub", met].dropna(),
    )
    print("metric: {}, ts: {}, pv: {}".format(met, ts, pval))
    fig.savefig(os.path.join(FIG_PATH, "{}.svg".format(met)), bbox_inches="tight")

fig = plot_metrics(pactive, y="corr_med")
ts, pval = ttest_ind(
    pactive.loc[pactive["cell_type"] == "drift", "corr_med"].dropna(),
    pactive.loc[pactive["cell_type"] == "hub", "corr_med"].dropna(),
)
print("metric: corr_med, ts: {}, pv: {}".format(ts, pval))
fig.savefig(os.path.join(FIG_PATH, "corr_med.svg"), bbox_inches="tight")
